import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_qkvpacked_func
    from flash_attn import flash_attn_varlen_qkvpacked_func
except ModuleNotFoundError:
    flash_attn_qkvpacked_func = None
    flash_attn_varlen_qkvpacked_func = None

from src.shared.model.kv_cache import LayerKeyValueCache

RotaryPositionCache = tuple[torch.Tensor, torch.Tensor]


class Attention(nn.Module):
    def __init__(self, d_model: int = 2, num_heads: int = 1) -> None:
        super().__init__()

        # ---------------------------------------------------------
        # Split the model dimension into multiple heads so the same
        # attention module can be reused in a more general structure.
        # ---------------------------------------------------------
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for rotary positional embeddings")

        # ---------------------------------------------------------
        # Project inputs into query, key, and value spaces and merge
        # the heads back into the model dimension after attention.
        # ---------------------------------------------------------
        self.W_qkv = nn.Linear(in_features=d_model, out_features=d_model * 3, bias=False)
        self.W_o = nn.Linear(in_features=d_model, out_features=d_model, bias=False)

    def _apply_rotary_position(
        self,
        x: torch.Tensor,
        rotary_position_cache: RotaryPositionCache,
    ) -> torch.Tensor:
        # ---------------------------------------------------------
        # Rotate query or key tensors in the FlashAttention layout:
        # sequence dimensions first, then heads, then head features.
        # ---------------------------------------------------------
        cosine, sine = rotary_position_cache
        cosine = cosine.to(device=x.device, dtype=x.dtype)
        sine = sine.to(device=x.device, dtype=x.dtype)
        even_values = x[..., 0::2]
        odd_values = x[..., 1::2]
        rotated = torch.stack(
            (
                even_values * cosine - odd_values * sine,
                even_values * sine + odd_values * cosine,
            ),
            dim=-1,
        )
        return rotated.flatten(start_dim=-2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_position_cache: RotaryPositionCache,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        if flash_attn_varlen_qkvpacked_func is None:
            raise RuntimeError("flash-attn is required for FlashAttention-2 varlen training")

        # ---------------------------------------------------------
        # Build packed QKV directly in the varlen FlashAttention
        # layout and rotate only the query and key slices.
        # ---------------------------------------------------------
        qkv = self.W_qkv(hidden_states).view(-1, 3, self.num_heads, self.head_dim)
        qkv[:, 0] = self._apply_rotary_position(
            qkv[:, 0],
            rotary_position_cache=rotary_position_cache,
        )
        qkv[:, 1] = self._apply_rotary_position(
            qkv[:, 1],
            rotary_position_cache=rotary_position_cache,
        )

        # ---------------------------------------------------------
        # Run causal varlen FlashAttention on packed tokens without
        # creating an explicit attention mask.
        # ---------------------------------------------------------
        attention_scores = flash_attn_varlen_qkvpacked_func(
            qkv,
            cu_seqlens,
            max_seqlen,
            dropout_p=0.0,
            causal=True,
        )
        merged_scores = attention_scores.reshape(-1, self.d_model)
        return self.W_o(merged_scores)

    def forward_with_flash_attention(
        self,
        hidden_states: torch.Tensor,
        rotary_position_cache: RotaryPositionCache,
    ) -> torch.Tensor:
        if flash_attn_qkvpacked_func is None:
            raise RuntimeError("flash-attn is required for FlashAttention-2 inference")

        # ---------------------------------------------------------
        # Build packed QKV in full-sequence FlashAttention layout and
        # keep RoPE broadcast on the sequence axis, not the head axis.
        # ---------------------------------------------------------
        batch_size, seq_len, _ = hidden_states.size()
        qkv = self.W_qkv(hidden_states).view(
            batch_size,
            seq_len,
            3,
            self.num_heads,
            self.head_dim,
        )
        qkv[:, :, 0] = self._apply_rotary_position(
            qkv[:, :, 0],
            rotary_position_cache=rotary_position_cache,
        )
        qkv[:, :, 1] = self._apply_rotary_position(
            qkv[:, :, 1],
            rotary_position_cache=rotary_position_cache,
        )

        # ---------------------------------------------------------
        # Run full-sequence causal FlashAttention and merge the head
        # dimension back into the model width.
        # ---------------------------------------------------------
        attention_scores = flash_attn_qkvpacked_func(
            qkv,
            dropout_p=0.0,
            causal=True,
        )
        merged_scores = attention_scores.reshape(batch_size, seq_len, self.d_model)
        return self.W_o(merged_scores)

    def forward_with_cache(
        self,
        hidden_states: torch.Tensor,
        rotary_position_cache: RotaryPositionCache,
        past_key_value: LayerKeyValueCache | None,
        is_causal: bool = False,
    ) -> tuple[torch.Tensor, LayerKeyValueCache]:
        # ---------------------------------------------------------
        # Build packed QKV in the same sequence-major layout used by
        # FlashAttention, then keep rotated keys in that cache layout.
        # ---------------------------------------------------------
        batch_size, seq_len, _ = hidden_states.size()
        qkv = self.W_qkv(hidden_states).view(
            batch_size,
            seq_len,
            3,
            self.num_heads,
            self.head_dim,
        )
        q = self._apply_rotary_position(qkv[:, :, 0], rotary_position_cache=rotary_position_cache)
        current_k = self._apply_rotary_position(
            qkv[:, :, 1],
            rotary_position_cache=rotary_position_cache,
        )
        current_v = qkv[:, :, 2]

        k = current_k
        v = current_v

        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat((past_k, current_k), dim=1)
            v = torch.cat((past_v, current_v), dim=1)

        # ---------------------------------------------------------
        # Keep the cache in FlashAttention layout, then transpose only
        # at the PyTorch attention boundary used for token decoding.
        # ---------------------------------------------------------
        attention_scores = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            is_causal=is_causal,
        )

        merged_scores = attention_scores.transpose(1, 2).contiguous().view(
            batch_size,
            seq_len,
            self.d_model,
        )
        return self.W_o(merged_scores), (k, v)
