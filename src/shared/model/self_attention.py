import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_varlen_qkvpacked_func
except ModuleNotFoundError:
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

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # ---------------------------------------------------------
        # Rearrange batched hidden states into attention heads for
        # the cached inference path.
        # ---------------------------------------------------------
        batch_size, seq_len, _ = x.size()
        reshaped = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return reshaped.transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # ---------------------------------------------------------
        # Restore the tensor to the original model dimension after
        # per-head attention has been combined.
        # ---------------------------------------------------------
        batch_size, _, seq_len, _ = x.size()
        transposed = x.transpose(1, 2).contiguous()
        return transposed.view(batch_size, seq_len, self.d_model)

    def _apply_rotary_position(
        self,
        x: torch.Tensor,
        rotary_position_cache: RotaryPositionCache,
    ) -> torch.Tensor:
        # ---------------------------------------------------------
        # Rotate each query/key pair with the shared RoPE cache while
        # keeping values unchanged for scaled dot-product attention.
        # ---------------------------------------------------------
        cosine, sine = rotary_position_cache
        cosine = cosine.to(device=x.device, dtype=x.dtype).unsqueeze(1)
        sine = sine.to(device=x.device, dtype=x.dtype).unsqueeze(1)
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
        # ---------------------------------------------------------
        # Project compact tokens into packed QKV and run CUDA
        # FlashAttention varlen without materializing masks.
        # ---------------------------------------------------------
        if flash_attn_varlen_qkvpacked_func is None:
            raise RuntimeError("flash-attn is required for FlashAttention-2 varlen training")

        qkv = self.W_qkv(hidden_states).view(-1, 3, self.num_heads, self.head_dim)
        q = self._apply_rotary_position(
            qkv[:, 0],
            rotary_position_cache=rotary_position_cache,
        )
        k = self._apply_rotary_position(
            qkv[:, 1],
            rotary_position_cache=rotary_position_cache,
        )
        qkv = torch.stack((q, k, qkv[:, 2]), dim=1)
        attention_scores = flash_attn_varlen_qkvpacked_func(
            qkv,
            cu_seqlens,
            max_seqlen,
            dropout_p=0.0,
            causal=True,
        )
        merged_scores = attention_scores.reshape(-1, self.d_model)
        return self.W_o(merged_scores)

    def forward_with_sdpa(
        self,
        hidden_states: torch.Tensor,
        rotary_position_cache: RotaryPositionCache,
        is_causal: bool = False,
    ) -> torch.Tensor:
        # ---------------------------------------------------------
        # Keep a standard batched attention path for non-cached full
        # sequence inference while training uses FlashAttention.
        # ---------------------------------------------------------
        qkv = self.W_qkv(hidden_states).chunk(3, dim=-1)
        q = self._apply_rotary_position(
            self._split_heads(qkv[0]),
            rotary_position_cache=rotary_position_cache,
        )
        k = self._apply_rotary_position(
            self._split_heads(qkv[1]),
            rotary_position_cache=rotary_position_cache,
        )
        v = self._split_heads(qkv[2])

        # ---------------------------------------------------------
        # Use PyTorch's fused scaled dot-product attention so large
        # score and softmax tensors do not need to be materialized.
        # ---------------------------------------------------------
        attention_scores = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=is_causal,
        )

        # ---------------------------------------------------------
        # Merge the attended heads and project the result back into
        # the model dimension for the next layer.
        # ---------------------------------------------------------
        merged_scores = self._merge_heads(attention_scores)
        return self.W_o(merged_scores)

    def forward_with_cache(
        self,
        hidden_states: torch.Tensor,
        rotary_position_cache: RotaryPositionCache,
        past_key_value: LayerKeyValueCache | None,
        is_causal: bool = False,
    ) -> tuple[torch.Tensor, LayerKeyValueCache]:
        # ---------------------------------------------------------
        # Project the current tokens, rotate the current keys, and
        # append cache entries that already include RoPE positions.
        # ---------------------------------------------------------
        qkv = self.W_qkv(hidden_states).chunk(3, dim=-1)
        q = self._apply_rotary_position(
            self._split_heads(qkv[0]),
            rotary_position_cache=rotary_position_cache,
        )
        current_k = self._apply_rotary_position(
            self._split_heads(qkv[1]),
            rotary_position_cache=rotary_position_cache,
        )
        current_v = self._split_heads(qkv[2])

        k = current_k
        v = current_v

        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat((past_k, current_k), dim=2)
            v = torch.cat((past_v, current_v), dim=2)

        # ---------------------------------------------------------
        # Attend the current query positions over cached and current
        # keys with the fused scaled dot-product implementation.
        # ---------------------------------------------------------
        attention_scores = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=is_causal,
        )

        # ---------------------------------------------------------
        # Return both the attention result and the updated cache for
        # this layer so the caller can feed the next token directly.
        # ---------------------------------------------------------
        merged_scores = self._merge_heads(attention_scores)
        return self.W_o(merged_scores), (k, v)
