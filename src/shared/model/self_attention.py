import torch
import torch.nn as nn
import torch.nn.functional as F

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
        # Rotate query or key tensors in sequence-major layout:
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
        is_causal: bool = False,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # ---------------------------------------------------------
        # Build batched QKV tensors and apply RoPE before entering
        # PyTorch SDPA with an optional packed-segment mask.
        # ---------------------------------------------------------
        batch_size, seq_len, _ = hidden_states.size()
        qkv = self.W_qkv(hidden_states).view(
            batch_size,
            seq_len,
            3,
            self.num_heads,
            self.head_dim,
        )
        q = self._apply_rotary_position(
            qkv[:, :, 0],
            rotary_position_cache=rotary_position_cache,
        )
        k = self._apply_rotary_position(
            qkv[:, :, 1],
            rotary_position_cache=rotary_position_cache,
        )
        v = qkv[:, :, 2]

        # ---------------------------------------------------------
        # Keep computation in the fixed batch layout so memory is
        # bounded by batch_size and max_len, not document_count.
        # ---------------------------------------------------------
        attention_scores = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            attn_mask=attention_mask,
            is_causal=is_causal,
        )
        merged_scores = attention_scores.transpose(1, 2).contiguous().view(
            batch_size,
            seq_len,
            self.d_model,
        )
        return self.W_o(merged_scores)

    def forward_with_sdpa(
        self,
        hidden_states: torch.Tensor,
        rotary_position_cache: RotaryPositionCache,
    ) -> torch.Tensor:
        # ---------------------------------------------------------
        # Use the same PyTorch SDPA path without a packed mask for
        # ordinary full-sequence inference.
        # ---------------------------------------------------------
        return self.forward(
            hidden_states=hidden_states,
            rotary_position_cache=rotary_position_cache,
            is_causal=True,
        )

    def forward_with_cache(
        self,
        hidden_states: torch.Tensor,
        rotary_position_cache: RotaryPositionCache,
        past_key_value: LayerKeyValueCache | None,
        is_causal: bool = False,
    ) -> tuple[torch.Tensor, LayerKeyValueCache]:
        # ---------------------------------------------------------
        # Build packed QKV in sequence-major layout, then keep
        # rotated keys in that cache layout.
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
        # Keep the cache in sequence-major layout, then transpose only
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
