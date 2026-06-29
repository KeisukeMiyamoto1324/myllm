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
        self.W_q = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.W_k = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.W_v = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.W_o = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        rotary_indexes = torch.arange(start=0, end=self.head_dim, step=2).float()
        rotary_frequencies = 1.0 / (10000.0 ** (rotary_indexes / self.head_dim))
        self.register_buffer("rotary_frequencies", rotary_frequencies)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # ---------------------------------------------------------
        # Rearrange the last dimension into head count and head size
        # so attention can be computed independently per head.
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
        # Create the projected queries, keys, and values for each
        # attention head and apply RoPE to queries and keys.
        # ---------------------------------------------------------
        q = self._apply_rotary_position(
            self._split_heads(self.W_q(hidden_states)),
            rotary_position_cache=rotary_position_cache,
        )
        k = self._apply_rotary_position(
            self._split_heads(self.W_k(hidden_states)),
            rotary_position_cache=rotary_position_cache,
        )
        v = self._split_heads(self.W_v(hidden_states))

        # ---------------------------------------------------------
        # Use PyTorch's fused scaled dot-product attention so large
        # score and softmax tensors do not need to be materialized.
        # ---------------------------------------------------------
        attention_scores = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
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
        q = self._apply_rotary_position(
            self._split_heads(self.W_q(hidden_states)),
            rotary_position_cache=rotary_position_cache,
        )
        current_k = self._apply_rotary_position(
            self._split_heads(self.W_k(hidden_states)),
            rotary_position_cache=rotary_position_cache,
        )
        current_v = self._split_heads(self.W_v(hidden_states))

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
