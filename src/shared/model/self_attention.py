import torch
import torch.nn as nn
import torch.nn.functional as F

from src.shared.model.kv_cache import LayerKeyValueCache


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

        # ---------------------------------------------------------
        # Project inputs into query, key, and value spaces and merge
        # the heads back into the model dimension after attention.
        # ---------------------------------------------------------
        self.W_q = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.W_k = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.W_v = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.W_o = nn.Linear(in_features=d_model, out_features=d_model, bias=False)

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

    def forward(
        self,
        hidden_states: torch.Tensor,
        is_causal: bool = False,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # ---------------------------------------------------------
        # Create the projected queries, keys, and values for each
        # attention head from the incoming hidden states.
        # ---------------------------------------------------------
        q = self._split_heads(self.W_q(hidden_states))
        k = self._split_heads(self.W_k(hidden_states))
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

    def forward_with_sdpa(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # ---------------------------------------------------------
        # Use the same PyTorch SDPA path without a packed mask for
        # ordinary full-sequence inference.
        # ---------------------------------------------------------
        return self.forward(
            hidden_states=hidden_states,
            is_causal=True,
        )

    def forward_with_cache(
        self,
        hidden_states: torch.Tensor,
        past_key_value: LayerKeyValueCache | None,
        is_causal: bool = False,
    ) -> tuple[torch.Tensor, LayerKeyValueCache]:
        # ---------------------------------------------------------
        # Project the current tokens and append previous keys and
        # values so generation can avoid recomputing old states.
        # ---------------------------------------------------------
        q = self._split_heads(self.W_q(hidden_states))
        current_k = self._split_heads(self.W_k(hidden_states))
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
