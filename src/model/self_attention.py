import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.kv_cache import LayerKeyValueCache


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
        encoding_for_q: torch.Tensor,
        encoding_for_k: torch.Tensor,
        encoding_for_v: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # ---------------------------------------------------------
        # Create the projected queries, keys, and values for each
        # attention head from the incoming hidden states.
        # ---------------------------------------------------------
        q = self._split_heads(self.W_q(encoding_for_q))
        k = self._split_heads(self.W_k(encoding_for_k))
        v = self._split_heads(self.W_v(encoding_for_v))

        # ---------------------------------------------------------
        # Compute scaled dot-product attention and block future tokens
        # with the causal mask when one is supplied.
        # ---------------------------------------------------------
        sims = torch.matmul(q, k.transpose(-2, -1))
        scaled_sims = sims / math.sqrt(self.head_dim)

        if mask is not None:
            scaled_sims = scaled_sims.masked_fill(mask=mask, value=torch.finfo(scaled_sims.dtype).min)

        attention_percents = F.softmax(scaled_sims, dim=-1)
        attention_scores = torch.matmul(attention_percents, v)

        # ---------------------------------------------------------
        # Merge the attended heads and project the result back into
        # the model dimension for the next layer.
        # ---------------------------------------------------------
        merged_scores = self._merge_heads(attention_scores)
        return self.W_o(merged_scores)

    def forward_with_cache(
        self,
        encoding_for_q: torch.Tensor,
        encoding_for_k: torch.Tensor,
        encoding_for_v: torch.Tensor,
        past_key_value: LayerKeyValueCache | None,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, LayerKeyValueCache]:
        # ---------------------------------------------------------
        # Project the current tokens and append previous keys and
        # values so generation can avoid recomputing old states.
        # ---------------------------------------------------------
        q = self._split_heads(self.W_q(encoding_for_q))
        current_k = self._split_heads(self.W_k(encoding_for_k))
        current_v = self._split_heads(self.W_v(encoding_for_v))

        k = current_k
        v = current_v

        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat((past_k, current_k), dim=2)
            v = torch.cat((past_v, current_v), dim=2)

        # ---------------------------------------------------------
        # Attend the current query positions over cached and current
        # keys, preserving the same masking behavior as full forward.
        # ---------------------------------------------------------
        sims = torch.matmul(q, k.transpose(-2, -1))
        scaled_sims = sims / math.sqrt(self.head_dim)

        if mask is not None:
            scaled_sims = scaled_sims.masked_fill(mask=mask, value=torch.finfo(scaled_sims.dtype).min)

        attention_percents = F.softmax(scaled_sims, dim=-1)
        attention_scores = torch.matmul(attention_percents, v)

        # ---------------------------------------------------------
        # Return both the attention result and the updated cache for
        # this layer so the caller can feed the next token directly.
        # ---------------------------------------------------------
        merged_scores = self._merge_heads(attention_scores)
        return self.W_o(merged_scores), (k, v)
