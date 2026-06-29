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

    def _run_packed_sdpa(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        # ---------------------------------------------------------
        # Use a direct view when every document already has the same
        # length, avoiding scatter overhead for fixed-size batches.
        # ---------------------------------------------------------
        document_lengths = cu_seqlens.diff().to(dtype=torch.long)
        document_count = int(document_lengths.numel())
        total_tokens = q.size(dim=0)

        if total_tokens == document_count * max_seqlen:
            q_padded = q.view(document_count, max_seqlen, self.num_heads, self.head_dim)
            k_padded = k.view(document_count, max_seqlen, self.num_heads, self.head_dim)
            v_padded = v.view(document_count, max_seqlen, self.num_heads, self.head_dim)
            attention_scores = F.scaled_dot_product_attention(
                q_padded.transpose(1, 2),
                k_padded.transpose(1, 2),
                v_padded.transpose(1, 2),
                is_causal=True,
            )
            return attention_scores.transpose(1, 2).contiguous().view(
                total_tokens,
                self.num_heads,
                self.head_dim,
            )

        # ---------------------------------------------------------
        # Scatter ragged packed documents into a padded batch so SDPA
        # can use causal kernels without a dense block-diagonal mask.
        # ---------------------------------------------------------
        starts = cu_seqlens[:-1].to(device=q.device, dtype=torch.long)
        document_ids = torch.repeat_interleave(
            torch.arange(document_count, device=q.device),
            document_lengths.to(device=q.device),
        )
        token_positions = torch.arange(total_tokens, device=q.device) - torch.repeat_interleave(
            starts,
            document_lengths.to(device=q.device),
        )
        padded_shape = (document_count, max_seqlen, self.num_heads, self.head_dim)
        q_padded = q.new_zeros(padded_shape)
        k_padded = k.new_zeros(padded_shape)
        v_padded = v.new_zeros(padded_shape)
        q_padded[document_ids, token_positions] = q
        k_padded[document_ids, token_positions] = k
        v_padded[document_ids, token_positions] = v

        # ---------------------------------------------------------
        # Run PyTorch SDPA on each document independently, then
        # gather only valid token rows back to the packed layout.
        # ---------------------------------------------------------
        attention_scores = F.scaled_dot_product_attention(
            q_padded.transpose(1, 2),
            k_padded.transpose(1, 2),
            v_padded.transpose(1, 2),
            is_causal=True,
        )
        packed_scores = attention_scores.transpose(1, 2).contiguous()
        return packed_scores[document_ids, token_positions]

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_position_cache: RotaryPositionCache,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        # ---------------------------------------------------------
        # Build packed QKV in sequence-major layout and rotate only
        # the query and key slices before PyTorch SDPA.
        # ---------------------------------------------------------
        qkv = self.W_qkv(hidden_states).view(-1, 3, self.num_heads, self.head_dim)
        q = self._apply_rotary_position(
            qkv[:, 0],
            rotary_position_cache=rotary_position_cache,
        )
        k = self._apply_rotary_position(
            qkv[:, 1],
            rotary_position_cache=rotary_position_cache,
        )
        v = qkv[:, 2]

        # ---------------------------------------------------------
        # Keep packed documents isolated while using PyTorch SDPA's
        # causal attention implementation for each document.
        # ---------------------------------------------------------
        attention_scores = self._run_packed_sdpa(
            q=q,
            k=k,
            v=v,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        merged_scores = attention_scores.reshape(-1, self.d_model)
        return self.W_o(merged_scores)

    def forward_with_sdpa(
        self,
        hidden_states: torch.Tensor,
        rotary_position_cache: RotaryPositionCache,
    ) -> torch.Tensor:
        # ---------------------------------------------------------
        # Build full-sequence QKV in sequence-major layout and keep
        # RoPE broadcast on the sequence axis, not the head axis.
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
        # Run full-sequence causal PyTorch SDPA and merge the head
        # dimension back into the model width.
        # ---------------------------------------------------------
        attention_scores = F.scaled_dot_product_attention(
            qkv[:, :, 0].transpose(1, 2),
            qkv[:, :, 1].transpose(1, 2),
            qkv[:, :, 2].transpose(1, 2),
            is_causal=True,
        )
        merged_scores = attention_scores.transpose(1, 2).contiguous().view(
            batch_size,
            seq_len,
            self.d_model,
        )
        return self.W_o(merged_scores)

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
