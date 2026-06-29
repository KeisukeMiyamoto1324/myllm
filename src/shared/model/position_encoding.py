import torch
import torch.nn as nn


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_len: int = 4096) -> None:
        super().__init__()

        # ---------------------------------------------------------
        # RoPE rotates pairs of head features, so each attention head
        # must expose an even number of dimensions.
        # ---------------------------------------------------------
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for rotary position embedding")

        # ---------------------------------------------------------
        # Precompute fixed trig tables once so training forwards do
        # not update module buffers or synchronize CUDA tensors.
        # ---------------------------------------------------------
        embedding_index = torch.arange(start=0, end=head_dim, step=2).float()
        inv_freq = 1 / torch.tensor(10000.0) ** (embedding_index / head_dim)
        positions = torch.arange(start=0, end=max_len, dtype=inv_freq.dtype)
        angles = positions.unsqueeze(-1) * inv_freq
        self.register_buffer("cos_cache", torch.cos(angles), persistent=False)
        self.register_buffer("sin_cache", torch.sin(angles), persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        position_offset: int = 0,
    ) -> torch.Tensor:
        # ---------------------------------------------------------
        # Resolve explicit packed positions or contiguous positions
        # for regular full-sequence and cached inference paths.
        # ---------------------------------------------------------
        seq_len = x.size(dim=2)

        if position_ids is None:
            position_ids = torch.arange(
                start=position_offset,
                end=position_offset + seq_len,
                device=x.device,
                dtype=torch.long,
            ).unsqueeze(0)

        if position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)

        position_ids = position_ids.to(device=x.device, dtype=torch.long)

        # ---------------------------------------------------------
        # Gather precomputed cos and sin rows without recomputing
        # trig functions or extending buffers inside the forward.
        # ---------------------------------------------------------
        cos = self.cos_cache[position_ids].to(dtype=x.dtype).unsqueeze(1)
        sin = self.sin_cache[position_ids].to(dtype=x.dtype).unsqueeze(1)

        # ---------------------------------------------------------
        # Rotate even and odd channels as complex-number pairs, then
        # flatten them back to the original head feature layout.
        # ---------------------------------------------------------
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        rotated = torch.stack(
            (
                x_even * cos - x_odd * sin,
                x_even * sin + x_odd * cos,
            ),
            dim=-1,
        )
        return rotated.flatten(start_dim=-2)
