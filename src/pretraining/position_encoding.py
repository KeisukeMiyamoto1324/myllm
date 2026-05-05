import torch
import torch.nn as nn


class PositionEncoding(nn.Module):
    def __init__(self, d_model: int = 2, max_len: int = 6) -> None:
        super().__init__()

        # ---------------------------------------------------------
        # Precompute sinusoidal positions once so token embeddings
        # can be shifted cheaply during training and inference.
        # ---------------------------------------------------------
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(start=0, end=max_len, step=1).float().unsqueeze(1)
        embedding_index = torch.arange(start=0, end=d_model, step=2).float()
        div_term = 1 / torch.tensor(10000.0) ** (embedding_index / d_model)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, word_embeddings: torch.Tensor, position_offset: int = 0) -> torch.Tensor:
        # ---------------------------------------------------------
        # Add positions for the visible slice, starting at the cache
        # length when incremental inference supplies an offset.
        # ---------------------------------------------------------
        seq_len = word_embeddings.size(1)
        position_end = position_offset + seq_len
        return word_embeddings + self.pe[position_offset:position_end, :].unsqueeze(0)


if __name__ == "__main__":
    n = PositionEncoding()
