import torch
from torch.utils.data import TensorDataset
from typing import Callable


def get_dataset(
    sentences: list[str],
    tokenizer: Callable[[str], torch.Tensor],
    pad_token_id: int,
) -> TensorDataset:
    # ---------------------------------------------------------
    # Tokenize each sentence and stack them into a batch-friendly
    # tensor for training.
    # ---------------------------------------------------------
    inputs = torch.stack([tokenizer(sentence) for sentence in sentences])

    # ---------------------------------------------------------
    # Create next-token labels by shifting the inputs and placing
    # the PAD token at the last position.
    # ---------------------------------------------------------
    labels = torch.cat(
        [
            inputs[:, 1:],
            torch.full((inputs.size(0), 1), pad_token_id, dtype=inputs.dtype),
        ],
        dim=1,
    )

    return TensorDataset(inputs, labels)
