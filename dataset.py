import torch
from torch.utils.data import TensorDataset
from typing import Callable


def get_dataset(
    sentences: list[str], 
    tokenizer: Callable[[list[str], dict[str, int]], torch.Tensor]
) -> TensorDataset:
    inputs = torch.stack([
        tokenizer(sentence) 
        for sentence in sentences
    ])

    labels = torch.stack([
        torch.cat([
            tokenizer(sentence)[1:],
            tokenizer("<EOS>")
        ])
        for sentence in sentences
    ])
    
    return TensorDataset(inputs, labels)
    