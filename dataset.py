import torch
from torch.utils.data import TensorDataset
from typing import Callable


def get_dataset(
    sentences: list[str], 
    token_map: dict[str, int], 
    tokenizer: Callable[[list[str], dict[str, int]], list[int]]
) -> TensorDataset:
    inputs = torch.tensor([tokenizer(sentence.split(), token_map) for sentence in sentences])
    labels = torch.tensor([tokenizer(sentence.split()[1:]+["<EOS>"], token_map) for sentence in sentences])
    
    return TensorDataset(inputs, labels)

