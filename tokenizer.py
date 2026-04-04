import torch


vocabulary = ["what", "is", "statquest", "awesome", "<EOS>"]
token_map = {w: i for i, w in enumerate(vocabulary)}

def tokenizer(words: list[str]) -> torch.Tensor:
    return torch.tensor([token_map[w] for w in words])
