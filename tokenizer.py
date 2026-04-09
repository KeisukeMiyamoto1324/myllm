import torch


class Tokenizer:
    def __init__(self):
        self.vocabulary: set[str] = set()
        self.token_map: dict[str, int] = {}
        self.id_to_token: list[str] = []
    
    def learn_vocab(self, sentences: list[str]) -> None:
        self.vocabulary = {word for sentence in sentences for word in sentence.split()}
        self.id_to_token = sorted(self.vocabulary)
        self.token_map = {word: i for i, word in enumerate(self.id_to_token)}

    def tokenizer(self, words: str) -> torch.Tensor:
        return torch.tensor([self.token_map[w] for w in words.split()])

    def detokenizer(self, tokens: torch.Tensor) -> list[str]:
        return [self.id_to_token[token.item()] for token in tokens]
