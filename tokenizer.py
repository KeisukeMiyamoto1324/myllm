import torch
import json
from pathlib import Path


class Tokenizer:
    def __init__(self):
        # ---------------------------------------------------------
        # Keep the unknown token as a fixed special token so
        # unseen words can be encoded safely at inference time.
        # ---------------------------------------------------------
        self.unknown_token: str = "<UNK>"
        self.vocabulary: set[str] = set()
        self.token_map: dict[str, int] = {}
        self.id_to_token: list[str] = []
    
    def learn_vocab(self, sentences: list[str]) -> None:
        # ---------------------------------------------------------
        # Learn the base vocabulary from the training sentences and
        # reserve the unknown token in the saved vocabulary.
        # ---------------------------------------------------------
        self.vocabulary = {word for sentence in sentences for word in sentence.split()}
        self.vocabulary.add(self.unknown_token)
        self.id_to_token = sorted(self.vocabulary)
        self.token_map = {word: i for i, word in enumerate(self.id_to_token)}

    def tokenizer(self, words: str) -> torch.Tensor:
        # ---------------------------------------------------------
        # Replace unseen words with the unknown token id so
        # tokenization stays valid for arbitrary input text.
        # ---------------------------------------------------------
        unknown_token_id = self.token_map[self.unknown_token]
        return torch.tensor([self.token_map.get(word, unknown_token_id) for word in words.split()])

    def detokenizer(self, tokens: torch.Tensor) -> list[str]:
        # ---------------------------------------------------------
        # Convert token ids back into their vocabulary strings in
        # the saved tokenizer order.
        # ---------------------------------------------------------
        return [self.id_to_token[token.item()] for token in tokens]

    def save(self, path: str | Path) -> None:
        # ---------------------------------------------------------
        # Save the learned vocabulary order so token ids stay
        # consistent between training and inference.
        # ---------------------------------------------------------
        with open(path, "w") as f:
            json.dump({"id_to_token": self.id_to_token}, f)

    @classmethod
    def load(cls, path: str | Path) -> "Tokenizer":
        # ---------------------------------------------------------
        # Restore the tokenizer state exactly as saved so model
        # weights and token ids remain aligned.
        # ---------------------------------------------------------
        with open(path) as f:
            data = json.load(f)

        tokenizer = cls()
        tokenizer.id_to_token = data["id_to_token"]

        # ---------------------------------------------------------
        # Reject tokenizer artifacts that were saved before the
        # unknown token was introduced in the vocabulary.
        # ---------------------------------------------------------
        if tokenizer.unknown_token not in tokenizer.id_to_token:
            raise ValueError("Tokenizer artifact must include <UNK>. Retrain the tokenizer and model.")

        tokenizer.vocabulary = set(tokenizer.id_to_token)
        tokenizer.token_map = {token: i for i, token in enumerate(tokenizer.id_to_token)}
        return tokenizer
