
# https://www.youtube.com/watch?v=HEikzVL-lZU&t=8s&pp=ygUOQnl0ZS1sZXZlbCBCUEU%3D

import json
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

from tqdm import tqdm


@dataclass
class ByteLevelBPE:
    vocab_size: int = 65536
    pad_token: str = "|<pad>|"
    unknown_token: str = "|<unknown>|"
    bos_token: str = "|<bos>|"
    eos_token: str = "|<eos>|"
    sep_token: str = "|<sep>|"
    cls_token: str = "|<cls>|"
    mask_token: str = "|<mask>|"
    extra_special_tokens: list[str] = field(default_factory=list)
    sentences: list[str] = field(default_factory=list, init=False)
    vocab: dict[bytes, int] = field(default_factory=dict, init=False)
    special_tokens: list[bytes] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        # ---------------------------------------------------------
        # Collect the special tokens configured at initialization
        # ---------------------------------------------------------
        initial_tokens = [
            self.pad_token,
            self.unknown_token,
            self.bos_token,
            self.eos_token,
            self.sep_token,
            self.cls_token,
            self.mask_token,
        ]

        # ---------------------------------------------------------
        # Register the init-time special tokens in a stable order
        # ---------------------------------------------------------
        self.special_tokens = []
        self.add_special_tokens(initial_tokens + self.extra_special_tokens)

    def train(self, sentences: list[str]) -> None:
        # ---------------------------------------------------------
        # Store the training corpus and reset the current vocabulary
        # ---------------------------------------------------------
        self.sentences = [sentence for sentence in sentences]
        self.vocab = {}

        # ---------------------------------------------------------
        # Register bytes and special tokens before BPE merges
        # ---------------------------------------------------------
        self.register_minimum_vocab()
        progress = tqdm(total=self.vocab_size - len(self.vocab), desc="ByteLevelBPE")

        # ---------------------------------------------------------
        # Repeat one merge at a time until the target size is reached
        # ---------------------------------------------------------
        while len(self.vocab) < self.vocab_size and self.merge_token():
            progress.update(1)

        # ---------------------------------------------------------
        # Close the progress bar after the merge loop finishes
        # ---------------------------------------------------------
        progress.close()

    def register_minimum_vocab(self) -> None:
        # ---------------------------------------------------------
        # Register all 256 byte values as the minimum vocabulary
        # ---------------------------------------------------------
        for i in range(256):
            byte_token = bytes([i])
            self.vocab.setdefault(byte_token, len(self.vocab) + 1)

        # ---------------------------------------------------------
        # Register all configured special tokens after byte tokens
        # ---------------------------------------------------------
        for special_token in self.special_tokens:
            self.vocab.setdefault(special_token, len(self.vocab) + 1)

    def add_special_token(self, token: str) -> bytes:
        # ---------------------------------------------------------
        # Encode and store one special token if it is not registered
        # ---------------------------------------------------------
        token_bytes = token.encode("utf-8")

        if token and token_bytes not in self.special_tokens:
            self.special_tokens.append(token_bytes)

        return token_bytes

    def add_special_tokens(self, tokens: list[str]) -> list[bytes]:
        # ---------------------------------------------------------
        # Register multiple special tokens with one consistent flow
        # ---------------------------------------------------------
        return [self.add_special_token(token) for token in tokens if token]

    def merge_token(self) -> bool:
        # ---------------------------------------------------------
        # Count adjacent token pairs across all current tokenizations
        # without materializing the full tokenized corpus in memory.
        # ---------------------------------------------------------
        pair_counts: dict[tuple[bytes, bytes], int] = {}

        # ---------------------------------------------------------
        # Aggregate pair frequencies over the entire training corpus
        # one sentence at a time to keep memory usage bounded.
        # ---------------------------------------------------------
        for sentence in self.sentences:
            tokens = self.split_into_tokens(sentence)

            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pair_counts[pair] = pair_counts.get(pair, 0) + 1

        # ---------------------------------------------------------
        # Stop when no merge candidate remains in the corpus
        # ---------------------------------------------------------
        if not pair_counts:
            return False

        # ---------------------------------------------------------
        # Select the most frequent pair that creates a new token
        # ---------------------------------------------------------
        sorted_pairs = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)
        best_pair = next(
            (pair for pair, _ in sorted_pairs if pair[0] + pair[1] not in self.vocab),
            None,
        )

        # ---------------------------------------------------------
        # Stop when every candidate has already been registered
        # ---------------------------------------------------------
        if best_pair is None:
            return False

        # ---------------------------------------------------------
        # Register the selected merged token into the vocabulary
        # ---------------------------------------------------------
        merged_token = best_pair[0] + best_pair[1]
        self.vocab.setdefault(merged_token, len(self.vocab) + 1)

        return True

    def split_into_tokens(self, sentence: str) -> list[bytes]:
        # ---------------------------------------------------------
        # Split the sentence bytes with the current greedy vocabulary
        # ---------------------------------------------------------
        if not sentence:
            return []

        if not self.vocab:
            raise ValueError("Tokenizer vocabulary is empty. Train or load the tokenizer before tokenizing.")

        b = sentence.encode("utf-8")
        tokens: list[bytes] = []
        start = 0

        # ---------------------------------------------------------
        # Select the longest token that already exists in the vocab
        # ---------------------------------------------------------
        while start < len(b):
            end = start + 1
            token = b[start:end]

            while end <= len(b) and token in self.vocab:
                end += 1
                token = b[start:end]

            selected_token = b[start:end - 1]

            if not selected_token:
                raise ValueError("Tokenizer vocabulary does not contain the required byte token.")

            tokens.append(selected_token)
            start = end - 1

        return tokens

    def tokenize(self, sentence: str) -> list[int]:
        # ---------------------------------------------------------
        # Convert the greedy byte split into vocabulary ids
        # ---------------------------------------------------------
        if not sentence:
            return []

        tokens = self.split_into_tokens(sentence)
        return [self.vocab[token] for token in tokens]

    def save(self, path: str | Path) -> None:
        vocab_items = [
            {"token_hex": token.hex(), "id": token_id}
            for token, token_id in sorted(self.vocab.items(), key=lambda x: x[1])
        ]
        data = {
            "vocab_size": self.vocab_size,
            "pad_token": self.pad_token,
            "unknown_token": self.unknown_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "sep_token": self.sep_token,
            "cls_token": self.cls_token,
            "mask_token": self.mask_token,
            "extra_special_tokens": self.extra_special_tokens,
            "special_tokens": [token.hex() for token in self.special_tokens],
            "vocab": vocab_items,
        }

        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str | Path) -> "ByteLevelBPE":
        with open(path) as f:
            data = json.load(f)

        tokenizer = cls(
            vocab_size=data["vocab_size"],
            pad_token=data["pad_token"],
            unknown_token=data["unknown_token"],
            bos_token=data["bos_token"],
            eos_token=data["eos_token"],
            sep_token=data["sep_token"],
            cls_token=data["cls_token"],
            mask_token=data["mask_token"],
            extra_special_tokens=data["extra_special_tokens"],
        )
        tokenizer.special_tokens = [bytes.fromhex(token) for token in data["special_tokens"]]
        tokenizer.vocab = {
            bytes.fromhex(item["token_hex"]): item["id"]
            for item in data["vocab"]
        }

        return tokenizer


if __name__ == "__main__":
    tokenizer = ByteLevelBPE(vocab_size=270)

    tokenizer.train(
        sentences=[
            "Hello",
            "Hello world",
            "Hello there",
            "Hi world",
            "Byte level BPE",
            "Byte pair encoding",
            "Tokenizer test",
            "Hello Hello",
        ]
    )

    print(dict(sorted(tokenizer.vocab.items(), key=lambda x: x[1])))
    print(tokenizer.tokenize("Hello Japan!!"))
    tokenizer.save("tokenizer.json")
