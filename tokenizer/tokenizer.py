
# https://www.youtube.com/watch?v=HEikzVL-lZU&t=8s&pp=ygUOQnl0ZS1sZXZlbCBCUEU%3D

class ByteLevelBPE:
    def __init__(self, vocab_size: int = 65536) -> None:
        self.sentences: list[str] = []
        self.vocab: dict[bytes, int] = {}
        self.vocab_size = vocab_size
        self.unknown_token: bytes = b"|<unknown>|"

    def train(self) -> None:
        self.register_minimum_vocab()

    def register_minimum_vocab(self) -> None:
        # ---------------------------------------------------------
        # Register all 256 byte values as the minimum vocabulary
        # ---------------------------------------------------------
        for i in range(256):
            byte_token = bytes([i])
            self.vocab.setdefault(byte_token, len(self.vocab) + 1)

        self.vocab.setdefault(self.unknown_token, len(self.vocab) + 1)

    def merge_token(self) -> None:
        

    def tokenize(self, sentence: str) -> list[int]:
        # ---------------------------------------------------------
        # Greedy longest-match tokenization on UTF-8 bytes
        # ---------------------------------------------------------
        if not sentence:
            return []

        b = sentence.encode("utf-8")
        res: list[int] = []
        start = 0

        while start < len(b):
            end = start + 1
            token = b[start:end]

            while end <= len(b) and token in self.vocab:
                end += 1
                token = b[start:end]

            token = b[start:end-1]
            res.append(self.vocab.get(token, self.vocab[self.unknown_token]))
            start = end - 1

        return res


if __name__ == "__main__":
    tokenizer = ByteLevelBPE()
    tokenizer.train(
        sentences=[
            "Hello"
        ]
    )

    print(dict(sorted(tokenizer.vocab.items(), key=lambda x: x[1])))
    print(tokenizer.tokenize("Hello"))