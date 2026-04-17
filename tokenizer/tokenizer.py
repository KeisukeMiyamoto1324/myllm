
# https://www.youtube.com/watch?v=HEikzVL-lZU&t=8s&pp=ygUOQnl0ZS1sZXZlbCBCUEU%3D

class ByteLevelBPE:
    def __init__(self, vocab_size: int = 65536) -> None:
        self.sentences: list[str] = []
        self.vocab: dict[bytes, int] = {}
        self.vocab_size = vocab_size
        self.unknown_token: bytes = b"|<unknown>|"

    def train(self, sentences: list[str]) -> None:
        self.sentences = sentences
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
        for sentence in self.sentences:
            pass

    def tokenize(self, sentence: str) -> list[int]:
        # ---------------------------------------------------------
        # Tokenize UTF-8 bytes by consuming the longest valid token
        # ---------------------------------------------------------
        if not sentence:
            return []

        byte_sequence = sentence.encode("utf-8")
        tokenized_sentence: list[int] = []
        start = 0

        while start < len(byte_sequence):
            end = start + 1
            current_vocab = bytes([byte_sequence[start]])

            while end < len(byte_sequence):
                next_vocab = current_vocab + bytes([byte_sequence[end]])
                if next_vocab not in self.vocab:
                    break

                current_vocab = next_vocab
                end += 1

            tokenized_sentence.append(
                self.vocab.get(current_vocab, self.vocab[self.unknown_token])
            )
            start = end

        return tokenized_sentence


if __name__ == "__main__":
    tokenizer = ByteLevelBPE()
    tokenizer.train(
        sentences=[
            "Hello"
        ]
    )

    print(dict(sorted(tokenizer.vocab.items(), key=lambda x: x[1])))
    print(tokenizer.tokenize("Hello"))