
# https://www.youtube.com/watch?v=HEikzVL-lZU&t=8s&pp=ygUOQnl0ZS1sZXZlbCBCUEU%3D

class ByteLevelBPE:
    def __init__(self, vocab_size: int = 65536) -> None:
        self.sentences: list[str] = []
        self.vocab: dict[bytes, int] = {}
        self.vocab_size = vocab_size
        self.unknown_token: bytes = b"|<unknown>|"

    def train(self, sentences: list[str]) -> None:
        self.sentences = [sentence for sentence in sentences]
        self.vocab = {}
        self.register_minimum_vocab()

        # ---------------------------------------------------------
        # Repeat one merge at a time until the target size is reached
        # ---------------------------------------------------------
        while len(self.vocab) < self.vocab_size and self.merge_token():
            continue

    def register_minimum_vocab(self) -> None:
        # ---------------------------------------------------------
        # Register all 256 byte values as the minimum vocabulary
        # ---------------------------------------------------------
        for i in range(256):
            byte_token = bytes([i])
            self.vocab.setdefault(byte_token, len(self.vocab) + 1)

        self.vocab.setdefault(self.unknown_token, len(self.vocab) + 1)

    def merge_token(self) -> bool:
        # ---------------------------------------------------------
        # Count adjacent token pairs across all current tokenizations
        # ---------------------------------------------------------
        pair_counts: dict[tuple[bytes, bytes], int] = {}
        tokenized_sentences = [self.split_into_tokens(sentence) for sentence in self.sentences]

        for tokens in tokenized_sentences:
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

        b = sentence.encode("utf-8")
        tokens: list[bytes] = []
        start = 0

        while start < len(b):
            end = start + 1
            token = b[start:end]

            while end <= len(b) and token in self.vocab:
                end += 1
                token = b[start:end]

            tokens.append(b[start:end - 1])
            start = end - 1

        return tokens

    def tokenize(self, sentence: str) -> list[int]:
        if not sentence:
            return []

        tokens = self.split_into_tokens(sentence)
        return [self.vocab.get(token, self.vocab[self.unknown_token]) for token in tokens]


if __name__ == "__main__":
    tokenizer = ByteLevelBPE()
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
    print(tokenizer.tokenize("Hello"))
