
# https://www.youtube.com/watch?v=HEikzVL-lZU&t=8s&pp=ygUOQnl0ZS1sZXZlbCBCUEU%3D

class ByteLevelBPE:
    def __init__(self) -> None:
        self.sentences: list[str] = []
        self.vocab: dict[str, int] = {}

    def train(self, sentences: list[str]) -> None:
        self.sentences = sentences
        self.register_minimum_vocab()
        
        

    def register_minimum_vocab(self) -> None:
        # ---------------------------------------------------------
        # Collect all unique characters once, then register them
        # ---------------------------------------------------------
        unique_chars: set[str] = set("".join(self.sentences))

        for char in unique_chars:
            self.vocab.setdefault(char, len(self.vocab) + 1)
            
            
if __name__ == "__main__":
    tokenizer = ByteLevelBPE()
    tokenizer.train(
        sentences=[
            "I like to study computer science at university.",
            "The weather is very nice today.",
            "She enjoys reading books in the library.",
            "We are working on a programming project together.",
            "He usually drinks coffee in the morning.",
            "GPT-2 は内部で、byte をそのまま扱うというより、byte を printable な unicode 文字へ写像してから BPE しています。",
            "これは正規表現や文字列処理をしやすくするためです。"
        ]
    )

    print(sorted(tokenizer.vocab))