
# https://www.youtube.com/watch?v=HEikzVL-lZU&t=8s&pp=ygUOQnl0ZS1sZXZlbCBCUEU%3D

class ByteLevelBPE:
    def __init__(self, vocab_size=65536) -> None:
        self.sentences: list[str] = []
        self.vocab: dict[str, int] = {}
        self.vocab_size = vocab_size
        self.unknown_token = "|<unknown>|"

    def train(self, sentences: list[str]) -> None:
        self.sentences = sentences
        self.register_minimum_vocab()
        
    def register_minimum_vocab(self) -> None:
        # ---------------------------------------------------------
        # Collect all unique characters once, then register them
        # ---------------------------------------------------------
        unique_chars: set[str] = set(" ".join(self.sentences))

        for char in unique_chars:
            self.vocab.setdefault(char, len(self.vocab) + 1)       
            
        self.vocab.setdefault(self.unknown_token, len(self.vocab) + 1) 
    
    def merge_token(self):
        for sentence in self.sentences:
             pass
         
    def tokenize(self, sentence: str) -> list[int]:
        # ---------------------------------------------------------
        # Tokenize by consuming the longest valid token each step
        # ---------------------------------------------------------
        if not sentence:
            return []

        chars = list(sentence)
        tokenized_sentence: list[int] = []
        start = 0

        while start < len(chars):
            end = start + 1
            current_vocab = chars[start]

            while end < len(chars) and current_vocab + chars[end] in self.vocab:
                current_vocab += chars[end]
                end += 1

            tokenized_sentence.append(self.vocab.get(current_vocab, 0))
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
    print(tokenizer.tokenize(sentence="Hello"))