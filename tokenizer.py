
def get_token_map(words: list[str]) -> dict[str, int]:
    return {w: i for i, w in enumerate(words)}

def tokenizer(words: list[str], token_map: dict[str, int]) -> dict[str, int]:
    return [token_map[w] for w in words]
