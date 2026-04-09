import json
from pathlib import Path


def load_sentences(path: str | Path) -> list[str]:
    # ---------------------------------------------------------
    # Read the jsonl file and collect the raw text values used
    # by tokenizer and model training.
    # ---------------------------------------------------------
    with open(path) as f:
        return [json.loads(line)["text"] for line in f]


def format_sentences(raw_sentences: list[str], max_len: int) -> list[str]:
    # ---------------------------------------------------------
    # Normalize each sentence into the fixed-length training
    # format with EOS and PAD tokens.
    # ---------------------------------------------------------
    return [_format_sentence(sentence=sentence, max_len=max_len) for sentence in raw_sentences]


def _format_sentence(sentence: str, max_len: int) -> str:
    # ---------------------------------------------------------
    # Truncate the sentence, append EOS, and pad the remainder
    # so every sample has the same token count.
    # ---------------------------------------------------------
    tokens = sentence.split()[: max_len - 1]
    padded_tokens = tokens + ["<EOS>"]
    padding_size = max_len - len(padded_tokens)
    return " ".join(padded_tokens + ["<PAD>"] * padding_size)
