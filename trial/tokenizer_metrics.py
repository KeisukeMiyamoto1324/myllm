from dataclasses import dataclass

from transformers import PreTrainedTokenizerBase


@dataclass
class CompressionResult:
    model_name: str
    language: str
    vocab_size: int
    article_count: int
    char_count: int
    byte_count: int
    token_count: int
    chars_per_token: float
    bytes_per_token: float


def measure_compression(
    model_name: str,
    language: str,
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
) -> CompressionResult:
    # ---------------------------------------------------------
    # Count text size and tokenize without special tokens so
    # only the raw article text compression is measured.
    # ---------------------------------------------------------
    char_count = sum(len(text) for text in texts)
    byte_count = sum(len(text.encode("utf-8")) for text in texts)
    token_count = sum(len(tokenizer.encode(text, add_special_tokens=False)) for text in texts)

    # ---------------------------------------------------------
    # Convert total counts into compression ratios where larger
    # values mean fewer tokens were needed for the same text.
    # ---------------------------------------------------------
    return CompressionResult(
        model_name=model_name,
        language=language,
        vocab_size=tokenizer.vocab_size,
        article_count=len(texts),
        char_count=char_count,
        byte_count=byte_count,
        token_count=token_count,
        chars_per_token=char_count / token_count,
        bytes_per_token=byte_count / token_count,
    )
