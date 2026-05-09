from dataclasses import asdict
from dataclasses import dataclass


@dataclass
class PretrainingCorpusCase:
    name: str
    genre: str
    language: str
    dataset_path: str
    config_name: str
    split: str
    text_column: str
    token_percentage: float


PRETRAINING_CORPUS_CASES = [
    PretrainingCorpusCase(
        name="smollm-cosmopedia-v2",
        genre="web",
        language="en",
        dataset_path="HuggingFaceTB/smollm-corpus",
        config_name="cosmopedia-v2",
        split="train",
        text_column="text",
        token_percentage=100.0,
    ),
]


def serialize_pretraining_corpus_cases(
    corpus_cases: list[PretrainingCorpusCase],
) -> list[dict[str, str | float]]:
    # ---------------------------------------------------------
    # Convert corpus cases to JSON-compatible dictionaries so
    # training artifacts can record the exact data mixture.
    # ---------------------------------------------------------
    return [asdict(corpus_case) for corpus_case in corpus_cases]
