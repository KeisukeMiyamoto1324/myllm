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
        name="fineweb-en",
        genre="web",
        language="en",
        dataset_path="HuggingFaceFW/fineweb",
        config_name="sample-10BT",
        split="train",
        text_column="text",
        token_percentage=30.0,
    ),
    PretrainingCorpusCase(
        name="fineweb2-ja",
        genre="web",
        language="ja",
        dataset_path="HuggingFaceFW/fineweb-2",
        config_name="jpn_Jpan",
        split="train",
        text_column="text",
        token_percentage=70.0,
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
