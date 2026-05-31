from dataclasses import asdict
from dataclasses import dataclass


WIKI_RAMP_START_PROGRESS = 0.5


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
    is_ramped: bool
    repeat_on_end: bool


PRETRAINING_CORPUS_CASES = [
    PretrainingCorpusCase(
        name="fineweb2-edu-ja",
        genre="web",
        language="ja",
        dataset_path="hotchpotch/fineweb-2-edu-japanese",
        config_name="default",
        split="train",
        text_column="text",
        token_percentage=30.0,
        is_ramped=False,
        repeat_on_end=True,
    ),
    PretrainingCorpusCase(
        name="cleanedwiki-jp",
        genre="wiki",
        language="ja",
        dataset_path="MK0727/CleanedWiki-jp",
        config_name="all",
        split="train",
        text_column="text",
        token_percentage=70.0,
        is_ramped=True,
        repeat_on_end=True,
    ),
]


def serialize_pretraining_corpus_cases(
    corpus_cases: list[PretrainingCorpusCase],
) -> list[dict[str, str | float | bool]]:
    # ---------------------------------------------------------
    # Convert corpus cases to JSON-compatible dictionaries so
    # training artifacts can record the exact data mixture.
    # ---------------------------------------------------------
    return [asdict(corpus_case) for corpus_case in corpus_cases]
