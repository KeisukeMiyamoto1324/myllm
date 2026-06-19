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


PRETRAINING_CORPUS_CASE = PretrainingCorpusCase(
    name="cleaned-fineweb2-edu-jp",
    genre="web",
    language="ja",
    dataset_path="MK0727/CleanedFineWeb2Edu-jp",
    config_name="default",
    split="train",
    text_column="text",
)


def serialize_pretraining_corpus_case(
    corpus_case: PretrainingCorpusCase,
) -> dict[str, str]:
    # ---------------------------------------------------------
    # Convert the corpus case to a JSON-compatible dictionary so
    # training artifacts can record the exact dataset source.
    # ---------------------------------------------------------
    return asdict(corpus_case)
