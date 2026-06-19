from dataclasses import asdict
from dataclasses import dataclass


@dataclass
class TrainingCorpusCase:
    name: str
    genre: str
    language: str
    dataset_path: str
    config_name: str
    split: str
    text_column: str


def serialize_training_corpus_case(
    corpus_case: TrainingCorpusCase,
) -> dict[str, str]:
    # ---------------------------------------------------------
    # Convert one corpus definition into JSON-compatible metadata
    # for validation caches and saved model configuration.
    # ---------------------------------------------------------
    return asdict(corpus_case)
