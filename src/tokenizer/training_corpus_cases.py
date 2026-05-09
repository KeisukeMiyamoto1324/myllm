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
    sample_count: int
    max_chars: int


TRAINING_CORPUS_CASES = [
    TrainingCorpusCase(
        name="smollm-cosmopedia-v2",
        genre="synthetic-textbook",
        language="en",
        dataset_path="HuggingFaceTB/smollm-corpus",
        config_name="cosmopedia-v2",
        split="train",
        text_column="text",
        sample_count=2560000,
        max_chars=4096,
    ),
]
