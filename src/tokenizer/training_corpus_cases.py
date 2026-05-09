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
        name="fineweb-en",
        genre="web",
        language="en",
        dataset_path="HuggingFaceFW/fineweb",
        config_name="sample-10BT",
        split="train",
        text_column="text",
        sample_count=128000,
        max_chars=4096,
    ),
    TrainingCorpusCase(
        name="fineweb-ja",
        genre="web",
        language="ja",
        dataset_path="HuggingFaceFW/fineweb-2",
        config_name="jpn_Jpan",
        split="train",
        text_column="text",
        sample_count=128000,
        max_chars=4096,
    ),
]
