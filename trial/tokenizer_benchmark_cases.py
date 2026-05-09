from dataclasses import dataclass


@dataclass
class BenchmarkCase:
    name: str
    genre: str
    language: str
    dataset_path: str
    config_name: str
    split: str
    text_column: str
    sample_count: int
    max_chars: int


BENCHMARK_CASES = [
    BenchmarkCase(
        name="wiki-en",
        genre="encyclopedia",
        language="en",
        dataset_path="wikimedia/wikipedia",
        config_name="20231101.en",
        split="train",
        text_column="text",
        sample_count=100,
        max_chars=1024,
    ),
    BenchmarkCase(
        name="wiki-ja",
        genre="encyclopedia",
        language="ja",
        dataset_path="wikimedia/wikipedia",
        config_name="20231101.ja",
        split="train",
        text_column="text",
        sample_count=100,
        max_chars=1024,
    ),
]
