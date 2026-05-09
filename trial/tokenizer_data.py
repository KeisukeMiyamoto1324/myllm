from datasets import load_dataset

from tokenizer_benchmark_cases import BenchmarkCase


def load_benchmark_texts(benchmark_case: BenchmarkCase) -> list[str]:
    # ---------------------------------------------------------
    # Stream the configured benchmark dataset and keep only the
    # first fixed number of characters from each sample.
    # ---------------------------------------------------------
    dataset = load_dataset(
        path=benchmark_case.dataset_path,
        name=benchmark_case.config_name,
        split=benchmark_case.split,
        streaming=True,
    )

    # ---------------------------------------------------------
    # Materialize the small sample because all tokenizers reuse
    # the same texts for a fair comparison.
    # ---------------------------------------------------------
    return [
        row[benchmark_case.text_column][:benchmark_case.max_chars]
        for row in dataset.take(benchmark_case.sample_count)
    ]
