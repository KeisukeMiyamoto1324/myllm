from datasets import load_dataset


WIKIPEDIA_DATASET_PATH = "wikimedia/wikipedia"


def load_wikipedia_texts(config_name: str, article_count: int, max_chars: int) -> list[str]:
    # ---------------------------------------------------------
    # Stream Wikipedia articles and keep only the first fixed
    # number of characters from each article text.
    # ---------------------------------------------------------
    dataset = load_dataset(
        path=WIKIPEDIA_DATASET_PATH,
        name=config_name,
        split="train",
        streaming=True,
    )

    # ---------------------------------------------------------
    # Materialize the small sample because all tokenizers reuse
    # the same texts for a fair comparison.
    # ---------------------------------------------------------
    return [row["text"][:max_chars] for row in dataset.take(article_count)]
