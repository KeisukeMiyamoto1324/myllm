from collections.abc import Iterator

from datasets import load_dataset
from tqdm import tqdm

from src.tokenizer.training_corpus_cases import TrainingCorpusCase


def stream_corpus_texts(corpus_case: TrainingCorpusCase) -> Iterator[str]:
    # ---------------------------------------------------------
    # Stream one configured corpus split and yield a bounded
    # number of truncated text samples for tokenizer training.
    # ---------------------------------------------------------
    dataset = load_dataset(
        path=corpus_case.dataset_path,
        name=corpus_case.config_name,
        split=corpus_case.split,
        streaming=True,
    )
    progress = tqdm(total=corpus_case.sample_count, desc=corpus_case.name)

    # ---------------------------------------------------------
    # Yield samples one by one so the tokenizer trainer can
    # consume large datasets without materializing all text.
    # ---------------------------------------------------------
    for row in dataset.take(corpus_case.sample_count):
        progress.update(1)
        yield row[corpus_case.text_column][:corpus_case.max_chars]

    # ---------------------------------------------------------
    # Close the corpus progress bar after the stream is consumed.
    # ---------------------------------------------------------
    progress.close()


def stream_training_texts(corpus_cases: list[TrainingCorpusCase]) -> Iterator[str]:
    # ---------------------------------------------------------
    # Chain all configured corpus streams in their declared order
    # so adding a new corpus only changes the case list.
    # ---------------------------------------------------------
    for corpus_case in corpus_cases:
        yield from stream_corpus_texts(corpus_case)
