from src.shared.training_corpus import serialize_training_corpus_case
from src.shared.training_corpus import TrainingCorpusCase


MIDTRAINING_CORPUS_CASE = TrainingCorpusCase(
    name="synthetic-textbook-jp",
    genre="textbook",
    language="ja",
    dataset_path="MK0727/SyntheticTextbook-jp",
    config_name="default",
    split="train",
    text_column="rewrite",
)


def serialize_midtraining_corpus_case(
    corpus_case: TrainingCorpusCase,
) -> dict[str, str]:
    # ---------------------------------------------------------
    # Serialize the fixed mid-training corpus definition for
    # validation cache and model artifact metadata.
    # ---------------------------------------------------------
    return serialize_training_corpus_case(corpus_case=corpus_case)
