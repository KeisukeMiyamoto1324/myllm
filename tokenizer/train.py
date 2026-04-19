from pathlib import Path
import sys
from dotenv import load_dotenv
from datasets import load_dataset
load_dotenv()

sys.path.append(str(Path(__file__).resolve().parent))

from tokenizer import ByteLevelBPE


def main() -> None:
    # ---------------------------------------------------------
    # Load the FineWeb-Edu sample-10BT training split and keep
    # only the text field for tokenizer learning.
    # ---------------------------------------------------------
    dataset = load_dataset(
        path="HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
    )
    sentences = [row["text"] for row in dataset]

    # ---------------------------------------------------------
    # Train the ByteLevelBPE tokenizer on the loaded corpus and
    # save the learned vocabulary as a JSON artifact.
    # ---------------------------------------------------------
    tokenizer = ByteLevelBPE(vocab_size=65536)
    tokenizer.train(sentences=sentences)
    tokenizer.save(path=Path("tokenizer/fineweb_edu_sample_10bt_tokenizer.json"))


if __name__ == "__main__":
    main()
