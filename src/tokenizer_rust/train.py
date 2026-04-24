import argparse
from collections.abc import Iterator
from pathlib import Path
import sys
from dotenv import load_dotenv
load_dotenv()

from datasets import load_dataset
from tqdm import tqdm

# ---------------------------------------------------------
# Add the project root so direct script execution can import
# modules through the src package path.
# ---------------------------------------------------------
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.tokenizer_rust.tokenizer import ByteLevelBPE


def parse_args() -> argparse.Namespace:
    # ---------------------------------------------------------
    # Define CLI arguments for streaming tokenizer training so
    # the dataset size and output path stay explicit.
    # ---------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-size", type=int, default=32768)
    parser.add_argument("--max-samples", type=int, default=200000)
    parser.add_argument("--max-chars", type=int, default=2048)
    parser.add_argument(
        "--output-path",
        type=str,
        default="models/tokenizer.json",
    )
    return parser.parse_args()


def build_text_iterator(max_samples: int, max_chars: int) -> Iterator[str]:
    # ---------------------------------------------------------
    # Stream the FineWeb-Edu sample-10BT split and yield only a
    # bounded number of truncated text samples.
    # ---------------------------------------------------------
    dataset = load_dataset(
        path="HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )
    progress = tqdm(total=max_samples, desc="CollectSamples")

    # ---------------------------------------------------------
    # Yield streamed samples one by one so Rust-side training
    # can consume them without building a large Python list.
    # ---------------------------------------------------------
    for row in dataset.take(max_samples):
        progress.update(1)
        yield row["text"][:max_chars]

    # ---------------------------------------------------------
    # Close the sampling progress bar once the iterator ends.
    # ---------------------------------------------------------
    progress.close()


def main() -> None:
    # ---------------------------------------------------------
    # Parse the arguments and prepare the output path for the
    # Rust-backed ByteLevel BPE training run.
    # ---------------------------------------------------------
    args = parse_args()
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # Train the tokenizer from the streamed FineWeb-Edu samples
    # and persist the resulting tokenizer artifact.
    # ---------------------------------------------------------
    tokenizer = ByteLevelBPE(vocab_size=args.vocab_size)
    tokenizer.train(sentences=build_text_iterator(args.max_samples, args.max_chars))
    tokenizer.save(path=output_path)


if __name__ == "__main__":
    main()
