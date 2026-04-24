import argparse
from pathlib import Path
import sys
from dotenv import load_dotenv
load_dotenv()

from datasets import load_dataset
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent))

from tokenizer import ByteLevelBPE


def parse_args() -> argparse.Namespace:
    # ---------------------------------------------------------
    # Define training arguments so the corpus size stays bounded
    # for the current in-memory ByteLevelBPE implementation.
    # ---------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--max-samples", type=int, default=20000)
    parser.add_argument("--max-chars", type=int, default=2048)
    parser.add_argument(
        "--output-path",
        type=str,
        default="tokenizer/fineweb_edu_sample_10bt_tokenizer.json",
    )
    return parser.parse_args()


def main() -> None:
    # ---------------------------------------------------------
    # Parse the training arguments and prepare the streaming
    # dataset source plus the output destination.
    # ---------------------------------------------------------
    args = parse_args()
    output_path = Path(args.output_path)
    dataset = load_dataset(
        path="HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )
    progress = tqdm(total=args.max_samples, desc="CollectSamples")
    sentences: list[str] = []

    # ---------------------------------------------------------
    # Read only a bounded number of streamed samples so the
    # tokenizer can train without exhausting local memory.
    # ---------------------------------------------------------
    for row in dataset.take(args.max_samples):
        text = row["text"][:args.max_chars]
        sentences.append(text)
        progress.update(1)

    progress.close()

    # ---------------------------------------------------------
    # Train the ByteLevelBPE tokenizer on the loaded corpus and
    # save the learned vocabulary as a JSON artifact.
    # ---------------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = ByteLevelBPE(vocab_size=args.vocab_size)
    tokenizer.train(sentences=sentences)
    tokenizer.save(path=output_path)


if __name__ == "__main__":
    main()
