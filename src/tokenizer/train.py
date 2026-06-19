import argparse
from pathlib import Path
import sys
from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------
# Add the project root so direct script execution can import
# modules through the src package path.
# ---------------------------------------------------------
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.shared.tokenizer import ByteLevelBPE
from src.tokenizer.training_corpus_cases import TRAINING_CORPUS_CASES
from src.tokenizer.training_corpus_data import stream_training_texts


def parse_args() -> argparse.Namespace:
    # ---------------------------------------------------------
    # Define CLI arguments for tokenizer training settings that
    # are not tied to a specific corpus source.
    # ---------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-size", type=int, default=65536)
    parser.add_argument(
        "--output-path",
        type=str,
        default="models/tokenizer",
    )
    return parser.parse_args()


def main() -> None:
    # ---------------------------------------------------------
    # Parse the arguments and resolve the output directory for
    # the Hugging Face compatible tokenizer artifact.
    # ---------------------------------------------------------
    args = parse_args()
    output_path = Path(args.output_path)

    # ---------------------------------------------------------
    # Train the tokenizer from all configured corpus cases and
    # persist it for AutoTokenizer.from_pretrained loading.
    # ---------------------------------------------------------
    tokenizer = ByteLevelBPE(vocab_size=args.vocab_size)
    tokenizer.train(sentences=stream_training_texts(TRAINING_CORPUS_CASES))
    tokenizer.save_pretrained(path=output_path)


if __name__ == "__main__":
    main()
