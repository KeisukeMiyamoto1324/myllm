
# https://www.youtube.com/watch?v=HEikzVL-lZU

import argparse
from pathlib import Path

from text_preprocessor import format_sentences, load_sentences
from tokenizer import Tokenizer


def parse_args() -> argparse.Namespace:
    # ---------------------------------------------------------
    # Define command line arguments so tokenizer training can be
    # run independently from model training.
    # ---------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", type=str, default="dataset/train.jsonl")
    parser.add_argument("--max-len", type=int, default=16)
    parser.add_argument("--output-path", type=str, default="model/tokenizer.json")
    return parser.parse_args()


def main() -> None:
    # ---------------------------------------------------------
    # Parse CLI arguments and resolve the input and output paths
    # used for tokenizer training.
    # ---------------------------------------------------------
    args = parse_args()
    train_data_path = Path(args.train_data)
    output_path = Path(args.output_path)

    # ---------------------------------------------------------
    # Load the raw training text and apply the shared sentence
    # formatting rule before vocabulary learning.
    # ---------------------------------------------------------
    raw_sentences = load_sentences(path=train_data_path)
    sentences = format_sentences(raw_sentences=raw_sentences, max_len=args.max_len)

    # ---------------------------------------------------------
    # Learn the vocabulary once and save it as a reusable
    # tokenizer artifact for later training and inference.
    # ---------------------------------------------------------
    output_path.parent.mkdir(exist_ok=True)

    tokenizer = Tokenizer()
    tokenizer.learn_vocab(sentences)
    tokenizer.save(output_path)


if __name__ == "__main__":
    main()
