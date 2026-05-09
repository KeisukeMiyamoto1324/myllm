import argparse
import json
from pathlib import Path
import sys

# ---------------------------------------------------------
# Add the project root so direct script execution can import
# modules through the src package path.
# ---------------------------------------------------------
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.tokenizer.tokenizer import ByteLevelBPE


def parse_args() -> argparse.Namespace:
    # ---------------------------------------------------------
    # Define CLI arguments so one input sentence can be encoded
    # with a trained tokenizer artifact from this folder.
    # ---------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("text", type=str)
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default="models/tokenizer",
    )
    return parser.parse_args()


def format_token(token: str) -> str:
    # ---------------------------------------------------------
    # Convert the ByteLevel-internal token display into a human-
    # readable text fragment with visible leading spaces.
    # ---------------------------------------------------------
    return token.replace("Ġ", " ").replace("Ċ", "\n")


def main() -> None:
    # ---------------------------------------------------------
    # Parse the CLI input and load the trained tokenizer from
    # the configured tokenizer artifact path.
    # ---------------------------------------------------------
    args = parse_args()
    tokenizer = ByteLevelBPE.load(path=Path(args.tokenizer_path))
    encoding = tokenizer.tokenizer.encode(args.text)
    formatted_tokens = [format_token(token) for token in encoding.tokens]
    decoded_text = tokenizer.detokenize([token_id for token_id in encoding.ids])

    # ---------------------------------------------------------
    # Print the encoded ids, a decoded round-trip result, and
    # human-readable token fragments as JSON.
    # ---------------------------------------------------------
    print(
        json.dumps(
            {
                "text": args.text,
                "ids": [token_id for token_id in encoding.ids],
                "decoded_text": decoded_text,
                "formatted_tokens": formatted_tokens,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
