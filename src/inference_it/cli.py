import argparse
from pathlib import Path

from src.inference_base.cli import positive_float


def probability_float(value: str) -> float:
    # ---------------------------------------------------------
    # Convert CLI input into a probability used by nucleus
    # sampling and reject values outside the valid range.
    # ---------------------------------------------------------
    parsed_value = float(value)

    if parsed_value <= 0.0 or parsed_value > 1.0:
        raise argparse.ArgumentTypeError("value must be greater than 0 and less than or equal to 1")

    return parsed_value


def parse_args(default_model_dir: Path) -> argparse.Namespace:
    # ---------------------------------------------------------
    # Define CLI arguments for an interactive text generation session
    # backed by a Hugging Face AutoModel-compatible model directory
    # or repo.
    # ---------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default=str(default_model_dir))
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--temperature", type=positive_float, default=0.7)
    parser.add_argument("--top-p", type=probability_float, default=0.9)
    parser.add_argument("--repetition-penalty", type=positive_float, default=1.05)
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
    )
    return parser.parse_args()
