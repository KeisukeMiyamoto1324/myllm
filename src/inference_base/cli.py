import argparse
from pathlib import Path


def positive_float(value: str) -> float:
    # ---------------------------------------------------------
    # Convert CLI input into a positive float used by generation
    # settings such as temperature and repetition penalty.
    # ---------------------------------------------------------
    parsed_value = float(value)

    if parsed_value <= 0.0:
        raise argparse.ArgumentTypeError("value must be greater than 0")

    return parsed_value


def probability_float(value: str) -> float:
    # ---------------------------------------------------------
    # Convert CLI input into a probability used by nucleus
    # sampling and reject values outside the valid range.
    # ---------------------------------------------------------
    parsed_value = float(value)

    if parsed_value <= 0.0 or parsed_value > 1.0:
        raise argparse.ArgumentTypeError("value must be greater than 0 and less than or equal to 1")

    return parsed_value


def non_negative_int(value: str) -> int:
    # ---------------------------------------------------------
    # Convert CLI input into a non-negative integer used by top-k
    # sampling, where 0 disables the top-k filter.
    # ---------------------------------------------------------
    parsed_value = int(value)

    if parsed_value < 0:
        raise argparse.ArgumentTypeError("value must be greater than or equal to 0")

    return parsed_value


def parse_args(default_model_dir: Path) -> argparse.Namespace:
    # ---------------------------------------------------------
    # Define CLI arguments for an interactive text generation session
    # backed by a Hugging Face AutoModel-compatible model directory
    # or repo.
    # ---------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default=str(default_model_dir))
    parser.add_argument("--prompt", "--promot", type=str, default="")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--temperature", type=positive_float, default=0.6)
    parser.add_argument("--top-p", type=probability_float, default=0.85)
    parser.add_argument("--top-k", type=non_negative_int, default=40)
    parser.add_argument("--repetition-penalty", type=positive_float, default=1.2)
    parser.add_argument("--no-repeat-ngram-size", type=non_negative_int, default=3)
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
    )
    return parser.parse_args()
