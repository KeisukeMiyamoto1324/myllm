import argparse
from pathlib import Path


def positive_float(value: str) -> float:
    # ---------------------------------------------------------
    # Convert CLI input into a positive float for generation
    # parameters that cannot accept zero or negative values.
    # ---------------------------------------------------------
    parsed_value = float(value)

    if parsed_value <= 0:
        raise argparse.ArgumentTypeError("value must be greater than 0")

    return parsed_value


def non_negative_int(value: str) -> int:
    # ---------------------------------------------------------
    # Convert CLI input into a non-negative integer for optional
    # generation limits where zero means disabled.
    # ---------------------------------------------------------
    parsed_value = int(value)

    if parsed_value < 0:
        raise argparse.ArgumentTypeError("value must be greater than or equal to 0")

    return parsed_value


def parse_args(default_model_dir: Path) -> argparse.Namespace:
    # ---------------------------------------------------------
    # Define CLI arguments for selecting prompt, model location,
    # generation length, and sampling behavior.
    # ---------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", type=str)
    parser.add_argument("--model-dir", type=str, default=str(default_model_dir))
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--top-k", type=non_negative_int, default=1)
    parser.add_argument("--temperature", type=positive_float, default=1.0)
    return parser.parse_args()
