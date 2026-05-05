import argparse
from pathlib import Path

from src.inference_base.cli import non_negative_int
from src.inference_base.cli import positive_float


def parse_args(default_model_dir: Path) -> argparse.Namespace:
    # ---------------------------------------------------------
    # Define CLI arguments for an interactive chat session backed
    # by a posttrained chat model directory.
    # ---------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default=str(default_model_dir))
    parser.add_argument("--system-prompt", type=str, default="")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--top-k", type=non_negative_int, default=32)
    parser.add_argument("--temperature", type=positive_float, default=0.7)
    return parser.parse_args()
