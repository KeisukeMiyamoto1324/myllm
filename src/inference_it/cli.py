import argparse
from pathlib import Path

from src.inference_base.cli import non_negative_int
from src.inference_base.cli import positive_float
from src.inference_base.cli import probability_float


def parse_args(default_model_dir: Path) -> argparse.Namespace:
    # ---------------------------------------------------------
    # Define CLI arguments for terminal chat with a PyTorch
    # instruction-tuned model directory or Hub repo.
    # ---------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default=str(default_model_dir))
    parser.add_argument("--prompt", "--promot", type=str, default="")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--temperature", type=positive_float, default=0.7)
    parser.add_argument("--top-p", type=probability_float, default=0.85)
    parser.add_argument("--top-k", type=non_negative_int, default=8)
    parser.add_argument("--repetition-penalty", type=positive_float, default=1.2)
    parser.add_argument("--no-repeat-ngram-size", type=non_negative_int, default=3)
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
    )
    return parser.parse_args()
