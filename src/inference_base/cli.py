import argparse
from pathlib import Path

from src.shared.cli import require


def parse_args(default_model_dir: Path) -> argparse.Namespace:
    # ---------------------------------------------------------
    # Define CLI arguments for an interactive text generation session
    # backed by a PyTorch model directory or Hub repo.
    # ---------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default=str(default_model_dir))
    parser.add_argument("--prompt", "--promot", type=str, default="")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--top-p", type=float, default=0.85)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--repetition-penalty", type=float, default=1.2)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=3)
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
    )
    args = parser.parse_args()

    require(args.temperature > 0.0, parser, "--temperature must be greater than 0")
    require(
        args.top_p > 0.0 and args.top_p <= 1.0,
        parser,
        "--top-p must be greater than 0 and less than or equal to 1",
    )
    require(args.top_k >= 0, parser, "--top-k must be greater than or equal to 0")
    require(args.repetition_penalty > 0.0, parser, "--repetition-penalty must be greater than 0")
    require(
        args.no_repeat_ngram_size >= 0,
        parser,
        "--no-repeat-ngram-size must be greater than or equal to 0",
    )

    return args
