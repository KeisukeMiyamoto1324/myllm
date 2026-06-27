import sys
from pathlib import Path

# ---------------------------------------------------------
# Add the project root to the import path so direct script
# execution can import the project packages consistently.
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference_it.cli import parse_args
from src.inference_it.runtime import run_inference


DEFAULT_MODEL_DIR = Path("models/lambda-1-160m-it")


def main() -> None:
    # ---------------------------------------------------------
    # Parse chat runtime settings and start terminal inference for
    # the instruction-tuned model artifact.
    # ---------------------------------------------------------
    args = parse_args(default_model_dir=DEFAULT_MODEL_DIR)
    run_inference(args=args)


if __name__ == "__main__":
    main()
