import sys
from pathlib import Path

# ---------------------------------------------------------
# Add the project root to the import path so direct script
# execution can import the project packages consistently.
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference_base.cli import parse_args
from src.inference_base.runtime import run_inference


def main() -> None:
    # ---------------------------------------------------------
    # Parse runtime settings and load PyTorch artifacts from a
    # local directory or Hub repository id.
    # ---------------------------------------------------------
    args = parse_args(default_model_dir=PROJECT_ROOT / "models" / "lambda-160m-midtrained")
    run_inference(args=args)


if __name__ == "__main__":
    main()
