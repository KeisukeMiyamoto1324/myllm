import os
import sys
from pathlib import Path

from dotenv import load_dotenv

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
    # Load the Hub repository id from .env and use it as the
    # default model source for inference.
    # ---------------------------------------------------------
    load_dotenv()

    args = parse_args(default_model_dir=Path(os.environ["HF_REPO"]))
    run_inference(args=args)


if __name__ == "__main__":
    main()
