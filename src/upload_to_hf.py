import os
from pathlib import Path
import sys

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.shared.pytorch_artifacts import push_pytorch_model_artifacts


def main() -> None:
    # ---------------------------------------------------------
    # Load the Hugging Face token and repository name from .env.
    # This script uploads the completed model without training.
    # ---------------------------------------------------------
    load_dotenv()

    hf_token = os.environ["HF_TOKEN"]
    hf_repo = os.environ["HF_REPO"]
    model_dir = Path("models/lambda-160m-midtrained")

    # ---------------------------------------------------------
    # Push only PyTorch weights, model config, and tokenizer files.
    # Python source files and training outputs are skipped.
    # ---------------------------------------------------------
    push_pytorch_model_artifacts(
        output_path=model_dir,
        repo_id=hf_repo,
        private=True,
        commit_message="Upload lambda-160m-midtrained pretrained model",
        token=hf_token,
    )


if __name__ == "__main__":
    main()
