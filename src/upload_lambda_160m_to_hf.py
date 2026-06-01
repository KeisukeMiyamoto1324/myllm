import os
from pathlib import Path
import sys

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pretraining.hf_artifacts import copy_inference_code
from src.pretraining.hf_artifacts import push_hf_pretrained_artifacts


def main() -> None:
    # ---------------------------------------------------------
    # Load the Hugging Face token and repository name from .env.
    # This script uploads the completed model without training.
    # ---------------------------------------------------------
    load_dotenv()

    hf_token = os.environ["HF_TOKEN"]
    hf_repo = os.environ["HF_REPO"]
    model_dir = Path("models/lambda-160m")

    # ---------------------------------------------------------
    # Refresh remote-code files before upload so Hub inference uses
    # the latest model wrapper implementation.
    # ---------------------------------------------------------
    copy_inference_code(output_path=model_dir)

    # ---------------------------------------------------------
    # Push only portable inference artifacts to Hugging Face.
    # Checkpoints, metrics, and validation cache files are skipped.
    # ---------------------------------------------------------
    push_hf_pretrained_artifacts(
        output_path=model_dir,
        repo_id=hf_repo,
        private=True,
        commit_message="Upload lambda-160m pretrained model",
        token=hf_token,
    )


if __name__ == "__main__":
    main()
