import argparse
import os
from pathlib import Path

from src.shared.cli import require
from src.shared.device_utils import resolve_devices


def parse_args() -> argparse.Namespace:
    # ---------------------------------------------------------
    # Define the input model, training runtime, validation budget,
    # output artifacts, and optional checkpoint resume.
    # ---------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--max-len", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=72)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=10240)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-split-modulo", type=int, default=100)
    parser.add_argument("--val-split-index", type=int, default=0)
    parser.add_argument("--val-batches", type=int, default=64)
    parser.add_argument("--validation-cache-path", type=str, default="")
    parser.add_argument("--val-check-interval", type=int, default=1000)
    parser.add_argument("--checkpoint-every-n-steps", type=int, default=5000)
    parser.add_argument("--metric-log-every-n-steps", type=int, default=500)
    parser.add_argument("--loss-chunk-size", type=int, default=32)
    parser.add_argument("--devices", type=str, default="auto")
    parser.add_argument("--output-path", type=str, default="models/lambda-160m-midtrained")
    parser.add_argument("--resume-from-checkpoint", type=str, default="")
    parser.add_argument("--push-to-hub", action="store_true")
    args = parser.parse_args()

    # ---------------------------------------------------------
    # Reject incomplete model directories and invalid runtime
    # values before opening the remote streaming dataset.
    # ---------------------------------------------------------
    model_path = Path(args.model_path)
    required_model_files = [
        model_path / "model.pth",
        model_path / "model_config.json",
        model_path / "tokenizer.json",
    ]

    require(
        model_path.is_dir() and all(path.is_file() for path in required_model_files),
        parser,
        "--model-path must contain model.pth, model_config.json, and tokenizer.json",
    )
    require(args.max_len is None or args.max_len > 0, parser, "--max-len must be greater than 0")
    require(args.learning_rate > 0.0, parser, "--learning-rate must be greater than 0")
    require(
        args.gradient_accumulation_steps >= 1,
        parser,
        "--gradient-accumulation-steps must be greater than or equal to 1",
    )

    try:
        resolve_devices(devices=args.devices)
    except ValueError as error:
        require(False, parser, str(error))

    require(args.max_steps > 0, parser, "--max-steps must be greater than 0")
    require(
        args.checkpoint_every_n_steps > 0,
        parser,
        "--checkpoint-every-n-steps must be greater than 0",
    )
    require(args.val_split_modulo > 1, parser, "--val-split-modulo must be greater than 1")
    require(
        0 <= args.val_split_index < args.val_split_modulo,
        parser,
        "--val-split-index must be within --val-split-modulo",
    )
    require(
        not args.resume_from_checkpoint or Path(args.resume_from_checkpoint).is_file(),
        parser,
        "--resume-from-checkpoint must point to an existing checkpoint file",
    )
    require(
        not args.push_to_hub or bool(os.environ.get("HF_REPO")),
        parser,
        "HF_REPO is required in the environment when --push-to-hub is set",
    )

    return args
