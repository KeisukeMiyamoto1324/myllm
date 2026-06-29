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
    # Reject incomplete source model directories before opening
    # remote streaming data or initializing training outputs.
    # ---------------------------------------------------------
    model_path = Path(args.model_path)
    required_model_files = [
        model_path / "model.pth",
        model_path / "model_config.json",
        model_path / "tokenizer.json",
    ]

    if not model_path.is_dir() or any(not path.is_file() for path in required_model_files):
        parser.error("--model-path must contain model.pth, model_config.json, and tokenizer.json")

    # ---------------------------------------------------------
    # Validate midtraining-specific runtime values. A larger
    # --max-len than pretraining is allowed when it is positive.
    # ---------------------------------------------------------
    try:
        require(args.max_len is None or args.max_len > 0, "--max-len must be greater than 0")
        require(args.learning_rate > 0.0, "--learning-rate must be greater than 0")
        require(args.batch_size > 0, "--batch-size must be greater than 0")
        require(
            args.gradient_accumulation_steps >= 1,
            "--gradient-accumulation-steps must be greater than or equal to 1",
        )
        require(args.max_steps > 0, "--max-steps must be greater than 0")
        require(args.num_workers >= 0, "--num-workers must be greater than or equal to 0")
        require(args.val_split_modulo > 1, "--val-split-modulo must be greater than 1")
        require(
            0 <= args.val_split_index < args.val_split_modulo,
            "--val-split-index must be within --val-split-modulo",
        )
        require(args.val_batches > 0, "--val-batches must be greater than 0")
        require(args.val_check_interval > 0, "--val-check-interval must be greater than 0")
        require(args.checkpoint_every_n_steps > 0, "--checkpoint-every-n-steps must be greater than 0")
        require(args.metric_log_every_n_steps > 0, "--metric-log-every-n-steps must be greater than 0")
        require(args.loss_chunk_size > 0, "--loss-chunk-size must be greater than 0")
    except ValueError as error:
        parser.error(str(error))

    # ---------------------------------------------------------
    # Validate device, resume, and Hub settings while the parser
    # can still report clear command line errors.
    # ---------------------------------------------------------
    try:
        resolve_devices(devices=args.devices)
    except ValueError as error:
        parser.error(str(error))

    if args.resume_from_checkpoint and not Path(args.resume_from_checkpoint).is_file():
        parser.error("--resume-from-checkpoint must point to an existing checkpoint file")

    if args.push_to_hub and not os.environ.get("HF_REPO"):
        parser.error("HF_REPO is required in the environment when --push-to-hub is set")

    return args
