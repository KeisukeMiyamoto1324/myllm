import argparse
import os
from pathlib import Path

from src.shared.cli import require
from src.shared.device_utils import resolve_devices


def parse_args() -> argparse.Namespace:
    # ---------------------------------------------------------
    # Define command line arguments used to configure the full
    # pretraining run and reject invalid values before training.
    # ---------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--num-layers", type=int, default=16)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--d-ff", type=int, default=3072)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lr-warmup-steps", type=int, default=2000)
    parser.add_argument("--min-learning-rate-ratio", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=20480)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-split-modulo", type=int, default=100)
    parser.add_argument("--val-split-index", type=int, default=0)
    parser.add_argument("--val-batches", type=int, default=64)
    parser.add_argument("--validation-cache-path", type=str, default="")
    parser.add_argument("--val-check-interval", type=int, default=1000)
    parser.add_argument("--checkpoint-every-n-steps", type=int, default=2000)
    parser.add_argument("--metric-log-every-n-steps", type=int, default=500)
    parser.add_argument("--loss-chunk-size", type=int, default=32)
    parser.add_argument("--devices", type=str, default="auto")
    parser.add_argument("--tokenizer-path", type=str, default="models/tokenizer")
    parser.add_argument("--output-path", type=str, default="models/lambda-160m")
    parser.add_argument("--push-to-hub", action="store_true")

    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument("--resume-from-checkpoint", type=str, default="")
    resume_group.add_argument("--continue-from-model", type=str, default="")

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Validate scheduler, split, device, resume, and upload inputs
    # before any dataset stream or model artifact is opened.
    # ---------------------------------------------------------
    require(
        args.lr_warmup_steps >= 0 and args.lr_warmup_steps < args.max_steps,
        parser,
        "--lr-warmup-steps must be greater than or equal to 0 and less than --max-steps",
    )
    require(
        0.0 <= args.min_learning_rate_ratio <= 1.0,
        parser,
        "--min-learning-rate-ratio must be between 0.0 and 1.0",
    )
    require(
        args.gradient_accumulation_steps >= 1,
        parser,
        "--gradient-accumulation-steps must be greater than or equal to 1",
    )
    require(args.max_len > 0, parser, "--max-len must be greater than 0")
    require(args.loss_chunk_size > 0, parser, "--loss-chunk-size must be greater than 0")
    require(args.val_split_modulo > 1, parser, "--val-split-modulo must be greater than 1")
    require(
        0 <= args.val_split_index < args.val_split_modulo,
        parser,
        "--val-split-index must be within --val-split-modulo",
    )

    try:
        resolve_devices(devices=args.devices)
    except ValueError as error:
        require(False, parser, str(error))

    require(
        not args.resume_from_checkpoint or Path(args.resume_from_checkpoint).is_file(),
        parser,
        "--resume-from-checkpoint must point to an existing checkpoint file",
    )
    require(
        not args.continue_from_model or Path(args.continue_from_model).is_file(),
        parser,
        "--continue-from-model must point to an existing model state file",
    )
    require(
        not args.push_to_hub or bool(os.environ.get("HF_REPO")),
        parser,
        "HF_REPO is required in the environment when --push-to-hub is set",
    )

    return args
