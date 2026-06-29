import argparse
import os
from pathlib import Path

from src.shared.cli import require
from src.shared.device_utils import resolve_devices


def parse_args() -> argparse.Namespace:
    # ---------------------------------------------------------
    # Define command line arguments used to configure the full
    # pretraining run and its output artifacts.
    # ---------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--num-layers", type=int, default=16)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--d-ff", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lr-warmup-steps", type=int, default=2000)
    parser.add_argument("--min-learning-rate-ratio", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=16)
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
    # Reject invalid architecture and runtime values before any
    # model, dataset, or output state is initialized.
    # ---------------------------------------------------------
    try:
        require(args.max_len > 0, "--max-len must be greater than 0")
        require(args.d_model > 0, "--d-model must be greater than 0")
        require(args.num_layers > 0, "--num-layers must be greater than 0")
        require(args.num_heads > 0, "--num-heads must be greater than 0")
        require(args.d_ff > 0, "--d-ff must be greater than 0")
        require(args.d_model % args.num_heads == 0, "--d-model must be divisible by --num-heads")
        require((args.d_model // args.num_heads) % 2 == 0, "--d-model divided by --num-heads must be even")
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
        require(
            0 <= args.lr_warmup_steps < args.max_steps,
            "--lr-warmup-steps must be greater than or equal to 0 and less than --max-steps",
        )
        require(
            0.0 <= args.min_learning_rate_ratio <= 1.0,
            "--min-learning-rate-ratio must be between 0.0 and 1.0",
        )
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

    if args.continue_from_model and not Path(args.continue_from_model).is_file():
        parser.error("--continue-from-model must point to an existing model state file")

    if args.push_to_hub and not os.environ.get("HF_REPO"):
        parser.error("HF_REPO is required in the environment when --push-to-hub is set")

    return args
