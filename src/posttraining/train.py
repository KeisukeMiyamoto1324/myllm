from pathlib import Path
import sys
import argparse

from dotenv import load_dotenv

# ---------------------------------------------------------
# Add the project root so direct script execution can import
# modules through the src package path.
# ---------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.posttraining.artifacts import save_chat_model
from src.posttraining.cli import validate_repeat_epochs
from src.posttraining.dataloaders import build_dataloaders
from src.posttraining.model_setup import build_tokenizer
from src.posttraining.model_setup import DEFAULT_BASE_MODEL_ID
from src.posttraining.model_setup import download_base_model
from src.posttraining.model_setup import load_base_model
from src.posttraining.trainer import train_stage
from src.shared.device_utils import resolve_accelerator
from src.shared.device_utils import resolve_device_count
from src.shared.device_utils import resolve_devices
from src.shared.device_utils import resolve_precision
from src.shared.device_utils import resolve_strategy
load_dotenv()


def parse_args() -> argparse.Namespace:
    # ---------------------------------------------------------
    # Define CLI arguments for Ichikara SFT from a pretrained base
    # model into a chat-oriented model artifact.
    # ---------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-id", type=str, default=DEFAULT_BASE_MODEL_ID)
    parser.add_argument("--output-path", type=str, default="models/lambda-1-160m-it")
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--repeat-epochs", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-batches", type=int, default=8)
    parser.add_argument("--val-check-interval", type=int, default=500)
    parser.add_argument("--checkpoint-every-n-steps", type=int, default=1000)
    parser.add_argument("--metric-log-every-n-steps", type=int, default=50)
    parser.add_argument("--devices", type=str, default="auto")
    args = parser.parse_args()

    try:
        resolve_devices(devices=args.devices)
    except ValueError as error:
        parser.error(str(error))

    return args


def main() -> None:
    # ---------------------------------------------------------
    # Parse CLI input, prepare output storage, and resolve the
    # active accelerator configuration.
    # ---------------------------------------------------------
    args = parse_args()
    validate_repeat_epochs(args=args)
    model_dir = Path(args.output_path)
    model_dir.mkdir(parents=True, exist_ok=True)
    accelerator = resolve_accelerator()
    devices = resolve_devices(devices=args.devices)
    device_count = resolve_device_count(accelerator=accelerator, devices=devices)
    strategy = resolve_strategy(accelerator=accelerator, device_count=device_count)
    precision = resolve_precision(accelerator=accelerator)

    # ---------------------------------------------------------
    # Download and load the base tokenizer and model, then build
    # all SFT dataloaders from the shared chat template.
    # ---------------------------------------------------------
    base_model_dir = download_base_model(base_model_id=args.base_model_id)
    tokenizer = build_tokenizer(base_model_dir=base_model_dir, output_path=model_dir)
    model, model_config = load_base_model(
        base_model_dir=base_model_dir,
        tokenizer=tokenizer,
        learning_rate=args.learning_rate,
        max_len=args.max_len,
        accelerator=accelerator,
    )
    train_dataloader, validation_dataloader, max_steps = build_dataloaders(
        tokenizer=tokenizer,
        max_len=args.max_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        accelerator=accelerator,
        repeat_epochs=args.repeat_epochs,
        device_count=device_count,
    )
    args.posttraining_steps = max_steps
    args.device_count = device_count
    args.global_batch_size = args.batch_size * device_count
    args.global_effective_batch_size = args.batch_size * device_count

    # ---------------------------------------------------------
    # Run Ichikara instruction tuning for the requested number of
    # passes through the train split.
    # ---------------------------------------------------------
    trainer = train_stage(
        model=model,
        model_dir=model_dir,
        stage_name="ichikara",
        max_steps=max_steps,
        train_dataloader=train_dataloader,
        validation_dataloader=validation_dataloader,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        precision=precision,
        args=args,
    )

    # ---------------------------------------------------------
    # Save the final model after Ichikara tuning completes.
    # ---------------------------------------------------------
    if not trainer.is_global_zero:
        return

    save_chat_model(
        model=model,
        model_dir=model_dir,
        model_config=model_config,
        args=args,
        pad_token_id=tokenizer.token_to_id(tokenizer.pad_token),
        bos_token_id=tokenizer.token_to_id(tokenizer.bos_token),
        eos_token_id=tokenizer.token_to_id(tokenizer.eos_token),
        end_of_turn_token_id=tokenizer.token_to_id(tokenizer.end_of_turn_token),
    )


if __name__ == "__main__":
    main()
