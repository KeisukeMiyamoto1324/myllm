from pathlib import Path
import sys

from dotenv import load_dotenv

# ---------------------------------------------------------
# Add the project root so direct script execution can import
# modules through the src package path.
# ---------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.posttraining.artifacts import save_chat_model
from src.posttraining.cli import parse_args
from src.posttraining.cli import validate_step_budget
from src.posttraining.dataloaders import build_dataloaders
from src.posttraining.model_setup import build_tokenizer
from src.posttraining.model_setup import load_base_model
from src.posttraining.trainer import train_stage
from src.pretraining.device_utils import resolve_accelerator
from src.pretraining.device_utils import resolve_precision


load_dotenv()


def main() -> None:
    # ---------------------------------------------------------
    # Parse CLI input, prepare output storage, and resolve the
    # active accelerator configuration.
    # ---------------------------------------------------------
    args = parse_args()
    validate_step_budget(args=args)
    base_model_dir = Path(args.base_model_dir)
    model_dir = Path(args.output_path)
    model_dir.mkdir(parents=True, exist_ok=True)
    accelerator = resolve_accelerator()
    precision = resolve_precision(accelerator=accelerator)

    # ---------------------------------------------------------
    # Load the base tokenizer and model, then build all SFT
    # dataloaders from the shared chat template.
    # ---------------------------------------------------------
    tokenizer = build_tokenizer(base_model_dir=base_model_dir, output_path=model_dir)
    model, model_config = load_base_model(
        base_model_dir=base_model_dir,
        tokenizer=tokenizer,
        learning_rate=args.learning_rate,
        max_len=args.max_len,
        accelerator=accelerator,
    )
    magpie_dataloader, everyday_train_dataloader, everyday_validation_dataloader = build_dataloaders(
        tokenizer=tokenizer,
        max_len=args.max_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        accelerator=accelerator,
    )

    # ---------------------------------------------------------
    # Run broad instruction tuning first, then finish on all
    # high-quality Everyday conversations.
    # ---------------------------------------------------------
    train_stage(
        model=model,
        model_dir=model_dir,
        stage_name="magpie",
        max_steps=args.magpie_steps,
        train_dataloader=magpie_dataloader,
        validation_dataloader=everyday_validation_dataloader,
        accelerator=accelerator,
        precision=precision,
        args=args,
    )
    train_stage(
        model=model,
        model_dir=model_dir,
        stage_name="everyday",
        max_steps=args.everyday_steps,
        train_dataloader=everyday_train_dataloader,
        validation_dataloader=everyday_validation_dataloader,
        accelerator=accelerator,
        precision=precision,
        args=args,
    )

    # ---------------------------------------------------------
    # Save the final model after both stages complete.
    # ---------------------------------------------------------
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
