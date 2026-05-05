import argparse
import json
from pathlib import Path
import shutil
import sys

import lightning as L
import torch
from dotenv import load_dotenv
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader

# ---------------------------------------------------------
# Add the project root so direct script execution can import
# modules through the src package path.
# ---------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.inference_base.model_loader import build_model
from src.inference_base.model_loader import load_model_config
from src.posttraining.dataset import EVERYDAY_DATASET_PATH
from src.posttraining.dataset import EVERYDAY_TRAIN_SPLIT
from src.posttraining.dataset import EVERYDAY_VALIDATION_SPLIT
from src.posttraining.dataset import MAGPIE_DATASET_PATH
from src.posttraining.dataset import MAGPIE_DATASET_SPLIT
from src.posttraining.dataset import EverydayChatDataset
from src.posttraining.dataset import MagpieChatDataset
from src.pretraining.device_utils import resolve_accelerator
from src.pretraining.device_utils import resolve_device
from src.pretraining.device_utils import resolve_precision
from src.pretraining.transformer import DecoderOnlyTransformer
from src.tokenizer.tokenizer import ByteLevelBPE


load_dotenv()


def parse_args() -> argparse.Namespace:
    # ---------------------------------------------------------
    # Define CLI arguments for two-stage SFT from a pretrained
    # base model into a chat-oriented model artifact.
    # ---------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-dir", type=str, required=True)
    parser.add_argument("--output-path", type=str, default="models/chat-model")
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=12000)
    parser.add_argument("--magpie-steps", type=int, default=11000)
    parser.add_argument("--everyday-steps", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-batches", type=int, default=8)
    parser.add_argument("--val-check-interval", type=int, default=500)
    parser.add_argument("--checkpoint-every-n-steps", type=int, default=1000)
    parser.add_argument("--metric-log-every-n-steps", type=int, default=50)
    return parser.parse_args()


def build_tokenizer(base_model_dir: Path, output_path: Path) -> ByteLevelBPE:
    # ---------------------------------------------------------
    # Load the base tokenizer and copy it beside the chat model
    # artifacts so inference can resolve a self-contained model.
    # ---------------------------------------------------------
    tokenizer_path = base_model_dir / "tokenizer.json"
    output_tokenizer_path = output_path / "tokenizer.json"
    shutil.copyfile(tokenizer_path, output_tokenizer_path)
    return ByteLevelBPE.load(output_tokenizer_path)


def load_base_model(
    base_model_dir: Path,
    tokenizer: ByteLevelBPE,
    learning_rate: float,
    max_len: int,
    accelerator: str,
) -> tuple[DecoderOnlyTransformer, dict[str, int | float]]:
    # ---------------------------------------------------------
    # Recreate the base Transformer from its saved configuration
    # and restore weights before continuing SFT.
    # ---------------------------------------------------------
    model_config = load_model_config(model_dir=base_model_dir)
    model_config["learning_rate"] = learning_rate
    model_config["max_len"] = max_len
    model = build_model(
        tokenizer=tokenizer,
        model_config=model_config,
        model_path=base_model_dir / "model.pth",
        device=resolve_device(),
    )
    model.learning_rate = learning_rate
    model.use_fused_optimizer = accelerator == "cuda"
    model.train()
    return model, model_config


def build_dataloaders(
    tokenizer: ByteLevelBPE,
    max_len: int,
    batch_size: int,
    num_workers: int,
    accelerator: str,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    # ---------------------------------------------------------
    # Resolve tokenizer ids shared by both SFT datasets and the
    # Transformer loss masking convention.
    # ---------------------------------------------------------
    pad_token_id = tokenizer.token_to_id(tokenizer.pad_token)
    bos_token_id = tokenizer.token_to_id(tokenizer.bos_token)
    eos_token_id = tokenizer.token_to_id(tokenizer.eos_token)
    end_of_turn_token_id = tokenizer.token_to_id(tokenizer.end_of_turn_token)

    # ---------------------------------------------------------
    # Build broad Magpie training, high-quality Everyday finishing,
    # and fixed Everyday validation datasets.
    # ---------------------------------------------------------
    magpie_dataset = MagpieChatDataset(
        tokenizer=tokenizer,
        max_len=max_len,
        pad_token_id=pad_token_id,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        end_of_turn_token_id=end_of_turn_token_id,
    )
    everyday_train_dataset = EverydayChatDataset(
        tokenizer=tokenizer,
        split=EVERYDAY_TRAIN_SPLIT,
        max_len=max_len,
        pad_token_id=pad_token_id,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        end_of_turn_token_id=end_of_turn_token_id,
    )
    everyday_validation_dataset = EverydayChatDataset(
        tokenizer=tokenizer,
        split=EVERYDAY_VALIDATION_SPLIT,
        max_len=max_len,
        pad_token_id=pad_token_id,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        end_of_turn_token_id=end_of_turn_token_id,
    )
    use_pin_memory = accelerator == "cuda"
    use_persistent_workers = num_workers > 0

    # ---------------------------------------------------------
    # Wrap datasets with DataLoaders configured consistently with
    # the existing pretraining pipeline.
    # ---------------------------------------------------------
    magpie_dataloader = DataLoader(
        magpie_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=use_persistent_workers,
    )
    everyday_train_dataloader = DataLoader(
        everyday_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=use_persistent_workers,
    )
    everyday_validation_dataloader = DataLoader(
        everyday_validation_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=use_persistent_workers,
    )
    return magpie_dataloader, everyday_train_dataloader, everyday_validation_dataloader


def build_trainer(
    model_dir: Path,
    stage_name: str,
    max_steps: int,
    accelerator: str,
    precision: str,
    val_check_interval: int,
    val_batches: int,
    checkpoint_every_n_steps: int,
    metric_log_every_n_steps: int,
) -> L.Trainer:
    # ---------------------------------------------------------
    # Create callbacks and metrics logging for one SFT stage while
    # keeping stage artifacts separated under the output directory.
    # ---------------------------------------------------------
    checkpoint_dir = model_dir / "checkpoints" / stage_name
    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="step-{step}",
            every_n_train_steps=checkpoint_every_n_steps,
            save_top_k=-1,
            save_on_train_epoch_end=False,
        ),
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="best-step={step}-val_loss={val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
        ),
    ]
    metrics_logger = CSVLogger(
        save_dir=model_dir,
        name=f"metrics-{stage_name}",
        version="",
        flush_logs_every_n_steps=metric_log_every_n_steps,
    )

    # ---------------------------------------------------------
    # Return a Lightning trainer that runs a bounded number of
    # optimizer steps for the requested SFT stage.
    # ---------------------------------------------------------
    return L.Trainer(
        max_steps=max_steps,
        accelerator=accelerator,
        devices=1,
        precision=precision,
        callbacks=callbacks,
        logger=metrics_logger,
        log_every_n_steps=metric_log_every_n_steps,
        val_check_interval=val_check_interval,
        limit_val_batches=val_batches,
        num_sanity_val_steps=0,
    )


def fit_stage(
    model: DecoderOnlyTransformer,
    trainer: L.Trainer,
    train_dataloader: DataLoader,
    validation_dataloader: DataLoader,
) -> None:
    # ---------------------------------------------------------
    # Run one trainer stage against the supplied train stream and
    # shared Everyday validation set.
    # ---------------------------------------------------------
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=validation_dataloader,
    )


def save_chat_model(
    model: DecoderOnlyTransformer,
    model_dir: Path,
    model_config: dict[str, int | float],
    args: argparse.Namespace,
    pad_token_id: int,
    bos_token_id: int,
    eos_token_id: int,
    end_of_turn_token_id: int,
) -> None:
    # ---------------------------------------------------------
    # Save the final chat-tuned weights and metadata needed by
    # inference to rebuild the same architecture.
    # ---------------------------------------------------------
    torch.save(model.state_dict(), model_dir / "model.pth")

    # ---------------------------------------------------------
    # Persist posttraining provenance alongside architecture fields
    # inherited from the base model configuration.
    # ---------------------------------------------------------
    payload = {
        **model_config,
        "max_len": args.max_len,
        "learning_rate": args.learning_rate,
        "pad_token_id": pad_token_id,
        "bos_token_id": bos_token_id,
        "eos_token_id": eos_token_id,
        "end_of_turn_token_id": end_of_turn_token_id,
        "base_model_dir": args.base_model_dir,
        "chat_template_version": 1,
        "posttraining_datasets": [
            f"{MAGPIE_DATASET_PATH}:{MAGPIE_DATASET_SPLIT}",
            f"{EVERYDAY_DATASET_PATH}:{EVERYDAY_TRAIN_SPLIT}",
        ],
        "validation_dataset": f"{EVERYDAY_DATASET_PATH}:{EVERYDAY_VALIDATION_SPLIT}",
        "magpie_steps": args.magpie_steps,
        "everyday_steps": args.everyday_steps,
    }

    with open(model_dir / "model_config.json", "w") as f:
        json.dump(payload, f)


def validate_step_budget(args: argparse.Namespace) -> None:
    # ---------------------------------------------------------
    # Keep the explicit stage budgets aligned with the advertised
    # total budget so command-line mistakes fail before training.
    # ---------------------------------------------------------
    if args.magpie_steps + args.everyday_steps != args.max_steps:
        raise ValueError("magpie_steps plus everyday_steps must equal max_steps")


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
    magpie_trainer = build_trainer(
        model_dir=model_dir,
        stage_name="magpie",
        max_steps=args.magpie_steps,
        accelerator=accelerator,
        precision=precision,
        val_check_interval=args.val_check_interval,
        val_batches=args.val_batches,
        checkpoint_every_n_steps=args.checkpoint_every_n_steps,
        metric_log_every_n_steps=args.metric_log_every_n_steps,
    )
    fit_stage(
        model=model,
        trainer=magpie_trainer,
        train_dataloader=magpie_dataloader,
        validation_dataloader=everyday_validation_dataloader,
    )
    everyday_trainer = build_trainer(
        model_dir=model_dir,
        stage_name="everyday",
        max_steps=args.everyday_steps,
        accelerator=accelerator,
        precision=precision,
        val_check_interval=args.val_check_interval,
        val_batches=args.val_batches,
        checkpoint_every_n_steps=args.checkpoint_every_n_steps,
        metric_log_every_n_steps=args.metric_log_every_n_steps,
    )
    fit_stage(
        model=model,
        trainer=everyday_trainer,
        train_dataloader=everyday_train_dataloader,
        validation_dataloader=everyday_validation_dataloader,
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
