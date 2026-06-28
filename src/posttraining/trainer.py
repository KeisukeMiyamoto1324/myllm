import argparse
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader

from src.shared.model.transformer import DecoderOnlyTransformer
from src.shared.training_progress import FullTrainingProgressBar


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
    devices: int | str = 1,
    strategy: str | None = None,
) -> L.Trainer:
    # ---------------------------------------------------------
    # Create callbacks and metrics logging for one SFT stage while
    # keeping stage artifacts separated under the output directory.
    # ---------------------------------------------------------
    checkpoint_dir = model_dir / "checkpoints" / stage_name
    callbacks = [
        FullTrainingProgressBar(),
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
    # optimizer steps and validates by global training step.
    # ---------------------------------------------------------
    strategy_kwargs = {"strategy": strategy} if strategy is not None else {}
    return L.Trainer(
        max_steps=max_steps,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        callbacks=callbacks,
        logger=metrics_logger,
        log_every_n_steps=metric_log_every_n_steps,
        val_check_interval=val_check_interval,
        check_val_every_n_epoch=None,
        limit_val_batches=val_batches,
        num_sanity_val_steps=0,
        enable_progress_bar=False,
        **strategy_kwargs,
    )


def fit_stage(
    model: DecoderOnlyTransformer,
    trainer: L.Trainer,
    train_dataloader: DataLoader,
    validation_dataloader: DataLoader,
) -> None:
    # ---------------------------------------------------------
    # Run one trainer stage against the supplied train stream and
    # shared validation set.
    # ---------------------------------------------------------
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=validation_dataloader,
    )


def train_stage(
    model: DecoderOnlyTransformer,
    model_dir: Path,
    stage_name: str,
    max_steps: int,
    train_dataloader: DataLoader,
    validation_dataloader: DataLoader,
    accelerator: str,
    devices: int | str,
    strategy: str | None,
    precision: str,
    args: argparse.Namespace,
) -> L.Trainer:
    # ---------------------------------------------------------
    # Build and execute one named SFT stage using the shared CLI
    # training controls for validation, checkpointing, and logging.
    # ---------------------------------------------------------
    trainer = build_trainer(
        model_dir=model_dir,
        stage_name=stage_name,
        max_steps=max_steps,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        precision=precision,
        val_check_interval=args.val_check_interval,
        val_batches=args.val_batches,
        checkpoint_every_n_steps=args.checkpoint_every_n_steps,
        metric_log_every_n_steps=args.metric_log_every_n_steps,
    )
    fit_stage(
        model=model,
        trainer=trainer,
        train_dataloader=train_dataloader,
        validation_dataloader=validation_dataloader,
    )
    return trainer
