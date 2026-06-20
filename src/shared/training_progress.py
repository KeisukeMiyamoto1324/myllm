from typing import Any

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback

from src.shared.console import progress_manager


class FullTrainingProgressBar(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.task_id: int | None = None

    def on_fit_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        # ---------------------------------------------------------
        # Track optimizer steps across all epochs so the displayed
        # ETA represents the remaining time for the complete run.
        # ---------------------------------------------------------
        del pl_module

        if not trainer.is_global_zero:
            return

        self.task_id = progress_manager.add_task(
            description="Full training",
            total=trainer.max_steps,
            completed=trainer.global_step,
        )

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        # ---------------------------------------------------------
        # Advance only when gradient accumulation produces one
        # optimizer update, matching Lightning's max_steps counter.
        # ---------------------------------------------------------
        del pl_module, outputs, batch, batch_idx

        if self.task_id is None or not trainer.is_global_zero:
            return

        progress_manager.update(
            task_id=self.task_id,
            completed=trainer.global_step,
        )
        self._update_metrics(trainer=trainer)

    def on_validation_epoch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        # ---------------------------------------------------------
        # Add the latest aggregated validation loss while retaining
        # the current training loss beside the full-run ETA.
        # ---------------------------------------------------------
        del pl_module
        self._update_metrics(trainer=trainer)

    def on_fit_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        # ---------------------------------------------------------
        # Close the full-run progress bar after the current training
        # stage and its validation work have completed.
        # ---------------------------------------------------------
        del pl_module

        if self.task_id is not None and trainer.is_global_zero:
            progress_manager.finish_task(task_id=self.task_id)

    def _update_metrics(self, trainer: L.Trainer) -> None:
        # ---------------------------------------------------------
        # Format Lightning metrics so values logged with prog_bar
        # remain visible on the shared Rich progress row.
        # ---------------------------------------------------------
        if self.task_id is None or not trainer.is_global_zero:
            return

        metrics = [
            f"{name}={value.item():.3f}" if isinstance(value, torch.Tensor) else f"{name}={float(value):.3f}"
            for name, value in trainer.progress_bar_metrics.items()
            if name in {"train_loss", "val_loss"}
        ]

        progress_manager.update(
            task_id=self.task_id,
            metrics=" ".join(metrics),
        )
