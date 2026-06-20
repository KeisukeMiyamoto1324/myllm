from typing import Any

import lightning as L
from lightning.pytorch.callbacks import Callback
from tqdm.auto import tqdm


class FullTrainingProgressBar(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.progress_bar: tqdm | None = None

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
        self.progress_bar = tqdm(
            total=trainer.max_steps,
            initial=trainer.global_step,
            desc="Full training",
            unit="step",
            dynamic_ncols=True,
            disable=not trainer.is_global_zero,
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

        if self.progress_bar is None:
            return

        completed_steps = trainer.global_step - self.progress_bar.n

        if completed_steps > 0:
            self.progress_bar.update(completed_steps)

    def on_fit_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        # ---------------------------------------------------------
        # Close the full-run progress bar after the current training
        # stage and its validation work have completed.
        # ---------------------------------------------------------
        del trainer, pl_module

        if self.progress_bar is not None:
            self.progress_bar.close()
