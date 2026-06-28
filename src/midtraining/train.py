import argparse
from hashlib import blake2b
import json
import os
from pathlib import Path
import sys

import lightning as L
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.midtraining.training_corpus_cases import MIDTRAINING_CORPUS_CASE
from src.midtraining.training_corpus_cases import serialize_midtraining_corpus_case
from src.shared.device_utils import is_global_zero_process
from src.shared.device_utils import resolve_accelerator
from src.shared.device_utils import resolve_device_count
from src.shared.device_utils import resolve_devices
from src.shared.device_utils import resolve_precision
from src.shared.device_utils import resolve_strategy
from src.shared.device_utils import wait_for_file
from src.shared.packed_dataset import build_tokenized_cache
from src.shared.packed_dataset import LocalTokenizedDataset
from src.shared.packed_dataset import PackedCorpusDataset
from src.shared.pytorch_artifacts import load_model_config
from src.shared.pytorch_artifacts import load_pytorch_model
from src.shared.pytorch_artifacts import push_pytorch_model_artifacts
from src.shared.tokenizer import ByteLevelBPE
from src.shared.training_progress import FullTrainingProgressBar
from src.shared.validation_generation import ValidationGenerationCallback

from dotenv import load_dotenv


load_dotenv()


PACKING_VERSION = "bucket-packing-v1"
SHUFFLE_BUFFER_SIZE = 10000
SHUFFLE_SEED = 17


class DatasetEpochCallback(Callback):
    def __init__(self, dataset: PackedCorpusDataset) -> None:
        super().__init__()
        self.dataset = dataset

    def on_train_epoch_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        # ---------------------------------------------------------
        # Use the Lightning epoch index to select the deterministic
        # shuffle order, including after checkpoint resume.
        # ---------------------------------------------------------
        del pl_module
        self.dataset.set_epoch(epoch_index=trainer.current_epoch)


def parse_args() -> argparse.Namespace:
    # ---------------------------------------------------------
    # Define the input model, training runtime, validation budget,
    # output artifacts, and optional epoch-checkpoint resume.
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

    if not model_path.is_dir() or any(not path.is_file() for path in required_model_files):
        parser.error("--model-path must contain model.pth, model_config.json, and tokenizer.json")

    if args.max_len is not None and args.max_len <= 0:
        parser.error("--max-len must be greater than 0")

    if args.learning_rate <= 0.0:
        parser.error("--learning-rate must be greater than 0")

    if args.gradient_accumulation_steps < 1:
        parser.error("--gradient-accumulation-steps must be greater than or equal to 1")

    try:
        resolve_devices(devices=args.devices)
    except ValueError as error:
        parser.error(str(error))

    if args.max_steps <= 0:
        parser.error("--max-steps must be greater than 0")

    if args.checkpoint_every_n_steps <= 0:
        parser.error("--checkpoint-every-n-steps must be greater than 0")

    if args.val_split_modulo <= 1:
        parser.error("--val-split-modulo must be greater than 1")

    if args.val_split_index < 0 or args.val_split_index >= args.val_split_modulo:
        parser.error("--val-split-index must be within --val-split-modulo")

    if args.resume_from_checkpoint and not Path(args.resume_from_checkpoint).is_file():
        parser.error("--resume-from-checkpoint must point to an existing checkpoint file")

    if args.push_to_hub and not os.environ.get("HF_REPO"):
        parser.error("HF_REPO is required in the environment when --push-to-hub is set")

    return args


def build_corpus_signature() -> str:
    # ---------------------------------------------------------
    # Build a stable validation cache key from the exact corpus
    # definition used by this mid-training stage.
    # ---------------------------------------------------------
    payload = serialize_midtraining_corpus_case(MIDTRAINING_CORPUS_CASE)
    encoded_payload = json.dumps(payload, sort_keys=True).encode("utf-8")
    return blake2b(encoded_payload, digest_size=8).hexdigest()


def main() -> None:
    # ---------------------------------------------------------
    # Load the pretrained artifacts and resolve the requested
    # context length before creating datasets or output files.
    # ---------------------------------------------------------
    args = parse_args()
    source_model_dir = Path(args.model_path)
    source_model_config = load_model_config(model_dir=source_model_dir)
    max_len = int(source_model_config["max_len"] if args.max_len is None else args.max_len)
    tokenizer = ByteLevelBPE.load(source_model_dir)
    accelerator = resolve_accelerator()
    devices = resolve_devices(devices=args.devices)
    device_count = resolve_device_count(accelerator=accelerator, devices=devices)
    strategy = resolve_strategy(accelerator=accelerator, device_count=device_count)
    precision = resolve_precision(accelerator=accelerator)
    pad_token_id = tokenizer.token_to_id(tokenizer.pad_token)
    bos_token_id = tokenizer.token_to_id(tokenizer.bos_token)
    eos_token_id = tokenizer.token_to_id(tokenizer.eos_token)

    # ---------------------------------------------------------
    # Prepare output, validation cache, and deterministic corpus
    # partitions shared by the full step-bounded training run.
    # ---------------------------------------------------------
    model_dir = Path(args.output_path)
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = model_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    validation_sample_count = args.batch_size * args.val_batches * device_count
    corpus_signature = build_corpus_signature()
    default_validation_cache_path = (
        model_dir
        / f"validation-cache-{corpus_signature}-{PACKING_VERSION}-len{max_len}-samples{validation_sample_count}"
        f"-split{args.val_split_modulo}-{args.val_split_index}.pt"
    )
    validation_cache_path = (
        Path(args.validation_cache_path) if args.validation_cache_path else default_validation_cache_path
    )
    train_split_indexes = tuple(
        split_index
        for split_index in range(args.val_split_modulo)
        if split_index != args.val_split_index
    )

    # ---------------------------------------------------------
    # Stream finite corpus passes until the configured optimizer
    # step budget is reached, changing shuffle order each pass.
    # ---------------------------------------------------------
    train_dataset = PackedCorpusDataset(
        corpus_case=MIDTRAINING_CORPUS_CASE,
        tokenizer=tokenizer,
        max_len=max_len,
        pad_token_id=pad_token_id,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        split_modulo=args.val_split_modulo,
        split_indexes=train_split_indexes,
        shuffle_buffer_size=SHUFFLE_BUFFER_SIZE,
        shuffle_seed=SHUFFLE_SEED,
    )
    validation_source_dataset = PackedCorpusDataset(
        corpus_case=MIDTRAINING_CORPUS_CASE,
        tokenizer=tokenizer,
        max_len=max_len,
        pad_token_id=pad_token_id,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        split_modulo=args.val_split_modulo,
        split_indexes=(args.val_split_index,),
    )
    validation_cache_metadata = {
        "packing_version": PACKING_VERSION,
        "corpus_signature": corpus_signature,
        "corpus_case": serialize_midtraining_corpus_case(MIDTRAINING_CORPUS_CASE),
    }

    if not validation_cache_path.exists() and is_global_zero_process():
        build_tokenized_cache(
            dataset=validation_source_dataset,
            path=validation_cache_path,
            num_samples=validation_sample_count,
            max_len=max_len,
            metadata=validation_cache_metadata,
        )

    if not is_global_zero_process():
        wait_for_file(path=validation_cache_path)

    val_dataset = LocalTokenizedDataset(
        path=validation_cache_path,
        max_len=max_len,
        num_samples=validation_sample_count,
        metadata=validation_cache_metadata,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=accelerator == "cuda",
        persistent_workers=False,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=accelerator == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    # ---------------------------------------------------------
    # Rebuild the pretrained architecture with optional context
    # length override and a fixed mid-training learning rate.
    # ---------------------------------------------------------
    model, _ = load_pytorch_model(
        model_dir=source_model_dir,
        vocab_size=tokenizer.get_vocab_size(),
        learning_rate=args.learning_rate,
        use_fused_optimizer=accelerator == "cuda",
        max_len=max_len,
    )
    model.loss_chunk_size = args.loss_chunk_size

    # ---------------------------------------------------------
    # Save resumable checkpoints at fixed step intervals, the best
    # validation checkpoint, and batched CSV metrics.
    # ---------------------------------------------------------
    callbacks = [
        FullTrainingProgressBar(),
        ValidationGenerationCallback(
            dataset=val_dataset,
            tokenizer=tokenizer,
            output_dir=model_dir / "validation-generations",
        ),
        DatasetEpochCallback(dataset=train_dataset),
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="step-{step}",
            every_n_train_steps=args.checkpoint_every_n_steps,
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
        LearningRateMonitor(logging_interval="step"),
    ]
    metrics_logger = CSVLogger(
        save_dir=model_dir,
        name="metrics",
        version="",
        flush_logs_every_n_steps=args.metric_log_every_n_steps,
    )
    strategy_kwargs = {"strategy": strategy} if strategy is not None else {}
    trainer = L.Trainer(
        max_steps=args.max_steps,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        callbacks=callbacks,
        logger=metrics_logger,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        log_every_n_steps=args.metric_log_every_n_steps,
        val_check_interval=args.val_check_interval,
        limit_val_batches=args.val_batches,
        num_sanity_val_steps=0,
        enable_progress_bar=False,
        **strategy_kwargs,
    )
    checkpoint_path = args.resume_from_checkpoint or None
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=checkpoint_path,
    )

    # ---------------------------------------------------------
    # Save final weights with inherited architecture metadata and
    # the exact fixed-LR, corpus, and step budget details.
    # ---------------------------------------------------------
    if not trainer.is_global_zero:
        return

    torch.save(model.state_dict(), model_dir / "model.pth")
    model_config = {
        **source_model_config,
        "max_len": max_len,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "devices": args.devices,
        "device_count": device_count,
        "global_batch_size": args.batch_size * device_count,
        "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
        "global_effective_batch_size": args.batch_size * args.gradient_accumulation_steps * device_count,
        "lr_schedule": "fixed",
        "loss_chunk_size": args.loss_chunk_size,
        "pad_token_id": pad_token_id,
        "bos_token_id": bos_token_id,
        "eos_token_id": eos_token_id,
        "midtraining_source_model": str(source_model_dir),
        "midtraining_max_steps": args.max_steps,
        "dataset_case": serialize_midtraining_corpus_case(MIDTRAINING_CORPUS_CASE),
        "corpus_signature": corpus_signature,
        "val_split_modulo": args.val_split_modulo,
        "val_split_index": args.val_split_index,
        "validation_cache_path": str(validation_cache_path),
        "validation_sample_count": validation_sample_count,
        "packing_version": PACKING_VERSION,
        "shuffle_buffer_size": SHUFFLE_BUFFER_SIZE,
        "shuffle_seed": SHUFFLE_SEED,
        "trained_steps": trainer.global_step,
    }

    # ---------------------------------------------------------
    # Remove the inherited pretraining scheduler fields because
    # mid-training uses one fixed learning rate for all epochs.
    # ---------------------------------------------------------
    obsolete_config_keys = [
        "lr_warmup_steps",
        "min_learning_rate",
        "min_learning_rate_ratio",
        "midtraining_epochs",
        "midtraining_completed_epochs",
    ]

    for key in obsolete_config_keys:
        model_config.pop(key, None)

    with open(model_dir / "model_config.json", "w") as file:
        json.dump(model_config, file)

    tokenizer.save_pretrained(path=model_dir)

    # ---------------------------------------------------------
    # Optionally publish the final PyTorch artifacts without
    # checkpoints, validation tensors, metrics, or source code.
    # ---------------------------------------------------------
    if args.push_to_hub:
        push_pytorch_model_artifacts(
            output_path=model_dir,
            repo_id=os.environ["HF_REPO"],
            private=True,
            commit_message="Upload mid-trained MyLLM model",
        )


if __name__ == "__main__":
    main()
