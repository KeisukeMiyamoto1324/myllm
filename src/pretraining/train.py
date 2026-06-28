import argparse
from hashlib import blake2b
import json
import os
from pathlib import Path
import sys

import lightning as L
import torch
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.shared.packed_dataset import build_tokenized_cache
from src.shared.packed_dataset import LocalTokenizedDataset
from src.shared.packed_dataset import PackedCorpusDataset
from src.shared.device_utils import is_global_zero_process
from src.shared.device_utils import resolve_accelerator
from src.shared.device_utils import resolve_device_count
from src.shared.device_utils import resolve_devices
from src.shared.device_utils import resolve_precision
from src.shared.device_utils import resolve_strategy
from src.shared.device_utils import wait_for_file
from src.shared.pytorch_artifacts import push_pytorch_model_artifacts
from src.shared.training_progress import FullTrainingProgressBar
from src.shared.validation_generation import ValidationGenerationCallback
from src.pretraining.training_corpus_cases import PRETRAINING_CORPUS_CASE
from src.pretraining.training_corpus_cases import PretrainingCorpusCase
from src.pretraining.training_corpus_cases import serialize_pretraining_corpus_case
from src.shared.tokenizer import ByteLevelBPE
from src.shared.model.transformer import DecoderOnlyTransformer

from dotenv import load_dotenv
load_dotenv()


PACKING_VERSION = "bucket-packing-v1"


def parse_args() -> argparse.Namespace:
    # ---------------------------------------------------------
    # Define command line arguments used to configure the full
    # training run. Each value has the following responsibility.
    #
    # --max-len:
    # Maximum token length of one training sample. A larger value
    # lets the model see longer context, but increases memory use.
    #
    # --d-model:
    # Hidden dimension size of the Transformer. This controls the
    # width of token representations across the whole network.
    #
    # --num-layers:
    # Number of stacked Transformer blocks. A larger value makes
    # the network deeper and usually increases expressiveness.
    #
    # --num-heads:
    # Number of attention heads used in each block. This decides
    # how many parallel attention patterns are learned at once.
    #
    # --d-ff:
    # Hidden size of the feed-forward sublayer inside each block.
    # This is the expansion dimension used after attention.
    #
    # --learning-rate:
    # Maximum optimizer step size after warmup. Larger values update
    # weights faster, while smaller values tend to be more stable.
    #
    # --lr-warmup-steps:
    # Number of optimizer steps used to linearly increase the
    # learning rate from zero to --learning-rate.
    #
    # --min-learning-rate-ratio:
    # Final cosine-decay learning rate as a ratio of --learning-rate.
    # A positive floor keeps small updates active near train end.
    #
    # --batch-size:
    # Number of samples processed in one forward and backward pass.
    # Larger batches improve throughput but require more device memory.
    #
    # --gradient-accumulation-steps:
    # Number of batches accumulated before one optimizer step. This
    # increases the effective batch size without increasing peak memory.
    #
    # --max-steps:
    # Total number of optimizer steps before training stops. This
    # is the main budget that limits the full training duration.
    #
    # --num-workers:
    # Number of DataLoader worker processes used to prepare data.
    # Increasing this can improve input throughput on CPU-heavy IO.
    #
    # --val-split-modulo:
    # Modulo base for deterministic dataset splitting. Samples are
    # partitioned by index remainder into train and validation sets.
    #
    # --val-split-index:
    # Remainder value reserved for validation samples. With modulo
    # 100 and index 0, roughly 1 percent of samples become validation.
    #
    # --val-batches:
    # Number of validation batches evaluated at each validation run.
    # This caps validation cost so streamed training stays bounded.
    #
    # --validation-cache-path:
    # Optional file path for fixed tokenized validation samples.
    # Empty value stores a cache under the output directory.
    #
    # --val-check-interval:
    # Training step interval used to trigger validation. Smaller
    # values monitor quality more often, with extra compute cost.
    #
    # --checkpoint-every-n-steps:
    # Step interval for saving periodic checkpoints. These files
    # allow training to resume or preserve intermediate states.
    #
    # --metric-log-every-n-steps:
    # Step interval for writing logged metrics to CSV. Larger values
    # reduce disk writes and keep training throughput stable.
    #
    # --loss-chunk-size:
    # Number of sequence positions projected to vocabulary logits at
    # once when computing loss for large vocabulary training.
    #
    # --tokenizer-path:
    # Directory path to the tokenizer artifact. This tokenizer
    # defines the vocabulary and special token ids used in training.
    #
    # --output-path:
    # Directory path used to save the trained model weights,
    # model configuration, and Lightning checkpoints.
    #
    # --resume-from-checkpoint:
    # Lightning checkpoint path used to resume interrupted training
    # with optimizer state, callback state, and global step.
    #
    # --continue-from-model:
    # Trained model.pth path used to initialize weights for a new
    # training run without restoring optimizer state.
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
    # Reject invalid LR schedule settings before any model or
    # streaming dataset state is initialized.
    # ---------------------------------------------------------
    if args.lr_warmup_steps < 0 or args.lr_warmup_steps >= args.max_steps:
        parser.error("--lr-warmup-steps must be greater than or equal to 0 and less than --max-steps")

    if args.min_learning_rate_ratio < 0.0 or args.min_learning_rate_ratio > 1.0:
        parser.error("--min-learning-rate-ratio must be between 0.0 and 1.0")

    if args.gradient_accumulation_steps < 1:
        parser.error("--gradient-accumulation-steps must be greater than or equal to 1")

    try:
        resolve_devices(devices=args.devices)
    except ValueError as error:
        parser.error(str(error))

    # ---------------------------------------------------------
    # Reject missing resume inputs before streaming datasets or
    # model artifacts are opened for the training run.
    # ---------------------------------------------------------
    if args.resume_from_checkpoint and not Path(args.resume_from_checkpoint).is_file():
        parser.error("--resume-from-checkpoint must point to an existing checkpoint file")

    if args.continue_from_model and not Path(args.continue_from_model).is_file():
        parser.error("--continue-from-model must point to an existing model state file")

    if args.push_to_hub and not os.environ.get("HF_REPO"):
        parser.error("HF_REPO is required in the environment when --push-to-hub is set")

    return args


def build_corpus_signature(
    corpus_case: PretrainingCorpusCase,
) -> str:
    # ---------------------------------------------------------
    # Hash the corpus configuration into a short stable cache key
    # so validation files change when the dataset source changes.
    # ---------------------------------------------------------
    payload = serialize_pretraining_corpus_case(corpus_case)
    encoded_payload = json.dumps(payload, sort_keys=True).encode("utf-8")
    return blake2b(encoded_payload, digest_size=8).hexdigest()


def main() -> None:
    # ---------------------------------------------------------
    # Parse the CLI input and load the tokenizer artifact that
    # defines the training vocabulary and special tokens.
    # ---------------------------------------------------------
    args = parse_args()
    tokenizer = ByteLevelBPE.load(Path(args.tokenizer_path))
    accelerator = resolve_accelerator()
    devices = resolve_devices(devices=args.devices)
    device_count = resolve_device_count(accelerator=accelerator, devices=devices)
    strategy = resolve_strategy(accelerator=accelerator, device_count=device_count)
    precision = resolve_precision(accelerator=accelerator)

    # ---------------------------------------------------------
    # Create the output directory and resolve the tokenizer ids
    # needed to stream fixed-length language modeling samples.
    # ---------------------------------------------------------
    model_dir = Path(args.output_path)
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = model_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    validation_sample_count = args.batch_size * args.val_batches * device_count
    corpus_signature = build_corpus_signature(
        corpus_case=PRETRAINING_CORPUS_CASE,
    )
    default_validation_cache_path = (
        model_dir
        / f"validation-cache-{corpus_signature}-{PACKING_VERSION}-len{args.max_len}-samples{validation_sample_count}"
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
    pad_token_id = tokenizer.token_to_id(tokenizer.pad_token)
    bos_token_id = tokenizer.token_to_id(tokenizer.bos_token)
    eos_token_id = tokenizer.token_to_id(tokenizer.eos_token)
    min_learning_rate = args.learning_rate * args.min_learning_rate_ratio

    # ---------------------------------------------------------
    # Stream the single training corpus and build a fixed validation
    # cache from its deterministic validation partition.
    # ---------------------------------------------------------
    train_dataset = PackedCorpusDataset(
        corpus_case=PRETRAINING_CORPUS_CASE,
        tokenizer=tokenizer,
        max_len=args.max_len,
        pad_token_id=pad_token_id,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        split_modulo=args.val_split_modulo,
        split_indexes=train_split_indexes,
    )
    validation_source_dataset = PackedCorpusDataset(
        corpus_case=PRETRAINING_CORPUS_CASE,
        tokenizer=tokenizer,
        max_len=args.max_len,
        pad_token_id=pad_token_id,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        split_modulo=args.val_split_modulo,
        split_indexes=(args.val_split_index,),
    )

    # ---------------------------------------------------------
    # Materialize validation samples once, then read validation
    # batches from local tensors during every validation pass.
    # ---------------------------------------------------------
    validation_cache_metadata = {
        "packing_version": PACKING_VERSION,
        "corpus_signature": corpus_signature,
        "corpus_case": serialize_pretraining_corpus_case(PRETRAINING_CORPUS_CASE),
    }

    if not validation_cache_path.exists() and is_global_zero_process():
        build_tokenized_cache(
            dataset=validation_source_dataset,
            path=validation_cache_path,
            num_samples=validation_sample_count,
            max_len=args.max_len,
            metadata=validation_cache_metadata,
        )

    if not is_global_zero_process():
        wait_for_file(path=validation_cache_path)

    val_dataset = LocalTokenizedDataset(
        path=validation_cache_path,
        max_len=args.max_len,
        num_samples=validation_sample_count,
        metadata=validation_cache_metadata,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=accelerator == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=accelerator == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    # ---------------------------------------------------------
    # Build the Transformer with the tokenizer vocabulary size
    # and train for a bounded number of optimizer steps.
    # ---------------------------------------------------------
    model = DecoderOnlyTransformer(
        num_tokens=tokenizer.get_vocab_size(),
        d_model=args.d_model,
        max_len=args.max_len,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        learning_rate=args.learning_rate,
        pad_token_id=pad_token_id,
        use_fused_optimizer=accelerator == "cuda",
        loss_chunk_size=args.loss_chunk_size,
        lr_warmup_steps=args.lr_warmup_steps,
        lr_total_steps=args.max_steps,
        min_learning_rate=min_learning_rate,
    )

    # ---------------------------------------------------------
    # Initialize a fresh training run from saved model weights
    # when continuing after a completed training job.
    # ---------------------------------------------------------
    if args.continue_from_model:
        model_state = torch.load(
            Path(args.continue_from_model),
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(model_state)

    # ---------------------------------------------------------
    # Save both periodic checkpoints and the best validation model
    # so training progress can be resumed or selected later.
    # ---------------------------------------------------------
    callbacks = [
        FullTrainingProgressBar(),
        ValidationGenerationCallback(
            dataset=val_dataset,
            tokenizer=tokenizer,
            output_dir=model_dir / "validation-generations",
        ),
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

    # ---------------------------------------------------------
    # Store train and validation metrics as CSV with batched writes
    # so loss history can be inspected without slowing training.
    # ---------------------------------------------------------
    metrics_logger = CSVLogger(
        save_dir=model_dir,
        name="metrics",
        version="",
        flush_logs_every_n_steps=args.metric_log_every_n_steps,
    )

    # ---------------------------------------------------------
    # Let Lightning place the model on CUDA or MPS when those
    # backends are available and choose precision for that backend.
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # Pass the checkpoint path to Lightning so interrupted runs
    # restore optimizer state, callbacks, and global step.
    # ---------------------------------------------------------
    checkpoint_path = None

    if args.resume_from_checkpoint:
        checkpoint_path = args.resume_from_checkpoint

    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=checkpoint_path,
    )

    # ---------------------------------------------------------
    # Save the trained weights and configuration so inference
    # can rebuild the same model with the same tokenizer ids.
    # ---------------------------------------------------------
    if not trainer.is_global_zero:
        return

    torch.save(model.state_dict(), model_dir / "model.pth")

    model_config = {
        "max_len": args.max_len,
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "d_ff": args.d_ff,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "devices": args.devices,
        "device_count": device_count,
        "global_batch_size": args.batch_size * device_count,
        "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
        "global_effective_batch_size": args.batch_size * args.gradient_accumulation_steps * device_count,
        "lr_schedule": "warmup_cosine",
        "lr_warmup_steps": args.lr_warmup_steps,
        "min_learning_rate": min_learning_rate,
        "min_learning_rate_ratio": args.min_learning_rate_ratio,
        "loss_chunk_size": args.loss_chunk_size,
        "pad_token_id": pad_token_id,
        "bos_token_id": bos_token_id,
        "eos_token_id": eos_token_id,
        "corpus_signature": corpus_signature,
        "dataset_case": serialize_pretraining_corpus_case(PRETRAINING_CORPUS_CASE),
        "val_split_modulo": args.val_split_modulo,
        "val_split_index": args.val_split_index,
        "validation_cache_path": str(validation_cache_path),
        "validation_sample_count": validation_sample_count,
        "packing_version": PACKING_VERSION,
        "trained_steps": trainer.global_step,
    }

    with open(model_dir / "model_config.json", "w") as f:
        json.dump(model_config, f)

    # ---------------------------------------------------------
    # Save the tokenizer beside the model so PyTorch-only inference
    # can load token ids without rebuilding the tokenizer.
    # ---------------------------------------------------------
    tokenizer.save_pretrained(path=model_dir)

    # ---------------------------------------------------------
    # ---------------------------------------------------------
    # Optionally publish only the PyTorch model artifacts and
    # tokenizer files. Python source files are not uploaded.
    # ---------------------------------------------------------
    if args.push_to_hub:
        push_pytorch_model_artifacts(
            output_path=model_dir,
            repo_id=os.environ["HF_REPO"],
            private=True,
            commit_message="Upload pretrained MyLLM model",
        )


if __name__ == "__main__":
    main()
