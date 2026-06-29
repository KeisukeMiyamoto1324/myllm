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

from src.pretraining.cli import parse_args
from src.shared.packed_dataset import build_tokenized_cache
from src.shared.packed_dataset import collate_packed_examples
from src.shared.packed_dataset import LocalTokenizedDataset
from src.shared.packed_dataset import PACKING_VERSION
from src.shared.packed_dataset import PackedCorpusDataset
from src.shared.packed_dataset import SHUFFLE_BUFFER_SIZE
from src.shared.packed_dataset import SHUFFLE_SEED
from src.shared.device_utils import is_global_zero_process
from src.shared.device_utils import resolve_accelerator
from src.shared.device_utils import resolve_device_count
from src.shared.device_utils import resolve_devices
from src.shared.device_utils import resolve_precision
from src.shared.device_utils import resolve_strategy
from src.shared.device_utils import wait_for_file
from src.shared.pytorch_artifacts import push_pytorch_model_artifacts
from src.shared.training_plan import show_training_token_plan
from src.shared.training_progress import FullTrainingProgressBar
from src.shared.validation_generation import ValidationGenerationCallback
from src.pretraining.training_corpus_cases import PRETRAINING_CORPUS_CASE
from src.pretraining.training_corpus_cases import PretrainingCorpusCase
from src.pretraining.training_corpus_cases import serialize_pretraining_corpus_case
from src.shared.tokenizer import ByteLevelBPE
from src.shared.model.transformer import DecoderOnlyTransformer

from dotenv import load_dotenv
load_dotenv()

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
        shuffle_buffer_size=SHUFFLE_BUFFER_SIZE,
        shuffle_seed=SHUFFLE_SEED,
        repeat=True,
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
        collate_fn=collate_packed_examples,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=accelerator == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_packed_examples,
    )

    if is_global_zero_process():
        estimated_train_tokens = show_training_token_plan(
            stage_name="pretraining",
            max_steps=args.max_steps,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            batch_size=args.batch_size,
            device_count=device_count,
            max_len=args.max_len,
        )
    else:
        estimated_train_tokens = 0

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
        "ffn_type": "gelu",
        "attention_backend": "pytorch_sdpa_masked",
        "requires_cuda": False,
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
        "shuffle_buffer_size": SHUFFLE_BUFFER_SIZE,
        "shuffle_seed": SHUFFLE_SEED,
        "estimated_train_tokens": estimated_train_tokens,
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
