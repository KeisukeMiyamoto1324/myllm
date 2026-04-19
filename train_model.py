import argparse
import json
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from dataset import FineWebEduDataset
from device_utils import resolve_accelerator
from tokenizer_rust.tokenizer import ByteLevelBPE
from transformer import DecoderOnlyTransformer

from dotenv import load_dotenv
load_dotenv()


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
    # Optimizer step size. Larger values update weights faster,
    # while smaller values tend to make training more stable.
    #
    # --batch-size:
    # Number of samples processed in one optimizer step. Larger
    # batches improve throughput but require more device memory.
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
    # --val-check-interval:
    # Training step interval used to trigger validation. Smaller
    # values monitor quality more often, with extra compute cost.
    #
    # --checkpoint-every-n-steps:
    # Step interval for saving periodic checkpoints. These files
    # allow training to resume or preserve intermediate states.
    #
    # --tokenizer-path:
    # File path to the tokenizer JSON artifact. This tokenizer
    # defines the vocabulary and special token ids used in training.
    # ---------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=16)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=25600)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-split-modulo", type=int, default=100)
    parser.add_argument("--val-split-index", type=int, default=0)
    parser.add_argument("--val-batches", type=int, default=64)
    parser.add_argument("--val-check-interval", type=int, default=1000)
    parser.add_argument("--checkpoint-every-n-steps", type=int, default=1000)
    parser.add_argument("--tokenizer-path", type=str, default="model/tokenizer.json")
    return parser.parse_args()


def main() -> None:
    # ---------------------------------------------------------
    # Parse the CLI input and load the tokenizer artifact that
    # defines the training vocabulary and special tokens.
    # ---------------------------------------------------------
    args = parse_args()
    tokenizer = ByteLevelBPE.load(Path(args.tokenizer_path))
    accelerator = resolve_accelerator()

    # ---------------------------------------------------------
    # Create the output directory and resolve the tokenizer ids
    # needed to stream fixed-length language modeling samples.
    # ---------------------------------------------------------
    model_dir = Path(__file__).with_name("model")
    model_dir.mkdir(exist_ok=True)
    checkpoint_dir = model_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    train_split_indexes = tuple(
        split_index
        for split_index in range(args.val_split_modulo)
        if split_index != args.val_split_index
    )
    pad_token_id = tokenizer.token_to_id(tokenizer.pad_token)
    eos_token_id = tokenizer.token_to_id(tokenizer.eos_token)

    # ---------------------------------------------------------
    # Stream FineWeb-Edu directly from Hugging Face and split the
    # stream deterministically into training and validation sets.
    # ---------------------------------------------------------
    train_dataset = FineWebEduDataset(
        tokenizer=tokenizer,
        max_len=args.max_len,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        split_modulo=args.val_split_modulo,
        split_indexes=train_split_indexes,
    )
    val_dataset = FineWebEduDataset(
        tokenizer=tokenizer,
        max_len=args.max_len,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        split_modulo=args.val_split_modulo,
        split_indexes=(args.val_split_index,),
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
    )

    # ---------------------------------------------------------
    # Save both periodic checkpoints and the best validation model
    # so training progress can be resumed or selected later.
    # ---------------------------------------------------------
    callbacks = [
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
    ]

    # ---------------------------------------------------------
    # Let Lightning place the model on CUDA or MPS when those
    # backends are available and run periodic validation checks.
    # ---------------------------------------------------------
    trainer = L.Trainer(
        max_steps=args.max_steps,
        accelerator=accelerator,
        devices=1,
        callbacks=callbacks,
        val_check_interval=args.val_check_interval,
        limit_val_batches=args.val_batches,
        num_sanity_val_steps=0,
    )
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    # ---------------------------------------------------------
    # Save the trained weights and configuration so inference
    # can rebuild the same model with the same tokenizer ids.
    # ---------------------------------------------------------
    torch.save(model.state_dict(), model_dir / "model.pth")

    with open(model_dir / "model_config.json", "w") as f:
        json.dump(
            {
                "max_len": args.max_len,
                "d_model": args.d_model,
                "num_layers": args.num_layers,
                "num_heads": args.num_heads,
                "d_ff": args.d_ff,
                "learning_rate": args.learning_rate,
                "pad_token_id": pad_token_id,
                "val_split_modulo": args.val_split_modulo,
                "val_split_index": args.val_split_index,
            },
            f,
        )


if __name__ == "__main__":
    main()
