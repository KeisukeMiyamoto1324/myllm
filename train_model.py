import argparse
import json
from pathlib import Path

import lightning as L
import torch
from torch.utils.data import DataLoader

from dataset import FineWebEduDataset
from device_utils import resolve_accelerator
from tokenizer_rust.tokenizer import ByteLevelBPE
from transformer import DecoderOnlyTransformer

from dotenv import load_dotenv
load_dotenv()


def parse_args() -> argparse.Namespace:
    # ---------------------------------------------------------
    # Define command line arguments so training can control the
    # model size and streamed optimization budget.
    # ---------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=4)
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
    pad_token_id = tokenizer.token_to_id(tokenizer.pad_token)
    eos_token_id = tokenizer.token_to_id(tokenizer.eos_token)

    # ---------------------------------------------------------
    # Stream FineWeb-Edu directly from Hugging Face so training
    # can scale without loading the full dataset into memory.
    # ---------------------------------------------------------
    dataset = FineWebEduDataset(
        tokenizer=tokenizer,
        max_len=args.max_len,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
    )
    dataloader = DataLoader(
        dataset,
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
    # Let Lightning place the model on CUDA or MPS when those
    # backends are available on the current machine.
    # ---------------------------------------------------------
    trainer = L.Trainer(
        max_steps=args.max_steps,
        accelerator=accelerator,
        devices=1,
    )
    trainer.fit(model, train_dataloaders=dataloader)

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
            },
            f,
        )


if __name__ == "__main__":
    main()
