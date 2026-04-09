import argparse
import json
from pathlib import Path

import lightning as L
import torch
from torch.utils.data import DataLoader

from dataset import get_dataset
from tokenizer import Tokenizer
from transformer import DecoderOnlyTransformer


def parse_args() -> argparse.Namespace:
    # ---------------------------------------------------------
    # Define command line arguments so training hyper-parameters
    # can be adjusted without editing the source file.
    # ---------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-len", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-epochs", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    # ---------------------------------------------------------
    # Parse CLI arguments and prepare the fixed training corpus
    # used in this minimal learning example.
    # ---------------------------------------------------------
    args = parse_args()
    sentences = [
        "what is statquest <EOS> awesome",
        "statquest is what <EOS> great",
    ]

    # ---------------------------------------------------------
    # Create the output directory and tokenizer before building
    # the dataset and data loader.
    # ---------------------------------------------------------
    model_dir = Path(__file__).with_name("model")
    model_dir.mkdir(exist_ok=True)

    tokenizer = Tokenizer()
    tokenizer.learn_vocab(sentences)

    dataset = get_dataset(sentences=sentences, tokenizer=tokenizer.tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size)

    # ---------------------------------------------------------
    # Build the Transformer from the requested hyper-parameters
    # and run Lightning training.
    # ---------------------------------------------------------
    model = DecoderOnlyTransformer(
        num_tokens=len(tokenizer.vocabulary),
        d_model=args.d_model,
        max_len=args.max_len,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        learning_rate=args.learning_rate,
    )

    trainer = L.Trainer(max_epochs=args.max_epochs)
    trainer.fit(model, train_dataloaders=dataloader)

    # ---------------------------------------------------------
    # Save the trained model weights and tokenizer artifacts for
    # later inference.
    # ---------------------------------------------------------
    torch.save(model.state_dict(), model_dir / "model.pth")
    tokenizer.save(model_dir / "tokenizer.json")

    with open(model_dir / "model_config.json", "w") as f:
        json.dump(
            {
                "max_len": args.max_len,
                "d_model": args.d_model,
                "num_layers": args.num_layers,
                "num_heads": args.num_heads,
                "d_ff": args.d_ff,
                "learning_rate": args.learning_rate,
            },
            f,
        )


if __name__ == "__main__":
    # ---------------------------------------------------------
    # Provide a single script entry point for command line usage.
    # ---------------------------------------------------------
    main()
