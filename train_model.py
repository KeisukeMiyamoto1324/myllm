import argparse
import json
from pathlib import Path

import lightning as L
import torch
from torch.utils.data import DataLoader

from dataset import get_dataset
from text_preprocessor import load_sentences
from tokenizer_rust.tokenizer import ByteLevelBPE
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
    parser.add_argument("--train-data", type=str, default="dataset/train.jsonl")
    parser.add_argument("--tokenizer-path", type=str, default="model/tokenizer.json")
    return parser.parse_args()


def format_sentences(
    raw_sentences: list[str],
    max_len: int,
    eos_token: str,
    pad_token: str,
) -> list[str]:
    # ---------------------------------------------------------
    # Normalize each sentence into the fixed-length training
    # format with tokenizer-defined EOS and PAD tokens.
    # ---------------------------------------------------------
    return [
        _format_sentence(
            sentence=sentence,
            max_len=max_len,
            eos_token=eos_token,
            pad_token=pad_token,
        )
        for sentence in raw_sentences
    ]


def _format_sentence(
    sentence: str,
    max_len: int,
    eos_token: str,
    pad_token: str,
) -> str:
    # ---------------------------------------------------------
    # Truncate the sentence, append EOS, and pad the remainder
    # so every sample keeps the same token count.
    # ---------------------------------------------------------
    tokens = sentence.split()[: max_len - 1]
    padded_tokens = tokens + [eos_token]
    padding_size = max_len - len(padded_tokens)
    return " ".join(padded_tokens + [pad_token for _ in range(padding_size)])


def encode_sentence(tokenizer: ByteLevelBPE, sentence: str) -> torch.Tensor:
    # ---------------------------------------------------------
    # Convert one sentence into a tensor of token ids so the
    # existing dataset builder can keep its interface.
    # ---------------------------------------------------------
    token_ids = tokenizer.tokenize(sentence=sentence)
    return torch.tensor(token_ids, dtype=torch.long)


def main() -> None:
    # ---------------------------------------------------------
    # Parse CLI arguments and resolve the training data and
    # tokenizer artifact paths.
    # ---------------------------------------------------------
    args = parse_args()
    train_data_path = Path(args.train_data)
    tokenizer_path = Path(args.tokenizer_path)

    # ---------------------------------------------------------
    # Load the raw training text and apply the shared sentence
    # formatting rule before dataset construction.
    # ---------------------------------------------------------
    tokenizer = ByteLevelBPE.load(tokenizer_path)
    raw_sentences = load_sentences(path=train_data_path)
    sentences = format_sentences(
        raw_sentences=raw_sentences,
        max_len=args.max_len,
        eos_token=tokenizer.eos_token,
        pad_token=tokenizer.pad_token,
    )

    # ---------------------------------------------------------
    # Create the output directory and derive the PAD token id
    # before building the dataset and data loader.
    # ---------------------------------------------------------
    model_dir = Path(__file__).with_name("model")
    model_dir.mkdir(exist_ok=True)

    pad_token_id = tokenizer.token_to_id(tokenizer.pad_token)

    dataset = get_dataset(
        sentences=sentences,
        tokenizer=lambda sentence: encode_sentence(tokenizer=tokenizer, sentence=sentence),
        pad_token_id=pad_token_id,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size)

    # ---------------------------------------------------------
    # Build the Transformer from the requested hyper-parameters
    # and run Lightning training.
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

    trainer = L.Trainer(max_epochs=args.max_epochs)
    trainer.fit(model, train_dataloaders=dataloader)

    # ---------------------------------------------------------
    # Save the trained model weights and configuration for later
    # inference with the existing tokenizer artifact.
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
