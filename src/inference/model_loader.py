import json
from pathlib import Path

import torch

from src.model.transformer import DecoderOnlyTransformer
from src.tokenizer_rust.tokenizer import ByteLevelBPE


def load_model_config(model_dir: Path) -> dict[str, int | float]:
    # ---------------------------------------------------------
    # Load the saved model configuration so inference rebuilds
    # the same Transformer architecture used for training.
    # ---------------------------------------------------------
    with open(model_dir / "model_config.json") as f:
        return json.load(f)


def build_model(
    tokenizer: ByteLevelBPE,
    model_config: dict[str, int | float],
    model_path: Path,
    device: torch.device,
) -> DecoderOnlyTransformer:
    # ---------------------------------------------------------
    # Recreate the Transformer from the saved hyper-parameters
    # and restore the trained weights onto the target device.
    # ---------------------------------------------------------
    model = DecoderOnlyTransformer(
        num_tokens=tokenizer.get_vocab_size(),
        d_model=int(model_config["d_model"]),
        max_len=int(model_config["max_len"]),
        num_layers=int(model_config["num_layers"]),
        num_heads=int(model_config["num_heads"]),
        d_ff=int(model_config["d_ff"]),
        learning_rate=float(model_config["learning_rate"]),
        pad_token_id=int(model_config["pad_token_id"]),
    )

    # ---------------------------------------------------------
    # Load weights, move the model to the selected device, and
    # switch it to evaluation mode before generation.
    # ---------------------------------------------------------
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
