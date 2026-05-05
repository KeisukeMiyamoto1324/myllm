import argparse
import json
from pathlib import Path

import torch

from src.posttraining.dataset import EVERYDAY_DATASET_PATH
from src.posttraining.dataset import EVERYDAY_TRAIN_SPLIT
from src.posttraining.dataset import EVERYDAY_VALIDATION_SPLIT
from src.posttraining.dataset import MAGPIE_DATASET_PATH
from src.posttraining.dataset import MAGPIE_DATASET_SPLIT
from src.pretraining.transformer import DecoderOnlyTransformer


def save_chat_model(
    model: DecoderOnlyTransformer,
    model_dir: Path,
    model_config: dict[str, int | float],
    args: argparse.Namespace,
    pad_token_id: int,
    bos_token_id: int,
    eos_token_id: int,
    end_of_turn_token_id: int,
) -> None:
    # ---------------------------------------------------------
    # Save the final chat-tuned weights and metadata needed by
    # inference to rebuild the same architecture.
    # ---------------------------------------------------------
    torch.save(model.state_dict(), model_dir / "model.pth")

    # ---------------------------------------------------------
    # Persist posttraining provenance alongside architecture fields
    # inherited from the base model configuration.
    # ---------------------------------------------------------
    payload = {
        **model_config,
        "max_len": args.max_len,
        "learning_rate": args.learning_rate,
        "pad_token_id": pad_token_id,
        "bos_token_id": bos_token_id,
        "eos_token_id": eos_token_id,
        "end_of_turn_token_id": end_of_turn_token_id,
        "base_model_dir": args.base_model_dir,
        "chat_template_version": 1,
        "posttraining_datasets": [
            f"{MAGPIE_DATASET_PATH}:{MAGPIE_DATASET_SPLIT}",
            f"{EVERYDAY_DATASET_PATH}:{EVERYDAY_TRAIN_SPLIT}",
        ],
        "validation_dataset": f"{EVERYDAY_DATASET_PATH}:{EVERYDAY_VALIDATION_SPLIT}",
        "magpie_steps": args.magpie_steps,
        "everyday_steps": args.everyday_steps,
    }

    with open(model_dir / "model_config.json", "w") as f:
        json.dump(payload, f)
