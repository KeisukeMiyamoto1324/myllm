import json
from pathlib import Path

import torch
from huggingface_hub import HfApi
from huggingface_hub import snapshot_download

from src.shared.model.transformer import DecoderOnlyTransformer


ModelConfig = dict[str, int | float | str | list[object]]


def resolve_model_dir(model_source: str | Path) -> Path:
    # ---------------------------------------------------------
    # Use local model directories directly. Download Hub model
    # snapshots before loading PyTorch-only model artifacts.
    # ---------------------------------------------------------
    model_path = Path(model_source)

    if model_path.exists():
        return model_path

    return Path(snapshot_download(repo_id=str(model_source), repo_type="model"))


def load_model_config(model_dir: Path) -> ModelConfig:
    # ---------------------------------------------------------
    # Read the model architecture metadata saved beside model.pth
    # so the PyTorch module can be rebuilt directly.
    # ---------------------------------------------------------
    config_path = model_dir / "model_config.json"

    with open(config_path) as f:
        return json.load(f)


def build_model_from_config(
    model_config: ModelConfig,
    vocab_size: int,
    learning_rate: float | None = None,
    use_fused_optimizer: bool = False,
    max_len: int | None = None,
    lr_warmup_steps: int | None = None,
    lr_total_steps: int | None = None,
    min_learning_rate: float | None = None,
) -> DecoderOnlyTransformer:
    # ---------------------------------------------------------
    # Recreate the local Transformer from saved architecture
    # values and optional caller-specific training settings.
    # ---------------------------------------------------------
    return DecoderOnlyTransformer(
        num_tokens=vocab_size,
        d_model=int(model_config["d_model"]),
        max_len=int(model_config["max_len"] if max_len is None else max_len),
        num_layers=int(model_config["num_layers"]),
        num_heads=int(model_config["num_heads"]),
        d_ff=int(model_config["d_ff"]),
        learning_rate=float(model_config["learning_rate"] if learning_rate is None else learning_rate),
        pad_token_id=int(model_config["pad_token_id"]),
        use_fused_optimizer=use_fused_optimizer,
        lr_warmup_steps=lr_warmup_steps,
        lr_total_steps=lr_total_steps,
        min_learning_rate=min_learning_rate,
    )


def load_pytorch_model(
    model_dir: Path,
    vocab_size: int,
    learning_rate: float | None = None,
    use_fused_optimizer: bool = False,
    map_location: str | torch.device = "cpu",
    max_len: int | None = None,
    lr_warmup_steps: int | None = None,
    lr_total_steps: int | None = None,
    min_learning_rate: float | None = None,
) -> tuple[DecoderOnlyTransformer, ModelConfig]:
    # ---------------------------------------------------------
    # Load PyTorch weights directly from model.pth and return both
    # the ready model and its saved configuration.
    # ---------------------------------------------------------
    model_config = load_model_config(model_dir=model_dir)
    model = build_model_from_config(
        model_config=model_config,
        vocab_size=vocab_size,
        learning_rate=learning_rate,
        use_fused_optimizer=use_fused_optimizer,
        max_len=max_len,
        lr_warmup_steps=lr_warmup_steps,
        lr_total_steps=lr_total_steps,
        min_learning_rate=min_learning_rate,
    )
    model_state = torch.load(
        model_dir / "model.pth",
        map_location=map_location,
        weights_only=True,
    )

    # ---------------------------------------------------------
    # Keep the regenerated sinusoidal table when callers request a
    # different context length from the saved model metadata.
    # ---------------------------------------------------------
    changed_max_len = max_len is not None and int(model_config["max_len"]) != max_len

    if changed_max_len:
        model_state.pop("pe.pe")

    model.load_state_dict(model_state, strict=not changed_max_len)
    return model, model_config


def push_pytorch_model_artifacts(
    output_path: Path,
    repo_id: str,
    private: bool,
    commit_message: str,
    token: str | None = None,
) -> None:
    # ---------------------------------------------------------
    # Upload only PyTorch model artifacts and tokenizer metadata.
    # Training outputs and Python source files stay local.
    # ---------------------------------------------------------
    api = HfApi(token=token)
    api.create_repo(
        repo_id=repo_id,
        private=private,
        repo_type="model",
        exist_ok=True,
    )
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=output_path,
        commit_message=commit_message,
        allow_patterns=[
            "model.pth",
            "model_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "added_tokens.json",
        ],
        ignore_patterns=[
            "*.py",
            "checkpoints/*",
            "metrics/*",
            "validation-cache-*",
        ],
    )
