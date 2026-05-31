import json
from pathlib import Path
import shutil

from huggingface_hub import HfApi

from .configuration_myllm import MyLLMConfig
from .modeling_myllm import MyLLMForCausalLM
from .transformer import DecoderOnlyTransformer


def build_hf_config(
    model_config: dict[str, int | float | str | list[object]],
    vocab_size: int,
) -> MyLLMConfig:
    # ---------------------------------------------------------
    # Convert the training metadata into the compact Transformers
    # config needed to rebuild the inference architecture.
    # ---------------------------------------------------------
    return MyLLMConfig(
        vocab_size=vocab_size,
        max_len=int(model_config["max_len"]),
        d_model=int(model_config["d_model"]),
        num_layers=int(model_config["num_layers"]),
        num_heads=int(model_config["num_heads"]),
        d_ff=int(model_config["d_ff"]),
        learning_rate=float(model_config["learning_rate"]),
        pad_token_id=int(model_config["pad_token_id"]),
        bos_token_id=int(model_config["bos_token_id"]),
        eos_token_id=int(model_config["eos_token_id"]),
        architectures=["MyLLMForCausalLM"],
    )


def save_hf_pretrained_artifacts(
    model: DecoderOnlyTransformer,
    model_config: dict[str, int | float | str | list[object]],
    vocab_size: int,
    output_path: Path,
) -> None:
    # ---------------------------------------------------------
    # Register custom classes so save_pretrained writes the AutoMap
    # entries required by trust_remote_code loading.
    # ---------------------------------------------------------
    MyLLMConfig.register_for_auto_class()
    MyLLMForCausalLM.register_for_auto_class("AutoModelForCausalLM")

    # ---------------------------------------------------------
    # Build the HF wrapper, copy trained weights into the wrapped
    # PyTorch model, and save safetensors plus remote-code files.
    # ---------------------------------------------------------
    hf_config = build_hf_config(
        model_config=model_config,
        vocab_size=vocab_size,
    )
    hf_model = MyLLMForCausalLM(config=hf_config)
    hf_model.transformer.load_state_dict(model.state_dict())
    hf_model.save_pretrained(
        save_directory=output_path,
        safe_serialization=True,
    )

    # ---------------------------------------------------------
    # Map both AutoModel and AutoModelForCausalLM to the same
    # wrapper so generic AutoModel calls also resolve successfully.
    # ---------------------------------------------------------
    config_path = output_path / "config.json"
    config_data = json.loads(config_path.read_text(encoding="utf-8"))
    auto_map = config_data["auto_map"]
    auto_map["AutoModel"] = auto_map["AutoModelForCausalLM"]
    config_path.write_text(
        json.dumps(config_data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def push_hf_pretrained_artifacts(
    output_path: Path,
    repo_id: str,
    private: bool,
    commit_message: str,
) -> None:
    # ---------------------------------------------------------
    # Create or reuse the model repository, then upload only the
    # portable inference artifacts and skip local training outputs.
    # ---------------------------------------------------------
    api = HfApi()
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
        ignore_patterns=[
            "checkpoints/*",
            "metrics/*",
            "validation-cache-*",
        ],
    )


def copy_inference_code(output_path: Path) -> None:
    # ---------------------------------------------------------
    # Keep direct file copies predictable for tests and for Hub
    # uploads where remote-code dependencies must sit together.
    # ---------------------------------------------------------
    source_dir = Path(__file__).resolve().parent
    file_names = [
        "configuration_myllm.py",
        "modeling_myllm.py",
        "transformer.py",
        "self_attention.py",
        "position_encoding.py",
        "kv_cache.py",
    ]

    for file_name in file_names:
        shutil.copyfile(source_dir / file_name, output_path / file_name)
