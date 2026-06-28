from pathlib import Path

from huggingface_hub import snapshot_download

from src.shared.device_utils import is_global_zero_process
from src.shared.device_utils import resolve_device
from src.shared.device_utils import wait_for_file
from src.shared.pytorch_artifacts import load_pytorch_model
from src.shared.model.transformer import DecoderOnlyTransformer
from src.shared.tokenizer import ByteLevelBPE


DEFAULT_BASE_MODEL_ID = "MK0727/lambda-1-160m-base"


def download_base_model(base_model_id: str) -> Path:
    # ---------------------------------------------------------
    # Download the Hub snapshot so tokenizer and model artifacts
    # are available through the existing local loaders.
    # ---------------------------------------------------------
    return Path(snapshot_download(repo_id=base_model_id, repo_type="model"))


def build_tokenizer(base_model_dir: Path, output_path: Path) -> ByteLevelBPE:
    # ---------------------------------------------------------
    # Load the base tokenizer and save it beside the chat model
    # artifacts as a Hugging Face tokenizer directory.
    # ---------------------------------------------------------
    tokenizer = ByteLevelBPE.load(base_model_dir)

    if is_global_zero_process():
        tokenizer.save_pretrained(output_path)

    if not is_global_zero_process():
        wait_for_file(path=output_path / "tokenizer.json")

    return tokenizer


def build_model_config(
    model: DecoderOnlyTransformer,
    learning_rate: float,
    pad_token_id: int,
    bos_token_id: int,
    eos_token_id: int,
) -> dict[str, int | float]:
    # ---------------------------------------------------------
    # Build the compact config used by legacy and Hugging Face
    # artifact writers after posttraining completes.
    # ---------------------------------------------------------
    first_block = model.blocks[0]
    return {
        "max_len": model.pe.pe.size(dim=0),
        "d_model": model.we.embedding_dim,
        "num_layers": len(model.blocks),
        "num_heads": first_block.attention.num_heads,
        "d_ff": first_block.feed_forward.linear_1.out_features,
        "learning_rate": learning_rate,
        "pad_token_id": pad_token_id,
        "bos_token_id": bos_token_id,
        "eos_token_id": eos_token_id,
    }


def load_base_model(
    base_model_dir: Path,
    tokenizer: ByteLevelBPE,
    learning_rate: float,
    max_len: int,
    accelerator: str,
) -> tuple[DecoderOnlyTransformer, dict[str, int | float]]:
    # ---------------------------------------------------------
    # Load PyTorch model artifacts directly and prepare every layer
    # for fine tuning without a model wrapper.
    # ---------------------------------------------------------
    model, _ = load_pytorch_model(
        model_dir=base_model_dir,
        vocab_size=tokenizer.get_vocab_size(),
        learning_rate=learning_rate,
        use_fused_optimizer=accelerator == "cuda",
    )
    model = model.to(resolve_device())
    model.learning_rate = learning_rate
    model.use_fused_optimizer = accelerator == "cuda"
    model.train()

    # ---------------------------------------------------------
    # Keep architecture metadata aligned with the downloaded model
    # and the tokenizer ids used by the SFT datasets.
    # ---------------------------------------------------------
    model_config = build_model_config(
        model=model,
        learning_rate=learning_rate,
        pad_token_id=tokenizer.token_to_id(tokenizer.pad_token),
        bos_token_id=tokenizer.token_to_id(tokenizer.bos_token),
        eos_token_id=tokenizer.token_to_id(tokenizer.eos_token),
    )
    return model, model_config
