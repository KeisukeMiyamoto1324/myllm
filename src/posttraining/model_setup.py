from pathlib import Path
import shutil

from src.inference_base.model_loader import build_model
from src.inference_base.model_loader import load_model_config
from src.pretraining.device_utils import resolve_device
from src.pretraining.transformer import DecoderOnlyTransformer
from src.tokenizer.tokenizer import ByteLevelBPE


def build_tokenizer(base_model_dir: Path, output_path: Path) -> ByteLevelBPE:
    # ---------------------------------------------------------
    # Load the base tokenizer and copy it beside the chat model
    # artifacts so inference can resolve a self-contained model.
    # ---------------------------------------------------------
    tokenizer_path = base_model_dir / "tokenizer.json"
    output_tokenizer_path = output_path / "tokenizer.json"
    shutil.copyfile(tokenizer_path, output_tokenizer_path)
    return ByteLevelBPE.load(output_tokenizer_path)


def load_base_model(
    base_model_dir: Path,
    tokenizer: ByteLevelBPE,
    learning_rate: float,
    max_len: int,
    accelerator: str,
) -> tuple[DecoderOnlyTransformer, dict[str, int | float]]:
    # ---------------------------------------------------------
    # Recreate the base Transformer from its saved configuration
    # and restore weights before continuing SFT.
    # ---------------------------------------------------------
    model_config = load_model_config(model_dir=base_model_dir)
    model_config["learning_rate"] = learning_rate
    model_config["max_len"] = max_len
    model = build_model(
        tokenizer=tokenizer,
        model_config=model_config,
        model_path=base_model_dir / "model.pth",
        device=resolve_device(),
    )
    model.learning_rate = learning_rate
    model.use_fused_optimizer = accelerator == "cuda"
    model.train()
    return model, model_config
