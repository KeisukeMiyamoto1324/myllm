import sys
from pathlib import Path

# ---------------------------------------------------------
# Add the project root to the import path so direct script
# execution can import the project packages consistently.
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.cli import parse_args
from src.inference.generation import generate_token_ids
from src.inference.model_loader import build_model, load_model_config
from src.model.device_utils import resolve_device
from src.tokenizer_rust.tokenizer import ByteLevelBPE


def main() -> None:
    # ---------------------------------------------------------
    # Parse the CLI arguments and resolve the saved model files
    # used to run inference with the trained artifacts.
    # ---------------------------------------------------------
    args = parse_args(default_model_dir=PROJECT_ROOT / "models" / "model-10m")
    model_dir = Path(args.model_dir)
    model_path = model_dir / "model.pth"
    tokenizer_path = model_dir / "tokenizer.json"
    device = resolve_device()

    # ---------------------------------------------------------
    # Load the tokenizer and model configuration before the
    # model weights are reconstructed on the active device.
    # ---------------------------------------------------------
    tokenizer = ByteLevelBPE.load(tokenizer_path)
    model_config = load_model_config(model_dir=model_dir)
    model = build_model(
        tokenizer=tokenizer,
        model_config=model_config,
        model_path=model_path,
        device=device,
    )

    # ---------------------------------------------------------
    # Generate token ids from the prompt and decode them into
    # text with the same tokenizer used during training.
    # ---------------------------------------------------------
    generated_token_ids = generate_token_ids(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_len=int(model_config["max_len"]),
        max_new_tokens=args.max_new_tokens,
        device=device,
        top_k=args.top_k,
        temperature=args.temperature,
    )
    generated_text = tokenizer.detokenize(token_ids=generated_token_ids)
    print(f"predicted tokens: \n{generated_text}")


if __name__ == "__main__":
    main()
