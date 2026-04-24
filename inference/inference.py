import argparse
import json
from pathlib import Path

import torch

from model.device_utils import resolve_device
from tokenizer_rust.tokenizer import ByteLevelBPE
from model.transformer import DecoderOnlyTransformer


def parse_args() -> argparse.Namespace:
    # ---------------------------------------------------------
    # Define the CLI arguments required to select the prompt,
    # model directory, and generation length for inference.
    # ---------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", type=str)
    parser.add_argument("--model-dir", type=str, default="../model-10m")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    return parser.parse_args()


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

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def generate_token_ids(
    model: DecoderOnlyTransformer,
    tokenizer: ByteLevelBPE,
    prompt: str,
    max_len: int,
    max_new_tokens: int,
    device: torch.device,
) -> list[int]:
    # ---------------------------------------------------------
    # Encode the prompt into token ids and move the initial
    # sequence onto the target device for autoregressive use.
    # ---------------------------------------------------------
    prompt_token_ids = tokenizer.tokenize(sentence=prompt)
    model_input = torch.tensor(prompt_token_ids, dtype=torch.long).unsqueeze(0).to(device)
    eos_token_id = tokenizer.token_to_id(tokenizer.eos_token)
    max_generation_steps = min(max_len, len(prompt_token_ids) + max_new_tokens)

    # ---------------------------------------------------------
    # Predict one token at a time until EOS appears or the
    # configured generation length limit is reached.
    # ---------------------------------------------------------
    with torch.no_grad():
        predictions = model(model_input)
        predicted_id = torch.argmax(predictions[:, -1, :], dim=-1, keepdim=True)
        generated_ids = predicted_id

        for _ in range(model_input.size(dim=1), max_generation_steps):
            if predicted_id.item() == eos_token_id:
                break

            model_input = torch.cat((model_input, predicted_id), dim=1)
            predictions = model(model_input)
            predicted_id = torch.argmax(predictions[:, -1, :], dim=-1, keepdim=True)
            generated_ids = torch.cat((generated_ids, predicted_id), dim=1)

    return generated_ids.squeeze(0).detach().cpu().tolist()


def main() -> None:
    # ---------------------------------------------------------
    # Parse the CLI arguments and resolve the saved model files
    # used to run inference with the trained artifacts.
    # ---------------------------------------------------------
    args = parse_args()
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
    )
    generated_text = tokenizer.detokenize(token_ids=generated_token_ids)
    print(f"predicted tokens: \n{generated_text}")


if __name__ == "__main__":
    main()
