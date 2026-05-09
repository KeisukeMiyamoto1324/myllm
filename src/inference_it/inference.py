import sys
from pathlib import Path

# ---------------------------------------------------------
# Add the project root to the import path so direct script
# execution can import the project packages consistently.
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference_base.model_loader import build_model
from src.inference_base.model_loader import load_model_config
from src.inference_it.cli import parse_args
from src.inference_it.generation import generate_chat_response_token_ids
from src.posttraining.chat_template import ChatMessage
from src.pretraining.device_utils import resolve_device
from src.tokenizer.tokenizer import ByteLevelBPE


def build_initial_messages(system_prompt: str) -> list[ChatMessage]:
    # ---------------------------------------------------------
    # Seed the chat history with a system prompt only when the
    # caller provided one on the command line.
    # ---------------------------------------------------------
    messages: list[ChatMessage] = []

    if system_prompt:
        messages.append(ChatMessage(role="system", content=system_prompt))

    return messages


def main() -> None:
    # ---------------------------------------------------------
    # Parse runtime settings and load the chat model artifacts from
    # the selected model directory.
    # ---------------------------------------------------------
    args = parse_args(default_model_dir=PROJECT_ROOT / "models" / "chat-model")
    model_dir = Path(args.model_dir)
    tokenizer = ByteLevelBPE.load(model_dir)
    model_config = load_model_config(model_dir=model_dir)
    device = resolve_device()
    model = build_model(
        tokenizer=tokenizer,
        model_config=model_config,
        model_path=model_dir / "model.pth",
        device=device,
    )
    messages = build_initial_messages(system_prompt=args.system_prompt)

    # ---------------------------------------------------------
    # Run an interactive chat loop that appends each user turn and
    # generated assistant response to the conversation history.
    # ---------------------------------------------------------
    while True:
        user_text = input("user> ").strip()

        if user_text in {"exit", "quit"}:
            break

        messages.append(ChatMessage(role="user", content=user_text))
        generated_token_ids = generate_chat_response_token_ids(
            model=model,
            tokenizer=tokenizer,
            messages=messages,
            max_len=int(model_config["max_len"]),
            max_new_tokens=args.max_new_tokens,
            device=device,
            top_k=args.top_k,
            temperature=args.temperature,
        )
        assistant_text = tokenizer.detokenize(token_ids=generated_token_ids).strip()
        messages.append(ChatMessage(role="assistant", content=assistant_text))
        print(f"assistant> {assistant_text}")


if __name__ == "__main__":
    main()
