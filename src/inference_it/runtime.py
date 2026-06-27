import argparse

from src.inference_it.generation import generate_chat_response
from src.inference_it.generation import resolve_torch_dtype
from src.posttraining.chat_template import ChatMessage
from src.shared.console import console
from src.shared.device_utils import resolve_device
from src.shared.model.transformer import DecoderOnlyTransformer
from src.shared.pytorch_artifacts import load_pytorch_model
from src.shared.pytorch_artifacts import resolve_model_dir
from src.shared.tokenizer import ByteLevelBPE


def generate_and_print_response(
    args: argparse.Namespace,
    model: DecoderOnlyTransformer,
    tokenizer: ByteLevelBPE,
    messages: list[ChatMessage],
) -> str:
    # ---------------------------------------------------------
    # Generate one assistant message from the current conversation
    # and print it in the same terminal style each time.
    # ---------------------------------------------------------
    response = generate_chat_response(
        model=model,
        tokenizer=tokenizer,
        messages=messages,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
    )
    console.print(f"[bold cyan]model>[/bold cyan] {response}")
    return response


def run_inference(args: argparse.Namespace) -> None:
    # ---------------------------------------------------------
    # Resolve a local directory or Hub repo id, then load tokenizer
    # and PyTorch chat model artifacts directly.
    # ---------------------------------------------------------
    model_dir = resolve_model_dir(model_source=args.model_dir)
    tokenizer = ByteLevelBPE.load(model_dir)
    model, _ = load_pytorch_model(
        model_dir=model_dir,
        vocab_size=tokenizer.get_vocab_size(),
    )
    device = resolve_device()
    torch_dtype = resolve_torch_dtype(torch_dtype=args.torch_dtype)
    model = model.to(device=device)

    if torch_dtype is not None:
        model = model.to(dtype=torch_dtype)

    model.eval()

    # ---------------------------------------------------------
    # Use a single prompt when provided, otherwise keep reading
    # terminal input and preserving chat history.
    # ---------------------------------------------------------
    messages: list[ChatMessage] = []
    user_text = args.prompt.strip()

    if user_text:
        messages.append(ChatMessage(role="user", content=user_text))
        generate_and_print_response(args=args, model=model, tokenizer=tokenizer, messages=messages)
        return

    while True:
        user_text = input("user> ").strip()

        if user_text in {"", "exit", "quit"}:
            return

        messages.append(ChatMessage(role="user", content=user_text))
        response = generate_and_print_response(
            args=args,
            model=model,
            tokenizer=tokenizer,
            messages=messages,
        )
        messages.append(ChatMessage(role="assistant", content=response))
