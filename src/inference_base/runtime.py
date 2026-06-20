import argparse

from src.inference_base.generation import generate_continuation_text
from src.inference_base.generation import resolve_torch_dtype
from src.shared.console import console
from src.shared.device_utils import resolve_device
from src.shared.pytorch_artifacts import load_pytorch_model
from src.shared.pytorch_artifacts import resolve_model_dir
from src.shared.tokenizer import ByteLevelBPE


def run_inference(args: argparse.Namespace) -> None:
    # ---------------------------------------------------------
    # Resolve a local directory or Hub repo id, then load tokenizer
    # and PyTorch model artifacts directly.
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
    # Use the CLI prompt directly when provided. Otherwise, read one
    # prompt from standard input for interactive use.
    # ---------------------------------------------------------
    user_text = args.prompt.strip()

    if not user_text:
        user_text = input("user> ").strip()

    # ---------------------------------------------------------
    # Generate a continuation using the native PyTorch generation
    # settings parsed by the inference CLI.
    # ---------------------------------------------------------
    generated_text = generate_continuation_text(
        model=model,
        tokenizer=tokenizer,
        prompt=user_text,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
    )
    console.print(f"[bold cyan]model>[/bold cyan] {generated_text}")
