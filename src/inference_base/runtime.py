import argparse

from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

from src.inference_base.generation import generate_continuation_text
from src.inference_base.generation import resolve_torch_dtype


def run_inference(args: argparse.Namespace) -> None:
    # ---------------------------------------------------------
    # Load Hugging Face-compatible tokenizer and model artifacts
    # from either a local directory or Hub repository id.
    # ---------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        device_map="auto",
        dtype=resolve_torch_dtype(torch_dtype=args.torch_dtype),
    )
    model.eval()

    # ---------------------------------------------------------
    # Use the CLI prompt directly when provided. Otherwise, read one
    # prompt from standard input for interactive use.
    # ---------------------------------------------------------
    user_text = args.prompt.strip()

    if not user_text:
        user_text = input("user> ").strip()

    # ---------------------------------------------------------
    # Generate a continuation using the shared generation settings
    # parsed by the inference CLI.
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
    )
    print(f"model> {generated_text}")
