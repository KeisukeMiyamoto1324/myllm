import sys
from pathlib import Path

from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

# ---------------------------------------------------------
# Add the project root to the import path so direct script
# execution can import the project packages consistently.
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference_base.cli import parse_args
from src.inference_base.generation import generate_continuation_text
from src.inference_base.generation import resolve_torch_dtype


def main() -> None:
    # ---------------------------------------------------------
    # Parse runtime settings and load Hugging Face-compatible
    # artifacts from a local directory or Hub repository id.
    # ---------------------------------------------------------
    args = parse_args(default_model_dir=PROJECT_ROOT / "models" / "model-160m-v1")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        device_map="auto",
        dtype=resolve_torch_dtype(torch_dtype=args.torch_dtype),
    )
    model.eval()

    # ---------------------------------------------------------
    # Run an interactive continuation loop where the user's text is
    # passed directly to the language model as the prompt.
    # ---------------------------------------------------------
    user_text = input("user> ").strip()

    generated_text = generate_continuation_text(
        model=model,
        tokenizer=tokenizer,
        prompt=user_text,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )
    print(f"model> {generated_text}")


if __name__ == "__main__":
    main()
