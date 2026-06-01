import torch
from transformers import PreTrainedModel
from transformers import PreTrainedTokenizerBase


def resolve_torch_dtype(torch_dtype: str) -> torch.dtype | str:
    # ---------------------------------------------------------
    # Convert CLI dtype names into values accepted by Transformers
    # from_pretrained while preserving its automatic dtype mode.
    # ---------------------------------------------------------
    dtype_by_name: dict[str, torch.dtype | str] = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return dtype_by_name[torch_dtype]


def generate_continuation_text(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
) -> str:
    # ---------------------------------------------------------
    # Tokenize the raw prompt and move tensors onto the model device
    # selected by Transformers.
    # ---------------------------------------------------------
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_token_count = inputs["input_ids"].size(dim=1)

    # ---------------------------------------------------------
    # Delegate continuation decoding to Hugging Face generation and
    # stop on the tokenizer EOS token.
    # ---------------------------------------------------------
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    # ---------------------------------------------------------
    # Decode only newly generated tokens so the continuation does
    # not repeat the original prompt.
    # ---------------------------------------------------------
    generated_ids = output_ids[0, prompt_token_count:]
    return tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()
