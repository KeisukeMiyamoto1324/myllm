import torch

from src.shared.model.kv_cache import KeyValueCache
from src.shared.model.transformer import DecoderOnlyTransformer
from src.shared.tokenizer import ByteLevelBPE


def resolve_torch_dtype(torch_dtype: str) -> torch.dtype | None:
    # ---------------------------------------------------------
    # Convert CLI dtype names into PyTorch dtypes. None keeps the
    # model weights in their saved default dtype.
    # ---------------------------------------------------------
    dtype_by_name: dict[str, torch.dtype | None] = {
        "auto": None,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return dtype_by_name[torch_dtype]


def suppress_repeated_ngrams(
    logits: torch.Tensor,
    generated_ids: list[int],
    no_repeat_ngram_size: int,
) -> torch.Tensor:
    # ---------------------------------------------------------
    # Block tokens that would recreate an existing n-gram in the
    # generated context, matching the CLI constraint.
    # ---------------------------------------------------------
    if no_repeat_ngram_size <= 0 or len(generated_ids) + 1 < no_repeat_ngram_size:
        return logits

    prefix_size = no_repeat_ngram_size - 1
    current_prefix = generated_ids[-prefix_size:]
    blocked_tokens = [
        generated_ids[index + prefix_size]
        for index in range(len(generated_ids) - no_repeat_ngram_size + 1)
        if generated_ids[index : index + prefix_size] == current_prefix
    ]
    logits[blocked_tokens] = -torch.inf
    return logits


def apply_repetition_penalty(
    logits: torch.Tensor,
    generated_ids: list[int],
    repetition_penalty: float,
) -> torch.Tensor:
    # ---------------------------------------------------------
    # Penalize tokens already seen in the context before sampling
    # or greedy selection chooses the next token.
    # ---------------------------------------------------------
    if repetition_penalty == 1.0:
        return logits

    used_token_ids = set(generated_ids)
    token_indexes = torch.tensor(list(used_token_ids), device=logits.device, dtype=torch.long)
    token_logits = logits[token_indexes]
    logits[token_indexes] = torch.where(
        token_logits < 0,
        token_logits * repetition_penalty,
        token_logits / repetition_penalty,
    )
    return logits


def filter_top_k_top_p(
    logits: torch.Tensor,
    top_k: int,
    top_p: float,
) -> torch.Tensor:
    # ---------------------------------------------------------
    # Keep only the configured top-k and nucleus candidates before
    # sampling the next token from the probability distribution.
    # ---------------------------------------------------------
    if top_k > 0:
        top_values = torch.topk(logits, k=min(top_k, logits.size(dim=-1))).values
        logits[logits < top_values[-1]] = -torch.inf

    sorted_logits, sorted_indexes = torch.sort(logits, descending=True)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    sorted_mask = cumulative_probs > top_p
    sorted_mask[1:] = sorted_mask[:-1].clone()
    sorted_mask[0] = False
    logits[sorted_indexes[sorted_mask]] = -torch.inf
    return logits


def select_next_token(
    logits: torch.Tensor,
    generated_ids: list[int],
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
) -> int:
    # ---------------------------------------------------------
    # Apply generation constraints and select the next token by
    # sampling or greedy argmax.
    # ---------------------------------------------------------
    next_logits = logits.clone()
    next_logits = apply_repetition_penalty(
        logits=next_logits,
        generated_ids=generated_ids,
        repetition_penalty=repetition_penalty,
    )
    next_logits = suppress_repeated_ngrams(
        logits=next_logits,
        generated_ids=generated_ids,
        no_repeat_ngram_size=no_repeat_ngram_size,
    )

    if not do_sample:
        return int(torch.argmax(next_logits).item())

    next_logits = filter_top_k_top_p(
        logits=next_logits / temperature,
        top_k=top_k,
        top_p=top_p,
    )
    probabilities = torch.softmax(next_logits, dim=-1)
    return int(torch.multinomial(probabilities, num_samples=1).item())


def generate_token_ids(
    model: DecoderOnlyTransformer,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    eos_token_id: int,
) -> list[int]:
    # ---------------------------------------------------------
    # Generate one token at a time using the model's native cache
    # path instead of Transformers generation utilities.
    # ---------------------------------------------------------
    generated_ids = [int(token_id) for token_id in input_ids[0].tolist()]
    past_key_values: KeyValueCache | None = None
    current_input_ids = input_ids
    new_token_ids: list[int] = []

    for _ in range(max_new_tokens):
        logits, past_key_values = model.forward_with_cache(
            token_ids=current_input_ids,
            past_key_values=past_key_values,
        )
        next_token_id = select_next_token(
            logits=logits[0, -1, :],
            generated_ids=generated_ids,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
        generated_ids.append(next_token_id)
        new_token_ids.append(next_token_id)

        if next_token_id == eos_token_id:
            break

        current_input_ids = torch.tensor(
            [[next_token_id]],
            dtype=torch.long,
            device=input_ids.device,
        )

    return new_token_ids


def generate_continuation_text(
    model: DecoderOnlyTransformer,
    tokenizer: ByteLevelBPE,
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
    # Tokenize the prompt with BOS and decode only newly generated
    # token ids from the PyTorch generation loop.
    # ---------------------------------------------------------
    bos_token_id = tokenizer.token_to_id(tokenizer.bos_token)
    eos_token_id = tokenizer.token_to_id(tokenizer.eos_token)
    prompt_ids = [bos_token_id, *tokenizer.tokenize(prompt)]
    input_ids = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=next(model.parameters()).device,
    )

    with torch.no_grad():
        generated_ids = generate_token_ids(
            model=model,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            eos_token_id=eos_token_id,
        )

    return tokenizer.detokenize(generated_ids).strip()
