import torch

from src.inference_base.generation import resolve_torch_dtype
from src.inference_base.generation import select_next_token
from src.posttraining.chat_template import ChatMessage
from src.posttraining.chat_template import get_role_token
from src.posttraining.chat_template import normalize_role
from src.shared.model.kv_cache import KeyValueCache
from src.shared.model.transformer import DecoderOnlyTransformer
from src.shared.tokenizer import ByteLevelBPE


def build_chat_input_ids(
    tokenizer: ByteLevelBPE,
    messages: list[ChatMessage],
) -> list[int]:
    # ---------------------------------------------------------
    # Serialize chat history with the same role and turn markers
    # used during instruction tuning.
    # ---------------------------------------------------------
    bos_token_id = tokenizer.token_to_id(tokenizer.bos_token)
    end_of_turn_token_id = tokenizer.token_to_id(tokenizer.end_of_turn_token)
    input_ids = [bos_token_id]

    for message in messages:
        role = normalize_role(role=message.role)
        role_token_id = tokenizer.token_to_id(get_role_token(tokenizer=tokenizer, role=role))
        content_token_ids = tokenizer.tokenize(sentence=message.content)
        input_ids.extend([role_token_id, *content_token_ids, end_of_turn_token_id])

    # ---------------------------------------------------------
    # Add the assistant role marker as the generation cue for the
    # next model response.
    # ---------------------------------------------------------
    assistant_token_id = tokenizer.token_to_id(tokenizer.assistant_token)
    input_ids.append(assistant_token_id)
    return input_ids


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
    stop_token_ids: set[int],
) -> list[int]:
    # ---------------------------------------------------------
    # Generate one assistant response and stop on EOS or the chat
    # end-of-turn marker.
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

        if next_token_id in stop_token_ids:
            break

        current_input_ids = torch.tensor(
            [[next_token_id]],
            dtype=torch.long,
            device=input_ids.device,
        )

    return new_token_ids


def generate_chat_response(
    model: DecoderOnlyTransformer,
    tokenizer: ByteLevelBPE,
    messages: list[ChatMessage],
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
) -> str:
    # ---------------------------------------------------------
    # Apply the chat template, generate the assistant span, and
    # decode only the response tokens.
    # ---------------------------------------------------------
    prompt_ids = build_chat_input_ids(tokenizer=tokenizer, messages=messages)
    input_ids = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=next(model.parameters()).device,
    )
    stop_token_ids = {
        tokenizer.token_to_id(tokenizer.eos_token),
        tokenizer.token_to_id(tokenizer.end_of_turn_token),
    }

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
            stop_token_ids=stop_token_ids,
        )

    response_ids = [token_id for token_id in generated_ids if token_id not in stop_token_ids]
    return tokenizer.detokenize(response_ids).strip()
