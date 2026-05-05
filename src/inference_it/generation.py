import torch

from src.inference_base.sampling import sample_next_token_id
from src.inference_it.chat_template import build_chat_prompt_token_ids
from src.posttraining.chat_template import ChatMessage
from src.pretraining.transformer import DecoderOnlyTransformer
from src.tokenizer.tokenizer import ByteLevelBPE


def generate_chat_response_token_ids(
    model: DecoderOnlyTransformer,
    tokenizer: ByteLevelBPE,
    messages: list[ChatMessage],
    max_len: int,
    max_new_tokens: int,
    device: torch.device,
    top_k: int,
    temperature: float,
) -> list[int]:
    # ---------------------------------------------------------
    # Serialize the conversation history into the same role-token
    # format used by posttraining.
    # ---------------------------------------------------------
    prompt_token_ids = build_chat_prompt_token_ids(tokenizer=tokenizer, messages=messages)
    model_input = torch.tensor(prompt_token_ids, dtype=torch.long).unsqueeze(0).to(device)
    eos_token_id = tokenizer.token_to_id(tokenizer.eos_token)
    end_of_turn_token_id = tokenizer.token_to_id(tokenizer.end_of_turn_token)
    max_generation_steps = min(max_new_tokens, max_len - len(prompt_token_ids))
    generated_token_ids: list[int] = []

    # ---------------------------------------------------------
    # Return no output when the serialized history already fills
    # the full context window.
    # ---------------------------------------------------------
    if max_generation_steps <= 0:
        return generated_token_ids

    # ---------------------------------------------------------
    # Fill the cache with the full prompt, then generate one token
    # at a time until the chat turn or model context ends.
    # ---------------------------------------------------------
    with torch.no_grad():
        predictions, key_value_cache = model.forward_with_cache(
            token_ids=model_input,
            past_key_values=None,
        )

        for step_index in range(max_generation_steps):
            predicted_id = sample_next_token_id(
                logits=predictions[:, -1, :],
                top_k=top_k,
                temperature=temperature,
            )
            predicted_token_id = int(predicted_id.item())

            if predicted_token_id in {eos_token_id, end_of_turn_token_id}:
                break

            generated_token_ids.append(predicted_token_id)

            if step_index + 1 < max_generation_steps:
                predictions, key_value_cache = model.forward_with_cache(
                    token_ids=predicted_id,
                    past_key_values=key_value_cache,
                )

    return generated_token_ids
