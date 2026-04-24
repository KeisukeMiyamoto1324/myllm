import torch

from src.inference.sampling import sample_next_token_id
from src.model.transformer import DecoderOnlyTransformer
from src.tokenizer_rust.tokenizer import ByteLevelBPE


def generate_token_ids(
    model: DecoderOnlyTransformer,
    tokenizer: ByteLevelBPE,
    prompt: str,
    max_len: int,
    max_new_tokens: int,
    device: torch.device,
    top_k: int,
    temperature: float,
) -> list[int]:
    # ---------------------------------------------------------
    # Encode the prompt into token ids and move the initial
    # sequence onto the target device for autoregressive use.
    # ---------------------------------------------------------
    prompt_token_ids = tokenizer.tokenize(sentence=prompt)
    model_input = torch.tensor(prompt_token_ids, dtype=torch.long).unsqueeze(0).to(device)
    eos_token_id = tokenizer.token_to_id(tokenizer.eos_token)
    max_generation_steps = min(max_new_tokens, max_len - len(prompt_token_ids))
    generated_token_ids: list[int] = []

    # ---------------------------------------------------------
    # Stop immediately when the prompt already consumes the full
    # context window configured for the trained model.
    # ---------------------------------------------------------
    if max_generation_steps <= 0:
        return generated_token_ids

    # ---------------------------------------------------------
    # Run the prompt once to fill the KV cache, then feed only the
    # latest sampled token on later generation steps.
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

            if predicted_token_id == eos_token_id:
                break

            generated_token_ids.append(predicted_token_id)

            if step_index + 1 < max_generation_steps:
                predictions, key_value_cache = model.forward_with_cache(
                    token_ids=predicted_id,
                    past_key_values=key_value_cache,
                )

    return generated_token_ids
