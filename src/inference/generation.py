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
    # Predict one token at a time until EOS appears or the
    # configured generation length limit is reached.
    # ---------------------------------------------------------
    with torch.no_grad():
        for _ in range(max_generation_steps):
            predictions = model(model_input)
            predicted_id = sample_next_token_id(
                logits=predictions[:, -1, :],
                top_k=top_k,
                temperature=temperature,
            )
            predicted_token_id = int(predicted_id.item())

            if predicted_token_id == eos_token_id:
                break

            generated_token_ids.append(predicted_token_id)
            model_input = torch.cat((model_input, predicted_id), dim=1)

    return generated_token_ids
