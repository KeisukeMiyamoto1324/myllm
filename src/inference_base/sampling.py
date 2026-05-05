import torch


def sample_next_token_id(
    logits: torch.Tensor,
    top_k: int,
    temperature: float,
) -> torch.Tensor:
    # ---------------------------------------------------------
    # Scale logits by temperature so lower values sharpen and
    # higher values flatten the next-token distribution.
    # ---------------------------------------------------------
    scaled_logits = logits / temperature

    # ---------------------------------------------------------
    # Keep only the top-k logits when requested, then sample a
    # token id from the resulting probability distribution.
    # ---------------------------------------------------------
    if top_k > logits.size(dim=-1):
        raise ValueError("top_k must be less than or equal to the vocabulary size")

    if top_k > 0:
        top_k_logits, top_k_indices = torch.topk(scaled_logits, k=top_k, dim=-1)
        top_k_probabilities = torch.softmax(top_k_logits, dim=-1)
        sampled_index = torch.multinomial(top_k_probabilities, num_samples=1)
        return top_k_indices.gather(dim=1, index=sampled_index)

    probabilities = torch.softmax(scaled_logits, dim=-1)
    return torch.multinomial(probabilities, num_samples=1)
