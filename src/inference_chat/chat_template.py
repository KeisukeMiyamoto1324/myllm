from src.posttraining.chat_template import ChatMessage
from src.posttraining.chat_template import get_role_token
from src.posttraining.chat_template import normalize_role
from src.tokenizer.tokenizer import ByteLevelBPE


def build_chat_prompt_token_ids(
    tokenizer: ByteLevelBPE,
    messages: list[ChatMessage],
) -> list[int]:
    # ---------------------------------------------------------
    # Start chat inference from BOS so the prompt matches the SFT
    # conversation boundary used during posttraining.
    # ---------------------------------------------------------
    token_ids = [tokenizer.token_to_id(tokenizer.bos_token)]

    # ---------------------------------------------------------
    # Serialize all previous messages with role markers and turn
    # delimiters before asking the model to complete the assistant.
    # ---------------------------------------------------------
    for message in messages:
        role = normalize_role(message.role)
        role_token_id = tokenizer.token_to_id(get_role_token(tokenizer=tokenizer, role=role))
        content_token_ids = tokenizer.tokenize(sentence=message.content)
        end_of_turn_token_id = tokenizer.token_to_id(tokenizer.end_of_turn_token)
        token_ids.extend([role_token_id, *content_token_ids, end_of_turn_token_id])

    # ---------------------------------------------------------
    # Add the assistant marker as the final prompt token so the
    # generated continuation is the next assistant response.
    # ---------------------------------------------------------
    assistant_token_id = tokenizer.token_to_id(tokenizer.assistant_token)
    token_ids.append(assistant_token_id)
    return token_ids
