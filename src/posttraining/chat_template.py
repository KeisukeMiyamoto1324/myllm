from dataclasses import dataclass

from src.shared.tokenizer import ByteLevelBPE


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


@dataclass(frozen=True)
class TokenizedChatExample:
    input_ids: list[int]
    labels: list[int]


def normalize_role(role: str) -> str:
    # ---------------------------------------------------------
    # Convert dataset-specific role names into the chat roles
    # represented by tokenizer special tokens.
    # ---------------------------------------------------------
    role_by_name = {
        "human": "user",
        "user": "user",
        "gpt": "assistant",
        "assistant": "assistant",
        "system": "system",
    }
    return role_by_name[role]


def get_role_token(tokenizer: ByteLevelBPE, role: str) -> str:
    # ---------------------------------------------------------
    # Resolve the special token string that marks each supported
    # chat role inside a serialized conversation.
    # ---------------------------------------------------------
    token_by_role = {
        "system": tokenizer.system_token,
        "user": tokenizer.user_token,
        "assistant": tokenizer.assistant_token,
    }
    return token_by_role[role]


def tokenize_chat_messages(
    tokenizer: ByteLevelBPE,
    messages: list[ChatMessage],
    max_len: int,
    pad_token_id: int,
    bos_token_id: int,
    eos_token_id: int,
    end_of_turn_token_id: int,
) -> TokenizedChatExample:
    # ---------------------------------------------------------
    # Start each serialized conversation with BOS and keep its
    # target masked because it is structural context only.
    # ---------------------------------------------------------
    token_ids = [bos_token_id]
    target_mask = [False]

    # ---------------------------------------------------------
    # Append each role marker, content span, and end-of-turn marker
    # while enabling loss only on assistant content and turn end.
    # ---------------------------------------------------------
    for message in messages:
        role = normalize_role(message.role)
        role_token_id = tokenizer.token_to_id(get_role_token(tokenizer=tokenizer, role=role))
        content_token_ids = tokenizer.tokenize(sentence=message.content)
        is_assistant = role == "assistant"
        token_ids.extend([role_token_id, *content_token_ids, end_of_turn_token_id])
        target_mask.extend([False, *[is_assistant for _ in content_token_ids], is_assistant])

    # ---------------------------------------------------------
    # Close the sample with EOS and train on it only when it follows
    # an assistant answer, matching chat generation stop behavior.
    # ---------------------------------------------------------
    token_ids.append(eos_token_id)
    target_mask.append(target_mask[-1])

    # ---------------------------------------------------------
    # Convert the full token stream into shifted language-modeling
    # inputs and labels, masking every non-assistant next token.
    # ---------------------------------------------------------
    input_token_ids = token_ids[:-1][:max_len]
    shifted_token_ids = token_ids[1:][:max_len]
    shifted_target_mask = target_mask[1:][:max_len]
    label_token_ids = [
        token_id if is_target else pad_token_id
        for token_id, is_target in zip(shifted_token_ids, shifted_target_mask, strict=True)
    ]

    # ---------------------------------------------------------
    # Pad both streams to the configured fixed context length so
    # batches can be stacked by the default PyTorch collator.
    # ---------------------------------------------------------
    padding_size = max_len - len(input_token_ids)
    padded_input_ids = input_token_ids + [pad_token_id for _ in range(padding_size)]
    padded_label_ids = label_token_ids + [pad_token_id for _ in range(padding_size)]
    return TokenizedChatExample(input_ids=padded_input_ids, labels=padded_label_ids)
