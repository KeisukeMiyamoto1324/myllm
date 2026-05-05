from collections.abc import Iterator

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from torch.utils.data import IterableDataset
from torch.utils.data import get_worker_info

from src.posttraining.chat_template import ChatMessage
from src.posttraining.chat_template import tokenize_chat_messages
from src.tokenizer.tokenizer import ByteLevelBPE


MAGPIE_DATASET_PATH = "Magpie-Align/Magpie-Pro-300K-Filtered"
MAGPIE_DATASET_SPLIT = "train"
EVERYDAY_DATASET_PATH = "HuggingFaceTB/everyday-conversations-llama3.1-2k"
EVERYDAY_TRAIN_SPLIT = "train_sft"
EVERYDAY_VALIDATION_SPLIT = "test_sft"


class MagpieChatDataset(IterableDataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        tokenizer: ByteLevelBPE,
        max_len: int,
        pad_token_id: int,
        bos_token_id: int,
        eos_token_id: int,
        end_of_turn_token_id: int,
    ) -> None:
        super().__init__()

        # ---------------------------------------------------------
        # Store the tokenization dependencies required to stream
        # Magpie conversations lazily during broad SFT.
        # ---------------------------------------------------------
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.end_of_turn_token_id = end_of_turn_token_id

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        # ---------------------------------------------------------
        # Open Magpie as a streaming source so the 300K examples do
        # not need to be materialized before training starts.
        # ---------------------------------------------------------
        dataset = load_dataset(
            path=MAGPIE_DATASET_PATH,
            split=MAGPIE_DATASET_SPLIT,
            streaming=True,
        )

        # ---------------------------------------------------------
        # Shard the stream across DataLoader workers to avoid each
        # worker replaying the same remote samples.
        # ---------------------------------------------------------
        worker_info = get_worker_info()

        if worker_info is not None:
            dataset = dataset.shard(num_shards=worker_info.num_workers, index=worker_info.id)

        # ---------------------------------------------------------
        # Convert each streamed record into one fixed-length SFT
        # example with labels only on assistant answer tokens.
        # ---------------------------------------------------------
        for sample in dataset:
            messages = [
                ChatMessage(role=message["from"], content=message["value"])
                for message in sample["conversations"]
            ]
            yield build_tensor_example(
                tokenizer=self.tokenizer,
                messages=messages,
                max_len=self.max_len,
                pad_token_id=self.pad_token_id,
                bos_token_id=self.bos_token_id,
                eos_token_id=self.eos_token_id,
                end_of_turn_token_id=self.end_of_turn_token_id,
            )


class EverydayChatDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        tokenizer: ByteLevelBPE,
        split: str,
        max_len: int,
        pad_token_id: int,
        bos_token_id: int,
        eos_token_id: int,
        end_of_turn_token_id: int,
    ) -> None:
        super().__init__()

        # ---------------------------------------------------------
        # Load the small high-quality Everyday split locally so the
        # final SFT stage can iterate over all examples repeatedly.
        # ---------------------------------------------------------
        dataset = load_dataset(path=EVERYDAY_DATASET_PATH, split=split)
        self.examples = [
            build_tensor_example(
                tokenizer=tokenizer,
                messages=[
                    ChatMessage(role=message["role"], content=message["content"])
                    for message in sample["messages"]
                ],
                max_len=max_len,
                pad_token_id=pad_token_id,
                bos_token_id=bos_token_id,
                eos_token_id=eos_token_id,
                end_of_turn_token_id=end_of_turn_token_id,
            )
            for sample in dataset
        ]

    def __len__(self) -> int:
        # ---------------------------------------------------------
        # Return the number of loaded Everyday conversation examples
        # available in the selected split.
        # ---------------------------------------------------------
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        # ---------------------------------------------------------
        # Return one pre-tokenized fixed-length chat example without
        # additional network or tokenizer work.
        # ---------------------------------------------------------
        return self.examples[index]


def build_tensor_example(
    tokenizer: ByteLevelBPE,
    messages: list[ChatMessage],
    max_len: int,
    pad_token_id: int,
    bos_token_id: int,
    eos_token_id: int,
    end_of_turn_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # ---------------------------------------------------------
    # Tokenize one chat record through the shared template and
    # convert its two streams into tensors for model training.
    # ---------------------------------------------------------
    example = tokenize_chat_messages(
        tokenizer=tokenizer,
        messages=messages,
        max_len=max_len,
        pad_token_id=pad_token_id,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        end_of_turn_token_id=end_of_turn_token_id,
    )
    input_ids = torch.tensor(example.input_ids, dtype=torch.long)
    labels = torch.tensor(example.labels, dtype=torch.long)
    return input_ids, labels
