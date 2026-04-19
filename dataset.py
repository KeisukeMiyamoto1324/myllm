from collections.abc import Iterator

import torch
from datasets import load_dataset
from torch.utils.data import IterableDataset
from torch.utils.data import get_worker_info

from tokenizer_rust.tokenizer import ByteLevelBPE


class FineWebEduDataset(IterableDataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        tokenizer: ByteLevelBPE,
        max_len: int,
        pad_token_id: int,
        eos_token_id: int,
    ) -> None:
        super().__init__()

        # ---------------------------------------------------------
        # Keep only the tokenizer and fixed-length sequence setup
        # needed to stream training examples lazily.
        # ---------------------------------------------------------
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        # ---------------------------------------------------------
        # Open the FineWeb-Edu training split as a streaming source
        # so samples are never materialized in local memory.
        # ---------------------------------------------------------
        dataset = load_dataset(
            path="HuggingFaceFW/fineweb-edu",
            name="sample-10BT",
            split="train",
            streaming=True,
        )

        # ---------------------------------------------------------
        # Shard the stream per worker so parallel data loading
        # does not duplicate the same records.
        # ---------------------------------------------------------
        worker_info = get_worker_info()
        if worker_info is not None:
            dataset = dataset.shard(num_shards=worker_info.num_workers, index=worker_info.id)

        # ---------------------------------------------------------
        # Tokenize each streamed document into one fixed-length
        # training example and yield it immediately.
        # ---------------------------------------------------------
        for sample in dataset:
            yield self._create_example(text=sample["text"])

    def _create_example(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        # ---------------------------------------------------------
        # Encode the text once, append EOS, and pad the remainder
        # so every sample matches the configured sequence length.
        # ---------------------------------------------------------
        token_ids = self.tokenizer.tokenize(sentence=text)
        input_ids = token_ids[: self.max_len - 1] + [self.eos_token_id]
        padding_size = self.max_len - len(input_ids)
        padded_input_ids = input_ids + [self.pad_token_id for _ in range(padding_size)]

        # ---------------------------------------------------------
        # Shift the padded sequence by one position to build the
        # next-token labels used by the decoder loss.
        # ---------------------------------------------------------
        label_ids = padded_input_ids[1:] + [self.pad_token_id]
        inputs = torch.tensor(padded_input_ids, dtype=torch.long)
        labels = torch.tensor(label_ids, dtype=torch.long)
        return inputs, labels
