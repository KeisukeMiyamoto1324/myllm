from collections.abc import Iterator
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import IterableDataset
from torch.utils.data import get_worker_info

from src.tokenizer_rust.tokenizer import ByteLevelBPE


class FineWebEduDataset(IterableDataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        tokenizer: ByteLevelBPE,
        max_len: int,
        pad_token_id: int,
        eos_token_id: int,
        split_modulo: int = 1,
        split_indexes: tuple[int, ...] = (0,),
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
        self.split_modulo = split_modulo
        self.split_indexes = split_indexes

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
        # Route each streamed document into the configured split so
        # training and validation consume disjoint samples.
        # ---------------------------------------------------------
        for sample_index, sample in enumerate(dataset):
            if sample_index % self.split_modulo not in self.split_indexes:
                continue

            # ---------------------------------------------------------
            # Tokenize each streamed document into fixed-length
            # training sequence chunks and yield them immediately.
            # ---------------------------------------------------------
            yield from self._create_examples(text=sample["text"])

    def _create_examples(self, text: str) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        # ---------------------------------------------------------
        # Encode the document once and split it into sequential
        # fixed-length chunks so long texts produce many samples.
        # ---------------------------------------------------------
        token_ids = self.tokenizer.tokenize(sentence=text)

        # ---------------------------------------------------------
        # Slice the token stream into chunks that reserve the last
        # position for EOS inside every yielded training example.
        # ---------------------------------------------------------
        chunk_size = self.max_len - 1
        chunk_starts = range(0, max(len(token_ids), 1), chunk_size)

        # ---------------------------------------------------------
        # Convert every chunk into one padded input-label pair and
        # stream it out without buffering the full document.
        # ---------------------------------------------------------
        for chunk_start in chunk_starts:
            chunk_token_ids = token_ids[chunk_start : chunk_start + chunk_size]
            yield self._build_example(chunk_token_ids=chunk_token_ids)

    def _build_example(self, chunk_token_ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        # ---------------------------------------------------------
        # Append EOS to the chunk and pad the remainder so every
        # sample matches the configured sequence length.
        # ---------------------------------------------------------
        input_ids = chunk_token_ids + [self.eos_token_id]
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


class LocalTokenizedDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        path: Path,
        max_len: int,
        num_samples: int,
    ) -> None:
        super().__init__()

        # ---------------------------------------------------------
        # Load fixed tokenized samples from local storage so
        # validation avoids streaming, skipping, and tokenization.
        # ---------------------------------------------------------
        payload = torch.load(path, map_location="cpu")
        self.inputs = payload["inputs"]
        self.labels = payload["labels"]
        self.metadata = payload["metadata"]

        # ---------------------------------------------------------
        # Reject stale caches because validation samples must match
        # the current model context length exactly.
        # ---------------------------------------------------------
        if self.metadata["max_len"] != max_len:
            raise ValueError("Validation cache max_len does not match the training configuration")

        if self.metadata["num_samples"] != num_samples:
            raise ValueError("Validation cache sample count does not match the training configuration")

    def __len__(self) -> int:
        # ---------------------------------------------------------
        # Return the number of fixed validation examples available
        # from the cached tensor file.
        # ---------------------------------------------------------
        return self.inputs.size(dim=0)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        # ---------------------------------------------------------
        # Return one cached input-label pair without additional
        # tokenization or network access.
        # ---------------------------------------------------------
        return self.inputs[index], self.labels[index]


def build_tokenized_cache(
    dataset: IterableDataset[tuple[torch.Tensor, torch.Tensor]],
    path: Path,
    num_samples: int,
    max_len: int,
) -> None:
    # ---------------------------------------------------------
    # Materialize a fixed number of examples from the streaming
    # source once so later validation reads local tensors only.
    # ---------------------------------------------------------
    path.parent.mkdir(parents=True, exist_ok=True)
    inputs: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []

    # ---------------------------------------------------------
    # Consume exactly the configured validation budget and stop
    # before extra remote samples are streamed or tokenized.
    # ---------------------------------------------------------
    progress_bar = tqdm(total=num_samples, desc="Building validation cache", unit="sample")

    for input_ids, label_ids in dataset:
        inputs.append(input_ids)
        labels.append(label_ids)
        progress_bar.update(1)

        if len(inputs) == num_samples:
            break

    progress_bar.close()

    # ---------------------------------------------------------
    # Fail loudly when the source stream cannot provide the
    # requested fixed validation budget.
    # ---------------------------------------------------------
    if len(inputs) != num_samples:
        raise ValueError("Validation cache source ended before enough samples were collected")

    # ---------------------------------------------------------
    # Store tensors with enough metadata to reject incompatible
    # validation caches on later training runs.
    # ---------------------------------------------------------
    payload = {
        "inputs": torch.stack(inputs),
        "labels": torch.stack(labels),
        "metadata": {
            "num_samples": num_samples,
            "max_len": max_len,
        },
    }
    torch.save(payload, path)
