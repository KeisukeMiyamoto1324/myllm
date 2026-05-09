from collections.abc import Iterator
from hashlib import blake2b
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import IterableDataset
from torch.utils.data import get_worker_info

from src.pretraining.training_corpus_cases import PretrainingCorpusCase
from src.tokenizer.tokenizer import ByteLevelBPE


class PretrainingCorpusDataset(IterableDataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        corpus_case: PretrainingCorpusCase,
        tokenizer: ByteLevelBPE,
        max_len: int,
        pad_token_id: int,
        bos_token_id: int,
        eos_token_id: int,
        split_modulo: int = 1,
        split_indexes: tuple[int, ...] = (0,),
    ) -> None:
        super().__init__()

        # ---------------------------------------------------------
        # Keep the corpus source and fixed-length sequence setup
        # needed to stream training examples lazily.
        # ---------------------------------------------------------
        self.corpus_case = corpus_case
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.split_modulo = split_modulo
        self.split_indexes = split_indexes

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        # ---------------------------------------------------------
        # Open the configured corpus split as a streaming source so
        # samples are never materialized in local memory.
        # ---------------------------------------------------------
        dataset = load_dataset(
            path=self.corpus_case.dataset_path,
            name=self.corpus_case.config_name,
            split=self.corpus_case.split,
            streaming=True,
        )

        # ---------------------------------------------------------
        # Partition documents per worker with the same stable hash
        # used for train-validation splitting.
        # ---------------------------------------------------------
        worker_info = get_worker_info()
        worker_modulo = worker_info.num_workers if worker_info is not None else 1
        worker_index = worker_info.id if worker_info is not None else 0

        # ---------------------------------------------------------
        # Route each streamed document into both the configured
        # dataset split and the current DataLoader worker partition.
        # ---------------------------------------------------------
        dataset = dataset.filter(
            lambda sample: self._contains_partition(
                sample=sample,
                worker_modulo=worker_modulo,
                worker_index=worker_index,
            )
        )

        for sample in dataset:
            # ---------------------------------------------------------
            # Tokenize each streamed document into fixed-length
            # training sequence chunks and yield them immediately.
            # ---------------------------------------------------------
            yield from self._create_examples(text=sample[self.corpus_case.text_column])

    def _contains_partition(
        self,
        sample: dict[str, str],
        worker_modulo: int,
        worker_index: int,
    ) -> bool:
        # ---------------------------------------------------------
        # Use one deterministic document-content hash so split and
        # worker membership stay independent of remote file shards.
        # ---------------------------------------------------------
        document_index = self._resolve_document_index(
            text=sample[self.corpus_case.text_column],
        )
        split_index = document_index % self.split_modulo
        partition_index = (document_index // self.split_modulo) % worker_modulo
        return split_index in self.split_indexes and partition_index == worker_index

    def _resolve_document_index(self, text: str) -> int:
        # ---------------------------------------------------------
        # Convert document text into a stable integer id without
        # relying on process-randomized Python hashing.
        # ---------------------------------------------------------
        encoded_text = text.encode("utf-8")
        digest = blake2b(encoded_text, digest_size=8).digest()
        return int.from_bytes(digest, byteorder="big")

    def _create_examples(self, text: str) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        # ---------------------------------------------------------
        # Encode the document with BOS at the document start and
        # EOS at the true document end before chunking.
        # ---------------------------------------------------------
        document_token_ids = [
            self.bos_token_id,
            *self.tokenizer.tokenize(sentence=text),
            self.eos_token_id,
        ]

        # ---------------------------------------------------------
        # Slice overlapping windows so the final token in each input
        # still learns to predict the next document token.
        # ---------------------------------------------------------
        chunk_starts = range(0, len(document_token_ids) - 1, self.max_len)

        # ---------------------------------------------------------
        # Convert every chunk into one padded input-label pair and
        # stream it out without buffering the full document.
        # ---------------------------------------------------------
        for chunk_start in chunk_starts:
            window_token_ids = document_token_ids[chunk_start : chunk_start + self.max_len + 1]
            yield self._build_example(
                input_token_ids=window_token_ids[:-1],
                label_token_ids=window_token_ids[1:],
            )

    def _build_example(
        self,
        input_token_ids: list[int],
        label_token_ids: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # ---------------------------------------------------------
        # Pad the input and label streams separately so every sample
        # matches the configured sequence length.
        # ---------------------------------------------------------
        padding_size = self.max_len - len(input_token_ids)
        padded_input_ids = input_token_ids + [self.pad_token_id for _ in range(padding_size)]
        padded_label_ids = label_token_ids + [self.pad_token_id for _ in range(padding_size)]
        inputs = torch.tensor(padded_input_ids, dtype=torch.long)
        labels = torch.tensor(padded_label_ids, dtype=torch.long)
        return inputs, labels


class MixedPretrainingDataset(IterableDataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        corpus_cases: list[PretrainingCorpusCase],
        tokenizer: ByteLevelBPE,
        max_len: int,
        pad_token_id: int,
        bos_token_id: int,
        eos_token_id: int,
        mix_cycle_tokens: int,
        split_modulo: int = 1,
        split_indexes: tuple[int, ...] = (0,),
    ) -> None:
        super().__init__()

        # ---------------------------------------------------------
        # Validate the token mixture once so every worker follows
        # the same deterministic per-cycle allocation.
        # ---------------------------------------------------------
        self.corpus_cases = corpus_cases
        self.token_targets = resolve_mix_token_targets(
            corpus_cases=corpus_cases,
            mix_cycle_tokens=mix_cycle_tokens,
        )

        # ---------------------------------------------------------
        # Store shared tokenization settings for the corpus streams
        # that are opened lazily inside each DataLoader worker.
        # ---------------------------------------------------------
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.split_modulo = split_modulo
        self.split_indexes = split_indexes

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        # ---------------------------------------------------------
        # Build one independent stream per configured corpus so the
        # mixer can consume each source according to token targets.
        # ---------------------------------------------------------
        corpus_iterators = [
            self._build_corpus_iterator(corpus_case=corpus_case)
            for corpus_case in self.corpus_cases
        ]

        # ---------------------------------------------------------
        # Repeat fixed token-budget cycles so the long-run stream
        # stays close to the configured corpus percentages.
        # ---------------------------------------------------------
        while True:
            for corpus_index, token_target in enumerate(self.token_targets):
                consumed_tokens = 0

                while consumed_tokens < token_target:
                    input_ids, label_ids = self._next_corpus_example(
                        corpus_iterators=corpus_iterators,
                        corpus_index=corpus_index,
                    )
                    consumed_tokens += count_label_tokens(
                        label_ids=label_ids,
                        pad_token_id=self.pad_token_id,
                    )
                    yield input_ids, label_ids

    def _next_corpus_example(
        self,
        corpus_iterators: list[Iterator[tuple[torch.Tensor, torch.Tensor]]],
        corpus_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # ---------------------------------------------------------
        # Read the next example and reopen finite streams at cycle
        # boundaries so long training runs can continue.
        # ---------------------------------------------------------
        try:
            return next(corpus_iterators[corpus_index])
        except StopIteration:
            corpus_iterators[corpus_index] = self._build_corpus_iterator(
                corpus_case=self.corpus_cases[corpus_index],
            )

        # ---------------------------------------------------------
        # Raise a clear configuration/data split error when a corpus
        # partition has no examples after filtering.
        # ---------------------------------------------------------
        try:
            return next(corpus_iterators[corpus_index])
        except StopIteration as exc:
            corpus_name = self.corpus_cases[corpus_index].name
            raise ValueError(
                f"Corpus stream has no samples after split filtering: {corpus_name}"
            ) from exc

    def _build_corpus_iterator(
        self,
        corpus_case: PretrainingCorpusCase,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        # ---------------------------------------------------------
        # Open a fresh stream for one corpus with the same split and
        # tokenization settings used by the mixed training stream.
        # ---------------------------------------------------------
        return iter(
            PretrainingCorpusDataset(
                corpus_case=corpus_case,
                tokenizer=self.tokenizer,
                max_len=self.max_len,
                pad_token_id=self.pad_token_id,
                bos_token_id=self.bos_token_id,
                eos_token_id=self.eos_token_id,
                split_modulo=self.split_modulo,
                split_indexes=self.split_indexes,
            )
        )


class LocalTokenizedDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        path: Path,
        max_len: int,
        num_samples: int,
        metadata: dict[str, object],
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

        # ---------------------------------------------------------
        # Reject explicitly provided cache files when their stored
        # corpus mixture does not match the current run.
        # ---------------------------------------------------------
        if any(self.metadata.get(key) != value for key, value in metadata.items()):
            raise ValueError("Validation cache metadata does not match the training configuration")

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
    metadata: dict[str, object],
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
            **metadata,
        },
    }
    torch.save(payload, path)


def resolve_mix_token_targets(
    corpus_cases: list[PretrainingCorpusCase],
    mix_cycle_tokens: int,
) -> list[int]:
    # ---------------------------------------------------------
    # Convert configured percentages into integer token budgets
    # for one deterministic mixing cycle.
    # ---------------------------------------------------------
    validate_mix_settings(corpus_cases=corpus_cases, mix_cycle_tokens=mix_cycle_tokens)
    raw_targets = [
        mix_cycle_tokens * corpus_case.token_percentage / 100.0
        for corpus_case in corpus_cases
    ]
    token_targets = [int(raw_target) for raw_target in raw_targets]
    remaining_tokens = mix_cycle_tokens - sum(token_targets)
    ranked_indexes = sorted(
        range(len(raw_targets)),
        key=lambda index: raw_targets[index] - token_targets[index],
        reverse=True,
    )

    # ---------------------------------------------------------
    # Assign leftover tokens to the largest fractional shares so
    # the integer cycle stays as close as possible to percentages.
    # ---------------------------------------------------------
    for corpus_index in ranked_indexes[:remaining_tokens]:
        token_targets[corpus_index] += 1

    if min(token_targets) == 0:
        raise ValueError("mix_cycle_tokens is too small for the configured corpus percentages")

    return token_targets


def validate_mix_settings(
    corpus_cases: list[PretrainingCorpusCase],
    mix_cycle_tokens: int,
) -> None:
    # ---------------------------------------------------------
    # Reject invalid corpus mixtures before any remote streaming
    # dataset is opened.
    # ---------------------------------------------------------
    if len(corpus_cases) == 0:
        raise ValueError("At least one pretraining corpus case is required")

    if mix_cycle_tokens <= 0:
        raise ValueError("mix_cycle_tokens must be positive")

    if any(corpus_case.token_percentage <= 0.0 for corpus_case in corpus_cases):
        raise ValueError("Each token_percentage must be positive")

    # ---------------------------------------------------------
    # Require a complete 100 percent allocation so adding or
    # removing corpora cannot silently change the mixture.
    # ---------------------------------------------------------
    total_percentage = sum(corpus_case.token_percentage for corpus_case in corpus_cases)

    if abs(total_percentage - 100.0) > 1e-6:
        raise ValueError("Pretraining corpus token percentages must sum to 100.0")


def count_label_tokens(label_ids: torch.Tensor, pad_token_id: int) -> int:
    # ---------------------------------------------------------
    # Count only real next-token targets so corpus percentages are
    # based on useful training tokens instead of padded positions.
    # ---------------------------------------------------------
    return int(label_ids.ne(pad_token_id).sum().item())
