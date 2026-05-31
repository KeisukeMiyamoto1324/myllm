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
        total_training_tokens: int | None = None,
        ramp_start_progress: float = 0.5,
        split_modulo: int = 1,
        split_indexes: tuple[int, ...] = (0,),
    ) -> None:
        super().__init__()

        # ---------------------------------------------------------
        # Validate the token mixture once so every worker can build
        # deterministic per-cycle allocations from the same config.
        # ---------------------------------------------------------
        self.corpus_cases = corpus_cases
        self.static_token_targets = resolve_mix_token_targets(
            corpus_cases=corpus_cases,
            mix_cycle_tokens=mix_cycle_tokens,
        )

        # ---------------------------------------------------------
        # Keep the ramp interval bounded so progress can be mapped
        # into a finite late-training percentage.
        # ---------------------------------------------------------
        if ramp_start_progress < 0.0 or ramp_start_progress >= 1.0:
            raise ValueError("ramp_start_progress must be greater than or equal to 0 and less than 1")

        # ---------------------------------------------------------
        # Store shared tokenization settings for the corpus streams
        # that are opened lazily inside each DataLoader worker.
        # ---------------------------------------------------------
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.mix_cycle_tokens = mix_cycle_tokens
        self.total_training_tokens = total_training_tokens
        self.ramp_start_progress = ramp_start_progress
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
        # Track worker-local progress because IterableDataset
        # workers independently produce their share of each batch.
        # ---------------------------------------------------------
        worker_info = get_worker_info()
        worker_modulo = worker_info.num_workers if worker_info is not None else 1
        training_token_budget = self._resolve_worker_training_token_budget(
            worker_modulo=worker_modulo,
        )
        emitted_tokens = 0
        exhausted_cases = [False for _ in self.corpus_cases]
        seen_cases = [False for _ in self.corpus_cases]

        # ---------------------------------------------------------
        # Repeat token-budget cycles and update the ramped corpus
        # percentage from the worker-local training progress.
        # ---------------------------------------------------------
        while True:
            progress = emitted_tokens / training_token_budget
            token_targets = self._resolve_cycle_token_targets(
                progress=progress,
                exhausted_cases=exhausted_cases,
            )

            for corpus_index, token_target in enumerate(token_targets):
                consumed_tokens = 0

                while consumed_tokens < token_target:
                    example = self._next_corpus_example(
                        corpus_iterators=corpus_iterators,
                        corpus_index=corpus_index,
                    )

                    if example is None:
                        if not seen_cases[corpus_index]:
                            corpus_name = self.corpus_cases[corpus_index].name
                            raise ValueError(
                                f"Corpus stream has no samples after split filtering: {corpus_name}"
                            )

                        exhausted_cases[corpus_index] = True
                        break

                    input_ids, label_ids = example
                    seen_cases[corpus_index] = True
                    example_tokens = count_label_tokens(
                        label_ids=label_ids,
                        pad_token_id=self.pad_token_id,
                    )
                    consumed_tokens += example_tokens
                    yield input_ids, label_ids
                    emitted_tokens += example_tokens

    def _resolve_worker_training_token_budget(self, worker_modulo: int) -> int:
        # ---------------------------------------------------------
        # Convert the global token estimate into a per-worker budget
        # so each worker reaches the ramp endpoint near train end.
        # ---------------------------------------------------------
        if self.total_training_tokens is None:
            return self.mix_cycle_tokens

        return max(1, self.total_training_tokens // worker_modulo)

    def _resolve_cycle_token_targets(
        self,
        progress: float,
        exhausted_cases: list[bool],
    ) -> list[int]:
        # ---------------------------------------------------------
        # Use static final percentages for validation and scheduled
        # percentages for training where total token budget exists.
        # ---------------------------------------------------------
        if self.total_training_tokens is None:
            return self.static_token_targets

        token_percentages = resolve_scheduled_mix_percentages(
            corpus_cases=self.corpus_cases,
            progress=progress,
            ramp_start_progress=self.ramp_start_progress,
        )
        token_percentages = redistribute_exhausted_percentages(
            corpus_cases=self.corpus_cases,
            token_percentages=token_percentages,
            exhausted_cases=exhausted_cases,
        )
        return resolve_token_targets(
            token_percentages=token_percentages,
            mix_cycle_tokens=self.mix_cycle_tokens,
            allow_zero_targets=True,
        )

    def _next_corpus_example(
        self,
        corpus_iterators: list[Iterator[tuple[torch.Tensor, torch.Tensor]]],
        corpus_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        # ---------------------------------------------------------
        # Read the next example and reopen repeatable streams at
        # cycle boundaries for long training runs.
        # ---------------------------------------------------------
        try:
            return next(corpus_iterators[corpus_index])
        except StopIteration:
            pass

        if not self.corpus_cases[corpus_index].repeat_on_end:
            return None

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
    token_percentages = [corpus_case.token_percentage for corpus_case in corpus_cases]
    return resolve_token_targets(
        token_percentages=token_percentages,
        mix_cycle_tokens=mix_cycle_tokens,
        allow_zero_targets=False,
    )


def resolve_scheduled_mix_percentages(
    corpus_cases: list[PretrainingCorpusCase],
    progress: float,
    ramp_start_progress: float,
) -> list[float]:
    # ---------------------------------------------------------
    # Increase ramped high-quality corpora from zero to their final
    # percentages over the configured late-training window.
    # ---------------------------------------------------------
    ramp_ratio = resolve_ramp_ratio(
        progress=progress,
        ramp_start_progress=ramp_start_progress,
    )
    ramped_percentages = [
        corpus_case.token_percentage * ramp_ratio if corpus_case.is_ramped else 0.0
        for corpus_case in corpus_cases
    ]
    filler_percentage = 100.0 - sum(ramped_percentages)
    filler_total = sum(
        corpus_case.token_percentage
        for corpus_case in corpus_cases
        if not corpus_case.is_ramped
    )

    # ---------------------------------------------------------
    # Allocate the remaining budget across non-ramped corpora using
    # their final static percentages as relative weights.
    # ---------------------------------------------------------
    return [
        ramped_percentage
        if corpus_case.is_ramped
        else filler_percentage * corpus_case.token_percentage / filler_total
        for corpus_case, ramped_percentage in zip(corpus_cases, ramped_percentages)
    ]


def resolve_ramp_ratio(progress: float, ramp_start_progress: float) -> float:
    # ---------------------------------------------------------
    # Clamp progress to the ramp interval so early training stays
    # at zero and the final stage stays at the target percentage.
    # ---------------------------------------------------------
    if progress <= ramp_start_progress:
        return 0.0

    return min(1.0, (progress - ramp_start_progress) / (1.0 - ramp_start_progress))


def redistribute_exhausted_percentages(
    corpus_cases: list[PretrainingCorpusCase],
    token_percentages: list[float],
    exhausted_cases: list[bool],
) -> list[float]:
    # ---------------------------------------------------------
    # Move exhausted finite-corpus budget to active repeatable
    # corpora so training continues after CleanedWiki is consumed.
    # ---------------------------------------------------------
    active_percentages = [
        0.0 if exhausted else token_percentage
        for token_percentage, exhausted in zip(token_percentages, exhausted_cases)
    ]
    missing_percentage = 100.0 - sum(active_percentages)
    repeatable_percentage = sum(
        token_percentage
        for corpus_case, token_percentage, exhausted in zip(
            corpus_cases,
            active_percentages,
            exhausted_cases,
        )
        if corpus_case.repeat_on_end and not exhausted
    )

    # ---------------------------------------------------------
    # Preserve the current active distribution when no finite
    # corpus has been exhausted during this worker stream.
    # ---------------------------------------------------------
    if missing_percentage <= 0.0:
        return active_percentages

    return [
        token_percentage + missing_percentage * token_percentage / repeatable_percentage
        if corpus_case.repeat_on_end and not exhausted
        else token_percentage
        for corpus_case, token_percentage, exhausted in zip(
            corpus_cases,
            active_percentages,
            exhausted_cases,
        )
    ]


def resolve_token_targets(
    token_percentages: list[float],
    mix_cycle_tokens: int,
    allow_zero_targets: bool,
) -> list[int]:
    # ---------------------------------------------------------
    # Convert percentages into integer token targets while keeping
    # the total token budget exact for each mixing cycle.
    # ---------------------------------------------------------
    raw_targets = [
        mix_cycle_tokens * token_percentage / 100.0
        for token_percentage in token_percentages
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

    if min(token_targets) == 0 and not allow_zero_targets:
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

    if not any(corpus_case.repeat_on_end for corpus_case in corpus_cases):
        raise ValueError("At least one repeatable pretraining corpus case is required")

    if any(corpus_case.is_ramped for corpus_case in corpus_cases) and not any(
        not corpus_case.is_ramped for corpus_case in corpus_cases
    ):
        raise ValueError("At least one filler corpus case is required for ramped mixing")

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
