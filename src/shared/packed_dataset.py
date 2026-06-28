from collections.abc import Iterator
from hashlib import blake2b
from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from torch.utils.data import IterableDataset
from torch.utils.data import get_worker_info

from src.shared.tokenizer import ByteLevelBPE
from src.shared.training_corpus import TrainingCorpusCase
from src.shared.console import progress_manager


PackedTrainingExample = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
PackedTrainingSegment = tuple[list[int], list[int], int]
BUCKET_PACKING_BUFFER_SEGMENTS = 4096
BUCKET_PACKING_SEED = 17


class PackedCorpusDataset(IterableDataset[PackedTrainingExample]):
    def __init__(
        self,
        corpus_case: TrainingCorpusCase,
        tokenizer: ByteLevelBPE,
        max_len: int,
        pad_token_id: int,
        bos_token_id: int,
        eos_token_id: int,
        split_modulo: int = 1,
        split_indexes: tuple[int, ...] = (0,),
        shuffle_buffer_size: int = 0,
        shuffle_seed: int = 0,
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
        self.shuffle_buffer_size = shuffle_buffer_size
        self.shuffle_seed = shuffle_seed
        self.epoch_index = 0

    def set_epoch(self, epoch_index: int) -> None:
        # ---------------------------------------------------------
        # Change only the shuffle seed between epochs so repeated
        # corpus passes remain deterministic and reproducible.
        # ---------------------------------------------------------
        self.epoch_index = epoch_index

    def __iter__(self) -> Iterator[PackedTrainingExample]:
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
        # Route each streamed document through the deterministic
        # train-validation split before optional training shuffle.
        # ---------------------------------------------------------
        dataset = dataset.filter(
            lambda sample: self._contains_partition(
                sample=sample,
            )
        )

        # ---------------------------------------------------------
        # Shuffle each training epoch with a distinct fixed seed,
        # then shard the same order across ranks and workers.
        # ---------------------------------------------------------
        if self.shuffle_buffer_size > 0:
            dataset = dataset.shuffle(
                seed=self.shuffle_seed + self.epoch_index,
                buffer_size=self.shuffle_buffer_size,
            )

        worker_info = get_worker_info()
        worker_count = 1 if worker_info is None else worker_info.num_workers
        worker_index = 0 if worker_info is None else worker_info.id
        rank_count = 1
        rank_index = 0

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank_count = torch.distributed.get_world_size()
            rank_index = torch.distributed.get_rank()

        shard_count = rank_count * worker_count
        shard_index = rank_index * worker_count + worker_index

        if shard_count > 1:
            dataset = dataset.shard(num_shards=shard_count, index=shard_index)

        segment_buffer: list[PackedTrainingSegment] = []
        segment_index = 0

        for sample in dataset:
            # ---------------------------------------------------------
            # Tokenize documents into independent segments and keep a
            # bounded local buffer for deterministic best-fit packing.
            # ---------------------------------------------------------
            for input_token_ids, label_token_ids in self._create_segments(
                text=sample[self.corpus_case.text_column]
            ):
                order_key = self._resolve_segment_order_key(segment_index=segment_index)
                segment_buffer.append((input_token_ids, label_token_ids, order_key))
                segment_index += 1

                if len(segment_buffer) >= BUCKET_PACKING_BUFFER_SEGMENTS:
                    yield self._build_bucket_packed_example(segment_buffer=segment_buffer)

        while len(segment_buffer) > 0:
            yield self._build_bucket_packed_example(segment_buffer=segment_buffer)

    def _contains_partition(
        self,
        sample: dict[str, str],
    ) -> bool:
        # ---------------------------------------------------------
        # Use one deterministic document-content hash so train and
        # validation membership stay independent of remote shards.
        # ---------------------------------------------------------
        document_index = self._resolve_document_index(
            text=sample[self.corpus_case.text_column],
        )
        split_index = document_index % self.split_modulo
        return split_index in self.split_indexes

    def _resolve_document_index(self, text: str) -> int:
        # ---------------------------------------------------------
        # Convert document text into a stable integer id without
        # relying on process-randomized Python hashing.
        # ---------------------------------------------------------
        encoded_text = text.encode("utf-8")
        digest = blake2b(encoded_text, digest_size=8).digest()
        return int.from_bytes(digest, byteorder="big")

    def _create_segments(self, text: str) -> Iterator[tuple[list[int], list[int]]]:
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
        # Slice long documents into independent packable segments
        # without creating labels that cross document boundaries.
        # ---------------------------------------------------------
        chunk_starts = range(0, len(document_token_ids) - 1, self.max_len)

        # ---------------------------------------------------------
        # Return unpadded segments so the outer iterator can combine
        # multiple short documents into one training example.
        # ---------------------------------------------------------
        for chunk_start in chunk_starts:
            window_token_ids = document_token_ids[chunk_start : chunk_start + self.max_len + 1]
            yield window_token_ids[:-1], window_token_ids[1:]

    def _resolve_segment_order_key(self, segment_index: int) -> int:
        # ---------------------------------------------------------
        # Create a fixed-seed deterministic tie breaker so bucket
        # packing can reorder segments without run-to-run drift.
        # ---------------------------------------------------------
        payload = f"{BUCKET_PACKING_SEED}:{self.corpus_case.name}:{segment_index}".encode("utf-8")
        digest = blake2b(payload, digest_size=8).digest()
        return int.from_bytes(digest, byteorder="big")

    def _build_bucket_packed_example(
        self,
        segment_buffer: list[PackedTrainingSegment],
    ) -> PackedTrainingExample:
        # ---------------------------------------------------------
        # Start from the longest segment, then repeatedly choose the
        # buffered segment that best fills the remaining context.
        # ---------------------------------------------------------
        selected_segments: list[PackedTrainingSegment] = []
        start_index = self._find_longest_segment_index(segment_buffer=segment_buffer)
        selected_segments.append(segment_buffer.pop(start_index))
        remaining_size = self.max_len - len(selected_segments[0][0])

        while remaining_size > 0:
            next_index = self._find_best_fit_segment_index(
                segment_buffer=segment_buffer,
                remaining_size=remaining_size,
            )

            if next_index is None:
                break

            selected_segment = segment_buffer.pop(next_index)
            selected_segments.append(selected_segment)
            remaining_size -= len(selected_segment[0])

        return self._build_example_from_segments(segments=selected_segments)

    def _find_longest_segment_index(
        self,
        segment_buffer: list[PackedTrainingSegment],
    ) -> int:
        # ---------------------------------------------------------
        # Pick the longest segment first. The fixed order key keeps
        # ties deterministic while allowing bounded reordering.
        # ---------------------------------------------------------
        return max(
            range(len(segment_buffer)),
            key=lambda index: (len(segment_buffer[index][0]), -segment_buffer[index][2]),
        )

    def _find_best_fit_segment_index(
        self,
        segment_buffer: list[PackedTrainingSegment],
        remaining_size: int,
    ) -> int | None:
        # ---------------------------------------------------------
        # Choose the largest segment that fits in the current gap.
        # Return None when every remaining segment is too large.
        # ---------------------------------------------------------
        candidate_indexes = [
            index
            for index, segment in enumerate(segment_buffer)
            if len(segment[0]) <= remaining_size
        ]

        if len(candidate_indexes) == 0:
            return None

        return max(
            candidate_indexes,
            key=lambda index: (len(segment_buffer[index][0]), -segment_buffer[index][2]),
        )

    def _build_example_from_segments(
        self,
        segments: list[PackedTrainingSegment],
    ) -> PackedTrainingExample:
        # ---------------------------------------------------------
        # Flatten selected segments into one packed sample while
        # resetting positions and segment ids for each segment.
        # ---------------------------------------------------------
        input_token_ids: list[int] = []
        label_token_ids: list[int] = []
        position_ids: list[int] = []
        segment_ids: list[int] = []

        for segment_id, segment in enumerate(segments):
            segment_input_ids, segment_label_ids, _ = segment
            segment_len = len(segment_input_ids)
            input_token_ids.extend(segment_input_ids)
            label_token_ids.extend(segment_label_ids)
            position_ids.extend(range(segment_len))
            segment_ids.extend([segment_id for _ in range(segment_len)])

        return self._build_example(
            input_token_ids=input_token_ids,
            label_token_ids=label_token_ids,
            position_ids=position_ids,
            segment_ids=segment_ids,
        )

    def _build_example(
        self,
        input_token_ids: list[int],
        label_token_ids: list[int],
        position_ids: list[int],
        segment_ids: list[int],
    ) -> PackedTrainingExample:
        # ---------------------------------------------------------
        # Pad all packed streams so every sample matches the fixed
        # context length expected by the model and DataLoader.
        # ---------------------------------------------------------
        padding_size = self.max_len - len(input_token_ids)
        padded_input_ids = input_token_ids + [self.pad_token_id for _ in range(padding_size)]
        padded_label_ids = label_token_ids + [self.pad_token_id for _ in range(padding_size)]
        padded_position_ids = position_ids + [0 for _ in range(padding_size)]
        padded_segment_ids = segment_ids + [-1 for _ in range(padding_size)]
        inputs = torch.tensor(padded_input_ids, dtype=torch.long)
        labels = torch.tensor(padded_label_ids, dtype=torch.long)
        positions = torch.tensor(padded_position_ids, dtype=torch.long)
        segments = torch.tensor(padded_segment_ids, dtype=torch.long)
        return inputs, labels, positions, segments


class LocalTokenizedDataset(Dataset[PackedTrainingExample]):
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
        self.position_ids = payload["position_ids"]
        self.segment_ids = payload["segment_ids"]
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

    def __getitem__(self, index: int) -> PackedTrainingExample:
        # ---------------------------------------------------------
        # Return one cached packed example without additional
        # tokenization or network access.
        # ---------------------------------------------------------
        return (
            self.inputs[index],
            self.labels[index],
            self.position_ids[index],
            self.segment_ids[index],
        )


def build_tokenized_cache(
    dataset: IterableDataset[PackedTrainingExample],
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
    position_ids: list[torch.Tensor] = []
    segment_ids: list[torch.Tensor] = []

    # ---------------------------------------------------------
    # Consume exactly the configured validation budget and stop
    # before extra remote samples are streamed or tokenized.
    # ---------------------------------------------------------
    task_id = progress_manager.add_task(
        description="Building validation cache",
        total=num_samples,
    )

    try:
        for input_ids, label_ids, example_position_ids, example_segment_ids in dataset:
            inputs.append(input_ids)
            labels.append(label_ids)
            position_ids.append(example_position_ids)
            segment_ids.append(example_segment_ids)
            progress_manager.update(task_id=task_id, advance=1)

            if len(inputs) == num_samples:
                break
    finally:
        progress_manager.finish_task(task_id=task_id)

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
        "position_ids": torch.stack(position_ids),
        "segment_ids": torch.stack(segment_ids),
        "metadata": {
            "num_samples": num_samples,
            "max_len": max_len,
            **metadata,
        },
    }
    torch.save(payload, path)
