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


PackedTrainingExample = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]
PackedTrainingSegment = tuple[list[int], list[int], int]
PACKING_VERSION = "bucket-packing-flash-varlen-v1"
SHUFFLE_BUFFER_SIZE = 10000
SHUFFLE_SEED = 17
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
        repeat: bool = False,
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
        self.repeat = repeat

    def __iter__(self) -> Iterator[PackedTrainingExample]:
        # ---------------------------------------------------------
        # Training can repeat the finite corpus inside each worker
        # while validation keeps one finite pass for cache checks.
        # ---------------------------------------------------------
        repeat_index = 0

        while True:
            yielded = False

            for example in self._iter_corpus_pass(repeat_index=repeat_index):
                yielded = True
                yield example

            if not self.repeat or not yielded:
                return

            repeat_index += 1

    def _iter_corpus_pass(
        self,
        repeat_index: int,
    ) -> Iterator[PackedTrainingExample]:
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
                seed=self.shuffle_seed + repeat_index,
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
        document_lengths: list[int] = []

        for segment in segments:
            segment_input_ids, segment_label_ids, _ = segment
            segment_len = len(segment_input_ids)
            input_token_ids.extend(segment_input_ids)
            label_token_ids.extend(segment_label_ids)
            position_ids.extend(range(segment_len))
            document_lengths.append(segment_len)

        return self._build_example(
            input_token_ids=input_token_ids,
            label_token_ids=label_token_ids,
            position_ids=position_ids,
            document_lengths=document_lengths,
        )

    def _build_example(
        self,
        input_token_ids: list[int],
        label_token_ids: list[int],
        position_ids: list[int],
        document_lengths: list[int],
    ) -> PackedTrainingExample:
        # ---------------------------------------------------------
        # Return compact packed streams and document offsets so
        # FlashAttention varlen can avoid dense attention masks.
        # ---------------------------------------------------------
        cumulative_lengths = torch.tensor(document_lengths, dtype=torch.int32).cumsum(dim=0)
        cu_seqlens = torch.tensor([0, *cumulative_lengths.tolist()], dtype=torch.int32)
        inputs = torch.tensor(input_token_ids, dtype=torch.long)
        labels = torch.tensor(label_token_ids, dtype=torch.long)
        positions = torch.tensor(position_ids, dtype=torch.long)
        max_seqlen = max(document_lengths)
        return inputs, labels, positions, cu_seqlens, max_seqlen


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
        self.cu_seqlens = payload["cu_seqlens"]
        self.max_seqlens = payload["max_seqlens"]
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
            self.cu_seqlens[index],
            int(self.max_seqlens[index]),
        )


def collate_packed_examples(
    examples: list[PackedTrainingExample],
) -> PackedTrainingExample:
    # ---------------------------------------------------------
    # Flatten variable-length packed samples into one global token
    # stream and rebuild document offsets for FlashAttention.
    # ---------------------------------------------------------
    input_ids = torch.cat([example[0] for example in examples])
    label_ids = torch.cat([example[1] for example in examples])
    position_ids = torch.cat([example[2] for example in examples])
    cu_seqlens_values = [0]
    total_tokens = 0

    for _, _, _, sample_cu_seqlens, _ in examples:
        document_lengths = sample_cu_seqlens.diff().tolist()

        for document_length in document_lengths:
            total_tokens += int(document_length)
            cu_seqlens_values.append(total_tokens)

    cu_seqlens = torch.tensor(cu_seqlens_values, dtype=torch.int32)
    max_seqlen = max(example[4] for example in examples)
    return input_ids, label_ids, position_ids, cu_seqlens, max_seqlen


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
    cu_seqlens: list[torch.Tensor] = []
    max_seqlens: list[int] = []

    # ---------------------------------------------------------
    # Consume exactly the configured validation budget and stop
    # before extra remote samples are streamed or tokenized.
    # ---------------------------------------------------------
    task_id = progress_manager.add_task(
        description="Building validation cache",
        total=num_samples,
    )

    try:
        for input_ids, label_ids, example_position_ids, example_cu_seqlens, example_max_seqlen in dataset:
            inputs.append(input_ids)
            labels.append(label_ids)
            position_ids.append(example_position_ids)
            cu_seqlens.append(example_cu_seqlens)
            max_seqlens.append(example_max_seqlen)
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
        "inputs": inputs,
        "labels": labels,
        "position_ids": position_ids,
        "cu_seqlens": cu_seqlens,
        "max_seqlens": max_seqlens,
        "metadata": {
            "num_samples": num_samples,
            "max_len": max_len,
            **metadata,
        },
    }
    torch.save(payload, path)
