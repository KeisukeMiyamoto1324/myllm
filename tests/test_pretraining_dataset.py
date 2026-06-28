from collections.abc import Callable
from collections.abc import Iterator
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import IterableDataset

from src.pretraining.dataset import build_tokenized_cache
from src.pretraining.dataset import LocalTokenizedDataset
from src.pretraining.dataset import PretrainingCorpusDataset
from src.pretraining.training_corpus_cases import PretrainingCorpusCase
from src.pretraining.training_corpus_cases import PRETRAINING_CORPUS_CASE


class FixedTokenDataset(IterableDataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        # ---------------------------------------------------------
        # Yield fixed examples so cache metadata can be tested
        # without opening remote datasets.
        # ---------------------------------------------------------
        for value in range(2):
            input_ids = torch.tensor([value, value + 1], dtype=torch.long)
            label_ids = torch.tensor([value + 1, 0], dtype=torch.long)
            position_ids = torch.tensor([0, 1], dtype=torch.long)
            segment_ids = torch.tensor([0, -1], dtype=torch.long)
            yield input_ids, label_ids, position_ids, segment_ids


class FixedTokenizer:
    def tokenize(self, sentence: str) -> list[int]:
        # ---------------------------------------------------------
        # Return integer tokens from whitespace-separated test text
        # so packing behavior can be asserted directly.
        # ---------------------------------------------------------
        return [int(token) for token in sentence.split()]


class FakeStreamingDataset:
    def __init__(self, samples: list[dict[str, str]]) -> None:
        # ---------------------------------------------------------
        # Keep a small in-memory sample list with the filter method
        # used by Hugging Face streaming datasets.
        # ---------------------------------------------------------
        self.samples = samples

    def filter(self, predicate: Callable[[dict[str, str]], bool]) -> "FakeStreamingDataset":
        # ---------------------------------------------------------
        # Apply the dataset predicate eagerly so tests can exercise
        # the same code path as the real streaming dataset.
        # ---------------------------------------------------------
        return FakeStreamingDataset(
            samples=[sample for sample in self.samples if predicate(sample)]
        )

    def shard(self, num_shards: int, index: int) -> "FakeStreamingDataset":
        # ---------------------------------------------------------
        # Return one deterministic worker shard so streaming worker
        # partitioning can be tested without remote datasets.
        # ---------------------------------------------------------
        return FakeStreamingDataset(samples=self.samples[index::num_shards])

    def shuffle(self, seed: int, buffer_size: int) -> "FakeStreamingDataset":
        # ---------------------------------------------------------
        # Shuffle the complete small test list while accepting the
        # same arguments as a streaming Hugging Face dataset.
        # ---------------------------------------------------------
        del buffer_size
        shuffled_samples = list(self.samples)
        random.Random(seed).shuffle(shuffled_samples)
        return FakeStreamingDataset(samples=shuffled_samples)

    def __iter__(self) -> Iterator[dict[str, str]]:
        # ---------------------------------------------------------
        # Yield samples one by one like a streaming dataset.
        # ---------------------------------------------------------
        return iter(self.samples)


def build_case(
    name: str,
) -> PretrainingCorpusCase:
    # ---------------------------------------------------------
    # Build a minimal corpus case for tests that do not open the
    # real remote dataset.
    # ---------------------------------------------------------
    return PretrainingCorpusCase(
        name=name,
        genre="test",
        language="en",
        dataset_path="unused",
        config_name="unused",
        split="train",
        text_column="text",
    )


class PretrainingDatasetTest(unittest.TestCase):
    def test_pretraining_corpus_case_uses_cleaned_fineweb2_edu_jp(self) -> None:
        # ---------------------------------------------------------
        # Keep pretraining pointed only at the requested Japanese
        # FineWeb dataset.
        # ---------------------------------------------------------
        self.assertEqual(
            (
                PRETRAINING_CORPUS_CASE.name,
                PRETRAINING_CORPUS_CASE.dataset_path,
                PRETRAINING_CORPUS_CASE.config_name,
                PRETRAINING_CORPUS_CASE.split,
                PRETRAINING_CORPUS_CASE.text_column,
            ),
            (
                "cleaned-fineweb2-edu-jp",
                "MK0727/CleanedFineWeb2Edu-jp",
                "default",
                "train",
                "text",
            ),
        )

    def test_pretraining_corpus_split_uses_text_column(self) -> None:
        # ---------------------------------------------------------
        # Split a sample by its configured text column instead of a
        # hard-coded column name.
        # ---------------------------------------------------------
        corpus_case = build_case(name="custom")
        corpus_case.text_column = "body"
        dataset = PretrainingCorpusDataset(
            corpus_case=corpus_case,
            tokenizer=None,
            max_len=4,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
        )
        sample = {"body": "hello", "text": "different"}
        document_index = dataset._resolve_document_index(text=sample["body"])
        split_index = document_index % dataset.split_modulo

        self.assertTrue(
            dataset._contains_partition(
                sample=sample,
            )
        )
        self.assertIn(split_index, dataset.split_indexes)

    def test_pretraining_corpus_shards_streaming_dataset_by_worker(self) -> None:
        # ---------------------------------------------------------
        # Read only the current worker shard so parallel workers do
        # not replay the same streaming samples.
        # ---------------------------------------------------------
        corpus_case = build_case(name="custom")
        dataset = PretrainingCorpusDataset(
            corpus_case=corpus_case,
            tokenizer=FixedTokenizer(),
            max_len=2,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
        )
        fake_dataset = FakeStreamingDataset(
            samples=[
                {"text": "10"},
                {"text": "20"},
                {"text": "30"},
                {"text": "40"},
            ]
        )
        worker_info = type("WorkerInfo", (), {"num_workers": 2, "id": 1})()

        with patch("src.shared.packed_dataset.load_dataset", return_value=fake_dataset):
            with patch("src.shared.packed_dataset.get_worker_info", return_value=worker_info):
                examples = list(iter(dataset))

        input_token_ids = [example[0][1].item() for example in examples]
        self.assertEqual(sorted(input_token_ids), [20, 40])

    def test_pretraining_corpus_shards_streaming_dataset_by_rank_and_worker(self) -> None:
        # ---------------------------------------------------------
        # Combine DDP rank and DataLoader worker ids so every
        # process reads a distinct streaming shard.
        # ---------------------------------------------------------
        corpus_case = build_case(name="custom")
        dataset = PretrainingCorpusDataset(
            corpus_case=corpus_case,
            tokenizer=FixedTokenizer(),
            max_len=2,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
        )
        fake_dataset = FakeStreamingDataset(
            samples=[
                {"text": "10"},
                {"text": "20"},
                {"text": "30"},
                {"text": "40"},
                {"text": "50"},
                {"text": "60"},
                {"text": "70"},
                {"text": "80"},
            ]
        )
        worker_info = type("WorkerInfo", (), {"num_workers": 2, "id": 1})()

        with (
            patch("src.shared.packed_dataset.load_dataset", return_value=fake_dataset),
            patch("src.shared.packed_dataset.get_worker_info", return_value=worker_info),
            patch("torch.distributed.is_available", return_value=True),
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.get_world_size", return_value=2),
            patch("torch.distributed.get_rank", return_value=1),
        ):
            examples = list(iter(dataset))

        input_token_ids = [example[0][1].item() for example in examples]
        self.assertEqual(sorted(input_token_ids), [40, 80])

    def test_pretraining_corpus_keeps_wikipedia_urls(self) -> None:
        # ---------------------------------------------------------
        # Keep Wikipedia documents because the single FineWeb corpus
        # no longer applies URL-domain exclusions.
        # ---------------------------------------------------------
        dataset = PretrainingCorpusDataset(
            corpus_case=build_case(name="custom"),
            tokenizer=FixedTokenizer(),
            max_len=2,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
        )
        fake_dataset = FakeStreamingDataset(
            samples=[
                {
                    "text": "10",
                    "url": "https://ja.wikipedia.org/wiki/Python",
                },
            ]
        )

        with patch("src.shared.packed_dataset.load_dataset", return_value=fake_dataset):
            examples = list(iter(dataset))

        self.assertEqual(examples[0][0].tolist(), [1, 10])

    def test_pretraining_corpus_bucket_packs_best_fit_documents(self) -> None:
        # ---------------------------------------------------------
        # Pack short documents by best fit so equal-length segments
        # can fill the context window without padding.
        # ---------------------------------------------------------
        corpus_case = build_case(name="custom")
        dataset = PretrainingCorpusDataset(
            corpus_case=corpus_case,
            tokenizer=FixedTokenizer(),
            max_len=6,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
        )
        fake_dataset = FakeStreamingDataset(
            samples=[
                {"text": "10 11"},
                {"text": "20"},
                {"text": "30 31"},
            ]
        )

        with patch("src.shared.packed_dataset.load_dataset", return_value=fake_dataset):
            input_ids, label_ids, position_ids, segment_ids = next(iter(dataset))

        self.assertEqual(input_ids.tolist(), [1, 10, 11, 1, 30, 31])
        self.assertEqual(label_ids.tolist(), [10, 11, 2, 30, 31, 2])
        self.assertEqual(position_ids.tolist(), [0, 1, 2, 0, 1, 2])
        self.assertEqual(segment_ids.tolist(), [0, 0, 0, 1, 1, 1])

    def test_pretraining_corpus_bucket_packs_remaining_segments(self) -> None:
        # ---------------------------------------------------------
        # Flush all buffered segments at stream end so short leftover
        # documents are still emitted as padded packed samples.
        # ---------------------------------------------------------
        corpus_case = build_case(name="custom")
        dataset = PretrainingCorpusDataset(
            corpus_case=corpus_case,
            tokenizer=FixedTokenizer(),
            max_len=6,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
        )
        fake_dataset = FakeStreamingDataset(
            samples=[
                {"text": "10 11"},
                {"text": "20"},
                {"text": "30 31"},
            ]
        )

        with patch("src.shared.packed_dataset.load_dataset", return_value=fake_dataset):
            examples = list(iter(dataset))

        input_ids, label_ids, position_ids, segment_ids = examples[1]
        self.assertEqual(input_ids.tolist(), [1, 20, 0, 0, 0, 0])
        self.assertEqual(label_ids.tolist(), [20, 2, 0, 0, 0, 0])
        self.assertEqual(position_ids.tolist(), [0, 1, 0, 0, 0, 0])
        self.assertEqual(segment_ids.tolist(), [0, 0, -1, -1, -1, -1])

    def test_pretraining_corpus_splits_oversized_documents(self) -> None:
        # ---------------------------------------------------------
        # Split a single long document into max_len-sized segments
        # so long samples are not dropped during packing.
        # ---------------------------------------------------------
        corpus_case = build_case(name="custom")
        dataset = PretrainingCorpusDataset(
            corpus_case=corpus_case,
            tokenizer=FixedTokenizer(),
            max_len=3,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
        )

        segments = list(dataset._create_segments(text="10 11 12 13"))

        self.assertEqual(
            segments,
            [
                ([1, 10, 11], [10, 11, 12]),
                ([12, 13], [13, 2]),
            ],
        )

    def test_packed_corpus_changes_shuffle_order_by_epoch(self) -> None:
        # ---------------------------------------------------------
        # Use a reproducible but distinct shuffle seed for every
        # corpus pass in multi-epoch mid-training.
        # ---------------------------------------------------------
        dataset = PretrainingCorpusDataset(
            corpus_case=build_case(name="custom"),
            tokenizer=FixedTokenizer(),
            max_len=2,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            shuffle_buffer_size=10000,
            shuffle_seed=17,
        )
        fake_dataset = FakeStreamingDataset(
            samples=[{"text": str(value)} for value in range(10, 20)]
        )

        with patch("src.shared.packed_dataset.load_dataset", return_value=fake_dataset):
            first_epoch = [example[0][1].item() for example in dataset]
            dataset.set_epoch(epoch_index=1)
            second_epoch = [example[0][1].item() for example in dataset]
            dataset.set_epoch(epoch_index=0)
            repeated_first_epoch = [example[0][1].item() for example in dataset]

        self.assertNotEqual(first_epoch, second_epoch)
        self.assertEqual(first_epoch, repeated_first_epoch)
        self.assertEqual(sorted(first_epoch), list(range(10, 20)))

    def test_build_tokenized_cache_keeps_metadata(self) -> None:
        # ---------------------------------------------------------
        # Store caller-provided corpus metadata beside validation
        # tensors so stale caches can be identified.
        # ---------------------------------------------------------
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "validation.pt"
            build_tokenized_cache(
                dataset=FixedTokenDataset(),
                path=path,
                num_samples=2,
                max_len=2,
                metadata={"corpus_signature": "abc123"},
            )
            payload = torch.load(path, map_location="cpu")

        self.assertEqual(payload["metadata"]["num_samples"], 2)
        self.assertEqual(payload["metadata"]["max_len"], 2)
        self.assertEqual(payload["metadata"]["corpus_signature"], "abc123")
        self.assertTrue(torch.equal(payload["position_ids"][0], torch.tensor([0, 1])))
        self.assertTrue(torch.equal(payload["segment_ids"][0], torch.tensor([0, -1])))

    def test_local_tokenized_dataset_rejects_metadata_mismatch(self) -> None:
        # ---------------------------------------------------------
        # Refuse a cache file when an explicit validation path points
        # to tensors from a different corpus.
        # ---------------------------------------------------------
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "validation.pt"
            build_tokenized_cache(
                dataset=FixedTokenDataset(),
                path=path,
                num_samples=2,
                max_len=2,
                metadata={"corpus_signature": "abc123"},
            )

            with self.assertRaises(ValueError):
                LocalTokenizedDataset(
                    path=path,
                    max_len=2,
                    num_samples=2,
                    metadata={"corpus_signature": "def456"},
                )


if __name__ == "__main__":
    unittest.main()
