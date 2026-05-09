from collections.abc import Iterator
from pathlib import Path
import tempfile
import unittest

import torch
from torch.utils.data import IterableDataset

from src.pretraining.dataset import build_tokenized_cache
from src.pretraining.dataset import LocalTokenizedDataset
from src.pretraining.dataset import MixedPretrainingDataset
from src.pretraining.dataset import PretrainingCorpusDataset
from src.pretraining.dataset import resolve_mix_token_targets
from src.pretraining.training_corpus_cases import PretrainingCorpusCase


class FixedTokenDataset(IterableDataset[tuple[torch.Tensor, torch.Tensor]]):
    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        # ---------------------------------------------------------
        # Yield fixed examples so cache metadata can be tested
        # without opening remote datasets.
        # ---------------------------------------------------------
        for value in range(2):
            input_ids = torch.tensor([value, value + 1], dtype=torch.long)
            label_ids = torch.tensor([value + 1, 0], dtype=torch.long)
            yield input_ids, label_ids


class EmptyMixedPretrainingDataset(MixedPretrainingDataset):
    def _build_corpus_iterator(
        self,
        corpus_case: PretrainingCorpusCase,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        # ---------------------------------------------------------
        # Return an empty stream so tests can verify the mixer error
        # path without opening a remote Hugging Face dataset.
        # ---------------------------------------------------------
        return iter(())


def build_case(name: str, token_percentage: float) -> PretrainingCorpusCase:
    # ---------------------------------------------------------
    # Build a minimal corpus case for tests that only need mixture
    # settings rather than real dataset streaming.
    # ---------------------------------------------------------
    return PretrainingCorpusCase(
        name=name,
        genre="test",
        language="en",
        dataset_path="unused",
        config_name="unused",
        split="train",
        text_column="text",
        token_percentage=token_percentage,
    )


class PretrainingDatasetTest(unittest.TestCase):
    def test_resolve_mix_token_targets_uses_percentages(self) -> None:
        # ---------------------------------------------------------
        # Convert 70/30 percentages into exact integer token targets
        # for a 10-token mixing cycle.
        # ---------------------------------------------------------
        corpus_cases = [
            build_case(name="a", token_percentage=70.0),
            build_case(name="b", token_percentage=30.0),
        ]
        token_targets = resolve_mix_token_targets(
            corpus_cases=corpus_cases,
            mix_cycle_tokens=10,
        )
        self.assertEqual(token_targets, [7, 3])

    def test_resolve_mix_token_targets_rejects_incomplete_percentages(self) -> None:
        # ---------------------------------------------------------
        # Reject mixtures that do not allocate the full 100 percent
        # token budget.
        # ---------------------------------------------------------
        corpus_cases = [
            build_case(name="a", token_percentage=60.0),
            build_case(name="b", token_percentage=30.0),
        ]

        with self.assertRaises(ValueError):
            resolve_mix_token_targets(
                corpus_cases=corpus_cases,
                mix_cycle_tokens=10,
        )

    def test_pretraining_corpus_partition_uses_text_column(self) -> None:
        # ---------------------------------------------------------
        # Split a sample by its configured text column instead of a
        # hard-coded column name.
        # ---------------------------------------------------------
        corpus_case = build_case(name="custom", token_percentage=100.0)
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
        worker_index = (document_index // dataset.split_modulo) % 2

        self.assertTrue(
            dataset._contains_partition(
                sample=sample,
                worker_modulo=2,
                worker_index=worker_index,
            )
        )
        self.assertFalse(
            dataset._contains_partition(
                sample=sample,
                worker_modulo=2,
                worker_index=1 - worker_index,
            )
        )
        self.assertIn(split_index, dataset.split_indexes)

    def test_mixed_dataset_reports_empty_filtered_corpus(self) -> None:
        # ---------------------------------------------------------
        # Convert a twice-empty corpus stream into a clear data split
        # error instead of leaking StopIteration from the generator.
        # ---------------------------------------------------------
        dataset = EmptyMixedPretrainingDataset(
            corpus_cases=[build_case(name="empty", token_percentage=100.0)],
            tokenizer=None,
            max_len=4,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            mix_cycle_tokens=4,
        )

        with self.assertRaisesRegex(ValueError, "empty"):
            next(iter(dataset))

    def test_build_tokenized_cache_keeps_metadata(self) -> None:
        # ---------------------------------------------------------
        # Store caller-provided corpus metadata beside validation
        # tensors so stale mixed caches can be identified.
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

    def test_local_tokenized_dataset_rejects_metadata_mismatch(self) -> None:
        # ---------------------------------------------------------
        # Refuse a cache file when an explicit validation path points
        # to tensors from a different corpus mixture.
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
