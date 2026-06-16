from collections.abc import Callable
from collections.abc import Iterator
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import IterableDataset

from src.pretraining.dataset import build_tokenized_cache
from src.pretraining.dataset import LocalTokenizedDataset
from src.pretraining.dataset import MixedPretrainingDataset
from src.pretraining.dataset import PretrainingCorpusDataset
from src.pretraining.dataset import resolve_mix_token_targets
from src.pretraining.dataset import resolve_scheduled_mix_percentages
from src.pretraining.training_corpus_cases import PretrainingCorpusCase
from src.pretraining.training_corpus_cases import PRETRAINING_CORPUS_CASES


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


class EmptyMixedPretrainingDataset(MixedPretrainingDataset):
    def _build_corpus_iterator(
        self,
        corpus_case: PretrainingCorpusCase,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        # ---------------------------------------------------------
        # Return an empty stream so tests can verify the mixer error
        # path without opening a remote Hugging Face dataset.
        # ---------------------------------------------------------
        return iter(())


class NamedTokenMixedPretrainingDataset(MixedPretrainingDataset):
    def _build_corpus_iterator(
        self,
        corpus_case: PretrainingCorpusCase,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        # ---------------------------------------------------------
        # Yield a source-specific token so tests can verify finite
        # corpus exhaustion without opening remote datasets.
        # ---------------------------------------------------------
        token_id = 1 if corpus_case.name == "fineweb" else 2
        input_ids = torch.tensor([token_id, 0], dtype=torch.long)
        label_ids = torch.tensor([token_id, 0], dtype=torch.long)
        position_ids = torch.tensor([0, 0], dtype=torch.long)
        segment_ids = torch.tensor([0, -1], dtype=torch.long)
        example = (input_ids, label_ids, position_ids, segment_ids)

        if corpus_case.name == "wiki":
            return iter([example])

        return iter([example for _ in range(10)])


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

    def __iter__(self) -> Iterator[dict[str, str]]:
        # ---------------------------------------------------------
        # Yield samples one by one like a streaming dataset.
        # ---------------------------------------------------------
        return iter(self.samples)


def build_case(
    name: str,
    token_percentage: float,
    is_ramped: bool = False,
    repeat_on_end: bool = True,
    excluded_url_domains: tuple[str, ...] = (),
) -> PretrainingCorpusCase:
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
        is_ramped=is_ramped,
        repeat_on_end=repeat_on_end,
        excluded_url_domains=excluded_url_domains,
    )


class PretrainingDatasetTest(unittest.TestCase):
    def test_pretraining_corpus_cases_use_requested_japanese_datasets(self) -> None:
        # ---------------------------------------------------------
        # Keep the production corpus list pointed at the requested
        # Japanese FineWeb and cleaned Wikipedia datasets.
        # ---------------------------------------------------------
        self.assertEqual(
            [
                (
                    corpus_case.name,
                    corpus_case.dataset_path,
                    corpus_case.config_name,
                    corpus_case.split,
                    corpus_case.text_column,
                    corpus_case.repeat_on_end,
                    corpus_case.excluded_url_domains,
                )
                for corpus_case in PRETRAINING_CORPUS_CASES
            ],
            [
                (
                    "fineweb2-edu-ja",
                    "hotchpotch/fineweb-2-edu-japanese",
                    "default",
                    "train",
                    "text",
                    True,
                    ("wikipedia.org",),
                ),
                (
                    "cleanedwiki-jp",
                    "MK0727/CleanedWiki-jp",
                    "all",
                    "train",
                    "text",
                    True,
                    (),
                ),
            ],
        )

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

    def test_resolve_scheduled_mix_percentages_ramps_late_wiki(self) -> None:
        # ---------------------------------------------------------
        # Keep CleanedWiki out of early training, then linearly
        # increase it to the final 70 percent by train end.
        # ---------------------------------------------------------
        corpus_cases = [
            build_case(name="fineweb", token_percentage=30.0),
            build_case(
                name="wiki",
                token_percentage=70.0,
                is_ramped=True,
                repeat_on_end=True,
            ),
        ]

        self.assertEqual(
            resolve_scheduled_mix_percentages(
                corpus_cases=corpus_cases,
                progress=0.0,
                ramp_start_progress=0.5,
            ),
            [100.0, 0.0],
        )
        self.assertEqual(
            resolve_scheduled_mix_percentages(
                corpus_cases=corpus_cases,
                progress=0.49,
                ramp_start_progress=0.5,
            ),
            [100.0, 0.0],
        )
        self.assertEqual(
            resolve_scheduled_mix_percentages(
                corpus_cases=corpus_cases,
                progress=0.75,
                ramp_start_progress=0.5,
            ),
            [65.0, 35.0],
        )
        self.assertEqual(
            resolve_scheduled_mix_percentages(
                corpus_cases=corpus_cases,
                progress=1.0,
                ramp_start_progress=0.5,
            ),
            [30.0, 70.0],
        )

    def test_repeatable_wiki_reopens_after_stream_end(self) -> None:
        # ---------------------------------------------------------
        # Reopen CleanedWiki after stream end so the late-training
        # mixture can keep using it at the scheduled percentage.
        # ---------------------------------------------------------
        dataset = NamedTokenMixedPretrainingDataset(
            corpus_cases=[
                build_case(name="fineweb", token_percentage=30.0),
                build_case(
                    name="wiki",
                    token_percentage=70.0,
                    is_ramped=True,
                    repeat_on_end=True,
                ),
            ],
            tokenizer=None,
            max_len=2,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            mix_cycle_tokens=10,
            total_training_tokens=1,
            ramp_start_progress=0.0,
        )

        dataset_iterator = iter(dataset)
        input_ids = [next(dataset_iterator)[0] for _ in range(25)]
        token_ids = [input_id[0].item() for input_id in input_ids]

        self.assertEqual(token_ids.count(2), 9)
        self.assertEqual(token_ids[:3], [1, 1, 1])

    def test_pretraining_corpus_split_uses_text_column(self) -> None:
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

        self.assertTrue(
            dataset._contains_partition(
                sample=sample,
            )
        )
        self.assertIn(split_index, dataset.split_indexes)

    def test_pretraining_corpus_filters_excluded_wikipedia_urls(self) -> None:
        # ---------------------------------------------------------
        # Drop only Wikipedia hostnames and subdomains from FineWeb
        # while keeping unrelated wiki-looking domains available.
        # ---------------------------------------------------------
        corpus_case = build_case(
            name="fineweb",
            token_percentage=100.0,
            excluded_url_domains=("wikipedia.org",),
        )
        dataset = PretrainingCorpusDataset(
            corpus_case=corpus_case,
            tokenizer=None,
            max_len=4,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
        )

        self.assertFalse(
            dataset._contains_allowed_url(
                sample={"url": "https://ja.wikipedia.org/wiki/Python"}
            )
        )
        self.assertFalse(
            dataset._contains_allowed_url(
                sample={"url": "https://www.wikipedia.org/portal/"}
            )
        )
        self.assertTrue(
            dataset._contains_allowed_url(
                sample={"url": "https://example.com/wiki/Python"}
            )
        )
        self.assertTrue(
            dataset._contains_allowed_url(
                sample={"url": "https://ja.lotr.wikia.com/wiki/Page"}
            )
        )
        self.assertTrue(dataset._contains_allowed_url(sample={"url": "not-a-url"}))
        self.assertTrue(dataset._contains_allowed_url(sample={}))

    def test_pretraining_corpus_bucket_packs_best_fit_documents(self) -> None:
        # ---------------------------------------------------------
        # Pack short documents by best fit so equal-length segments
        # can fill the context window without padding.
        # ---------------------------------------------------------
        corpus_case = build_case(name="custom", token_percentage=100.0)
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

        with patch("src.pretraining.dataset.load_dataset", return_value=fake_dataset):
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
        corpus_case = build_case(name="custom", token_percentage=100.0)
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

        with patch("src.pretraining.dataset.load_dataset", return_value=fake_dataset):
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
        corpus_case = build_case(name="custom", token_percentage=100.0)
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
        self.assertTrue(torch.equal(payload["position_ids"][0], torch.tensor([0, 1])))
        self.assertTrue(torch.equal(payload["segment_ids"][0], torch.tensor([0, -1])))

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
