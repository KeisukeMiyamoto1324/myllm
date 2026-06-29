import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

import torch

from src.shared.validation_generation import ValidationGenerationCallback


class ValidationDataset:
    def __init__(self) -> None:
        self.samples = [
            (
                torch.tensor([1, 10, 11, 12, 13, 2, 20, 21]),
                torch.tensor([10, 11, 12, 13, 2, 0, 21, 2]),
                torch.tensor([0, 1, 2, 3, 4, 5, 0, 1]),
                torch.tensor([0, 0, 0, 0, 0, 0, 1, 1], dtype=torch.long),
            ),
            (
                torch.tensor([1, 30, 31, 2]),
                torch.tensor([30, 31, 2, 0]),
                torch.tensor([0, 1, 2, 3]),
                torch.tensor([0, 0, 0, 0], dtype=torch.long),
            ),
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.samples[index]


class ValidationTokenizer:
    eos_token = "|<eos>|"

    def token_to_id(self, token: str) -> int:
        del token
        return 2

    def detokenize(self, token_ids: list[int]) -> str:
        return " ".join(str(token_id) for token_id in token_ids)


class ValidationGenerationCallbackTest(unittest.TestCase):
    def test_validation_generation_saves_only_preview_samples(self) -> None:
        # ---------------------------------------------------------
        # Generate only the preview budget and save each generated
        # continuation with its prompt in the step JSONL file.
        # ---------------------------------------------------------
        with tempfile.TemporaryDirectory() as temp_dir:
            callback = ValidationGenerationCallback(
                dataset=ValidationDataset(),
                tokenizer=ValidationTokenizer(),
                output_dir=Path(temp_dir),
                preview_count=1,
            )
            trainer = MagicMock()
            trainer.global_step = 1000
            trainer.is_global_zero = True

            with patch.object(callback, "_generate_ids", return_value=[40, 41]) as generate_ids:
                callback.on_validation_epoch_end(
                    trainer=trainer,
                    pl_module=MagicMock(),
                )

            output_path = Path(temp_dir) / "step-1000.jsonl"
            results = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(len(results), 1)
        self.assertEqual(generate_ids.call_count, 1)
        self.assertEqual(results[0]["prompt"], "1 10 11")
        self.assertEqual(results[0]["generated_text"], "40 41")
        self.assertEqual(results[0]["global_step"], 1000)


if __name__ == "__main__":
    unittest.main()
