import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.midtraining.cli import parse_args
from src.midtraining.training_corpus_cases import MIDTRAINING_CORPUS_CASE


class MidtrainingTest(unittest.TestCase):
    def test_corpus_uses_synthetic_textbook_rewrite_column(self) -> None:
        # ---------------------------------------------------------
        # Train only on the synthetic textbook rewrite text from
        # the single default training split.
        # ---------------------------------------------------------
        self.assertEqual(
            (
                MIDTRAINING_CORPUS_CASE.dataset_path,
                MIDTRAINING_CORPUS_CASE.config_name,
                MIDTRAINING_CORPUS_CASE.split,
                MIDTRAINING_CORPUS_CASE.text_column,
            ),
            (
                "MK0727/SyntheticTextbook-jp",
                "default",
                "train",
                "rewrite",
            ),
        )

    def test_parse_args_requires_pretrained_model_directory(self) -> None:
        # ---------------------------------------------------------
        # Reject a missing source model before any dataset stream or
        # training output is initialized.
        # ---------------------------------------------------------
        with patch("sys.argv", ["train.py", "--model-path", "missing"]):
            with patch("sys.stderr", io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args()

    def test_parse_args_rejects_invalid_step_budget(self) -> None:
        # ---------------------------------------------------------
        # Reject non-positive step values before creating datasets
        # or loading the pretrained model weights.
        # ---------------------------------------------------------
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)

            for file_name in ["model.pth", "model_config.json", "tokenizer.json"]:
                (model_dir / file_name).touch()

            argv = [
                "train.py",
                "--model-path",
                str(model_dir),
                "--max-steps",
                "0",
            ]

            with patch("sys.argv", argv), patch("sys.stderr", io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args()

    def test_parse_args_rejects_invalid_runtime_values(self) -> None:
        # ---------------------------------------------------------
        # Reject values that would otherwise fail later in dataset
        # packing, DataLoader setup, or chunked loss computation.
        # ---------------------------------------------------------
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)

            for file_name in ["model.pth", "model_config.json", "tokenizer.json"]:
                (model_dir / file_name).touch()

            invalid_cases = [
                ("--max-len", "0"),
                ("--batch-size", "0"),
                ("--val-split-modulo", "0"),
                ("--val-batches", "0"),
                ("--val-check-interval", "0"),
                ("--checkpoint-every-n-steps", "0"),
                ("--metric-log-every-n-steps", "0"),
                ("--loss-chunk-size", "0"),
            ]

            for flag, value in invalid_cases:
                argv = [
                    "train.py",
                    "--model-path",
                    str(model_dir),
                    flag,
                    value,
                ]

                with self.subTest(flag=flag), patch("sys.argv", argv), patch("sys.stderr", io.StringIO()):
                    with self.assertRaises(SystemExit):
                        parse_args()

    def test_parse_args_accepts_longer_context_than_source_model(self) -> None:
        # ---------------------------------------------------------
        # Allow midtraining to request longer context than the saved
        # pretraining config because the model rebuild handles RoPE.
        # ---------------------------------------------------------
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)

            for file_name in ["model.pth", "model_config.json", "tokenizer.json"]:
                (model_dir / file_name).touch()

            argv = [
                "train.py",
                "--model-path",
                str(model_dir),
                "--max-len",
                "4096",
            ]

            with patch("sys.argv", argv):
                args = parse_args()

        self.assertEqual(args.max_len, 4096)


if __name__ == "__main__":
    unittest.main()
