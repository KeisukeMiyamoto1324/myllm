import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.midtraining.train import parse_args
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

    def test_parse_args_uses_midtraining_defaults(self) -> None:
        # ---------------------------------------------------------
        # Keep fixed LR, inherited max length, and separate output
        # directory as the standard mid-training behavior.
        # ---------------------------------------------------------
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)

            for file_name in ["model.pth", "model_config.json", "tokenizer.json"]:
                (model_dir / file_name).touch()

            with patch("sys.argv", ["train.py", "--model-path", str(model_dir)]):
                args = parse_args()

        self.assertIsNone(args.max_len)
        self.assertEqual(args.learning_rate, 2e-5)
        self.assertEqual(args.max_steps, 40960)
        self.assertEqual(args.checkpoint_every_n_steps, 5000)
        self.assertEqual(args.output_path, "models/lambda-160m-midtrained")

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


if __name__ == "__main__":
    unittest.main()
