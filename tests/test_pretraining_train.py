import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.pretraining.train import parse_args
from src.pretraining.transformer import resolve_warmup_cosine_learning_rate


class PretrainingTrainTest(unittest.TestCase):
    def test_parse_args_uses_warmup_cosine_lr_defaults(self) -> None:
        # ---------------------------------------------------------
        # Keep the existing learning rate as the maximum value while
        # enabling warmup and cosine decay by default.
        # ---------------------------------------------------------
        with patch("sys.argv", ["train.py"]):
            args = parse_args()

        self.assertEqual(args.learning_rate, 2e-4)
        self.assertEqual(args.lr_warmup_steps, 2000)
        self.assertEqual(args.min_learning_rate_ratio, 0.1)

    def test_parse_args_rejects_invalid_lr_schedule(self) -> None:
        # ---------------------------------------------------------
        # Reject schedule values that cannot form a bounded warmup
        # and decay interval before training starts.
        # ---------------------------------------------------------
        argv = [
            "train.py",
            "--max-steps",
            "2000",
            "--lr-warmup-steps",
            "2000",
        ]

        with patch("sys.argv", argv), patch("sys.stderr", io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args()

        argv = [
            "train.py",
            "--min-learning-rate-ratio",
            "1.1",
        ]

        with patch("sys.argv", argv), patch("sys.stderr", io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args()

    def test_resolve_warmup_cosine_learning_rate(self) -> None:
        # ---------------------------------------------------------
        # Match the modern pretraining schedule: zero start, linear
        # warmup to max LR, then cosine decay to the minimum LR.
        # ---------------------------------------------------------
        max_learning_rate = 2e-4
        min_learning_rate = 2e-5

        self.assertEqual(
            resolve_warmup_cosine_learning_rate(
                step=0,
                max_learning_rate=max_learning_rate,
                min_learning_rate=min_learning_rate,
                warmup_steps=2000,
                total_steps=600000,
            ),
            0.0,
        )
        self.assertEqual(
            resolve_warmup_cosine_learning_rate(
                step=1000,
                max_learning_rate=max_learning_rate,
                min_learning_rate=min_learning_rate,
                warmup_steps=2000,
                total_steps=600000,
            ),
            1e-4,
        )
        self.assertEqual(
            resolve_warmup_cosine_learning_rate(
                step=2000,
                max_learning_rate=max_learning_rate,
                min_learning_rate=min_learning_rate,
                warmup_steps=2000,
                total_steps=600000,
            ),
            max_learning_rate,
        )
        self.assertAlmostEqual(
            resolve_warmup_cosine_learning_rate(
                step=600000,
                max_learning_rate=max_learning_rate,
                min_learning_rate=min_learning_rate,
                warmup_steps=2000,
                total_steps=600000,
            ),
            min_learning_rate,
        )

    def test_parse_args_accepts_resume_checkpoint_path(self) -> None:
        # ---------------------------------------------------------
        # Accept an existing Lightning checkpoint path so interrupted
        # training can resume through trainer.fit ckpt_path.
        # ---------------------------------------------------------
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "last.ckpt"
            checkpoint_path.touch()
            argv = [
                "train.py",
                "--resume-from-checkpoint",
                str(checkpoint_path),
            ]

            with patch("sys.argv", argv):
                args = parse_args()

        self.assertEqual(args.resume_from_checkpoint, str(checkpoint_path))
        self.assertEqual(args.continue_from_model, "")

    def test_parse_args_accepts_continue_model_path(self) -> None:
        # ---------------------------------------------------------
        # Accept an existing model state path so completed training
        # weights can initialize a fresh optimizer run.
        # ---------------------------------------------------------
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "model.pth"
            model_path.touch()
            argv = [
                "train.py",
                "--continue-from-model",
                str(model_path),
            ]

            with patch("sys.argv", argv):
                args = parse_args()

        self.assertEqual(args.continue_from_model, str(model_path))
        self.assertEqual(args.resume_from_checkpoint, "")

    def test_parse_args_rejects_two_resume_sources(self) -> None:
        # ---------------------------------------------------------
        # Keep checkpoint resume and model-weight continuation
        # mutually exclusive so training state is unambiguous.
        # ---------------------------------------------------------
        argv = [
            "train.py",
            "--resume-from-checkpoint",
            "last.ckpt",
            "--continue-from-model",
            "model.pth",
        ]

        with patch("sys.argv", argv), patch("sys.stderr", io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args()

    def test_parse_args_accepts_hub_push_settings(self) -> None:
        # ---------------------------------------------------------
        # Accept explicit Hub publishing settings so completed
        # training artifacts can be uploaded after local saving.
        # ---------------------------------------------------------
        argv = [
            "train.py",
            "--push-to-hub",
            "--hub-repo-id",
            "user/myllm",
            "--hub-private",
            "--hub-commit-message",
            "test upload",
        ]

        with patch("sys.argv", argv):
            args = parse_args()

        self.assertTrue(args.push_to_hub)
        self.assertEqual(args.hub_repo_id, "user/myllm")
        self.assertTrue(args.hub_private)
        self.assertEqual(args.hub_commit_message, "test upload")

    def test_parse_args_rejects_missing_hub_repo_id(self) -> None:
        # ---------------------------------------------------------
        # Reject Hub publishing without a destination repository so
        # training never finishes with an ambiguous upload target.
        # ---------------------------------------------------------
        argv = [
            "train.py",
            "--push-to-hub",
        ]

        with patch("sys.argv", argv), patch("sys.stderr", io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args()


if __name__ == "__main__":
    unittest.main()
