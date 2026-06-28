import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from src.pretraining.train import parse_args
from src.shared.model.transformer import DecoderOnlyTransformer
from src.shared.model.transformer import build_packed_attention_mask
from src.shared.model.transformer import resolve_warmup_cosine_learning_rate


class PretrainingTrainTest(unittest.TestCase):
    def test_parse_args_uses_160m_model_defaults(self) -> None:
        # ---------------------------------------------------------
        # Keep the default pretraining run near the 160M class while
        # following the OpenCALM small width and FFN structure.
        # ---------------------------------------------------------
        with patch("sys.argv", ["train.py"]):
            args = parse_args()

        self.assertEqual(args.max_len, 1024)
        self.assertEqual(args.d_model, 768)
        self.assertEqual(args.num_layers, 16)
        self.assertEqual(args.num_heads, 12)
        self.assertEqual(args.d_ff, 3072)
        self.assertEqual(args.batch_size, 96)
        self.assertEqual(args.devices, "auto")
        self.assertEqual(args.output_path, "models/lambda-160m")

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

    def test_build_packed_attention_mask_blocks_other_segments(self) -> None:
        # ---------------------------------------------------------
        # Keep packed attention causal inside each segment and block
        # tokens from other packed documents.
        # ---------------------------------------------------------
        segment_ids = torch.tensor([[0, 0, 1, -1]], dtype=torch.long)
        attention_mask = build_packed_attention_mask(segment_ids=segment_ids)

        self.assertEqual(attention_mask.shape, (1, 1, 4, 4))
        self.assertEqual(
            attention_mask[0, 0].tolist(),
            [
                [True, False, False, False],
                [True, True, False, False],
                [False, False, True, False],
                [True, True, True, False],
            ],
        )

    def test_transformer_computes_loss_for_packed_batch(self) -> None:
        # ---------------------------------------------------------
        # Run one packed batch through the model using explicit
        # position ids and segment ids.
        # ---------------------------------------------------------
        model = DecoderOnlyTransformer(
            num_tokens=16,
            d_model=8,
            max_len=4,
            num_layers=1,
            num_heads=2,
            d_ff=16,
            pad_token_id=0,
        )
        input_tokens = torch.tensor([[1, 3, 1, 4]], dtype=torch.long)
        labels = torch.tensor([[3, 2, 4, 2]], dtype=torch.long)
        position_ids = torch.tensor([[0, 1, 0, 1]], dtype=torch.long)
        segment_ids = torch.tensor([[0, 0, 1, 1]], dtype=torch.long)
        loss = model.compute_chunked_loss(
            input_tokens=input_tokens,
            labels=labels,
            position_ids=position_ids,
            segment_ids=segment_ids,
        )

        self.assertTrue(torch.isfinite(loss))

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

    def test_parse_args_accepts_hub_push_with_env_repo(self) -> None:
        # ---------------------------------------------------------
        # Accept Hub publishing only when the destination repository
        # is provided through the environment.
        # ---------------------------------------------------------
        argv = [
            "train.py",
            "--push-to-hub",
        ]

        with patch("sys.argv", argv), patch.dict("os.environ", {"HF_REPO": "user/myllm"}):
            args = parse_args()

        self.assertTrue(args.push_to_hub)

    def test_parse_args_rejects_missing_hf_repo(self) -> None:
        # ---------------------------------------------------------
        # Reject Hub publishing without HF_REPO so training never
        # finishes with an ambiguous upload target.
        # ---------------------------------------------------------
        argv = [
            "train.py",
            "--push-to-hub",
        ]

        with patch("sys.argv", argv), patch.dict("os.environ", {}, clear=True), patch("sys.stderr", io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args()


if __name__ == "__main__":
    unittest.main()
