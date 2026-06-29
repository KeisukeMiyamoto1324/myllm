import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from src.pretraining.train import parse_args
from src.shared.model.transformer import build_packed_attention_mask
from src.shared.model.transformer import DecoderOnlyTransformer
from src.shared.model.transformer import normalize_training_batch
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
        self.assertEqual(args.batch_size, 24)
        self.assertEqual(args.devices, "auto")
        self.assertEqual(args.output_path, "models/lambda-160m")

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

    def test_normalize_training_batch_keeps_regular_batch(self) -> None:
        # ---------------------------------------------------------
        # Keep regular fixed-length batches in batch-major layout so
        # posttraining uses the simple causal PyTorch SDPA path.
        # ---------------------------------------------------------
        input_tokens = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
        labels = torch.tensor([[2, 3, 0], [5, 6, 0]], dtype=torch.long)
        normalized = normalize_training_batch(batch=(input_tokens, labels))

        self.assertEqual(normalized[0].tolist(), [[1, 2, 3], [4, 5, 6]])
        self.assertEqual(normalized[1].tolist(), [[2, 3, 0], [5, 6, 0]])
        self.assertIsNone(normalized[2])
        self.assertIsNone(normalized[3])

    def test_transformer_computes_loss_for_packed_batch(self) -> None:
        # ---------------------------------------------------------
        # Run one packed batch through the PyTorch SDPA path using
        # explicit position ids and segment ids.
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

    def test_packed_sdpa_keeps_documents_isolated(self) -> None:
        # ---------------------------------------------------------
        # Match a ragged packed batch against the same tokens
        # represented as padded independent rows.
        # ---------------------------------------------------------
        torch.manual_seed(3)
        model = DecoderOnlyTransformer(
            num_tokens=16,
            d_model=8,
            max_len=4,
            num_layers=1,
            num_heads=2,
            d_ff=16,
            pad_token_id=0,
        )
        packed_tokens = torch.tensor([[1, 3, 4, 2, 0, 0]], dtype=torch.long)
        row_tokens = torch.tensor([[1, 3, 4], [2, 0, 0]], dtype=torch.long)
        position_ids = torch.tensor([[0, 1, 2, 0, 0, 0]], dtype=torch.long)
        segment_ids = torch.tensor([[0, 0, 0, 1, -1, -1]], dtype=torch.long)

        with torch.no_grad():
            packed_hidden = model.forward_hidden(
                token_ids=packed_tokens,
                position_ids=position_ids,
                attention_mask=None,
            )
            masked_hidden = model.forward_hidden(
                token_ids=packed_tokens,
                position_ids=position_ids,
                attention_mask=build_packed_attention_mask(segment_ids),
            )
            row_hidden = model.forward_hidden(token_ids=row_tokens)

        self.assertFalse(torch.allclose(packed_hidden[:, 3], masked_hidden[:, 3]))
        torch.testing.assert_close(masked_hidden[:, :3], row_hidden[:1], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(masked_hidden[:, 3], row_hidden[1, 0].unsqueeze(0), atol=1e-6, rtol=1e-6)

    def test_explicit_position_ids_match_absolute_positions(self) -> None:
        # ---------------------------------------------------------
        # Keep packed position ids aligned with the absolute
        # sinusoidal position encoding.
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
        token_ids = torch.tensor([[1, 3, 4], [1, 3, 4]], dtype=torch.long)
        position_ids = torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long)

        with torch.no_grad():
            default_hidden = model.forward_hidden(token_ids=token_ids)
            explicit_hidden = model.forward_hidden(
                token_ids=token_ids,
                position_ids=position_ids,
            )

        torch.testing.assert_close(explicit_hidden, default_hidden)

    def test_forward_with_cache_uses_head_major_cache(self) -> None:
        # ---------------------------------------------------------
        # Store cached keys and values in the same head-major layout
        # used by PyTorch scaled dot-product attention.
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
        token_ids = torch.tensor([[1, 3], [4, 2]], dtype=torch.long)

        logits, past_key_values = model.forward_with_cache(
            token_ids=token_ids,
            past_key_values=None,
        )

        self.assertEqual(logits.shape, (2, 2, 16))
        self.assertEqual(past_key_values[0][0].shape, (2, 2, 2, 4))
        self.assertEqual(past_key_values[0][1].shape, (2, 2, 2, 4))

    def test_forward_with_cache_matches_full_forward(self) -> None:
        # ---------------------------------------------------------
        # Verify one-token cached inference uses the same absolute
        # positions as the full causal forward pass.
        # ---------------------------------------------------------
        torch.manual_seed(7)
        model = DecoderOnlyTransformer(
            num_tokens=16,
            d_model=8,
            max_len=4,
            num_layers=1,
            num_heads=2,
            d_ff=16,
            pad_token_id=0,
        )
        model.eval()
        token_ids = torch.tensor([[1, 3, 4, 2]], dtype=torch.long)

        with torch.no_grad():
            full_logits = model(token_ids)
            past_key_values = None
            cached_logits = []

            for token_index in range(token_ids.size(dim=1)):
                logits, past_key_values = model.forward_with_cache(
                    token_ids=token_ids[:, token_index : token_index + 1],
                    past_key_values=past_key_values,
                )
                cached_logits.append(logits)

        stacked_cached_logits = torch.cat(cached_logits, dim=1)
        torch.testing.assert_close(stacked_cached_logits, full_logits, atol=1e-5, rtol=1e-5)

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
