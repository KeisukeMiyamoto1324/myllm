import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from src.pretraining.cli import parse_args
from src.shared.model.position_encoding import RotaryPositionEmbedding
from src.shared.model.self_attention import Attention
from src.shared.model.transformer import DecoderOnlyTransformer
from src.shared.model.transformer import build_packed_attention_mask
from src.shared.model.transformer import resolve_warmup_cosine_learning_rate


class PretrainingTrainTest(unittest.TestCase):
    def test_rotary_position_embedding_keeps_shape_dtype_and_device(self) -> None:
        # ---------------------------------------------------------
        # Rotate query or key heads without changing the tensor
        # contract expected by scaled dot-product attention.
        # ---------------------------------------------------------
        rotary_position_embedding = RotaryPositionEmbedding(head_dim=4)
        x = torch.ones((2, 3, 5, 4), dtype=torch.float32)
        rotated = rotary_position_embedding(x)

        self.assertEqual(rotated.shape, x.shape)
        self.assertEqual(rotated.dtype, x.dtype)
        self.assertEqual(rotated.device, x.device)

    def test_rotary_position_embedding_accepts_packed_position_ids(self) -> None:
        # ---------------------------------------------------------
        # Use explicit per-token positions so packed samples can
        # reset each segment to position zero inside one batch.
        # ---------------------------------------------------------
        rotary_position_embedding = RotaryPositionEmbedding(head_dim=4)
        x = torch.ones((1, 2, 4, 4), dtype=torch.float32)
        position_ids = torch.tensor([[0, 1, 0, 1]], dtype=torch.long)
        rotated = rotary_position_embedding(x, position_ids=position_ids)

        self.assertEqual(rotated.shape, x.shape)
        self.assertFalse(torch.equal(rotated[:, :, 0, :], rotated[:, :, 1, :]))
        self.assertTrue(torch.equal(rotated[:, :, 0, :], rotated[:, :, 2, :]))

    def test_rotary_position_embedding_reuses_precomputed_trig_tables(self) -> None:
        # ---------------------------------------------------------
        # Build cos and sin tables during initialization and reuse
        # them without reallocating buffers inside forward calls.
        # ---------------------------------------------------------
        rotary_position_embedding = RotaryPositionEmbedding(head_dim=4)
        long_x = torch.ones((1, 2, 5, 4), dtype=torch.float32)
        short_x = torch.ones((1, 2, 3, 4), dtype=torch.float32)
        longer_x = torch.ones((1, 2, 6, 4), dtype=torch.float32)

        rotary_position_embedding(long_x)
        first_cos_pointer = rotary_position_embedding.cos_cache.data_ptr()
        rotary_position_embedding(short_x)
        second_cos_pointer = rotary_position_embedding.cos_cache.data_ptr()
        rotary_position_embedding(longer_x)
        third_cos_pointer = rotary_position_embedding.cos_cache.data_ptr()

        self.assertEqual(first_cos_pointer, second_cos_pointer)
        self.assertEqual(second_cos_pointer, third_cos_pointer)
        self.assertNotIn("cos_cache", rotary_position_embedding.state_dict())
        self.assertNotIn("sin_cache", rotary_position_embedding.state_dict())

    def test_attention_cache_appends_rotated_keys(self) -> None:
        # ---------------------------------------------------------
        # Store RoPE-applied keys in the cache and append only the
        # newly generated token states on the next inference step.
        # ---------------------------------------------------------
        attention = Attention(d_model=8, num_heads=2)
        first_input = torch.randn((1, 2, 8), dtype=torch.float32)
        _, first_cache = attention.forward_with_cache(
            first_input,
            first_input,
            first_input,
            past_key_value=None,
        )
        next_input = torch.randn((1, 1, 8), dtype=torch.float32)
        _, next_cache = attention.forward_with_cache(
            next_input,
            next_input,
            next_input,
            past_key_value=first_cache,
            position_offset=2,
        )

        self.assertEqual(first_cache[0].size(dim=2), 2)
        self.assertEqual(next_cache[0].size(dim=2), 3)

    def test_attention_rejects_odd_head_dim_for_rope(self) -> None:
        # ---------------------------------------------------------
        # Require pairwise head features because RoPE rotates even
        # and odd channels together.
        # ---------------------------------------------------------
        with self.assertRaises(ValueError):
            Attention(d_model=6, num_heads=2)

    def test_parse_args_uses_160m_model_defaults(self) -> None:
        # ---------------------------------------------------------
        # Keep the default pretraining run near the 160M class while
        # using the reduced SwiGLU width for parameter parity.
        # ---------------------------------------------------------
        with patch("sys.argv", ["train.py"]):
            args = parse_args()

        self.assertEqual(args.max_len, 1024)
        self.assertEqual(args.d_model, 768)
        self.assertEqual(args.num_layers, 16)
        self.assertEqual(args.num_heads, 12)
        self.assertEqual(args.d_ff, 2048)
        self.assertEqual(args.batch_size, 16)
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

    def test_parse_args_rejects_invalid_runtime_values(self) -> None:
        # ---------------------------------------------------------
        # Reject values that would otherwise fail later in dataset
        # packing, DataLoader setup, or chunked loss computation.
        # ---------------------------------------------------------
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
                flag,
                value,
            ]

            with self.subTest(flag=flag), patch("sys.argv", argv), patch("sys.stderr", io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args()

    def test_parse_args_rejects_invalid_transformer_shape(self) -> None:
        # ---------------------------------------------------------
        # Reject incompatible model dimensions before the attention
        # and RoPE modules are constructed.
        # ---------------------------------------------------------
        invalid_cases = [
            ["--d-model", "10", "--num-heads", "3"],
            ["--d-model", "6", "--num-heads", "2"],
        ]

        for cli_values in invalid_cases:
            argv = [
                "train.py",
                *cli_values,
            ]

            with self.subTest(cli_values=cli_values), patch("sys.argv", argv), patch("sys.stderr", io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args()

    def test_optimizer_excludes_embedding_norm_and_bias_from_weight_decay(self) -> None:
        # ---------------------------------------------------------
        # Keep regular matrix weights decayed while excluding token
        # embeddings, normalization parameters, and bias parameters.
        # ---------------------------------------------------------
        model = DecoderOnlyTransformer(
            num_tokens=12,
            d_model=8,
            num_layers=2,
            num_heads=2,
            d_ff=16,
            pad_token_id=0,
        )
        optimizer = model.configure_optimizers()
        decay_group = next(group for group in optimizer.param_groups if group["weight_decay"] > 0.0)
        no_decay_group = next(group for group in optimizer.param_groups if group["weight_decay"] == 0.0)
        decay_parameter_ids = {id(parameter) for parameter in decay_group["params"]}
        no_decay_parameter_ids = {id(parameter) for parameter in no_decay_group["params"]}

        self.assertIn(id(model.blocks[0].attention.W_q.weight), decay_parameter_ids)
        self.assertIn(id(model.blocks[0].feed_forward.gate_proj.weight), decay_parameter_ids)
        self.assertIn(id(model.we.weight), no_decay_parameter_ids)
        self.assertIn(id(model.final_norm.weight), no_decay_parameter_ids)
        self.assertIn(id(model.blocks[0].norm_1.weight), no_decay_parameter_ids)
        self.assertIn(id(model.blocks[0].feed_forward.gate_proj.bias), no_decay_parameter_ids)

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
