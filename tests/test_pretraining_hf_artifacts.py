import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

import torch

from src.shared.pytorch_artifacts import build_model_from_config
from src.shared.pytorch_artifacts import load_pytorch_model
from src.shared.pytorch_artifacts import push_pytorch_model_artifacts
from src.shared.model.transformer import DecoderOnlyTransformer


def build_model_config() -> dict[str, int | float]:
    # ---------------------------------------------------------
    # Build a compact model config shared by PyTorch artifact tests
    # so direct model loading stays fast.
    # ---------------------------------------------------------
    return {
        "max_len": 16,
        "d_model": 8,
        "num_layers": 2,
        "num_heads": 2,
        "d_ff": 16,
        "learning_rate": 0.1,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
    }


class PretrainingPytorchArtifactsTest(unittest.TestCase):
    def test_build_model_from_config_recreates_transformer(self) -> None:
        # ---------------------------------------------------------
        # Rebuild the native PyTorch architecture directly from the
        # saved config values without any wrapper class.
        # ---------------------------------------------------------
        model = build_model_from_config(
            model_config=build_model_config(),
            vocab_size=12,
        )

        self.assertEqual(model.we.num_embeddings, 12)
        self.assertEqual(model.we.embedding_dim, 8)
        self.assertEqual(len(model.blocks), 2)

    def test_load_pytorch_model_restores_state_dict(self) -> None:
        # ---------------------------------------------------------
        # Save model.pth and model_config.json, then verify direct
        # PyTorch loading restores the same weights.
        # ---------------------------------------------------------
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir)
            model = DecoderOnlyTransformer(
                num_tokens=12,
                d_model=8,
                max_len=16,
                num_layers=2,
                num_heads=2,
                d_ff=16,
                pad_token_id=0,
            )
            torch.save(model.state_dict(), output_path / "model.pth")
            (output_path / "model_config.json").write_text(
                json.dumps(build_model_config()),
                encoding="utf-8",
            )

            loaded_model, loaded_config = load_pytorch_model(
                model_dir=output_path,
                vocab_size=12,
            )

        self.assertEqual(loaded_config["max_len"], 16)
        self.assertTrue(torch.equal(model.we.weight, loaded_model.we.weight))

    def test_load_pytorch_model_rebuilds_position_buffer_for_new_max_len(self) -> None:
        # ---------------------------------------------------------
        # Keep learned weights while replacing only the deterministic
        # sinusoidal position buffer for a new context length.
        # ---------------------------------------------------------
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir)
            model = DecoderOnlyTransformer(
                num_tokens=12,
                d_model=8,
                max_len=16,
                num_layers=2,
                num_heads=2,
                d_ff=16,
                pad_token_id=0,
            )
            torch.save(model.state_dict(), output_path / "model.pth")
            (output_path / "model_config.json").write_text(
                json.dumps(build_model_config()),
                encoding="utf-8",
            )

            loaded_model, _ = load_pytorch_model(
                model_dir=output_path,
                vocab_size=12,
                max_len=32,
            )

        self.assertEqual(loaded_model.pe.pe.size(dim=0), 32)
        self.assertTrue(torch.equal(model.we.weight, loaded_model.we.weight))

    def test_push_uses_hub_model_repo_and_only_allows_artifacts(self) -> None:
        # ---------------------------------------------------------
        # Mock the Hub client so publishing behavior is verified
        # without requiring credentials or network access.
        # ---------------------------------------------------------
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir)
            api = MagicMock()

            with patch("src.shared.pytorch_artifacts.HfApi", return_value=api):
                push_pytorch_model_artifacts(
                    output_path=output_path,
                    repo_id="user/myllm",
                    private=True,
                    commit_message="upload",
                )

        api.create_repo.assert_called_once_with(
            repo_id="user/myllm",
            private=True,
            repo_type="model",
            exist_ok=True,
        )
        api.upload_folder.assert_called_once_with(
            repo_id="user/myllm",
            repo_type="model",
            folder_path=output_path,
            commit_message="upload",
            allow_patterns=[
                "model.pth",
                "model_config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "added_tokens.json",
            ],
            ignore_patterns=[
                "*.py",
                "checkpoints/*",
                "metrics/*",
                "validation-cache-*",
            ],
        )


if __name__ == "__main__":
    unittest.main()
