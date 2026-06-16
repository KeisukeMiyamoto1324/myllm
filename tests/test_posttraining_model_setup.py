import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from src.posttraining.artifacts import save_chat_model
from src.posttraining.model_setup import DEFAULT_BASE_MODEL_ID
from src.posttraining.model_setup import download_base_model
from src.posttraining.model_setup import load_base_model
from src.posttraining.train import parse_args
from src.pretraining.transformer import DecoderOnlyTransformer


class FakeTokenizer:
    pad_token = "|<pad>|"
    bos_token = "|<bos>|"
    eos_token = "|<eos>|"

    def get_vocab_size(self) -> int:
        # ---------------------------------------------------------
        # Match the saved test model vocabulary size.
        # ---------------------------------------------------------
        return 12

    def token_to_id(self, token: str) -> int:
        # ---------------------------------------------------------
        # Return stable ids for the special tokens needed by model
        # setup without loading a real tokenizer file.
        # ---------------------------------------------------------
        token_ids = {
            self.pad_token: 0,
            self.bos_token: 1,
            self.eos_token: 2,
        }
        return token_ids[token]

class PosttrainingModelSetupTest(unittest.TestCase):
    def test_parse_args_uses_lambda_hub_model_default(self) -> None:
        # ---------------------------------------------------------
        # Keep posttraining pointed at the published lambda-160m Hub
        # model unless the user overrides it.
        # ---------------------------------------------------------
        with patch("sys.argv", ["train.py"]):
            args = parse_args()

        self.assertEqual(args.base_model_id, DEFAULT_BASE_MODEL_ID)

    def test_download_base_model_uses_hub_snapshot(self) -> None:
        # ---------------------------------------------------------
        # Resolve Hub model ids through snapshot_download so training
        # always works from local artifacts after download.
        # ---------------------------------------------------------
        with patch("src.posttraining.model_setup.snapshot_download", return_value="/tmp/model") as mocked_download:
            model_dir = download_base_model(base_model_id="user/model")

        self.assertEqual(model_dir, Path("/tmp/model"))
        mocked_download.assert_called_once_with(repo_id="user/model", repo_type="model")

    def test_optimizer_uses_all_trainable_parameters(self) -> None:
        # ---------------------------------------------------------
        # Keep every parameter trainable so full-model fine tuning
        # sends the whole model into the optimizer.
        # ---------------------------------------------------------
        model = DecoderOnlyTransformer(
            num_tokens=12,
            d_model=8,
            max_len=16,
            num_layers=4,
            num_heads=2,
            d_ff=16,
            pad_token_id=0,
        )
        optimizer = model.configure_optimizers()

        model_parameter_ids = {id(parameter) for parameter in model.parameters()}
        optimizer_parameter_ids = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }

        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))
        self.assertEqual(optimizer_parameter_ids, model_parameter_ids)

    def test_load_base_model_loads_pytorch_weights_and_trains_all_layers(self) -> None:
        # ---------------------------------------------------------
        # Load PyTorch weights into the local Lightning model and
        # return metadata for the downloaded architecture.
        # ---------------------------------------------------------
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            model = DecoderOnlyTransformer(
                num_tokens=12,
                d_model=8,
                max_len=16,
                num_layers=4,
                num_heads=2,
                d_ff=16,
                pad_token_id=0,
            )
            torch.save(model.state_dict(), model_dir / "model.pth")
            (model_dir / "model_config.json").write_text(
                json.dumps(
                    {
                        "max_len": 16,
                        "d_model": 8,
                        "num_layers": 4,
                        "num_heads": 2,
                        "d_ff": 16,
                        "learning_rate": 5e-5,
                        "pad_token_id": 0,
                        "bos_token_id": 1,
                        "eos_token_id": 2,
                    }
                ),
                encoding="utf-8",
            )

            with patch("src.posttraining.model_setup.resolve_device", return_value=torch.device("cpu")):
                loaded_model, model_config = load_base_model(
                    base_model_dir=model_dir,
                    tokenizer=FakeTokenizer(),
                    learning_rate=5e-5,
                    max_len=8,
                    accelerator="cpu",
                )

        self.assertEqual(model_config["max_len"], 16)
        self.assertEqual(model_config["num_layers"], 4)
        self.assertTrue(all(parameter.requires_grad for parameter in loaded_model.parameters()))

    def test_save_chat_model_persists_pytorch_metadata(self) -> None:
        # ---------------------------------------------------------
        # Save PyTorch metadata with trainable layer provenance
        # included without extra converted artifacts.
        # ---------------------------------------------------------
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            model = DecoderOnlyTransformer(
                num_tokens=12,
                d_model=8,
                max_len=16,
                num_layers=4,
                num_heads=2,
                d_ff=16,
                pad_token_id=0,
            )
            args = argparse.Namespace(
                max_len=8,
                learning_rate=5e-5,
                base_model_id=DEFAULT_BASE_MODEL_ID,
                magpie_steps=1,
                everyday_steps=1,
            )
            model_config = {
                "max_len": 16,
                "d_model": 8,
                "num_layers": 4,
                "num_heads": 2,
                "d_ff": 16,
                "learning_rate": 5e-5,
                "pad_token_id": 0,
                "bos_token_id": 1,
                "eos_token_id": 2,
            }

            save_chat_model(
                model=model,
                model_dir=model_dir,
                model_config=model_config,
                args=args,
                pad_token_id=0,
                bos_token_id=1,
                eos_token_id=2,
                end_of_turn_token_id=11,
            )

            payload = json.loads((model_dir / "model_config.json").read_text())
            model_path_exists = (model_dir / "model.pth").is_file()

        self.assertEqual(payload["base_model_id"], DEFAULT_BASE_MODEL_ID)
        self.assertEqual(payload["training_max_len"], 8)
        self.assertEqual(payload["trainable_layers"], "all")
        self.assertTrue(model_path_exists)


if __name__ == "__main__":
    unittest.main()
