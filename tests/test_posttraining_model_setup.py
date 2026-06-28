import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import Dataset

from src.posttraining.artifacts import save_chat_model
from src.posttraining.dataloaders import build_dataloaders
from src.posttraining.dataset import IchikaraInstructionDataset
from src.posttraining.model_setup import DEFAULT_BASE_MODEL_ID
from src.posttraining.model_setup import download_base_model
from src.posttraining.model_setup import load_base_model
from src.posttraining.train import parse_args
from src.posttraining.trainer import build_trainer
from src.shared.model.transformer import DecoderOnlyTransformer


class FakeTokenizer:
    pad_token = "|<pad>|"
    bos_token = "|<bos>|"
    eos_token = "|<eos>|"
    end_of_turn_token = "|<end_of_turn>|"

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
            self.end_of_turn_token: 3,
        }
        return token_ids[token]


class FakeDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, size: int) -> None:
        # ---------------------------------------------------------
        # Store a fixed dataset size so dataloader step math can be
        # tested without loading the remote SFT dataset.
        # ---------------------------------------------------------
        self.size = size

    def __len__(self) -> int:
        # ---------------------------------------------------------
        # Return the configured number of examples.
        # ---------------------------------------------------------
        return self.size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        # ---------------------------------------------------------
        # Return one tiny tensor pair for DataLoader construction.
        # ---------------------------------------------------------
        del index
        tensor = torch.tensor([1], dtype=torch.long)
        return tensor, tensor


class PosttrainingModelSetupTest(unittest.TestCase):
    def test_parse_args_uses_lambda_hub_model_default(self) -> None:
        # ---------------------------------------------------------
        # Keep posttraining pointed at the published lambda base Hub
        # model unless the user overrides it.
        # ---------------------------------------------------------
        with patch("sys.argv", ["train.py"]):
            args = parse_args()

        self.assertEqual(args.base_model_id, DEFAULT_BASE_MODEL_ID)

    def test_parse_args_uses_three_ichikara_repeat_epochs(self) -> None:
        # ---------------------------------------------------------
        # Use three passes over Ichikara as the default posttraining
        # budget instead of a fixed two-stage step budget.
        # ---------------------------------------------------------
        with patch("sys.argv", ["train.py"]):
            args = parse_args()

        self.assertEqual(args.repeat_epochs, 3)
        self.assertEqual(args.devices, "auto")

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
                repeat_epochs=3,
                posttraining_steps=9,
                devices="auto",
                device_count=1,
                global_batch_size=16,
                global_effective_batch_size=16,
                batch_size=16,
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
        self.assertEqual(payload["posttraining_datasets"], ["msfm/ichikara-instruction-all:train"])
        self.assertEqual(payload["validation_dataset"], "msfm/ichikara-instruction-all:test")
        self.assertEqual(payload["repeat_epochs"], 3)
        self.assertEqual(payload["posttraining_steps"], 9)
        self.assertEqual(payload["devices"], "auto")
        self.assertEqual(payload["device_count"], 1)
        self.assertEqual(payload["global_batch_size"], 16)
        self.assertTrue(model_path_exists)

    def test_ichikara_dataset_maps_text_and_output_to_chat_messages(self) -> None:
        # ---------------------------------------------------------
        # Convert Ichikara text/output columns into the user and
        # assistant roles expected by the shared chat template.
        # ---------------------------------------------------------
        sample = {"text": "質問", "output": "回答", "ID": "id-1", "category": 1}

        with patch("src.posttraining.dataset.load_dataset", return_value=[sample]) as mocked_load:
            with patch(
                "src.posttraining.dataset.build_tensor_example",
                return_value=(torch.tensor([1]), torch.tensor([2])),
            ) as mocked_build:
                dataset = IchikaraInstructionDataset(
                    tokenizer=FakeTokenizer(),
                    split="train",
                    max_len=8,
                    pad_token_id=0,
                    bos_token_id=1,
                    eos_token_id=2,
                    end_of_turn_token_id=3,
                )

        messages = mocked_build.call_args.kwargs["messages"]
        mocked_load.assert_called_once_with(path="msfm/ichikara-instruction-all", split="train")
        self.assertEqual(len(dataset), 1)
        self.assertEqual(messages[0].role, "user")
        self.assertEqual(messages[0].content, "質問")
        self.assertEqual(messages[1].role, "assistant")
        self.assertEqual(messages[1].content, "回答")

    def test_build_dataloaders_computes_three_epoch_steps(self) -> None:
        # ---------------------------------------------------------
        # Derive Lightning max_steps from dataloader length times
        # repeat epochs so batch size changes keep epoch semantics.
        # ---------------------------------------------------------
        fake_datasets = [FakeDataset(size=5), FakeDataset(size=2)]

        with patch("src.posttraining.dataloaders.IchikaraInstructionDataset", side_effect=fake_datasets):
            _, _, max_steps = build_dataloaders(
                tokenizer=FakeTokenizer(),
                max_len=8,
                batch_size=2,
                num_workers=0,
                accelerator="cpu",
                repeat_epochs=3,
            )

        self.assertEqual(max_steps, 9)

    def test_build_dataloaders_computes_multi_gpu_epoch_steps(self) -> None:
        # ---------------------------------------------------------
        # Keep repeat_epochs tied to full dataset passes when each
        # optimizer step consumes batches from multiple GPUs.
        # ---------------------------------------------------------
        fake_datasets = [FakeDataset(size=5), FakeDataset(size=2)]

        with patch("src.posttraining.dataloaders.IchikaraInstructionDataset", side_effect=fake_datasets):
            _, _, max_steps = build_dataloaders(
                tokenizer=FakeTokenizer(),
                max_len=8,
                batch_size=2,
                num_workers=0,
                accelerator="cpu",
                repeat_epochs=3,
                device_count=2,
            )

        self.assertEqual(max_steps, 6)

    def test_build_trainer_validates_by_global_step(self) -> None:
        # ---------------------------------------------------------
        # Allow validation intervals larger than one epoch by using
        # Lightning's global-step validation mode.
        # ---------------------------------------------------------
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = build_trainer(
                model_dir=Path(temp_dir),
                stage_name="ichikara",
                max_steps=723,
                accelerator="cpu",
                precision="32-true",
                val_check_interval=500,
                val_batches=8,
                checkpoint_every_n_steps=1000,
                metric_log_every_n_steps=50,
            )

        self.assertIsNone(trainer.check_val_every_n_epoch)
        self.assertEqual(trainer.val_check_interval, 500)


if __name__ == "__main__":
    unittest.main()
