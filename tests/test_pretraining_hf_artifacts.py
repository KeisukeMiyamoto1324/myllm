from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

import torch
from transformers import AutoConfig
from transformers import AutoModel
from transformers import AutoModelForCausalLM

from src.pretraining.configuration_myllm import MyLLMConfig
from src.pretraining.hf_artifacts import copy_inference_code
from src.pretraining.hf_artifacts import push_hf_pretrained_artifacts
from src.pretraining.hf_artifacts import save_hf_pretrained_artifacts
from src.pretraining.modeling_myllm import MyLLMForCausalLM
from src.pretraining.transformer import DecoderOnlyTransformer


def build_model_config() -> dict[str, int | float]:
    # ---------------------------------------------------------
    # Build a compact model config shared by the HF artifact tests
    # so local AutoModel loading stays fast.
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


class PretrainingHfArtifactsTest(unittest.TestCase):
    def test_config_round_trips_with_auto_config(self) -> None:
        # ---------------------------------------------------------
        # Save the custom config with AutoClass metadata and load it
        # through the same trust_remote_code path used from Hub.
        # ---------------------------------------------------------
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir)
            MyLLMConfig.register_for_auto_class()
            config = MyLLMConfig(
                vocab_size=12,
                max_len=16,
                d_model=8,
                num_layers=2,
                num_heads=2,
                d_ff=16,
                pad_token_id=0,
                bos_token_id=1,
                eos_token_id=2,
            )
            config.save_pretrained(output_path)
            copy_inference_code(output_path=output_path)

            loaded_config = AutoConfig.from_pretrained(
                output_path,
                trust_remote_code=True,
            )

        self.assertEqual(loaded_config.model_type, "myllm")
        self.assertEqual(loaded_config.vocab_size, 12)
        self.assertEqual(loaded_config.num_hidden_layers, 2)

    def test_model_forward_returns_causal_lm_output(self) -> None:
        # ---------------------------------------------------------
        # Verify the HF wrapper delegates to the PyTorch model while
        # returning the standard causal LM output fields.
        # ---------------------------------------------------------
        config = MyLLMConfig(
            vocab_size=12,
            max_len=16,
            d_model=8,
            num_layers=2,
            num_heads=2,
            d_ff=16,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
        )
        model = MyLLMForCausalLM(config=config)
        input_ids = torch.tensor([[1, 3, 4]], dtype=torch.long)
        output = model(input_ids=input_ids, labels=input_ids)

        self.assertEqual(tuple(output.logits.shape), (1, 3, 12))
        self.assertIsNotNone(output.loss)

    def test_saved_model_loads_with_auto_model_and_generates(self) -> None:
        # ---------------------------------------------------------
        # Save trained-style artifacts and load them through the
        # public AutoModelForCausalLM trust_remote_code interface.
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
            save_hf_pretrained_artifacts(
                model=model,
                model_config=build_model_config(),
                vocab_size=12,
                output_path=output_path,
            )
            copy_inference_code(output_path=output_path)

            loaded_model = AutoModelForCausalLM.from_pretrained(
                output_path,
                trust_remote_code=True,
            )
            generic_model = AutoModel.from_pretrained(
                output_path,
                trust_remote_code=True,
            )
            input_ids = torch.tensor([[1, 3, 4]], dtype=torch.long)
            output_ids = loaded_model.generate(
                input_ids=input_ids,
                max_new_tokens=2,
                do_sample=False,
                pad_token_id=0,
                eos_token_id=2,
            )

        self.assertEqual(tuple(output_ids.shape), (1, 5))
        self.assertEqual(generic_model.__class__.__name__, "MyLLMForCausalLM")

    def test_push_uses_hub_model_repo_and_ignores_training_outputs(self) -> None:
        # ---------------------------------------------------------
        # Mock the Hub client so publishing behavior is verified
        # without requiring credentials or network access.
        # ---------------------------------------------------------
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir)
            api = MagicMock()

            with patch("src.pretraining.hf_artifacts.HfApi", return_value=api):
                push_hf_pretrained_artifacts(
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
            ignore_patterns=[
                "checkpoints/*",
                "metrics/*",
                "validation-cache-*",
            ],
        )


if __name__ == "__main__":
    unittest.main()
