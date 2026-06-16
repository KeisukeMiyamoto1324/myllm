import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

from src.inference_base.cli import parse_args
from src.inference_base.generation import generate_continuation_text
from src.inference_base.generation import generate_token_ids
from src.inference_base.generation import resolve_torch_dtype
from src.pretraining.kv_cache import KeyValueCache


class FakeTokenizer:
    bos_token = "|<bos>|"
    eos_token = "|<eos>|"

    def __init__(self) -> None:
        # ---------------------------------------------------------
        # Store generated decode input for assertions after the
        # helper returns continuation text.
        # ---------------------------------------------------------
        self.decoded_ids: list[int] = []

    def token_to_id(self, token: str) -> int:
        # ---------------------------------------------------------
        # Return stable ids for generation special tokens.
        # ---------------------------------------------------------
        token_ids = {
            self.bos_token: 1,
            self.eos_token: 2,
        }
        return token_ids[token]

    def tokenize(self, sentence: str) -> list[int]:
        # ---------------------------------------------------------
        # Return a fixed prompt so tests can verify continuation
        # decoding receives only newly generated ids.
        # ---------------------------------------------------------
        self.prompt = sentence
        return [10, 11, 12]

    def detokenize(self, token_ids: list[int]) -> str:
        # ---------------------------------------------------------
        # Decode receives only the generated suffix from the helper.
        # ---------------------------------------------------------
        self.decoded_ids = token_ids
        return " continuation text "


class FakeModel(nn.Module):
    def __init__(self, next_token_ids: list[int]) -> None:
        super().__init__()

        # ---------------------------------------------------------
        # Keep one parameter for device resolution and return fixed
        # next-token logits for each generation step.
        # ---------------------------------------------------------
        self.probe = nn.Parameter(torch.zeros(1))
        self.next_token_ids = next_token_ids
        self.calls: list[list[int]] = []

    def forward_with_cache(
        self,
        token_ids: torch.Tensor,
        past_key_values: KeyValueCache | None,
    ) -> tuple[torch.Tensor, KeyValueCache]:
        # ---------------------------------------------------------
        # Emit logits that greedily select the configured token id
        # while recording each input passed by generation.
        # ---------------------------------------------------------
        del past_key_values
        self.calls.append([int(token_id) for token_id in token_ids[0].tolist()])
        token_id = self.next_token_ids[len(self.calls) - 1]
        logits = torch.zeros((1, token_ids.size(dim=1), 16), dtype=torch.float32)
        logits[0, -1, token_id] = 100.0
        return logits, []


class InferenceItTest(unittest.TestCase):
    def test_generate_continuation_text_uses_pytorch_cache_generation(self) -> None:
        # ---------------------------------------------------------
        # Verify direct prompt continuation uses the PyTorch cache
        # path and decodes only newly generated tokens.
        # ---------------------------------------------------------
        model = FakeModel(next_token_ids=[13, 14])
        tokenizer = FakeTokenizer()
        text = generate_continuation_text(
            model=model,
            tokenizer=tokenizer,
            prompt="prompt",
            max_new_tokens=2,
            do_sample=False,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            repetition_penalty=1.05,
            no_repeat_ngram_size=3,
        )

        self.assertEqual(text, "continuation text")
        self.assertEqual(tokenizer.prompt, "prompt")
        self.assertEqual(tokenizer.decoded_ids, [13, 14])
        self.assertEqual(model.calls, [[1, 10, 11, 12], [13]])

    def test_generate_token_ids_stops_on_eos(self) -> None:
        # ---------------------------------------------------------
        # Stop generation as soon as EOS is selected so the caller
        # does not produce extra tokens.
        # ---------------------------------------------------------
        model = FakeModel(next_token_ids=[13, 2, 14])
        input_ids = torch.tensor([[1, 10]], dtype=torch.long)
        generated_ids = generate_token_ids(
            model=model,
            input_ids=input_ids,
            max_new_tokens=8,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            repetition_penalty=1.0,
            no_repeat_ngram_size=0,
            eos_token_id=2,
        )

        self.assertEqual(generated_ids, [13, 2])

    def test_parse_args_uses_pytorch_generation_defaults(self) -> None:
        # ---------------------------------------------------------
        # Keep instruction inference defaults aligned with the
        # native PyTorch generation path.
        # ---------------------------------------------------------
        with tempfile.TemporaryDirectory() as temp_dir:
            argv = [
                "inference.py",
                "--model-dir",
                "user/myllm",
                "--prompt",
                "hello",
                "--no-do-sample",
                "--top-p",
                "0.8",
                "--top-k",
                "32",
                "--repetition-penalty",
                "1.2",
                "--no-repeat-ngram-size",
                "4",
                "--torch-dtype",
                "float16",
            ]

            with patch("sys.argv", argv):
                args = parse_args(default_model_dir=Path(temp_dir))

        self.assertEqual(args.model_dir, "user/myllm")
        self.assertEqual(args.prompt, "hello")
        self.assertFalse(args.do_sample)
        self.assertEqual(args.top_p, 0.8)
        self.assertEqual(args.top_k, 32)
        self.assertEqual(args.repetition_penalty, 1.2)
        self.assertEqual(args.no_repeat_ngram_size, 4)
        self.assertEqual(args.torch_dtype, "float16")

    def test_parse_args_rejects_invalid_top_p(self) -> None:
        # ---------------------------------------------------------
        # Reject invalid nucleus sampling probabilities before
        # model loading begins.
        # ---------------------------------------------------------
        argv = [
            "inference.py",
            "--top-p",
            "1.5",
        ]

        with patch("sys.argv", argv), patch("sys.stderr", io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args(default_model_dir=Path("models/model"))

    def test_resolve_torch_dtype_maps_cli_values(self) -> None:
        # ---------------------------------------------------------
        # Convert dtype CLI names into values used by PyTorch model
        # casting.
        # ---------------------------------------------------------
        self.assertIsNone(resolve_torch_dtype("auto"))
        self.assertEqual(resolve_torch_dtype("float16"), torch.float16)
        self.assertEqual(resolve_torch_dtype("bfloat16"), torch.bfloat16)
        self.assertEqual(resolve_torch_dtype("float32"), torch.float32)


if __name__ == "__main__":
    unittest.main()
