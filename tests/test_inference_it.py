import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from src.inference_base.cli import parse_args
from src.inference_base.generation import generate_continuation_text
from src.inference_base.generation import resolve_torch_dtype


class FakeBatch(dict[str, torch.Tensor]):
    def to(self, device: torch.device) -> "FakeBatch":
        # ---------------------------------------------------------
        # Match the Transformers BatchEncoding interface needed by
        # inference while keeping tests independent of tokenizers.
        # ---------------------------------------------------------
        del device
        return self


class FakeTokenizer:
    eos_token_id = 2
    pad_token_id = 0

    def __init__(self) -> None:
        # ---------------------------------------------------------
        # Store generated decode input for assertions after the
        # helper returns continuation text.
        # ---------------------------------------------------------
        self.decoded_ids: list[int] = []

    def __call__(self, prompt: str, return_tensors: str) -> FakeBatch:
        # ---------------------------------------------------------
        # Return a fixed prompt length so tests can verify only new
        # generated ids are decoded.
        # ---------------------------------------------------------
        self.prompt = prompt
        self.return_tensors = return_tensors
        return FakeBatch({"input_ids": torch.tensor([[10, 11, 12]], dtype=torch.long)})

    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool) -> str:
        # ---------------------------------------------------------
        # Decode receives only the generated suffix after prompt ids
        # are removed by the helper.
        # ---------------------------------------------------------
        self.decoded_ids = [int(token_id) for token_id in token_ids.tolist()]
        self.skip_special_tokens = skip_special_tokens
        return " continuation text "


class FakeModel:
    def __init__(self) -> None:
        # ---------------------------------------------------------
        # Expose a device attribute and capture generation kwargs in
        # the same shape as a Transformers model.
        # ---------------------------------------------------------
        self.device = torch.device("cpu")
        self.generate_kwargs: dict[str, object] = {}

    def generate(self, **kwargs: object) -> torch.Tensor:
        # ---------------------------------------------------------
        # Return prompt ids plus two generated ids so helper slicing
        # can be tested without loading a real model.
        # ---------------------------------------------------------
        self.generate_kwargs = kwargs
        return torch.tensor([[10, 11, 12, 13, 14]], dtype=torch.long)


class InferenceItTest(unittest.TestCase):
    def test_generate_continuation_text_uses_hf_generate(self) -> None:
        # ---------------------------------------------------------
        # Verify inference delegates direct prompt continuation to
        # the model and decodes only newly generated tokens.
        # ---------------------------------------------------------
        model = FakeModel()
        tokenizer = FakeTokenizer()
        text = generate_continuation_text(
            model=model,
            tokenizer=tokenizer,
            prompt="prompt",
            max_new_tokens=8,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            repetition_penalty=1.05,
            no_repeat_ngram_size=3,
        )

        self.assertEqual(text, "continuation text")
        self.assertEqual(tokenizer.prompt, "prompt")
        self.assertEqual(tokenizer.decoded_ids, [13, 14])
        self.assertEqual(model.generate_kwargs["max_new_tokens"], 8)
        self.assertEqual(model.generate_kwargs["do_sample"], True)
        self.assertEqual(model.generate_kwargs["temperature"], 0.7)
        self.assertEqual(model.generate_kwargs["top_p"], 0.9)
        self.assertEqual(model.generate_kwargs["top_k"], 40)
        self.assertEqual(model.generate_kwargs["repetition_penalty"], 1.05)
        self.assertEqual(model.generate_kwargs["no_repeat_ngram_size"], 3)
        self.assertEqual(model.generate_kwargs["eos_token_id"], 2)
        self.assertEqual(model.generate_kwargs["pad_token_id"], 0)

    def test_parse_args_uses_hf_generation_defaults(self) -> None:
        # ---------------------------------------------------------
        # Keep instruction inference defaults aligned with the new
        # AutoModel generate path.
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
        # Convert dtype CLI names into the values passed to
        # AutoModelForCausalLM.from_pretrained.
        # ---------------------------------------------------------
        self.assertEqual(resolve_torch_dtype("auto"), "auto")
        self.assertEqual(resolve_torch_dtype("float16"), torch.float16)
        self.assertEqual(resolve_torch_dtype("bfloat16"), torch.bfloat16)
        self.assertEqual(resolve_torch_dtype("float32"), torch.float32)


if __name__ == "__main__":
    unittest.main()
