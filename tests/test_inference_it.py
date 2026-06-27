import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

from src.inference_it.cli import parse_args
from src.inference_it.generation import build_chat_input_ids
from src.inference_it.generation import generate_chat_response
from src.inference_it.generation import generate_token_ids
from src.inference_it.generation import resolve_torch_dtype
from src.posttraining.chat_template import ChatMessage
from src.shared.model.kv_cache import KeyValueCache


class FakeTokenizer:
    bos_token = "|<bos>|"
    eos_token = "|<eos>|"
    system_token = "|<system>|"
    user_token = "|<user>|"
    assistant_token = "|<assistant>|"
    end_of_turn_token = "|<end_of_turn>|"

    def __init__(self) -> None:
        # ---------------------------------------------------------
        # Store generated decode input for assertions after the
        # helper returns assistant response text.
        # ---------------------------------------------------------
        self.decoded_ids: list[int] = []
        self.prompts: list[str] = []

    def token_to_id(self, token: str) -> int:
        # ---------------------------------------------------------
        # Return stable ids for chat and generation special tokens.
        # ---------------------------------------------------------
        token_ids = {
            self.bos_token: 1,
            self.eos_token: 2,
            self.user_token: 3,
            self.assistant_token: 4,
            self.end_of_turn_token: 5,
            self.system_token: 6,
        }
        return token_ids[token]

    def tokenize(self, sentence: str) -> list[int]:
        # ---------------------------------------------------------
        # Return fixed content token ids so tests can verify the
        # serialized chat template order.
        # ---------------------------------------------------------
        self.prompts.append(sentence)
        token_ids_by_sentence = {
            "hello": [10, 11],
            "hi": [12],
        }
        return token_ids_by_sentence[sentence]

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
    def test_build_chat_input_ids_applies_chat_template(self) -> None:
        # ---------------------------------------------------------
        # Serialize user and assistant turns with role markers,
        # content tokens, and end-of-turn markers.
        # ---------------------------------------------------------
        tokenizer = FakeTokenizer()
        input_ids = build_chat_input_ids(
            tokenizer=tokenizer,
            messages=[
                ChatMessage(role="user", content="hello"),
                ChatMessage(role="assistant", content="hi"),
            ],
        )

        self.assertEqual(input_ids, [1, 3, 10, 11, 5, 4, 12, 5, 4])
        self.assertEqual(tokenizer.prompts, ["hello", "hi"])

    def test_generate_chat_response_uses_chat_template_generation(self) -> None:
        # ---------------------------------------------------------
        # Verify chat generation uses the PyTorch cache path and
        # decodes only assistant response tokens.
        # ---------------------------------------------------------
        model = FakeModel(next_token_ids=[13, 14])
        tokenizer = FakeTokenizer()
        text = generate_chat_response(
            model=model,
            tokenizer=tokenizer,
            messages=[ChatMessage(role="user", content="hello")],
            max_new_tokens=2,
            do_sample=False,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            repetition_penalty=1.05,
            no_repeat_ngram_size=3,
        )

        self.assertEqual(text, "continuation text")
        self.assertEqual(tokenizer.decoded_ids, [13, 14])
        self.assertEqual(model.calls, [[1, 3, 10, 11, 5, 4], [13]])

    def test_generate_token_ids_stops_on_end_of_turn(self) -> None:
        # ---------------------------------------------------------
        # Stop generation as soon as the chat end-of-turn marker is
        # selected so the next user turn can begin cleanly.
        # ---------------------------------------------------------
        model = FakeModel(next_token_ids=[13, 5, 14])
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
            stop_token_ids={2, 5},
        )

        self.assertEqual(generated_ids, [13, 5])

    def test_parse_args_uses_pytorch_generation_defaults(self) -> None:
        # ---------------------------------------------------------
        # Keep instruction chat defaults aligned with the native
        # PyTorch generation path.
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

    def test_parse_args_defaults_to_posttraining_model_dir(self) -> None:
        # ---------------------------------------------------------
        # Use the local instruction-tuned artifact path by default
        # for terminal chat.
        # ---------------------------------------------------------
        argv = ["inference.py"]

        with patch("sys.argv", argv):
            args = parse_args(default_model_dir=Path("models/lambda-1-160m-it"))

        self.assertEqual(args.model_dir, "models/lambda-1-160m-it")


if __name__ == "__main__":
    unittest.main()
