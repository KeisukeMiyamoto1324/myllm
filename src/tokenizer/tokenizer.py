import json
from collections.abc import Iterable
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import ByteLevel as ByteLevelProcessor
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast


@dataclass
class ByteLevelBPE:
    vocab_size: int = 65536
    pad_token: str = "|<pad>|"
    unknown_token: str = "|<unknown>|"
    bos_token: str = "|<bos>|"
    eos_token: str = "|<eos>|"
    sep_token: str = "|<sep>|"
    cls_token: str = "|<cls>|"
    mask_token: str = "|<mask>|"
    system_token: str = "|<system>|"
    user_token: str = "|<user>|"
    assistant_token: str = "|<assistant>|"
    thinking_token: str = "|<thinking>|"
    end_of_thinking_token: str = "|<end_of_thinking>|"
    end_of_turn_token: str = "|<end_of_turn>|"
    extra_special_tokens: list[str] = field(default_factory=list)
    tokenizer: Tokenizer = field(init=False)
    special_tokens: list[str] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        # ---------------------------------------------------------
        # Build the special token list once and initialize the
        # Hugging Face tokenizer with ByteLevel BPE components.
        # ---------------------------------------------------------
        self.special_tokens = [
            self.pad_token,
            self.unknown_token,
            self.bos_token,
            self.eos_token,
            self.sep_token,
            self.cls_token,
            self.mask_token,
            self.system_token,
            self.user_token,
            self.assistant_token,
            self.thinking_token,
            self.end_of_thinking_token,
            self.end_of_turn_token,
            *self.extra_special_tokens,
        ]
        self.tokenizer = Tokenizer(BPE(unk_token=self.unknown_token))
        self.tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        self.tokenizer.decoder = ByteLevelDecoder()
        self.tokenizer.post_processor = ByteLevelProcessor(trim_offsets=True)

    def train(self, sentences: Iterable[str]) -> None:
        # ---------------------------------------------------------
        # Train the Rust-backed tokenizer directly from the text
        # iterator without materializing the full corpus.
        # ---------------------------------------------------------
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=self.special_tokens,
            show_progress=True,
            initial_alphabet=ByteLevel.alphabet(),
        )
        self.tokenizer.train_from_iterator(iterator=sentences, trainer=trainer)

    def tokenize(self, sentence: str) -> list[int]:
        # ---------------------------------------------------------
        # Encode one input sentence into token ids using the
        # trained tokenizer state.
        # ---------------------------------------------------------
        encoding = self.tokenizer.encode(sentence)
        return [token_id for token_id in encoding.ids]

    def detokenize(self, token_ids: list[int]) -> str:
        # ---------------------------------------------------------
        # Decode token ids back into text with the byte-level
        # decoder configured on the tokenizer.
        # ---------------------------------------------------------
        return self.tokenizer.decode(token_ids)

    def token_to_id(self, token: str) -> int:
        # ---------------------------------------------------------
        # Resolve one token string into its vocabulary id so
        # training and inference can reuse tokenizer metadata.
        # ---------------------------------------------------------
        token_id = self.tokenizer.token_to_id(token)

        # ---------------------------------------------------------
        # Reject missing tokens because the saved tokenizer must
        # already include the configured special tokens.
        # ---------------------------------------------------------
        if token_id is None:
            raise ValueError(f"Token is not registered in the tokenizer: {token}")

        return token_id

    def get_vocab_size(self) -> int:
        # ---------------------------------------------------------
        # Return the serialized vocabulary size so model shapes
        # stay aligned with the tokenizer artifact.
        # ---------------------------------------------------------
        return self.tokenizer.get_vocab_size()

    def named_special_tokens(self) -> dict[str, str]:
        # ---------------------------------------------------------
        # Build the standard special token mapping expected by
        # Hugging Face tokenizer configuration files.
        # ---------------------------------------------------------
        return {
            "pad_token": self.pad_token,
            "unk_token": self.unknown_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "sep_token": self.sep_token,
            "cls_token": self.cls_token,
            "mask_token": self.mask_token,
        }

    def extra_special_token_list(self) -> list[str]:
        # ---------------------------------------------------------
        # Keep chat and reasoning control tokens separate from the
        # standard Hugging Face special token names.
        # ---------------------------------------------------------
        return [
            self.system_token,
            self.user_token,
            self.assistant_token,
            self.thinking_token,
            self.end_of_thinking_token,
            self.end_of_turn_token,
            *self.extra_special_tokens,
        ]

    def to_pretrained_tokenizer(self) -> PreTrainedTokenizerFast:
        # ---------------------------------------------------------
        # Wrap the Rust tokenizer with the Transformers interface
        # so it can be saved for AutoTokenizer.from_pretrained.
        # ---------------------------------------------------------
        return PreTrainedTokenizerFast(
            tokenizer_object=self.tokenizer,
            **self.named_special_tokens(),
            extra_special_tokens=self.extra_special_token_list(),
        )

    def save(self, path: str | Path) -> None:
        # ---------------------------------------------------------
        # Save the tokenizer as a single JSON artifact that can
        # be restored without rebuilding the trainer settings.
        # ---------------------------------------------------------
        self.tokenizer.save(str(path))

    def save_pretrained(self, path: str | Path) -> None:
        # ---------------------------------------------------------
        # Save the tokenizer as a Hugging Face directory artifact
        # that AutoTokenizer.from_pretrained can load directly.
        # ---------------------------------------------------------
        output_path = Path(path)
        output_path.mkdir(parents=True, exist_ok=True)
        pretrained_tokenizer = self.to_pretrained_tokenizer()
        pretrained_tokenizer.save_pretrained(str(output_path))

        # ---------------------------------------------------------
        # Write the legacy special token map explicitly so older
        # and newer Transformers loaders get the same metadata.
        # ---------------------------------------------------------
        special_tokens_map = {
            **self.named_special_tokens(),
            "extra_special_tokens": self.extra_special_token_list(),
        }
        special_tokens_map_path = output_path / "special_tokens_map.json"
        special_tokens_map_path.write_text(
            json.dumps(special_tokens_map, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def resolve_tokenizer_file(path: str | Path) -> Path:
        # ---------------------------------------------------------
        # Resolve either a Hugging Face tokenizer directory or a
        # direct tokenizer JSON path into the serialized JSON file.
        # ---------------------------------------------------------
        tokenizer_path = Path(path)
        return tokenizer_path / "tokenizer.json" if tokenizer_path.is_dir() else tokenizer_path

    @classmethod
    def load(cls, path: str | Path) -> "ByteLevelBPE":
        # ---------------------------------------------------------
        # Restore the saved tokenizer and keep the wrapper fields
        # aligned with the serialized special token settings.
        # ---------------------------------------------------------
        tokenizer_path = cls.resolve_tokenizer_file(path)

        with open(tokenizer_path) as f:
            data = json.load(f)

        tokenizer = cls()
        tokenizer.tokenizer = Tokenizer.from_file(str(tokenizer_path))

        # ---------------------------------------------------------
        # Recover the configured special tokens from the saved
        # added token entries so the wrapper stays consistent.
        # ---------------------------------------------------------
        added_tokens = data.get("added_tokens", [])
        tokenizer.special_tokens = [item["content"] for item in added_tokens]
        return tokenizer
