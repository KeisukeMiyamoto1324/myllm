from transformers import PreTrainedConfig


class MyLLMConfig(PreTrainedConfig):
    model_type = "myllm"

    def __init__(
        self,
        vocab_size: int = 4,
        max_len: int = 6,
        d_model: int = 2,
        num_layers: int = 2,
        num_heads: int = 1,
        d_ff: int = 8,
        learning_rate: float = 0.1,
        pad_token_id: int = 0,
        bos_token_id: int = 2,
        eos_token_id: int = 3,
        tie_word_embeddings: bool = True,
        **kwargs: object,
    ) -> None:
        # ---------------------------------------------------------
        # Store the architecture values needed to rebuild the
        # PyTorch decoder-only Transformer during AutoModel loading.
        # ---------------------------------------------------------
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.learning_rate = learning_rate
        self.tie_word_embeddings = tie_word_embeddings
        self.hidden_size = d_model
        self.num_hidden_layers = num_layers
        self.num_attention_heads = num_heads
        self.intermediate_size = d_ff
        self.max_position_embeddings = max_len

        # ---------------------------------------------------------
        # Pass standard token ids to the Transformers base config so
        # generation utilities can resolve special tokens normally.
        # ---------------------------------------------------------
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
