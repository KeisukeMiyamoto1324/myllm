import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration_myllm import MyLLMConfig
from .kv_cache import KeyValueCache
from .position_encoding import PositionEncoding
from .self_attention import Attention
from .transformer import DecoderOnlyTransformer

# ---------------------------------------------------------
# Reference nested remote-code dependencies directly so local
# AutoModel loading copies every file needed by relative imports.
# ---------------------------------------------------------
REMOTE_CODE_DEPENDENCIES = (Attention, PositionEncoding)


class MyLLMForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MyLLMConfig
    main_input_name = "input_ids"
    _tied_weights_keys = {"transformer.fc_layer.weight": "transformer.we.weight"}

    def __init__(self, config: MyLLMConfig) -> None:
        super().__init__(config)

        # ---------------------------------------------------------
        # Reuse the existing PyTorch Transformer implementation and
        # keep the HF wrapper responsible only for AutoModel APIs.
        # ---------------------------------------------------------
        self.transformer = DecoderOnlyTransformer(
            num_tokens=config.vocab_size,
            d_model=config.d_model,
            max_len=config.max_len,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            d_ff=config.d_ff,
            learning_rate=config.learning_rate,
            pad_token_id=config.pad_token_id,
        )
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        # ---------------------------------------------------------
        # Expose input embeddings through the standard Transformers
        # interface used by resizing and generation helpers.
        # ---------------------------------------------------------
        return self.transformer.we

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        # ---------------------------------------------------------
        # Keep tied output weights aligned when callers replace the
        # token embedding module through the Transformers interface.
        # ---------------------------------------------------------
        self.transformer.we = value
        self.transformer.fc_layer.weight = value.weight

    def get_output_embeddings(self) -> nn.Linear:
        # ---------------------------------------------------------
        # Expose the tied LM head through the standard Transformers
        # interface used by causal language model utilities.
        # ---------------------------------------------------------
        return self.transformer.fc_layer

    def set_output_embeddings(self, value: nn.Linear) -> None:
        # ---------------------------------------------------------
        # Allow Transformers utilities to replace the LM head while
        # preserving the module expected by the existing model.
        # ---------------------------------------------------------
        self.transformer.fc_layer = value

    def _supports_default_dynamic_cache(self) -> bool:
        # ---------------------------------------------------------
        # Use the existing list-based KV cache instead of letting
        # Transformers allocate its DynamicCache implementation.
        # ---------------------------------------------------------
        return False

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values: KeyValueCache | None = None,
        **kwargs: object,
    ) -> dict[str, torch.Tensor | KeyValueCache | bool | None]:
        # ---------------------------------------------------------
        # Feed only the newest token after the cache is populated so
        # generate can reuse the existing incremental forward path.
        # ---------------------------------------------------------
        del kwargs
        model_input_ids = input_ids[:, -1:] if past_key_values is not None else input_ids
        return {
            "input_ids": model_input_ids,
            "past_key_values": past_key_values,
            "use_cache": True,
        }

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        past_key_values: KeyValueCache | None = None,
        use_cache: bool | None = None,
        return_dict: bool | None = None,
        **kwargs: object,
    ) -> CausalLMOutputWithPast | tuple[torch.Tensor, ...]:
        # ---------------------------------------------------------
        # Accept the standard AutoModelForCausalLM argument names and
        # delegate the actual tensor computation to the PyTorch model.
        # ---------------------------------------------------------
        del kwargs

        if input_ids is None:
            raise ValueError("input_ids is required")

        should_use_cache = bool(use_cache)

        if past_key_values is not None or should_use_cache:
            logits, next_key_values = self.transformer.forward_with_cache(
                token_ids=input_ids,
                past_key_values=past_key_values,
            )
        else:
            logits = self.transformer(token_ids=input_ids)
            next_key_values = None

        # ---------------------------------------------------------
        # Follow causal LM convention for labels supplied by HF
        # Trainer and examples: predict token n+1 from position n.
        # ---------------------------------------------------------
        loss = None

        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=self.config.pad_token_id,
            )

        # ---------------------------------------------------------
        # Return either the standard modeling output or a tuple for
        # callers that explicitly disable dictionary-style outputs.
        # ---------------------------------------------------------
        if return_dict is False:
            output = (logits,)
            return (loss, *output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=next_key_values,
        )
