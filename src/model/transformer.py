import torch
import torch.nn as nn
from torch.optim import Adam
import lightning as L

from src.model.kv_cache import KeyValueCache, LayerKeyValueCache
from src.model.position_encoding import PositionEncoding
from src.model.self_attention import Attention


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()

        # ---------------------------------------------------------
        # Use the standard Transformer feed-forward sublayer so each
        # token can be transformed independently after attention.
        # ---------------------------------------------------------
        self.linear_1 = nn.Linear(in_features=d_model, out_features=d_ff)
        self.activation = nn.GELU()
        self.linear_2 = nn.Linear(in_features=d_ff, out_features=d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ---------------------------------------------------------
        # Expand the channel dimension, apply a non-linearity, and
        # project back to the model dimension.
        # ---------------------------------------------------------
        hidden = self.linear_1(x)
        activated = self.activation(hidden)
        return self.linear_2(activated)


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int) -> None:
        super().__init__()

        # ---------------------------------------------------------
        # Compose one decoder block from attention, feed-forward, and
        # normalization layers with residual connections.
        # ---------------------------------------------------------
        self.norm_1 = nn.LayerNorm(normalized_shape=d_model)
        self.attention = Attention(d_model=d_model, num_heads=num_heads)
        self.norm_2 = nn.LayerNorm(normalized_shape=d_model)
        self.feed_forward = FeedForward(d_model=d_model, d_ff=d_ff)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # ---------------------------------------------------------
        # Apply pre-norm self-attention so multiple decoder blocks can
        # be stacked without changing the external interface.
        # ---------------------------------------------------------
        attention_input = self.norm_1(x)
        attention_output = self.attention(
            attention_input,
            attention_input,
            attention_input,
            mask,
        )
        attention_residual = x + attention_output

        # ---------------------------------------------------------
        # Apply the position-wise feed-forward network as the second
        # sublayer inside the decoder block.
        # ---------------------------------------------------------
        feed_forward_input = self.norm_2(attention_residual)
        feed_forward_output = self.feed_forward(feed_forward_input)
        return attention_residual + feed_forward_output

    def forward_with_cache(
        self,
        x: torch.Tensor,
        past_key_value: LayerKeyValueCache | None,
        mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, LayerKeyValueCache]:
        # ---------------------------------------------------------
        # Apply self-attention with a layer-local cache, then keep the
        # feed-forward path identical to the full sequence forward.
        # ---------------------------------------------------------
        attention_input = self.norm_1(x)
        attention_output, key_value_cache = self.attention.forward_with_cache(
            attention_input,
            attention_input,
            attention_input,
            past_key_value,
            mask,
        )
        attention_residual = x + attention_output

        # ---------------------------------------------------------
        # Transform only the visible token states because old states
        # have already been folded into the cached keys and values.
        # ---------------------------------------------------------
        feed_forward_input = self.norm_2(attention_residual)
        feed_forward_output = self.feed_forward(feed_forward_input)
        return attention_residual + feed_forward_output, key_value_cache


class DecoderOnlyTransformer(L.LightningModule):
    def __init__(
        self,
        num_tokens: int = 4,
        d_model: int = 2,
        max_len: int = 6,
        num_layers: int = 2,
        num_heads: int = 1,
        d_ff: int = 8,
        learning_rate: float = 0.1,
        pad_token_id: int = 0,
    ) -> None:
        super().__init__()

        # ---------------------------------------------------------
        # Embed tokens and positions before passing them through a
        # stack of decoder blocks.
        # ---------------------------------------------------------
        self.we = nn.Embedding(num_embeddings=num_tokens, embedding_dim=d_model)
        self.pe = PositionEncoding(d_model=d_model, max_len=max_len)
        self.blocks = nn.ModuleList(
            [DecoderBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(normalized_shape=d_model)
        self.fc_layer = nn.Linear(in_features=d_model, out_features=num_tokens)
        self.learning_rate = learning_rate
        self.pad_token_id = pad_token_id

        # ---------------------------------------------------------
        # Keep the loss local to the model so Lightning can call the
        # training step without extra setup.
        # ---------------------------------------------------------
        self.loss = nn.CrossEntropyLoss(ignore_index=pad_token_id)

    def _create_causal_mask(self, token_ids: torch.Tensor) -> torch.Tensor:
        # ---------------------------------------------------------
        # Build a mask that hides future tokens for autoregressive
        # decoding in every decoder block.
        # ---------------------------------------------------------
        seq_len = token_ids.size(dim=1)
        lower_triangular = torch.tril(torch.ones((seq_len, seq_len), device=token_ids.device, dtype=torch.bool))
        return (~lower_triangular).unsqueeze(0).unsqueeze(0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # ---------------------------------------------------------
        # Convert token ids into hidden states and apply positional
        # information before the decoder stack.
        # ---------------------------------------------------------
        word_embeddings = self.we(token_ids)
        hidden_states = self.pe(word_embeddings)
        mask = self._create_causal_mask(token_ids)

        # ---------------------------------------------------------
        # Reuse the same decoder block interface for every layer to
        # make the model depth configurable.
        # ---------------------------------------------------------
        for block in self.blocks:
            hidden_states = block(hidden_states, mask)

        # ---------------------------------------------------------
        # Normalize the final hidden states and map them into token
        # logits for next-token prediction.
        # ---------------------------------------------------------
        normalized_hidden_states = self.final_norm(hidden_states)
        return self.fc_layer(normalized_hidden_states)

    def forward_with_cache(
        self,
        token_ids: torch.Tensor,
        past_key_values: KeyValueCache | None,
    ) -> tuple[torch.Tensor, KeyValueCache]:
        # ---------------------------------------------------------
        # Offset positions by the cached sequence length so one-token
        # inference matches full-sequence absolute positions.
        # ---------------------------------------------------------
        position_offset = 0

        if past_key_values is not None:
            position_offset = past_key_values[0][0].size(dim=2)

        word_embeddings = self.we(token_ids)
        hidden_states = self.pe(word_embeddings, position_offset=position_offset)
        mask = self._create_causal_mask(token_ids) if past_key_values is None else None
        next_key_values: KeyValueCache = []

        # ---------------------------------------------------------
        # Pass each layer its own cache entry and collect the updated
        # entries in the same order for the next generation step.
        # ---------------------------------------------------------
        for layer_index, block in enumerate(self.blocks):
            past_key_value = None if past_key_values is None else past_key_values[layer_index]
            hidden_states, key_value_cache = block.forward_with_cache(
                hidden_states,
                past_key_value,
                mask,
            )
            next_key_values.append(key_value_cache)

        # ---------------------------------------------------------
        # Produce logits only for the currently supplied token slice
        # while returning cache tensors that include all past tokens.
        # ---------------------------------------------------------
        normalized_hidden_states = self.final_norm(hidden_states)
        return self.fc_layer(normalized_hidden_states), next_key_values

    def configure_optimizers(self) -> Adam:
        # ---------------------------------------------------------
        # Use the same optimizer setup as before while keeping the
        # deeper architecture trainable with one entry point.
        # ---------------------------------------------------------
        return Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        # ---------------------------------------------------------
        # Run the forward pass and compute token-level cross-entropy
        # against the shifted labels.
        # ---------------------------------------------------------
        del batch_idx
        input_tokens, labels = batch
        output = self.forward(input_tokens)
        loss = self.loss(output.transpose(1, 2), labels)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        # ---------------------------------------------------------
        # Reuse the same autoregressive loss during validation so
        # checkpoints can monitor held-out next-token accuracy.
        # ---------------------------------------------------------
        del batch_idx
        input_tokens, labels = batch
        output = self.forward(input_tokens)
        loss = self.loss(output.transpose(1, 2), labels)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss
