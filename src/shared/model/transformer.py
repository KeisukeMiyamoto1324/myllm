import math

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import lightning as L

from src.shared.model.kv_cache import KeyValueCache, LayerKeyValueCache
from src.shared.model.position_encoding import RotaryPositionEmbedding
from src.shared.model.self_attention import Attention


WEIGHT_DECAY = 0.01


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()

        # ---------------------------------------------------------
        # Use SwiGLU so each token is transformed through separate
        # value and gate projections before returning to model width.
        # ---------------------------------------------------------
        self.gate_proj = nn.Linear(in_features=d_model, out_features=d_ff)
        self.up_proj = nn.Linear(in_features=d_model, out_features=d_ff)
        self.activation = nn.SiLU()
        self.down_proj = nn.Linear(in_features=d_ff, out_features=d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ---------------------------------------------------------
        # Activate the gate projection, multiply it with the value
        # projection, and map the gated states back to model width.
        # ---------------------------------------------------------
        gate = self.activation(self.gate_proj(x))
        value = self.up_proj(x)
        return self.down_proj(gate * value)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rotary_position_embedding: RotaryPositionEmbedding,
    ) -> None:
        super().__init__()

        # ---------------------------------------------------------
        # Compose one decoder block from attention, feed-forward, and
        # RMS normalization layers with residual connections.
        # ---------------------------------------------------------
        self.norm_1 = nn.RMSNorm(normalized_shape=d_model)
        self.attention = Attention(
            d_model=d_model,
            num_heads=num_heads,
            rotary_position_embedding=rotary_position_embedding,
        )
        self.norm_2 = nn.RMSNorm(normalized_shape=d_model)
        self.feed_forward = FeedForward(d_model=d_model, d_ff=d_ff)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # ---------------------------------------------------------
        # Apply pre-norm self-attention so multiple decoder blocks can
        # be stacked without changing the external interface.
        # ---------------------------------------------------------
        attention_input = self.norm_1(x)
        attention_output = self.attention(
            attention_input,
            attention_input,
            attention_input,
            is_causal=attention_mask is None,
            attention_mask=attention_mask,
            position_ids=position_ids,
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
        position_offset: int,
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
            is_causal=past_key_value is None,
            position_offset=position_offset,
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
        use_fused_optimizer: bool = False,
        loss_chunk_size: int = 32,
        lr_warmup_steps: int | None = None,
        lr_total_steps: int | None = None,
        min_learning_rate: float | None = None,
    ) -> None:
        super().__init__()

        # ---------------------------------------------------------
        # Embed tokens before passing them through a stack of decoder
        # blocks that apply rotary positions inside attention.
        # ---------------------------------------------------------
        self.we = nn.Embedding(num_embeddings=num_tokens, embedding_dim=d_model)
        self.rotary_position_embedding = RotaryPositionEmbedding(
            head_dim=d_model // num_heads,
            max_len=max_len,
        )
        self.blocks = nn.ModuleList(
            [
                DecoderBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    rotary_position_embedding=self.rotary_position_embedding,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.RMSNorm(normalized_shape=d_model)
        self.fc_layer = nn.Linear(in_features=d_model, out_features=num_tokens)

        # ---------------------------------------------------------
        # Share token embedding weights with the output projection
        # so small models spend more parameters inside the blocks.
        # ---------------------------------------------------------
        self.fc_layer.weight = self.we.weight
        self.learning_rate = learning_rate
        self.pad_token_id = pad_token_id
        self.use_fused_optimizer = use_fused_optimizer
        self.loss_chunk_size = loss_chunk_size
        self.lr_warmup_steps = lr_warmup_steps
        self.lr_total_steps = lr_total_steps
        self.min_learning_rate = min_learning_rate

        # ---------------------------------------------------------
        # Reject partially configured schedules so posttraining can
        # keep fixed LR while pretraining opts into full scheduling.
        # ---------------------------------------------------------
        lr_schedule_values = [lr_warmup_steps, lr_total_steps, min_learning_rate]

        if any(value is None for value in lr_schedule_values) and any(
            value is not None for value in lr_schedule_values
        ):
            raise ValueError("LR schedule requires warmup steps, total steps, and minimum learning rate")

        # ---------------------------------------------------------
        # Keep summed token loss local so large vocabulary logits
        # can be reduced chunk by chunk during training.
        # ---------------------------------------------------------
        self.loss = nn.CrossEntropyLoss(ignore_index=pad_token_id, reduction="sum")

    def forward_hidden(
        self,
        token_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # ---------------------------------------------------------
        # Convert token ids into hidden states before the decoder
        # stack applies attention-local rotary positions.
        # ---------------------------------------------------------
        hidden_states = self.we(token_ids)

        # ---------------------------------------------------------
        # Reuse the same decoder block interface for every layer to
        # make the model depth configurable.
        # ---------------------------------------------------------
        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )

        # ---------------------------------------------------------
        # Normalize the final hidden states and map them into token
        # logits for next-token prediction.
        # ---------------------------------------------------------
        return self.final_norm(hidden_states)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # ---------------------------------------------------------
        # Keep the public forward path returning full vocabulary
        # logits for inference and compatibility with callers.
        # ---------------------------------------------------------
        hidden_states = self.forward_hidden(token_ids)
        return self.fc_layer(hidden_states)

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

        hidden_states = self.we(token_ids)
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
                position_offset=position_offset,
            )
            next_key_values.append(key_value_cache)

        # ---------------------------------------------------------
        # Produce logits only for the currently supplied token slice
        # while returning cache tensors that include all past tokens.
        # ---------------------------------------------------------
        normalized_hidden_states = self.final_norm(hidden_states)
        return self.fc_layer(normalized_hidden_states), next_key_values

    def configure_optimizers(self) -> AdamW | dict[str, object]:
        # ---------------------------------------------------------
        # Use AdamW with decay only on matrix weights while keeping
        # embeddings, norms, and biases out of weight decay.
        # ---------------------------------------------------------
        parameter_groups = build_weight_decay_parameter_groups(
            model=self,
            weight_decay=WEIGHT_DECAY,
        )
        optimizer = AdamW(
            parameter_groups,
            lr=self.learning_rate,
            fused=self.use_fused_optimizer,
        )

        # ---------------------------------------------------------
        # Keep callers without scheduler settings on fixed learning
        # rate while pretraining uses step-wise warmup and cosine decay.
        # ---------------------------------------------------------
        if self.lr_warmup_steps is None or self.lr_total_steps is None or self.min_learning_rate is None:
            return optimizer

        scheduler = LambdaLR(
            optimizer=optimizer,
            lr_lambda=lambda step: resolve_warmup_cosine_learning_rate(
                step=step,
                max_learning_rate=self.learning_rate,
                min_learning_rate=self.min_learning_rate,
                warmup_steps=self.lr_warmup_steps,
                total_steps=self.lr_total_steps,
            )
            / self.learning_rate,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def compute_chunked_loss(
        self,
        input_tokens: torch.Tensor,
        labels: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        segment_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # ---------------------------------------------------------
        # Run the Transformer stack once, then split only the large
        # vocabulary projection and cross-entropy over token positions.
        # ---------------------------------------------------------
        attention_mask = None

        if segment_ids is not None:
            attention_mask = build_packed_attention_mask(segment_ids=segment_ids)

        hidden_states = self.forward_hidden(
            token_ids=input_tokens,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )
        seq_len = hidden_states.size(dim=1)
        chunk_starts = range(0, seq_len, self.loss_chunk_size)

        # ---------------------------------------------------------
        # Accumulate summed token losses so padding can be ignored
        # with the same weighting as a single full cross-entropy call.
        # ---------------------------------------------------------
        loss_chunks = [
            self.loss(
                self.fc_layer(
                    hidden_states[:, chunk_start : chunk_start + self.loss_chunk_size, :]
                ).transpose(1, 2),
                labels[:, chunk_start : chunk_start + self.loss_chunk_size],
            )
            for chunk_start in chunk_starts
        ]
        total_loss = torch.stack(loss_chunks).sum()
        valid_token_count = labels.ne(self.pad_token_id).sum()
        return total_loss / valid_token_count

    def training_step(self, batch: tuple[torch.Tensor, ...], batch_idx: int) -> torch.Tensor:
        # ---------------------------------------------------------
        # Run the forward pass and compute token-level cross-entropy
        # against the shifted labels.
        # ---------------------------------------------------------
        del batch_idx
        input_tokens, labels, position_ids, segment_ids = normalize_training_batch(batch=batch)
        loss = self.compute_chunked_loss(
            input_tokens=input_tokens,
            labels=labels,
            position_ids=position_ids,
            segment_ids=segment_ids,
        )
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, ...], batch_idx: int) -> torch.Tensor:
        # ---------------------------------------------------------
        # Reuse the same autoregressive loss during validation so
        # checkpoints can monitor held-out next-token accuracy.
        # ---------------------------------------------------------
        del batch_idx
        input_tokens, labels, position_ids, segment_ids = normalize_training_batch(batch=batch)
        loss = self.compute_chunked_loss(
            input_tokens=input_tokens,
            labels=labels,
            position_ids=position_ids,
            segment_ids=segment_ids,
        )
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return loss


def normalize_training_batch(
    batch: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    # ---------------------------------------------------------
    # Support packed pretraining batches while keeping existing
    # posttraining batches on the two-tensor path.
    # ---------------------------------------------------------
    if len(batch) == 2:
        input_tokens, labels = batch
        return input_tokens, labels, None, None

    input_tokens, labels, position_ids, segment_ids = batch
    return input_tokens, labels, position_ids, segment_ids


def build_weight_decay_parameter_groups(
    model: nn.Module,
    weight_decay: float,
) -> list[dict[str, object]]:
    # ---------------------------------------------------------
    # Apply decay to projection matrices only. Keep embeddings,
    # normalization parameters, and biases free from weight decay.
    # ---------------------------------------------------------
    no_decay_parameter_ids = {
        id(parameter)
        for module in model.modules()
        if isinstance(module, (nn.Embedding, nn.RMSNorm, nn.LayerNorm))
        for parameter in module.parameters(recurse=False)
    }
    decay_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and id(parameter) not in no_decay_parameter_ids and not name.endswith(".bias")
    ]
    no_decay_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and (id(parameter) in no_decay_parameter_ids or name.endswith(".bias"))
    ]
    return [
        {"params": decay_parameters, "weight_decay": weight_decay},
        {"params": no_decay_parameters, "weight_decay": 0.0},
    ]


def build_packed_attention_mask(segment_ids: torch.Tensor) -> torch.Tensor:
    # ---------------------------------------------------------
    # Allow each token to attend only to earlier tokens from the
    # same packed segment. Padding query rows attend to valid keys.
    # ---------------------------------------------------------
    batch_size, seq_len = segment_ids.size()
    causal_mask = torch.ones(
        seq_len,
        seq_len,
        dtype=torch.bool,
        device=segment_ids.device,
    ).tril()
    valid_tokens = segment_ids.ge(0)
    same_segment_mask = segment_ids.unsqueeze(2).eq(segment_ids.unsqueeze(1))
    packed_mask = same_segment_mask & causal_mask.unsqueeze(0) & valid_tokens.unsqueeze(1)
    padding_query_mask = causal_mask.unsqueeze(0) & valid_tokens.unsqueeze(1)
    attention_mask = torch.where(
        valid_tokens.view(batch_size, seq_len, 1),
        packed_mask,
        padding_query_mask,
    )
    return attention_mask.unsqueeze(1)


def resolve_warmup_cosine_learning_rate(
    step: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_steps: int,
    total_steps: int,
) -> float:
    # ---------------------------------------------------------
    # Raise the learning rate linearly at the start, then decay it
    # smoothly to the configured minimum by the final training step.
    # ---------------------------------------------------------
    if step < warmup_steps:
        return max_learning_rate * step / warmup_steps

    decay_progress = min(1.0, (step - warmup_steps) / (total_steps - warmup_steps))
    cosine_scale = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
    return min_learning_rate + (max_learning_rate - min_learning_rate) * cosine_scale
