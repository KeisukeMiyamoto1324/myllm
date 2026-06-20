import json
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback
from rich.table import Table
from torch.utils.data import Dataset

from src.inference_base.generation import generate_token_ids
from src.shared.console import console
from src.shared.console import progress_manager
from src.shared.model.transformer import DecoderOnlyTransformer
from src.shared.packed_dataset import PackedTrainingExample
from src.shared.tokenizer import ByteLevelBPE


class ValidationGenerationCallback(Callback):
    def __init__(
        self,
        dataset: Dataset[PackedTrainingExample],
        tokenizer: ByteLevelBPE,
        output_dir: Path,
        prompt_tokens: int = 128,
        max_new_tokens: int = 128,
        preview_count: int = 5,
        preview_characters: int = 100,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.prompt_tokens = prompt_tokens
        self.max_new_tokens = max_new_tokens
        self.preview_count = preview_count
        self.preview_characters = preview_characters

    def on_validation_epoch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        # ---------------------------------------------------------
        # Generate deterministic continuations for every validation
        # sample and store one complete JSONL file per global step.
        # ---------------------------------------------------------
        if not trainer.is_global_zero:
            return

        model = pl_module
        output_path = self.output_dir / f"step-{trainer.global_step}.jsonl"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        preview_texts: list[str] = []
        task_id = progress_manager.add_task(
            description="Validation generation",
            total=len(self.dataset),
        )

        try:
            with output_path.open("w", encoding="utf-8") as output_file:
                for sample_index in range(len(self.dataset)):
                    input_ids, _, _, segment_ids = self.dataset[sample_index]
                    prompt_ids = self._build_prompt_ids(
                        input_ids=input_ids,
                        segment_ids=segment_ids,
                    )
                    generated_ids = self._generate_ids(
                        trainer=trainer,
                        model=model,
                        prompt_ids=prompt_ids,
                    )
                    prompt_text = self.tokenizer.detokenize(prompt_ids)
                    generated_text = self.tokenizer.detokenize(generated_ids).strip()
                    result = {
                        "sample_index": sample_index,
                        "global_step": trainer.global_step,
                        "prompt": prompt_text,
                        "generated_text": generated_text,
                        "generated_token_ids": generated_ids,
                    }
                    output_file.write(json.dumps(result, ensure_ascii=False) + "\n")

                    if sample_index < self.preview_count:
                        preview_texts.append(generated_text[: self.preview_characters])

                    progress_manager.update(task_id=task_id, advance=1)
        finally:
            progress_manager.finish_task(task_id=task_id)

        # ---------------------------------------------------------
        # Print only the first few generated continuations in a
        # compact table while preserving every result in JSONL.
        # ---------------------------------------------------------
        self._print_previews(
            global_step=trainer.global_step,
            preview_texts=preview_texts,
            output_path=output_path,
        )

    def _build_prompt_ids(
        self,
        input_ids: torch.Tensor,
        segment_ids: torch.Tensor,
    ) -> list[int]:
        # ---------------------------------------------------------
        # Use the first packed document and stop halfway through
        # short segments so the model always predicts unseen text.
        # ---------------------------------------------------------
        first_segment_ids = input_ids[segment_ids.eq(0)].tolist()
        prompt_size = min(self.prompt_tokens, max(1, len(first_segment_ids) // 2))
        return [int(token_id) for token_id in first_segment_ids[:prompt_size]]

    def _generate_ids(
        self,
        trainer: L.Trainer,
        model: DecoderOnlyTransformer,
        prompt_ids: list[int],
    ) -> list[int]:
        # ---------------------------------------------------------
        # Reuse the inference generation path with greedy decoding
        # so validation outputs remain reproducible across runs.
        # ---------------------------------------------------------
        input_ids = torch.tensor(
            [prompt_ids],
            dtype=torch.long,
            device=next(model.parameters()).device,
        )
        eos_token_id = self.tokenizer.token_to_id(self.tokenizer.eos_token)

        with torch.inference_mode(), trainer.precision_plugin.forward_context():
            return generate_token_ids(
                model=model,
                input_ids=input_ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                top_k=0,
                repetition_penalty=1.0,
                no_repeat_ngram_size=0,
                eos_token_id=eos_token_id,
            )

    def _print_previews(
        self,
        global_step: int,
        preview_texts: list[str],
        output_path: Path,
    ) -> None:
        # ---------------------------------------------------------
        # Render generated text without references because the table
        # is intended for a quick qualitative training check.
        # ---------------------------------------------------------
        table = Table(title=f"Validation generation at step {global_step}")
        table.add_column("Sample", justify="right", style="cyan")
        table.add_column("Generated text", overflow="fold")

        for sample_index, generated_text in enumerate(preview_texts):
            table.add_row(str(sample_index), generated_text)

        console.print(table)
        console.print(f"Saved validation generations: {output_path}")
