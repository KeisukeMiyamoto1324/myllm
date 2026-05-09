from rich import box
from rich.console import Console
from rich.table import Table
from transformers import PreTrainedTokenizerBase

from tokenizer_metrics import CompressionResult


def build_vocab_table(tokenizers: list[tuple[str, PreTrainedTokenizerBase]]) -> Table:
    # ---------------------------------------------------------
    # Build a compact table for tokenizer vocabulary sizes.
    # ---------------------------------------------------------
    table = Table(title="Tokenizer Vocabulary Size", box=box.SIMPLE)
    table.add_column("model", no_wrap=True)
    table.add_column("vocab_size", justify="right")

    for model_name, tokenizer in tokenizers:
        table.add_row(model_name, str(tokenizer.vocab_size))

    return table


def build_compression_table(results: list[CompressionResult]) -> Table:
    # ---------------------------------------------------------
    # Build a readable table for language compression metrics.
    # ---------------------------------------------------------
    table = Table(title="Tokenizer Compression Benchmark", box=box.SIMPLE)
    table.add_column("model", no_wrap=True)
    table.add_column("benchmark", no_wrap=True)
    table.add_column("genre", no_wrap=True)
    table.add_column("lang")
    table.add_column("chars", justify="right")
    table.add_column("bytes", justify="right")
    table.add_column("tokens", justify="right")
    table.add_column("chars/token", justify="right")
    table.add_column("bytes/token", justify="right")

    for result in results:
        table.add_row(
            result.model_name,
            result.benchmark_name,
            result.genre,
            result.language,
            str(result.char_count),
            str(result.byte_count),
            str(result.token_count),
            f"{result.chars_per_token:.4f}",
            f"{result.bytes_per_token:.4f}",
        )

    return table


def print_report(tokenizers: list[tuple[str, PreTrainedTokenizerBase]], results: list[CompressionResult]) -> None:
    # ---------------------------------------------------------
    # Print the tokenizer summary and compression comparison
    # with a fixed console width to avoid truncated columns.
    # ---------------------------------------------------------
    console = Console(width=140)
    console.print(build_vocab_table(tokenizers))
    console.print(build_compression_table(results))
