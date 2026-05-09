from rich import box
from rich.console import Console
from rich.rule import Rule
from rich.table import Table
from transformers import PreTrainedTokenizerBase

from tokenizer_metrics import CompressionResult


def build_vocab_table(tokenizers: list[tuple[str, PreTrainedTokenizerBase]]) -> Table:
    # ---------------------------------------------------------
    # Build a compact table for tokenizer vocabulary sizes.
    # ---------------------------------------------------------
    table = Table(box=box.SIMPLE)
    table.add_column("model", no_wrap=True)
    table.add_column("vocab_size", justify="right")

    for model_name, tokenizer in tokenizers:
        table.add_row(model_name, str(tokenizer.vocab_size))

    return table


def build_compression_table(results: list[CompressionResult]) -> Table:
    # ---------------------------------------------------------
    # Build a readable table for one benchmark compression
    # metric group.
    # ---------------------------------------------------------
    table = Table(box=box.SIMPLE)
    table.add_column("model", no_wrap=True)
    table.add_column("tokens", justify="right")
    table.add_column("chars/token", justify="right")
    table.add_column("bytes/token", justify="right")

    for result in results:
        table.add_row(
            result.model_name,
            str(result.token_count),
            f"{result.chars_per_token:.4f}",
            f"{result.bytes_per_token:.4f}",
        )

    return table


def build_benchmark_title(benchmark_name: str, results: list[CompressionResult]) -> str:
    # ---------------------------------------------------------
    # Build a compact separator title from benchmark metadata
    # shared across all tokenizer rows in the same group.
    # ---------------------------------------------------------
    result = results[0]

    return (
        f"{benchmark_name} | {result.genre} | {result.language} | "
        f"chars={result.char_count} | bytes={result.byte_count}"
    )


def group_results_by_benchmark(results: list[CompressionResult]) -> dict[str, list[CompressionResult]]:
    # ---------------------------------------------------------
    # Group compression results by benchmark while preserving
    # the original benchmark order from the result list.
    # ---------------------------------------------------------
    grouped_results: dict[str, list[CompressionResult]] = {}

    for result in results:
        grouped_results.setdefault(result.benchmark_name, []).append(result)

    return grouped_results


def print_report(tokenizers: list[tuple[str, PreTrainedTokenizerBase]], results: list[CompressionResult]) -> None:
    # ---------------------------------------------------------
    # Print the tokenizer summary and compression comparison
    # with compact separators between each table.
    # ---------------------------------------------------------
    console = Console()
    console.print(Rule("Tokenizer Vocabulary Size", style="dim"))
    console.print(build_vocab_table(tokenizers))

    for benchmark_name, benchmark_results in group_results_by_benchmark(results).items():
        console.print()
        console.print(Rule(build_benchmark_title(benchmark_name, benchmark_results), style="dim"))
        console.print(build_compression_table(benchmark_results))
