from transformers import AutoTokenizer

from tokenizer_benchmark_cases import BENCHMARK_CASES
from tokenizer_data import load_benchmark_texts
from tokenizer_metrics import measure_compression
from tokenizer_report import print_report


def main() -> None:
    # ---------------------------------------------------------
    # Load the tokenizer trained in this project and comparison
    # tokenizers from Hugging Face model repositories.
    # ---------------------------------------------------------
    tokenizer_my_llm = AutoTokenizer.from_pretrained("models/tokenizer")
    tokenizer_qwen_3_5_0_8b = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
    tokenizer_gemma_3_270m = AutoTokenizer.from_pretrained("google/gemma-3-270m")
    tokenizer_lfm_2_5_350m = AutoTokenizer.from_pretrained("LiquidAI/LFM2.5-350M")

    # ---------------------------------------------------------
    # Group the explicitly loaded tokenizers for shared reporting
    # and compression measurement.
    # ---------------------------------------------------------
    tokenizers = [
        ("my-llm", tokenizer_my_llm),
        ("qwen3.5-0.8b", tokenizer_qwen_3_5_0_8b),
        ("gemma-3-270m", tokenizer_gemma_3_270m),
        ("lfm2.5-350m", tokenizer_lfm_2_5_350m),
    ]

    # ---------------------------------------------------------
    # Load every configured benchmark sample set once so all
    # tokenizers are measured against identical input text.
    # ---------------------------------------------------------
    benchmark_texts = [
        (benchmark_case, load_benchmark_texts(benchmark_case))
        for benchmark_case in BENCHMARK_CASES
    ]

    # ---------------------------------------------------------
    # Measure compression for every tokenizer and benchmark
    # case pair.
    # ---------------------------------------------------------
    results = [
        measure_compression(model_name, benchmark_case, tokenizer, texts)
        for model_name, tokenizer in tokenizers
        for benchmark_case, texts in benchmark_texts
    ]

    # ---------------------------------------------------------
    # Print the tokenizer summary and compression comparison.
    # ---------------------------------------------------------
    print_report(tokenizers, results)


if __name__ == "__main__":
    main()
