from transformers import AutoTokenizer

from tokenizer_data import load_wikipedia_texts
from tokenizer_metrics import measure_compression
from tokenizer_report import print_report


WIKIPEDIA_ARTICLE_COUNT = 100
WIKIPEDIA_MAX_CHARS = 1024


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
    # Load the English and Japanese Wikipedia article samples
    # used by every tokenizer in the comparison.
    # ---------------------------------------------------------
    texts_en = load_wikipedia_texts("20231101.en", WIKIPEDIA_ARTICLE_COUNT, WIKIPEDIA_MAX_CHARS)
    texts_ja = load_wikipedia_texts("20231101.ja", WIKIPEDIA_ARTICLE_COUNT, WIKIPEDIA_MAX_CHARS)

    # ---------------------------------------------------------
    # Measure compression for every tokenizer and language pair.
    # ---------------------------------------------------------
    results = [
        measure_compression(model_name, "en", tokenizer, texts_en)
        for model_name, tokenizer in tokenizers
    ] + [
        measure_compression(model_name, "ja", tokenizer, texts_ja)
        for model_name, tokenizer in tokenizers
    ]

    # ---------------------------------------------------------
    # Print the tokenizer summary and compression comparison.
    # ---------------------------------------------------------
    print_report(tokenizers, results)


if __name__ == "__main__":
    main()
