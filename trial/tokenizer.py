from transformers import AutoTokenizer


# ---------------------------------------------------------
# Load the tokenizer trained in this project from the local
# Hugging Face compatible tokenizer directory.
# ---------------------------------------------------------
tokenizer_my_llm = AutoTokenizer.from_pretrained("models/tokenizer")
tokenizer_qwen_3_5_0_8b = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
tokenizer_gemma_3_270m = AutoTokenizer.from_pretrained("google/gemma-3-270m")
tokenizer_lfm_2_5_350m = AutoTokenizer.from_pretrained("LiquidAI/LFM2.5-350M")

# ---------------------------------------------------------
# Print each tokenizer vocabulary size with its model name.
# ---------------------------------------------------------
print(f"my-llm tokenizer vocab_size: {tokenizer_my_llm.vocab_size}")
print(f"qwen3.5-0.8b tokenizer vocab_size: {tokenizer_qwen_3_5_0_8b.vocab_size}")
print(f"gemma-3-270m tokenizer vocab_size: {tokenizer_gemma_3_270m.vocab_size}")
print(f"lfm2.5-350m tokenizer vocab_size: {tokenizer_lfm_2_5_350m.vocab_size}")
