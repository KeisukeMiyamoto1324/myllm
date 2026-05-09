from transformers import AutoTokenizer




# ---------------------------------------------------------
# Load the tokenizer trained in this project from the local
# Hugging Face compatible tokenizer directory.
# ---------------------------------------------------------
tokenizer_my_llm = AutoTokenizer.from_pretrained("models/tokenizer")

# ---------------------------------------------------------
# Print each tokenizer vocabulary size with its model name.
# ---------------------------------------------------------
print(f"my-llm tokenizer vocab_size: {tokenizer_my_llm.vocab_size}")
