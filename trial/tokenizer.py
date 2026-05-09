# !pip install -U transformers
# !huggingface-cli login
from transformers import AutoTokenizer

# llama2
# tokenizer_llama2 = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

# 日本語LLM
tokenizer_elyza = AutoTokenizer.from_pretrained("elyza/ELYZA-japanese-Llama-2-7b-fast-instruct")
tokenizer_llmjp = AutoTokenizer.from_pretrained("llm-jp/llm-jp-13b-v1.0")

# cohere
tokenizer_aya = AutoTokenizer.from_pretrained("CohereForAI/aya-101")
tokenizer_commandr = AutoTokenizer.from_pretrained("CohereForAI/c4ai-command-r-plus")

# ---------------------------------------------------------
# Print each tokenizer vocabulary size with its model name.
# ---------------------------------------------------------
# print(f"llama2 tokenizer vocab_size: {tokenizer_llama2.vocab_size}")
print(f"elyza tokenizer vocab_size: {tokenizer_elyza.vocab_size}")
print(f"llm-jp tokenizer vocab_size: {tokenizer_llmjp.vocab_size}")
print(f"aya tokenizer vocab_size: {tokenizer_aya.vocab_size}")
print(f"command-r-plus tokenizer vocab_size: {tokenizer_commandr.vocab_size}")
