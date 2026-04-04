import torch

from transformer import DecoderOnlyTransformer
from tokenizer import tokenizer, vocabulary


prompt = "what is statquest <EOS> awesome"

model = DecoderOnlyTransformer(num_tokens=len(vocabulary), d_model=2, max_len=6)
model_input = tokenizer(words=prompt.split())
input_length = model_input.size(dim=0)

predictions = model(model_input)

print(predictions)
# predicted_id = torch.tensor([torch.argmax(predictions[-1, :])])
