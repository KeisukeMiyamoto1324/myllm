import torch

from transformer import DecoderOnlyTransformer
from tokenizer import tokenizer, vocabulary, detokenizer


prompt = "what is statquest <EOS> awesome"
max_len = 6

model = DecoderOnlyTransformer(num_tokens=len(vocabulary), d_model=2, max_len=max_len)
model_input = tokenizer(words=prompt.split())
input_length = model_input.size(dim=0)

predictions = model(model_input)
predicted_id = torch.tensor([torch.argmax(predictions[-1, :])])
predicted_ids = predicted_id

for i in range(input_length, max_len):
    if predicted_id == tokenizer(["<EOS>"]):
        break
    
    model_input = torch.cat((model_input, predicted_id))
    
    predictions = model(model_input)
    predicted_id = torch.tensor([torch.argmax(predictions[-1, :])])
    predicted_ids = torch.cat((predicted_ids, predicted_id))

print(f"predicted tokens: {detokenizer(tokens=predicted_ids)}")

