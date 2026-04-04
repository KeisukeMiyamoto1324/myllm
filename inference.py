import torch
from pathlib import Path

from transformer import DecoderOnlyTransformer
from tokenizer import tokenizer, vocabulary, detokenizer


prompt = "what is statquest <EOS> awesome"
max_len = 6

model = DecoderOnlyTransformer(num_tokens=len(vocabulary), d_model=2, max_len=max_len)
model_path = Path(__file__).with_name("model") / "model.pth"

state_dict = torch.load(model_path, map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

model_input = tokenizer(words=prompt.split())
input_length = model_input.size(dim=0)

with torch.no_grad():
    predictions = model(model_input)
    predicted_id = torch.tensor([torch.argmax(predictions[-1, :])])
    predicted_ids = predicted_id

    for i in range(input_length, max_len):
        if predicted_id.item() == tokenizer(["<EOS>"]).item():
            break
        
        model_input = torch.cat((model_input, predicted_id))
        
        predictions = model(model_input)
        predicted_id = torch.tensor([torch.argmax(predictions[-1, :])])
        predicted_ids = torch.cat((predicted_ids, predicted_id))

print(f"predicted tokens: {detokenizer(tokens=predicted_ids)}")
