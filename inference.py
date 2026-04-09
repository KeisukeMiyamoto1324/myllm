import torch
from pathlib import Path

from transformer import DecoderOnlyTransformer
from tokenizer import Tokenizer


prompt = "statquest is what <EOS>"
max_len = 6

model_dir = Path(__file__).with_name("model")
model_path = model_dir / "model.pth"
tokenizer = Tokenizer.load(model_dir / "tokenizer.json")

model = DecoderOnlyTransformer(num_tokens=len(tokenizer.vocabulary), d_model=2, max_len=max_len)

state_dict = torch.load(model_path, map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

model_input = tokenizer.tokenizer(words=prompt).unsqueeze(0)
input_length = model_input.size(dim=1)
eos_token_id = tokenizer.tokenizer("<EOS>").item()

with torch.no_grad():
    predictions = model(model_input)
    predicted_id = torch.argmax(predictions[:, -1, :], dim=-1, keepdim=True)
    predicted_ids = predicted_id

    for i in range(input_length, max_len):
        if predicted_id.item() == eos_token_id:
            break
        
        model_input = torch.cat((model_input, predicted_id), dim=1)
        
        predictions = model(model_input)
        predicted_id = torch.argmax(predictions[:, -1, :], dim=-1, keepdim=True)
        predicted_ids = torch.cat((predicted_ids, predicted_id), dim=1)

print(f"predicted tokens: {tokenizer.detokenizer(tokens=predicted_ids.squeeze(0))}")
