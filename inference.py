import json
from pathlib import Path

import torch

from device_utils import resolve_device
from tokenizer_rust.tokenizer import ByteLevelBPE
from transformer import DecoderOnlyTransformer


# prompt = "what is statquest <EOS>"
prompt = "Dogs are"

model_dir = Path(__file__).with_name("model")
model_path = model_dir / "model.pth"
device = resolve_device()

# ---------------------------------------------------------
# Load the tokenizer and the saved model configuration so
# inference uses the same architecture as training.
# ---------------------------------------------------------
tokenizer = ByteLevelBPE.load(model_dir / "tokenizer.json")

with open(model_dir / "model_config.json") as f:
    model_config = json.load(f)

# ---------------------------------------------------------
# Rebuild the Transformer with the saved hyper-parameters
# before restoring the trained weights.
# ---------------------------------------------------------
model = DecoderOnlyTransformer(
    num_tokens=tokenizer.get_vocab_size(),
    d_model=model_config["d_model"],
    max_len=model_config["max_len"],
    num_layers=model_config["num_layers"],
    num_heads=model_config["num_heads"],
    d_ff=model_config["d_ff"],
    learning_rate=model_config["learning_rate"],
    pad_token_id=model_config["pad_token_id"],
)

state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# ---------------------------------------------------------
# Encode the prompt and keep the saved sequence length as the
# upper limit for autoregressive token generation.
# ---------------------------------------------------------
prompt_token_ids = tokenizer.tokenize(sentence=prompt)
model_input = torch.tensor(prompt_token_ids, dtype=torch.long).unsqueeze(0).to(device)
input_length = model_input.size(dim=1)
max_len = model_config["max_len"]
eos_token_id = tokenizer.token_to_id(tokenizer.eos_token)

# ---------------------------------------------------------
# Generate one token at a time until EOS appears or the saved
# maximum sequence length is reached.
# ---------------------------------------------------------
with torch.no_grad():
    predictions = model(model_input)
    predicted_id = torch.argmax(predictions[:, -1, :], dim=-1, keepdim=True)
    predicted_ids = predicted_id

    for _ in range(input_length, max_len):
        if predicted_id.item() == eos_token_id:
            break

        model_input = torch.cat((model_input, predicted_id), dim=1)
        predictions = model(model_input)
        predicted_id = torch.argmax(predictions[:, -1, :], dim=-1, keepdim=True)
        predicted_ids = torch.cat((predicted_ids, predicted_id), dim=1)

generated_token_ids = predicted_ids.squeeze(0).detach().cpu().tolist()
print(f"predicted tokens: {tokenizer.detokenize(token_ids=generated_token_ids)}")
