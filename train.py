import torch
from torch.utils.data import DataLoader
import lightning as L
from pathlib import Path

from transformer import DecoderOnlyTransformer
from tokenizer import Tokenizer
from dataset import get_dataset


sentences = [
    "what is statquest <EOS> awesome",
    "statquest is what <EOS> great",
]
max_len = 6
num_layers = 2
num_heads = 1
d_ff = 8
model_dir = Path(__file__).with_name("model")
model_dir.mkdir(exist_ok=True)

tokenizer = Tokenizer()
tokenizer.learn_vocab(sentences)

dataset = get_dataset(sentences=sentences, tokenizer=tokenizer.tokenizer)
dataloader = DataLoader(dataset)

model = DecoderOnlyTransformer(
    num_tokens=len(tokenizer.vocabulary),
    d_model=2,
    max_len=max_len,
    num_layers=num_layers,
    num_heads=num_heads,
    d_ff=d_ff,
)

trainer = L.Trainer(max_epochs=30)
trainer.fit(model, train_dataloaders=dataloader)

torch.save(model.state_dict(), model_dir / "model.pth")
tokenizer.save(model_dir / "tokenizer.json")
