import torch
from torch.utils.data import DataLoader
import lightning as L

from transformer import DecoderOnlyTransformer
from tokenizer import tokenizer, vocabulary
from dataset import get_dataset


sentences = [
    "what is statquest <EOS> awesome",
    "statquest is what <EOS> awesome",
]
max_len = 6

dataset = get_dataset(sentences=sentences, tokenizer=tokenizer)
dataloader = DataLoader(dataset)

model = DecoderOnlyTransformer(num_tokens=len(vocabulary), d_model=2, max_len=max_len)

trainer = L.Trainer(max_epochs=30)
trainer.fit(model, train_dataloaders=dataloader)

torch.save(model.state_dict(), "model/model.pth")
