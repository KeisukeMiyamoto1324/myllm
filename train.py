from torch.utils.data import DataLoader

from transformer import DecoderOnlyTransformer
from tokenizer import tokenizer
from dataset import get_dataset


sentences = [
    "what is statquest <EOS> awesome",
    "statquest is what <EOS> awesome",
]

dataset = get_dataset(sentences=sentences, tokenizer=tokenizer)
dataloader = DataLoader(dataset)

