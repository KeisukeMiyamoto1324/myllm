from tokenizer import get_token_map, tokenizer
from dataset import get_dataset
from torch.utils.data import DataLoader


words = ["what", "is", "statquest", "awesome", "<EOS>"]
token_map = get_token_map(words=words)

sentences = [
    "what is statquest <EOS> awesome",
    "statquest is what <EOS> awesome",
]

dataset = get_dataset(sentences=sentences, token_map=token_map, tokenizer=tokenizer)
dataloader = DataLoader(dataset)

