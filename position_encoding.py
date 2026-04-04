import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader

import lightning as L


class PositionEncoding(nn.Module):
    def __init__(self, d_model=2, max_len=6):
        # d_model: dimention of the model, the number of word embedding value per token
        # max_len: maximum number of tokens our Transformer can process
    
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        
        position = torch.arange(start=0, end=max_len, step=1).float().unsqueeze(1)
        embedding_index = torch.arange(start=0, end=d_model, step=2).float()
        
        div_term = 1 / torch.tensor(10000.0) ** (embedding_index / d_model)
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
        
    def forward(self, word_embeddings: torch.Tensor):
        return word_embeddings + self.pe[:word_embeddings.size(0), :]
        
        
if __name__ == "__main__":
    n = PositionEncoding()