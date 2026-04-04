import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, d_model=2):
        super().__init__()
        
        self.W_q = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.W_k = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.W_v = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        
        self.row_dim = 0
        self.col_dim = 1
        
    def forward(self, encoding_for_q: torch.Tensor, encoding_for_k: torch.Tensor, encoding_for_v: torch.Tensor, mask=None):
        q: torch.Tensor = self.W_q(encoding_for_q)
        k: torch.Tensor = self.W_k(encoding_for_k)
        v: torch.Tensor = self.W_v(encoding_for_v)
        
        sims = torch.matmul(q, k.transpose(dim0=self.row_dim, dim1=self.col_dim))
        scaled_sims = sims / torch.tensor(k.size(self.col_dim)**0.5)
        
        if mask is not None:
            scaled_sims = scaled_sims.masked_fill(mask=mask, value=-1e9)
            
        attention_percents = F.softmax(scaled_sims, dim=self.col_dim)
        attention_scores = torch.matmul(attention_percents, v)
        
        return attention_scores
    