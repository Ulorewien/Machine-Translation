import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class InputEmbedding(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * self.d_model**(0.5)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len, dropout=0.5):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len

        positional_embedding = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        denominator = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        positional_embedding[:, 0::2] = torch.sin(position * denominator)
        positional_embedding[:, 1::2] = torch.cos(position * denominator)

        positional_embedding = positional_embedding.unsqueeze(0)
        
        self.register_buffer("positional_embedding", positional_embedding)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + (self.positional_embedding[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)
