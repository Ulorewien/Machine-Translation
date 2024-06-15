import math
import torch
import torch.nn as nn
from pathlib import Path

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

class LayerNormalization(nn.Module):
    def __init__(self, eps=10e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return (self.alpha * (x - mean) / (std + self.eps)) + self.bias
    
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.5):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.5):
        super().__init__()
        assert d_model % num_heads == 0, "The embedding dimension should be divisible by the number of heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_per_head = d_model // num_heads

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def calculate_attention(q, k, v, mask=None, dropout_layer=None):
        d_per_head = q.shape[-1]

        attention_scores = q @ k.transpose(-2, -1)
        attention_scores = attention_scores * (d_per_head**(-0.5))

        if mask is not None:
            attention_scores.masked_fill_(mask == 0, "-inf")
        
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout_layer is not None:
            attention_scores = dropout_layer(attention_scores)

        ouptut = attention_scores @ v

        return ouptut, attention_scores

    def forward(self, q, k, v, mask):
        query = self.query(q)
        key = self.key(k)
        value = self.value(v)

        query = query.view(query.shape[0], query.shape[1], self.num_heads, self.d_per_head).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.num_heads, self.d_per_head).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.num_heads, self.d_per_head).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttention.calculate_attention(query, key, value, mask, self.dropout)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.num_heads * self.d_per_head)
        x = self.projection(x)

        return x
    
class ResidualBlock(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.layernorm = LayerNormalization()

    def forward(self, x, layer):
        return x + self.dropout(layer(self.layernorm(x)))

class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block, feed_forward_block, dropout=0.5):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_block = nn.ModuleList([ResidualBlock(dropout) for _ in range(2)])

    def forward(self, x, source_mask):
        x = self.residual_block[0](x, lambda x: self.self_attention_block(x, x, x, source_mask))
        x = self.residual_block[1](x, self.feed_forward_block(x))
        return x

class DecoderBlock(nn.Module):
    def __init__(self, self_attention, cross_attention, feed_forward, dropout=0.5):
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.feed_forward = feed_forward
        self.residual_connection = nn.ModuleList([ResidualBlock(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, source_mask, target_mask):
        x = self.residual_connection[0](x, lambda x: self.self_attention(x, x, x, target_mask))
        x = self.residual_connection[1](x, lambda x: self.cross_attention(x, encoder_output, encoder_output, source_mask))
        x = self.residual_connection[2](x, self.feed_forward)
        return x

class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.projection = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.projection(x)
        x = torch.log_softmax(x, dim=-1)
        return x
        
def get_weights_file_path(config, epoch):
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path(".") / model_folder / model_filename)