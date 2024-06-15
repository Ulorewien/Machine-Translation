import torch
import torch.nn as nn
from util import *

class Encoder(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers
        self.layernorm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.layernorm(x)
    
class Decoder(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers
        self.layernorm = LayerNormalization()

    def forward(self, x, encoder_output, source_mask, target_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, source_mask, target_mask)
        return self.layernorm(x)
    
class Transformer(nn.Module):
    def __init__(self, encoder, decoder, source_embd, target_embd, source_pos, target_pos, projection_layer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.source_embd = source_embd
        self.target_embd = target_embd
        self.source_pos = source_pos
        self.target_pos = target_pos
        self.projection_layer = projection_layer

    def encode(self, x, source_mask):
        x = self.source_embd(x)
        x = self.source_pos(x)
        x = self.encoder(x)
        return x
    
    def decode(self, encoder_output, source_mask, x, target_mask):
        x = self.target_embd(x)
        x = self.target_pos(x)
        x = self.decoder(x, encoder_output, source_mask, target_mask)
        return x
    
    def project(self, x):
        return self.projection_layer(x)
    
def buildTransformer(source_vocab_size, target_vocab_size, source_seq_len, target_seq_len, d_model=512, n_layer=6, n_heads=8, dropout=0.5, d_ff=2048):
    source_embd = InputEmbedding(d_model, source_vocab_size)
    target_embd = InputEmbedding(d_model, target_vocab_size)

    source_pos = PositionalEncoding(d_model, source_seq_len, dropout)
    target_pos = PositionalEncoding(d_model, target_seq_len, dropout)

    encoder_blocks = list()
    for _ in range(n_layer):
        encoder_self_attention_blocks = MultiHeadAttention(d_model, n_heads, dropout)
        feed_forward_block = FeedForward(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_blocks, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    decoder_blocks = list()
    for _ in range(n_layer):
        decoder_self_attention_blocks = MultiHeadAttention(d_model, n_heads, dropout)
        decoder_cross_attention_blocks = MultiHeadAttention(d_model, n_heads, dropout)
        feed_forward_block = FeedForward(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_blocks, decoder_cross_attention_blocks, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    projection_layer = ProjectionLayer(d_model, target_vocab_size)

    transformer = Transformer(encoder, decoder, source_embd, target_embd, source_pos, target_pos, projection_layer)

    for parameter in transformer.parameters():
        if parameter.dim() > 1:
            nn.init.xavier_uniform_(parameter)

    return transformer
