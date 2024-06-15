import torch
import torch.nn as nn
from util import EncoderBlock, LayerNormalization

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
    
def buildTransformer():
    