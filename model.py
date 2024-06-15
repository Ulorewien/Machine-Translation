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