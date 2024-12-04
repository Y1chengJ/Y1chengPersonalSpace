import torch
from torch import nn
import torch.nn.functional as F
import math
from multihead_attention import multi_head_attention
from Transformer.Encoder import PositionalEmbedding, LayerNorm, PositionWiseFeedForward, TokenEmbedding

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, ffn_hidden, drop_prob):
        super(DecoderLayer, self).__init__()
        self.attention1 = multi_head_attention(d_model, n_head)
        self.norm1 = LayerNorm(d_model)
        self.drop1 = nn.Dropout(drop_prob)

        self.cross_attention = multi_head_attention(d_model, n_head)
        self.norm2 = LayerNorm(d_model)
        self.drop2 = nn.Dropout(drop_prob)

        self.ffn = PositionWiseFeedForward(d_model, ffn_hidden)
        self.norm3 = LayerNorm(d_model)
        self.drop3 = nn.Dropout(drop_prob)

    def forward(self, dec, enc, t_mask, s_mask):
        _x = dec
        # since we don't want to look into the future, we mask the upper diagonal part of the matrix
        x = self.attention1(dec, dec, dec, t_mask) 

        x = self.drop1(x)
        x = self.norm1(x + _x)

        if enc is not None:
            _x = x
            # s_mask is used to mask the padding tokens, since we don't want to care about them
            x = self.cross_attention(x, enc, enc, s_mask)
            x = self.drop2(x)
            x = self.norm2(x + _x)

        _x = x
        x = self.ffn(x)
        x = self.drop3(x)
        return x
    

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, n_layers, n_head, ffn_hidden, drop_prob, device):
        super(Decoder, self).__init__()
        self.embedding = TokenEmbedding(vocab_size, d_model)
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_head, ffn_hidden, drop_prob) for _ in range(n_layers)])
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x, enc, t_mask, s_mask):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, enc, t_mask, s_mask)
        x = self.fc(x)
        return x
