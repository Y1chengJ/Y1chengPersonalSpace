import torch
from torch import nn
import torch.nn.functional as F
import math
from multihead_attention import multi_head_attention

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len, device):
        super(PositionalEmbedding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False

        pos = torch.arrange(0, max_len, device)
        pos = pos.float().unsqueeze(dim=1)
        _2i = torch.arrange(0, d_model, step=2, device=device).float()

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    def forward(self, x):
        seq_len = x.shape[1]
        return self.encoding[:seq_len, :]
    
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super(LayerNorm, self).__init__()

        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, unbiased=False, keepdim=True)
        out = (x - mean) / torch.sqrt(std + self.eps)
        out = self.gamma * out + self.beta  

        return out
    

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(0.1)  

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x
    

class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, d_model):
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)


class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, drop_prob, device):
        super(TransformerEmbedding, self).__init__()
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.position_embedding = PositionalEmbedding(d_model, max_len, device)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x):
        x = self.token_embedding(x)
        x = x + self.position_embedding(x)
        x = self.dropout(x)
        return x
    

class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = multi_head_attention(d_model, n_head)
        self.norm1 = LayerNorm(d_model)
        self.drop1 = nn.Dropout(drop_prob)
        self.ffn = PositionWiseFeedForward(d_model, ffn_hidden, drop_prob)
        self.norm2 = LayerNorm(d_model)
        self.drop2 = nn.Dropout(drop_prob)

    def forward(self, x, mask):
        _x = x
        x = self.attention(x, x, x, mask)
        x = self.drop1(x)
        x = self.norm1(x + _x)

        _x = x
        x = self.ffn(x)
        x = self.drop2(x)
        x = self.norm2(x + _x)

        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, ffn_hidden, n_head, n_layers, drop_prob, device):
        super(Encoder, self).__init__()
        self.embedding = TransformerEmbedding(vocab_size, d_model, max_len, drop_prob, device)
        self.layers = nn.ModuleList([EncoderLayer(d_model, ffn_hidden, n_head, drop_prob) for _ in range(n_layers)])

    def forward(self, x, mask):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x