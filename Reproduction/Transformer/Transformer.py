import torch
from torch import nn
import torch.nn.functional as F
import math
from multihead_attention import multi_head_attention
from Encoder import Encoder
from Decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, src_pad_idx, trg_pad_idx, src_vocab_size, trg_vocab_size, d_model, max_len, n_layers, n_head, ffn_hidden, drop_prob, device):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_pad_idx, src_vocab_size, d_model, max_len, n_layers, n_head, ffn_hidden, drop_prob, device)
        self.decoder = Decoder(trg_pad_idx, trg_vocab_size, d_model, max_len, n_layers, n_head, ffn_hidden, drop_prob, device)
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_casual_mask(self, q, k):
        len_q, len_k = q.size(1), k.size(1)
        mask = (torch.triu(torch.ones(len_q, len_k))).type(torch.BoolTensor).to(self.device)
        return mask
    
    def make_pad_mask(self, q, k, pad_idx_q, pad_idx_k):
        len_q, len_k = q.size(1), k.size(1)
        q = q.ne(pad_idx_q).unsqueeze(1).unsqueeze(3)
        q.repeat(1, 1, 1, len_k)

        k = k.ne(pad_idx_k).unsqueeze(1).unsqueeze(2)
        k.repeat(1, 1, len_q, 1)

        mask = q & k
        return mask
    
    def forward(self, src, trg):
        src_mask = self.make_pad_mask(src, src, self.src_pad_idx, self.src_pad_idx)
        trg_mask = self.make_casual_mask(trg, trg) * self.make_pad_mask(trg, trg, self.trg_pad_idx, self.trg_pad_idx)
        src_trg_mask = self.make_pad_mask(trg, src, self.trg_pad_idx, self.src_pad_idx)

        enc = self.encoder(src, src_mask)
        output = self.decoder(trg, enc, trg_mask, src_trg_mask)
        return output
