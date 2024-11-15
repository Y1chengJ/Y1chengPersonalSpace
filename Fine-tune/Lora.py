import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LoraLinear(nn.Module):
    def __init__(self, in_features, out_features, merge, rank, lora_alpha, dropout):
        super(LoraLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.merge = merge
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.dropout_rate = dropout

        self.linear = nn.Linear(in_features, out_features)  # This is the original linear layer (used to mimic the original layer)
        if rank > 0:
            self.lora_b = nn.Parameter(torch.zeros(out_features, rank))
            self.lora_a = nn.Parameter(torch.zeros(rank, in_features))
            self.scale = self.rank / self.lora_alpha
            self.linear.weight.requires_grad = False
        
        if self.dropout > 0:
            self.dropout = nn.Dropout(self.dropout_rate)
        else:
            self.dropout = nn.Identity()

    def initial_weights(self):
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5)) 
        nn.init.zeros_(self.lora_b)

    def forward(self, x):
        if self.rank > 0 and self.merge:
            output = F.linear(x, self.linear.weight + self.lora_b @ self.lora_a * self.scale, self.linear.bias)
        else:
            output = self.linear(x)
        output = self.dropout(output)
        return output


