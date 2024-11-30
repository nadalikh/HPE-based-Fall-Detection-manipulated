import math
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from typing import Tuple, List


def get_positional_encoding(seq_length: int, d_model: int) -> torch.Tensor:
    pos = torch.arange(seq_length).unsqueeze(1)
    i = torch.arange(d_model).unsqueeze(0)
    angle_rates = 1 / torch.pow(10000, (2 * (i // 2)) / d_model).float()
    angle_rads = pos * angle_rates

    sines = torch.sin(angle_rads[:, 0::2])
    cosines = torch.cos(angle_rads[:, 1::2])

    pos_encoding = torch.empty((seq_length, d_model))
    pos_encoding[:, 0::2] = sines
    pos_encoding[:, 1::2] = cosines

    return pos_encoding.unsqueeze(0)


class ProbSparseAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super(ProbSparseAttention, self).__init__()
        self.d_model = d_model
        self.nhead = nhead

    def forward(self, query, key, value, mask=None):
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_model)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Select top-k sparse scores (ProbSparse logic)
        k = max(1, int(scores.size(-1) * 0.1))  # Keep top 10%
        top_scores, top_indices = scores.topk(k, dim=-1)

        # Create sparse attention weights
        sparse_attn_weights = torch.zeros_like(scores)
        batch_indices = torch.arange(scores.size(0)).unsqueeze(-1).unsqueeze(-1).expand_as(top_indices)
        sparse_attn_weights[batch_indices, torch.arange(scores.size(1)).unsqueeze(-1), top_indices] = top_scores

        # Normalize sparse attention weights
        attn_weights = torch.softmax(sparse_attn_weights, dim=-1)

        # Perform attention using sparse weights
        return torch.matmul(attn_weights, value)


class InformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_ff, dropout):
        super(InformerEncoderLayer, self).__init__()
        self.attention = ProbSparseAttention(d_model, nhead)
        self.linear1 = nn.Linear(d_model, dim_ff)
        self.linear2 = nn.Linear(dim_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output = self.attention(x, x, x, mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        feedforward_output = self.linear2(torch.relu(self.linear1(x)))
        x = x + self.dropout(feedforward_output)
        x = self.norm2(x)
        return x


class Informer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        num_features: int,
        dropout: float = 0.1,
        dim_ff: int = 2048,
        num_classes: int = 2,
    ):
        super(Informer, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.num_features = num_features
        self.num_classes = num_classes

        self.pos_encoding = get_positional_encoding(1000, d_model)
        self.encoder = nn.Linear(num_features, d_model)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [
                InformerEncoderLayer(d_model, nhead, dim_ff, dropout)
                for _ in range(num_layers)
            ]
        )

        self.decoder = nn.Linear(self.d_model, self.num_classes)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = x.permute(1, 0, 2)
        x = self.encoder(x) * math.sqrt(self.d_model)

        x += self.pos_encoding[:, : x.size(1), :].type_as(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, mask)

        x = x.permute(1, 0, 2)
        x = self.decoder(x[:, -1, :])
        return x
