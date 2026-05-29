"""Isolated head-dim-divisibility fixture. Only this rule should fire."""
import torch.nn as nn


class HeadDimBug(nn.Module):
    def __init__(self):
        super().__init__()
        # embed_dim=384, num_heads=5 -> head_dim=76.8 (non-integer)
        self.attn = nn.MultiheadAttention(embed_dim=384, num_heads=5)

    def forward(self, x):
        out, _ = self.attn(x, x, x)
        return out
