"""The corrected version of buggy_transformer.py.

All four structural bugs are fixed, so `node lint.mjs examples/fixed_transformer.py`
reports zero findings and exits 0.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.act = nn.GELU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class EncoderBlock(nn.Module):
    def __init__(self, dim=400, num_heads=8, num_groups=8):
        super().__init__()
        # FIX: 400 / 8 = 50, an integer head_dim.
        self.attn = nn.MultiheadAttention(embed_dim=400, num_heads=8, batch_first=True)
        self.ff = FeedForward(dim, dim * 4)
        # FIX: 400 / 8 = 50, channels divisible by num_groups.
        self.norm = nn.GroupNorm(num_groups=8, num_channels=400)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = x + attn_out
        x = x + self.ff(x)
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        return x


class TicketClassifier(nn.Module):
    """Classifies a token sequence into 5 categories."""

    def __init__(self, vocab_size=30000, dim=400, num_classes=5):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.block = EncoderBlock(dim=dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, tokens):
        x = self.embed(tokens)
        x = self.block(x)
        pooled = x.mean(dim=1)
        # FIX: return raw logits. BCEWithLogitsLoss applies the sigmoid
        # internally, and the loss expects logits rather than probabilities,
        # so no explicit Sigmoid or Softmax is needed here.
        logits = self.head(pooled)
        return logits


loss_fn = nn.BCEWithLogitsLoss()
