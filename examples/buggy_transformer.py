"""A small transformer encoder block for sequence classification.

Looks fine at a glance and imports cleanly, but it carries four structural
bugs that neurarch-lint catches before you spend a GPU hour finding them by
hand. Run `node lint.mjs examples/buggy_transformer.py` to see them.
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
    def __init__(self, dim=400, num_heads=6, num_groups=6):
        super().__init__()
        # BUG: embed_dim 400 is not divisible by num_heads 6 (head_dim 66.67).
        self.attn = nn.MultiheadAttention(embed_dim=400, num_heads=6, batch_first=True)
        self.ff = FeedForward(dim, dim * 4)
        # BUG: GroupNorm channels 400 not divisible by num_groups 6.
        self.norm = nn.GroupNorm(num_groups=6, num_channels=400)

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
        # BUG: Softmax with no explicit dim. The implicit dim is deprecated.
        self.softmax = nn.Softmax()
        # BUG: explicit Sigmoid gate double-applies with BCEWithLogitsLoss below.
        self.gate = nn.Sigmoid()

    def forward(self, tokens):
        x = self.embed(tokens)
        x = self.block(x)
        pooled = x.mean(dim=1)
        gated = self.gate(pooled)
        logits = self.head(gated)
        return self.softmax(logits)


loss_fn = nn.BCEWithLogitsLoss()
