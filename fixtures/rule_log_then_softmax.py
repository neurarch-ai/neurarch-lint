"""Fixture: manual log on a softmax output (log-then-softmax)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(128, 10)

    def forward(self, x):
        logits = self.head(x)
        # Numerically unstable: log(softmax(x)) instead of F.log_softmax.
        return torch.log(F.softmax(logits, dim=-1))
