"""Fixture: ReLU directly before Softmax in a Sequential (relu-then-softmax)."""
import torch.nn as nn


# ReLU clamps the logits to non-negative, which distorts the softmax output.
classifier = nn.Sequential(
    nn.Linear(128, 10),
    nn.ReLU(),
    nn.Softmax(dim=-1),
)
