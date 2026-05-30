"""Isolated softmax-no-dim fixture: nn.Softmax with no explicit dim argument."""
import torch.nn as nn


class SoftmaxNoDimBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(128, 10)
        # No dim= passed: the implicit dimension is ambiguous and deprecated.
        self.softmax = nn.Softmax()

    def forward(self, x):
        return self.softmax(self.fc(x))
