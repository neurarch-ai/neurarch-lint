"""Isolated zero-features fixture: nn.Linear with a zero dim."""
import torch.nn as nn


class ZeroFeaturesBug(nn.Module):
    def __init__(self):
        super().__init__()
        # in_features=0 -> construction fails at runtime.
        self.fc = nn.Linear(in_features=0, out_features=10)

    def forward(self, x):
        return self.fc(x)
