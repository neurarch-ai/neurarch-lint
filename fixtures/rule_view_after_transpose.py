"""Fixture: view() chained on a transpose result (view-after-transpose)."""
import torch.nn as nn


class Reshaper(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(64, 64)

    def forward(self, x):
        x = self.proj(x)
        # transpose makes x non-contiguous; .view() will raise at runtime.
        return x.transpose(1, 2).view(x.size(0), -1)
