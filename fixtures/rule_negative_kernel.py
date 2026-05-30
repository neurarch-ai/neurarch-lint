"""Isolated negative-or-zero-kernel fixture: Conv2d with kernel_size=0."""
import torch.nn as nn


class ZeroKernelBug(nn.Module):
    def __init__(self):
        super().__init__()
        # kernel_size=0 is not a valid window -> construction fails.
        self.conv = nn.Conv2d(3, 16, kernel_size=0)

    def forward(self, x):
        return self.conv(x)
