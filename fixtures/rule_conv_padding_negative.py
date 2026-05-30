"""Fixture: Conv2d with negative padding (conv-padding-negative)."""
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # padding=-1 is invalid and raises at construction.
        self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=-1)

    def forward(self, x):
        return self.conv(x)
