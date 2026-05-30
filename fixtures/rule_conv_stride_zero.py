"""Isolated conv-stride-zero fixture: Conv2d with stride=0."""
import torch.nn as nn


class ConvStrideZeroBug(nn.Module):
    def __init__(self):
        super().__init__()
        # stride=0 divides by zero in the output-size formula -> runtime error.
        self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=0)

    def forward(self, x):
        return self.conv(x)
