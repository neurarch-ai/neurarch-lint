"""Isolated linear-bias-before-norm fixture.

A Conv2d with explicit bias=True sits directly before a BatchNorm2d inside an
nn.Sequential, so the bias is redundant (BatchNorm has its own beta).
"""
import torch.nn as nn


class BiasBeforeNormBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, bias=True),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)
