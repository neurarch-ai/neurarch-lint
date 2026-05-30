"""Isolated groupnorm-channel-divisibility fixture. Only this rule should fire."""
import torch.nn as nn


class GroupNormBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        # num_channels=16 is not divisible by num_groups=3 -> construction crash.
        self.gn = nn.GroupNorm(3, 16)

    def forward(self, x):
        return self.gn(self.conv(x))
