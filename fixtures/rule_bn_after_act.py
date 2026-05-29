"""Isolated bn-after-activation fixture: norm wired after activation in forward."""
import torch.nn as nn


class BnAfterActBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        # BatchNorm SHOULD come before relu; here it's after.
        self.bn = nn.BatchNorm2d(64)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.bn(x)
        return x
