"""Isolated dropout-p-range fixture: Dropout with p out of [0, 1)."""
import torch.nn as nn


class DropoutRangeBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(128, 64)
        # p=1.0 zeros the entire signal during training.
        self.drop = nn.Dropout(p=1.0)

    def forward(self, x):
        return self.drop(self.fc(x))
