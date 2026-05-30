"""Fixture: nn.BCELoss with no Sigmoid in the file (bceloss-without-sigmoid)."""
import torch.nn as nn


class BinaryHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        # No Sigmoid: this returns raw logits, which BCELoss cannot consume.
        return self.fc(x)


loss_fn = nn.BCELoss()
