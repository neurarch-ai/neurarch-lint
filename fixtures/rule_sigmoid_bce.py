"""Isolated sigmoid-bce-with-logits fixture: nn.Sigmoid + nn.BCEWithLogitsLoss in same file."""
import torch.nn as nn


class SigmoidBceBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(128, 1)
        # Explicit sigmoid in the head double-applies with BCEWithLogitsLoss.
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc(x))


loss_fn = nn.BCEWithLogitsLoss()
