"""Isolated softmax-cross-entropy fixture: nn.Softmax + nn.CrossEntropyLoss in same file."""
import torch.nn as nn


class SoftmaxCeBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(128, 10)
        # Explicit softmax in the head double-applies with CrossEntropyLoss.
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.fc(x))


loss_fn = nn.CrossEntropyLoss()
