"""Fixture for the deep-no-residual rule. 10 linear layers, no skip add."""
import torch.nn as nn
import torch.nn.functional as F


class DeepMLP(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.fc3 = nn.Linear(dim, dim)
        self.fc4 = nn.Linear(dim, dim)
        self.fc5 = nn.Linear(dim, dim)
        self.fc6 = nn.Linear(dim, dim)
        self.fc7 = nn.Linear(dim, dim)
        self.fc8 = nn.Linear(dim, dim)
        self.fc9 = nn.Linear(dim, dim)
        self.classifier = nn.Linear(dim, 10)

    def forward(self, x):
        # No x = x + ... anywhere. Gradient signal degrades through depth.
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = F.relu(self.fc8(x))
        x = F.relu(self.fc9(x))
        return self.classifier(x)
