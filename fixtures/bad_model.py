"""Test fixture for neurarch-lint: deliberate bugs across the v1 rules."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BrokenAttention(nn.Module):
    def __init__(self):
        super().__init__()
        # BUG 1: head_dim non-integer (384 / 5 = 76.8)
        self.attn = nn.MultiheadAttention(embed_dim=384, num_heads=5)
        # BUG 2: GQA ratio invalid (32 % 7 != 0)
        self.gqa = nn.GroupedQueryAttention(
            num_heads=32,
            num_kv_heads=7,
        )
        self.classifier = nn.Linear(384, 10)
        # BUG 3: Softmax + CrossEntropyLoss double-apply (Softmax below + CE later)
        self.softmax = nn.Softmax(dim=-1)

        # BUG 4: BatchNorm placed AFTER an activation in forward (R-bn-after-activation)
        self.conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(64)

    def forward(self, x):
        # Adjacent activation -> normalization. Norm should be BEFORE relu.
        x = self.conv(x)
        x = self.relu(x)
        x = self.bn(x)

        x, _ = self.attn(x, x, x)
        x = self.gqa(x)
        x = self.classifier(x)
        return self.softmax(x)


loss_fn = nn.CrossEntropyLoss()
