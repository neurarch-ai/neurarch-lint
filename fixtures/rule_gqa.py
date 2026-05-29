"""Isolated gqa-head-divisibility fixture."""
import torch.nn as nn


class GqaBug(nn.Module):
    def __init__(self):
        super().__init__()
        # 32 % 7 != 0
        self.gqa = nn.GroupedQueryAttention(
            num_heads=32,
            num_kv_heads=7,
        )

    def forward(self, x):
        return self.gqa(x)
