"""Test fixture for neurarch-lint — clean model, should produce zero findings."""
import torch.nn as nn


class CleanAttention(nn.Module):
    def __init__(self):
        super().__init__()
        # head_dim = 384 / 6 = 64, integer
        self.attn = nn.MultiheadAttention(embed_dim=384, num_heads=6)
        # GQA ratio 32 / 8 = 4
        self.gqa = nn.GroupedQueryAttention(num_heads=32, num_kv_heads=8)
        self.classifier = nn.Linear(384, 10)

    def forward(self, x):
        x, _ = self.attn(x, x, x)
        x = self.gqa(x)
        return self.classifier(x)


loss_fn = nn.CrossEntropyLoss()
