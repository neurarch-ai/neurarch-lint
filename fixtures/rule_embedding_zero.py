"""Isolated embedding-zero-size fixture: Embedding with num_embeddings=0."""
import torch.nn as nn


class EmbeddingZeroBug(nn.Module):
    def __init__(self):
        super().__init__()
        # num_embeddings=0 -> a zero-row table; construction is meaningless.
        self.embed = nn.Embedding(num_embeddings=0, embedding_dim=128)

    def forward(self, x):
        return self.embed(x)
