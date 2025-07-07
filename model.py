import torch.nn as nn
import torch

class TagMLP(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.importance = nn.Embedding(vocab_size, 1)  # Learnable importance for each tag

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, tag_indices):
        tag_embeds = self.embedding(tag_indices)                     # [batch, seq, dim]
        tag_weights = self.importance(tag_indices).squeeze(-1)       # [batch, seq]

        weighted_sum = (tag_embeds * tag_weights.unsqueeze(-1)).sum(dim=1)
        weight_norm = tag_weights.sum(dim=1, keepdim=True) + 1e-8
        pooled = weighted_sum / weight_norm                         # [batch, dim]

        return self.mlp(pooled)                                     # [batch, 1]
