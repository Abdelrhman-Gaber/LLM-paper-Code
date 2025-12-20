# models.py
import torch.nn as nn


class SimpleNN(nn.Module):
    """
    Used for both LLM embeddings and tabular baselines (with different input_dim).
    """
    def __init__(self, input_dim: int, hidden_dim: int = 16, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).view(-1)
