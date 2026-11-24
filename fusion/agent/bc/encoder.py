"""State-only Gaussian action encoder used for Safe BC."""
import torch
from torch import nn

class StateEncoder(nn.Module):
    def __init__(self, hidden_dim=16, output_dim=16):
        super().__init__()
        self.output_dim = output_dim
        self.encoder = nn.Sequential(
            *[nn.Linear(35, hidden_dim),
              nn.ReLU(),
              nn.Linear(hidden_dim, hidden_dim),
              nn.ReLU(),
              nn.Linear(hidden_dim, output_dim*2)]
        )

    def forward(self, s):
        output = self.encoder(s)
        mu = output[:, :self.output_dim]
        log_std = output[:, self.output_dim:]

        return mu, torch.exp(log_std)
