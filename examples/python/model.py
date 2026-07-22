import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class CNFNet(nn.Module):
    def __init__(self, dim: int = 2, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            spectral_norm(nn.Linear(dim + 1, hidden)),
            nn.Tanh(),
            spectral_norm(nn.Linear(hidden, hidden)),
            nn.Tanh(),
            spectral_norm(nn.Linear(hidden, dim)),
        )

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if t.dim() == 0:
            t = t.expand(x.shape[0], 1)
        elif t.dim() == 1:
            t = t.unsqueeze(-1)
        tx = torch.cat([t, x], dim=-1)
        return self.net(tx)