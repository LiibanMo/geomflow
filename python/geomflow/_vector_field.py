import torch
import torch.nn as nn
from typing import List


class CNFVectorField(nn.Module):
    """PyTorch neural network modelling the vector field f(t, x) on a manifold.

    The network maps a time-augmented point [t; x_1; ...; x_D] on the manifold
    to a tangent vector at that point. This is the core NN that the C++ ODE
    solver integrates.

    Args:
        manifold_dim: Dimension of the manifold (e.g. 2 for R^2, 3 for R^3).
        hidden_dims: List of hidden layer widths (default [64, 64, 64]).
        activation: Activation function class (default nn.Tanh).
    """

    def __init__(
        self,
        manifold_dim: int,
        hidden_dims: List[int] | None = None,
        activation: type = nn.Tanh,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 64, 64]

        self.manifold_dim = manifold_dim
        input_dim = manifold_dim + 1  # +1 for time

        layers: List[nn.Module] = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(activation())
            in_dim = h
        layers.append(nn.Linear(in_dim, manifold_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the vector field f(t, x).

        Args:
            t: Scalar time value (shape: () or (1,)).
            x: Point on the manifold (shape: (..., D)).

        Returns:
            Tangent vector at x (same shape as x).
        """
        t_tensor = torch.as_tensor(t, dtype=x.dtype, device=x.device)
        if t_tensor.dim() == 0:
            t_tensor = t_tensor.expand(*x.shape[:-1], 1)
        elif t_tensor.dim() == 1 and t_tensor.shape[0] == 1:
            t_tensor = t_tensor.expand(*x.shape[:-1], 1)
        else:
            t_tensor = t_tensor.unsqueeze(-1)

        tx = torch.cat([t_tensor, x], dim=-1)
        return self.net(tx)

