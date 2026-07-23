"""Parametrised vector field on a Riemannian manifold (single chart)."""

from __future__ import annotations

import torch
import torch.nn as nn


class ManifoldVectorField(nn.Module):
    """MLP mapping ``(t, x) → f(t, x)`` with chart-coordinate outputs.

    Architecture: ``[dim+1 → hidden_dim (× n_layers) → dim]``.

    Parameters
    ----------
    dim : int
        Manifold dimension.
    hidden_dim : int
        Width of the hidden layers.
    n_layers : int
        Number of hidden layers.
    activation : type[nn.Module]
        Activation class.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int = 64,
        n_layers: int = 2,
        activation: type[nn.Module] = nn.SiLU,
    ):
        super().__init__()
        self.dim = dim

        layers: list[nn.Module] = [nn.Linear(dim + 1, hidden_dim), activation()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), activation()]
        layers.append(nn.Linear(hidden_dim, dim))
        self.net = nn.Sequential(*layers)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the vector field.

        Parameters
        ----------
        t : Tensor
            Time(s), broadcastable to ``(..., 1)`` or ``(...,)``.
        x : Tensor
            Chart coordinate(s), shape ``(..., dim)``.

        Returns
        -------
        f : Tensor
            Vector components, shape ``(..., dim)``.
        """
        if t.dim() == x.dim() - 1:
            t = t.unsqueeze(-1)
        tx = torch.cat([t, x], dim=-1)
        return self.net(tx)


def weight_decay_loss(vf: ManifoldVectorField) -> torch.Tensor:
    """Sum of squared parameters, for use as an explicit L2 weight-decay term.

    Complements (does not replace) optimizer-level ``weight_decay``; useful
    when the regularization coefficient needs to be scheduled/warmed-up
    alongside other loss terms (e.g. in :func:`cnf_nll`).
    """
    total = None
    for p in vf.parameters():
        term = (p * p).sum()
        total = term if total is None else total + term
    if total is None:
        return torch.tensor(0.0)
    return total


def lipschitz_regularizer(
    vf: ManifoldVectorField,
    x: torch.Tensor,
    t: torch.Tensor = 0.5,
    eps: float = 1e-3,
) -> torch.Tensor:
    """Rough spectral-norm-ish regulariser for the vector field.

    Penalises the Frobenius norm of the spatial Jacobian at a fixed time,
    encouraging smooth trajectories.
    """
    x.requires_grad_(True)
    t_ten = torch.full(x.shape[:-1], t, device=x.device, dtype=x.dtype)
    f = vf(t_ten, x)
    jac_norm = torch.zeros_like(x[..., 0])
    for i in range(vf.dim):
        (grad_i,) = torch.autograd.grad(f[..., i].sum(), x, retain_graph=True, create_graph=True)
        jac_norm = jac_norm + (grad_i * grad_i).sum(dim=-1)
    return jac_norm.mean()