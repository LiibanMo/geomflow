"""Parametrised vector field on a Riemannian manifold (single chart)."""

from __future__ import annotations

import torch
import torch.nn as nn
import warnings

from .operators import covariant_derivative_tensor
from ._utils import validate_tensor_module_compatibility


_NONSMOOTH_ACTIVATIONS = (nn.ReLU, nn.ReLU6, nn.LeakyReLU, nn.PReLU, nn.RReLU, nn.Hardtanh)
_KNOWN_SMOOTH_ACTIVATIONS = (nn.Tanh, nn.Sigmoid, nn.SiLU, nn.Softplus, nn.GELU)


def _validate_activation(activation: type[nn.Module]) -> None:
    if issubclass(activation, _NONSMOOTH_ACTIVATIONS):
        raise ValueError(
            "activation must be twice differentiable for divergence gradients and "
            "the intrinsic adjoint; use SiLU, Tanh, Softplus, Sigmoid, or GELU"
        )
    if not issubclass(activation, _KNOWN_SMOOTH_ACTIVATIONS):
        warnings.warn(
            "geomflow cannot verify that this activation is twice differentiable; "
            "the caller must ensure the required smoothness",
            UserWarning,
            stacklevel=3,
        )


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
        periodic: bool = False,
    ):
        super().__init__()
        _validate_activation(activation)
        self.dim = dim
        self.periodic = periodic

        input_dim = 2 * dim + 1 if periodic else dim + 1
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), activation()]
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
        validate_tensor_module_compatibility(x, self, "ManifoldVectorField.forward")
        if t.device != x.device or t.dtype != x.dtype:
            raise ValueError(
                "ManifoldVectorField.forward: expected time "
                f"device={x.device}, dtype={x.dtype}; got "
                f"device={t.device}, dtype={t.dtype}"
            )
        if t.dim() == x.dim() - 1:
            t = t.unsqueeze(-1)
        coordinates = torch.cat([torch.sin(x), torch.cos(x)], dim=-1) if self.periodic else x
        tx = torch.cat([t, coordinates], dim=-1)
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
        raise ValueError("weight_decay_loss: module has no parameters")
    return total


def coordinate_jacobian_regularizer(
    vf: ManifoldVectorField,
    x: torch.Tensor,
    t: torch.Tensor = 0.5,
) -> torch.Tensor:
    """Penalise the coordinate Frobenius norm of the spatial Jacobian.

    This chart-dependent engineering penalty is not an intrinsic Lipschitz
    norm and is not an assumption of Mohamud's theorem.
    """
    x_for_grad = x if x.requires_grad else x.clone().requires_grad_(True)
    t_ten = torch.full(x.shape[:-1], t, device=x.device, dtype=x.dtype)
    f = vf(t_ten, x_for_grad)
    jac_norm = torch.zeros_like(x_for_grad[..., 0])
    for i in range(vf.dim):
        (grad_i,) = torch.autograd.grad(
            f[..., i].sum(), x_for_grad, retain_graph=True, create_graph=True
        )
        jac_norm = jac_norm + (grad_i * grad_i).sum(dim=-1)
    return jac_norm.mean()


def intrinsic_covariant_regularizer(
    vf: ManifoldVectorField,
    metric,
    x: torch.Tensor,
    t: torch.Tensor | float = 0.5,
) -> torch.Tensor:
    r"""Return the mean intrinsic squared norm ``||nabla f||_g^2``.

    The contraction is ``g_ij g^kl (nabla_k f^i) (nabla_l f^j)`` and is
    invariant under smooth chart transitions when the field representations
    and metric are compatible.
    """
    x_for_grad = x if x.requires_grad else x.clone().requires_grad_(True)
    t_tensor = torch.as_tensor(t, device=x.device, dtype=x.dtype)
    if t_tensor.dim() == 0:
        t_tensor = t_tensor.expand(x.shape[:-1])
    nabla = covariant_derivative_tensor(
        lambda point: vf(t_tensor, point), x_for_grad, metric
    )
    g = metric.metric(x_for_grad)
    g_inv = metric.inverse(x_for_grad)
    return torch.einsum("...ij,...kl,...ik,...jl->...", g, g_inv, nabla, nabla).mean()


def lipschitz_regularizer(
    vf: ManifoldVectorField,
    x: torch.Tensor,
    t: torch.Tensor = 0.5,
    eps: float = 1e-3,
) -> torch.Tensor:
    """Deprecated alias for :func:`coordinate_jacobian_regularizer`."""
    del eps
    warnings.warn(
        "lipschitz_regularizer is chart-dependent; use "
        "coordinate_jacobian_regularizer or intrinsic_covariant_regularizer",
        DeprecationWarning,
        stacklevel=2,
    )
    return coordinate_jacobian_regularizer(vf, x, t)
