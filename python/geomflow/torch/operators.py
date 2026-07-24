"""Differential-geometric operators on a single coordinate chart.

All functions assume the metric is an :class:`AnalyticMetric` and return
tensors compatible with ``torch.autograd``.

Points and contravariant tangent vectors use a final coordinate dimension.
Metrics use ``g[..., i, j] = g_ij``, metric derivatives use
``dg[..., i, j, k] = partial_k g_ij``, and Christoffel symbols use
``Gamma[..., k, i, j] = Gamma^k_ij``.
"""

from __future__ import annotations

from typing import Callable

import torch


def _coordinate_derivative(output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Differentiate a scalar batch output, including constant outputs."""
    if output.requires_grad:
        (result,) = torch.autograd.grad(
            output.sum(),
            x,
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )
        if result is not None:
            return result
    return torch.zeros_like(x)


def christoffel(
    metric: "AnalyticMetric",  # type: ignore[name-defined]  # noqa: F821
    x: torch.Tensor,
) -> torch.Tensor:
    """Levi-Civita Christoffel symbols of the second kind.

    Parameters
    ----------
    metric : AnalyticMetric
        Metric object providing ``.inverse(x)``, ``.derivative(x)``.
    x : Tensor
        Point(s) in chart coordinates, shape ``(..., dim)`` with
        ``requires_grad=True``.

    Returns
    -------
    Gamma : Tensor
        Shape ``(..., dim, dim, dim)``.  ``Gamma[..., k, i, j] = Γ^k_ij``.
    """
    dim = metric.dim
    g_inv = metric.inverse(x)  # (..., dim, dim)
    dg = metric.derivative(x)  # (..., dim, dim, dim)

    Gamma = torch.zeros(*x.shape[:-1], dim, dim, dim, device=x.device, dtype=x.dtype)
    for k in range(dim):
        for i in range(dim):
            for j in range(dim):
                s = torch.zeros_like(x[..., 0])
                for l in range(dim):
                    s = s + g_inv[..., k, l] * (
                        dg[..., j, l, i] + dg[..., i, l, j] - dg[..., i, j, l]
                    )
                Gamma[..., k, i, j] = 0.5 * s
    return Gamma


def divergence(
    vf: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    metric: "AnalyticMetric",  # type: ignore[name-defined]  # noqa: F821
) -> torch.Tensor:
    """Divergence of a vector field `vf` on the manifold.

    Uses the coordinate formula:

        div V = (1/√|g|) ∂_i (√|g| V^i)

    which is coordinate-invariant.

    Parameters
    ----------
    vf : callable
        Function that maps ``(..., dim)`` to ``(..., dim)``.
    x : Tensor
        Field point(s), shape ``(..., dim)`` with ``requires_grad=True``.
    metric : AnalyticMetric
        The analytic metric object.

    Returns
    -------
    div : Tensor
        Scalar divergence, shape ``(...,)``.
    """
    dim = metric.dim
    sqrtg = metric.sqrt_det(x)  # (...,)
    V = vf(x)  # (..., dim)
    if V.shape != x.shape:
        raise ValueError(f"vf returned shape {V.shape}; expected {x.shape}")

    div = torch.zeros_like(sqrtg)
    for i in range(dim):
        sqrtg_Vi = sqrtg * V[..., i]
        d_i = _coordinate_derivative(sqrtg_Vi, x)
        div = div + d_i[..., i]
    return div / sqrtg


def gradient(
    scalar_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    metric: "AnalyticMetric",  # type: ignore[name-defined]  # noqa: F821
) -> torch.Tensor:
    """Riemannian gradient of a scalar function.

    grad h = g^{ij} ∂_j h

    Parameters
    ----------
    scalar_fn : callable
        Maps ``(..., dim)`` to ``(...,)``.
    x : Tensor
        Field point(s), shape ``(..., dim)`` with ``requires_grad=True``.
    metric : AnalyticMetric
        The analytic metric object.

    Returns
    -------
    result : Tensor
        Gradient vector components, shape ``(..., dim)``.
    """
    h = scalar_fn(x)  # (...,)
    if h.shape != x.shape[:-1]:
        raise ValueError(f"scalar_fn returned shape {h.shape}; expected {x.shape[:-1]}")
    dh = _coordinate_derivative(h, x)
    g_inv = metric.inverse(x)  # (..., dim, dim)
    return (g_inv * dh.unsqueeze(-2)).sum(dim=-1)  # Einstein sum over j


def covariant_derivative_tensor(
    vf: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    metric: "AnalyticMetric",  # type: ignore[name-defined]  # noqa: F821
) -> torch.Tensor:
    """Covariant derivative of a vector field: ∇_j V^i.

    ∇_j V^i = ∂_j V^i + Γ^i_kj V^k

    Parameters
    ----------
    vf : callable
        Maps ``(..., dim)`` to ``(..., dim)``.
    x : Tensor
        Field point(s), shape ``(..., dim)``, requires_grad=True.
    metric : AnalyticMetric
        The analytic metric object.

    Returns
    -------
    nabla_V : Tensor
        Shape ``(..., dim, dim)`` where ``nabla_V[..., i, j] = ∇_j V^i``.
    """
    dim = metric.dim
    V = vf(x)  # (..., dim)
    if V.shape != x.shape:
        raise ValueError(f"vf returned shape {V.shape}; expected {x.shape}")
    Gamma = christoffel(metric, x)  # (..., dim, dim, dim)

    dV_rows: list[torch.Tensor] = []
    for i in range(dim):
        dV_rows.append(_coordinate_derivative(V[..., i], x))
    dV = torch.stack(dV_rows, dim=-2)

    Gamma_V = torch.einsum("...ikj,...k->...ij", Gamma, V)

    return dV + Gamma_V  # ∇_j V^i
