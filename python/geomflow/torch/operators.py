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

from ._utils import batched_jacobian


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
    g_inv = metric.inverse(x)  # (..., dim, dim)
    dg = metric.derivative(x)  # (..., dim, dim, dim)
    first_kind = (
        dg.permute(*range(dg.ndim - 3), -1, -3, -2)
        + dg.transpose(-2, -1)
        - dg
    )
    return 0.5 * torch.einsum("...kl,...ijl->...kij", g_inv, first_kind)


def divergence(
    vf: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    metric: "AnalyticMetric",  # type: ignore[name-defined]  # noqa: F821
    *,
    divergence_mode: str = "exact",
    generator: torch.Generator | None = None,
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
    if divergence_mode not in {"exact", "hutchinson"}:
        raise ValueError("divergence_mode must be 'exact' or 'hutchinson'")

    if x.device.type == "cpu" and divergence_mode == "exact":
        sqrtg = metric.sqrt_det(x)
        value = vf(x)
        if value.shape != x.shape:
            raise ValueError(f"vf returned shape {value.shape}; expected {x.shape}")
        derivatives = [
            _coordinate_derivative(sqrtg * value[..., i], x)[..., i]
            for i in range(metric.dim)
        ]
        return torch.stack(derivatives, -1).sum(-1) / sqrtg

    def volume_weighted_field(point: torch.Tensor) -> torch.Tensor:
        value = vf(point)
        if value.shape != point.shape:
            raise ValueError(f"vf returned shape {value.shape}; expected {point.shape}")
        return metric.sqrt_det(point).unsqueeze(-1) * value

    sqrtg = metric.sqrt_det(x)
    jacobian = batched_jacobian(volume_weighted_field, x)
    if divergence_mode == "exact":
        trace = jacobian.diagonal(dim1=-2, dim2=-1).sum(-1)
    else:
        probe = torch.empty_like(x).bernoulli_(0.5, generator=generator)
        probe = probe.mul_(2.0).sub_(1.0)
        trace = torch.einsum("...i,...ij,...j->...", probe, jacobian, probe)
    return trace / sqrtg


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
    return torch.einsum("...ij,...j->...i", g_inv, dh)


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
    V = vf(x)  # (..., dim)
    if V.shape != x.shape:
        raise ValueError(f"vf returned shape {V.shape}; expected {x.shape}")
    Gamma = christoffel(metric, x)  # (..., dim, dim, dim)

    dV = batched_jacobian(vf, x)

    Gamma_V = torch.einsum("...ikj,...k->...ij", Gamma, V)

    return dV + Gamma_V  # ∇_j V^i
