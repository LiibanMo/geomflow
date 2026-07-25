"""Built-in manifold presets and analytic metric constructors for geomflow.torch.

Provides ready-to-use AnalyticMetric instances and Atlas multi-chart setups for:
- Euclidean space R^d
- Sphere S^2 (single chart and 2-chart stereographic Atlas)
- Torus T^2 (2D torus in standard coordinates)
- Hyperbolic space H^d (Poincaré disk model)
- Induced metric on embedded submanifolds phi: R^d -> R^N
"""

from __future__ import annotations

import torch

from .analytic_metric import AnalyticMetric
from .atlas import Atlas, Chart, Transition
from .base_distribution import (
    PoincareDiskCoordinateBase,
    UniformAngleCoordinateBase,
)


def EuclideanSpace(dim: int = 2) -> AnalyticMetric:
    """Euclidean space R^d with constant identity metric tensor."""

    def metric_fn(x: torch.Tensor) -> torch.Tensor:
        eye = torch.eye(dim, device=x.device, dtype=x.dtype)
        return eye.expand(*x.shape[:-1], -1, -1)

    def inverse_fn(x: torch.Tensor) -> torch.Tensor:
        eye = torch.eye(dim, device=x.device, dtype=x.dtype)
        return eye.expand(*x.shape[:-1], -1, -1)

    def sqrt_det_fn(x: torch.Tensor) -> torch.Tensor:
        return torch.ones(x.shape[:-1], device=x.device, dtype=x.dtype)

    return AnalyticMetric(dim, metric_fn, inverse_fn, sqrt_det_fn)


def SphereStereographicMetric(dim: int = 2) -> AnalyticMetric:
    """Stereographic coordinate metric on the sphere S^d.

    g_ij(x) = [4 / (1 + ||x||^2)^2] * delta_ij
    """

    def metric_fn(x: torch.Tensor) -> torch.Tensor:
        r2 = (x * x).sum(dim=-1, keepdim=True)
        lam = 4.0 / ((1.0 + r2) ** 2)
        eye = torch.eye(dim, device=x.device, dtype=x.dtype)
        return lam.unsqueeze(-1) * eye

    def inverse_fn(x: torch.Tensor) -> torch.Tensor:
        r2 = (x * x).sum(dim=-1, keepdim=True)
        lam_inv = ((1.0 + r2) ** 2) / 4.0
        eye = torch.eye(dim, device=x.device, dtype=x.dtype)
        return lam_inv.unsqueeze(-1) * eye

    def sqrt_det_fn(x: torch.Tensor) -> torch.Tensor:
        r2 = (x * x).sum(dim=-1)
        return (2.0 / (1.0 + r2)) ** dim

    return AnalyticMetric(dim, metric_fn, inverse_fn, sqrt_det_fn)


def Sphere2DAtlas(n_samples: int = 500, seed: int = 42) -> Atlas:
    """2-chart stereographic Atlas covering the sphere S^2.

    Chart 0: Stereographic projection from North pole.
    Chart 1: Stereographic projection from South pole.
    Transition: x_B = x_A / ||x_A||^2
    """
    del n_samples, seed
    metric_a = SphereStereographicMetric(2)
    metric_b = SphereStereographicMetric(2)

    def transition_a_to_b(x: torch.Tensor) -> torch.Tensor:
        r2 = (x * x).sum(dim=-1, keepdim=True)
        return x / r2

    def transition_b_to_a(x: torch.Tensor) -> torch.Tensor:
        return transition_a_to_b(x)

    def chart_domain(x: torch.Tensor) -> torch.Tensor:
        return torch.isfinite(x).all(dim=-1)

    def overlap_domain(x: torch.Tensor) -> torch.Tensor:
        return chart_domain(x) & ((x * x).sum(dim=-1) > 0.0)

    chart_a = Chart(
        0,
        2,
        None,
        metric_a,
        transitions={1: Transition(transition_a_to_b, overlap_domain)},
        domain=chart_domain,
    )
    chart_b = Chart(
        1,
        2,
        None,
        metric_b,
        transitions={0: Transition(transition_b_to_a, overlap_domain)},
        domain=chart_domain,
    )

    return Atlas([chart_a, chart_b], reference_chart_id=0)


def Torus2D(R: float = 2.0, r: float = 1.0) -> AnalyticMetric:
    """2D Torus T^2 in angle coordinates (theta, phi) in (-pi, pi]^2.

    Metric in (theta, phi) coordinates:
    G = diag((R + r * cos(phi))^2, r^2)
    """

    def metric_fn(x: torch.Tensor) -> torch.Tensor:
        phi = x[..., 1]
        g11 = (R + r * torch.cos(phi)) ** 2
        g22 = torch.full_like(phi, r**2)
        zero = torch.zeros_like(phi)

        row1 = torch.stack([g11, zero], dim=-1)
        row2 = torch.stack([zero, g22], dim=-1)
        return torch.stack([row1, row2], dim=-2)

    def inverse_fn(x: torch.Tensor) -> torch.Tensor:
        phi = x[..., 1]
        inv_g11 = 1.0 / ((R + r * torch.cos(phi)) ** 2).clamp_min(1e-8)
        inv_g22 = torch.full_like(phi, 1.0 / (r**2))
        zero = torch.zeros_like(phi)

        row1 = torch.stack([inv_g11, zero], dim=-1)
        row2 = torch.stack([zero, inv_g22], dim=-1)
        return torch.stack([row1, row2], dim=-2)

    def sqrt_det_fn(x: torch.Tensor) -> torch.Tensor:
        phi = x[..., 1]
        return r * (R + r * torch.cos(phi))

    metric = AnalyticMetric(2, metric_fn, inverse_fn, sqrt_det_fn)
    metric.default_base_distribution = UniformAngleCoordinateBase(2)
    return metric


def PoincareDisk(dim: int = 2) -> AnalyticMetric:
    """Poincaré disk model of hyperbolic space H^d on ||x|| < 1.

    g_ij(x) = [4 / (1 - ||x||^2)^2] * delta_ij
    """

    def metric_fn(x: torch.Tensor) -> torch.Tensor:
        r2 = (x * x).sum(dim=-1, keepdim=True).clamp(max=0.9999)
        lam = 4.0 / ((1.0 - r2) ** 2)
        eye = torch.eye(dim, device=x.device, dtype=x.dtype)
        return lam.unsqueeze(-1) * eye

    def inverse_fn(x: torch.Tensor) -> torch.Tensor:
        r2 = (x * x).sum(dim=-1, keepdim=True).clamp(max=0.9999)
        lam_inv = ((1.0 - r2) ** 2) / 4.0
        eye = torch.eye(dim, device=x.device, dtype=x.dtype)
        return lam_inv.unsqueeze(-1) * eye

    def sqrt_det_fn(x: torch.Tensor) -> torch.Tensor:
        r2 = (x * x).sum(dim=-1).clamp(max=0.9999)
        return (2.0 / (1.0 - r2)) ** dim

    metric = AnalyticMetric(dim, metric_fn, inverse_fn, sqrt_det_fn)
    metric.default_base_distribution = PoincareDiskCoordinateBase(dim)
    return metric


# Alias
HyperbolicSpace = PoincareDisk


def InducedMetric(
    dim: int,
    immersion_fn: callable,
    debug: bool = False,
) -> AnalyticMetric:
    """Analytic metric induced by an immersion map phi: R^d -> R^N.

    G(x) = J_phi(x)^T * J_phi(x). The immersion must act pointwise over
    leading batch dimensions. If ``debug`` is true, rank-deficient Jacobians
    raise ``ValueError``.
    """
    from ._utils import batched_jacobian

    def metric_fn(x: torch.Tensor) -> torch.Tensor:
        J = batched_jacobian(immersion_fn, x)  # (..., N, d)
        if debug and torch.any(torch.linalg.matrix_rank(J) < dim):
            raise ValueError("immersion Jacobian must have full column rank")
        return torch.matmul(J.transpose(-1, -2), J)  # (..., d, d)

    return AnalyticMetric(dim, metric_fn)
