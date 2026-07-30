"""Built-in manifold presets and analytic metric constructors for geomflow.torch.

Provides ready-to-use AnalyticMetric instances and Atlas multi-chart setups for:
- Euclidean space R^d
- Sphere S^2 (single chart and 2-chart stereographic Atlas)
- Torus T^2 (2D torus in standard coordinates)
- Hyperbolic space H^d (Poincaré disk model)
- Induced metric on embedded submanifolds phi: R^d -> R^N
"""

from __future__ import annotations

import math

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

    def log_volume_gradient_fn(x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)

    return AnalyticMetric(
        dim,
        metric_fn,
        inverse_fn,
        sqrt_det_fn,
        log_volume_gradient_fn=log_volume_gradient_fn,
    )


def SphereStereographicMetric(dim: int = 2, radius: float = 1.0) -> AnalyticMetric:
    """Stereographic coordinate metric on the sphere S^d.

    g_ij(x) = [4 / (1 + ||x||^2)^2] * delta_ij
    """

    if dim < 1:
        raise ValueError("sphere dimension must be positive")
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("sphere radius must be finite and positive")

    def metric_fn(x: torch.Tensor) -> torch.Tensor:
        r2 = (x * x).sum(dim=-1, keepdim=True)
        lam = 4.0 * radius**2 / ((1.0 + r2) ** 2)
        eye = torch.eye(dim, device=x.device, dtype=x.dtype)
        return lam.unsqueeze(-1) * eye

    def inverse_fn(x: torch.Tensor) -> torch.Tensor:
        r2 = (x * x).sum(dim=-1, keepdim=True)
        lam_inv = ((1.0 + r2) ** 2) / (4.0 * radius**2)
        eye = torch.eye(dim, device=x.device, dtype=x.dtype)
        return lam_inv.unsqueeze(-1) * eye

    def sqrt_det_fn(x: torch.Tensor) -> torch.Tensor:
        r2 = (x * x).sum(dim=-1)
        return (2.0 * radius / (1.0 + r2)) ** dim

    def log_volume_gradient_fn(x: torch.Tensor) -> torch.Tensor:
        r2 = (x * x).sum(dim=-1, keepdim=True)
        return (-2.0 * dim / (1.0 + r2)) * x

    def domain_fn(x: torch.Tensor) -> torch.Tensor:
        limit = torch.finfo(x.dtype).max ** 0.25 / math.sqrt(dim)
        return x.abs().amax(dim=-1) < limit

    return AnalyticMetric(
        dim,
        metric_fn,
        inverse_fn,
        sqrt_det_fn,
        domain_fn=domain_fn,
        log_volume_gradient_fn=log_volume_gradient_fn,
    )


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

    def transition_jacobian(x: torch.Tensor) -> torch.Tensor:
        r2 = (x * x).sum(dim=-1, keepdim=True)
        eye = torch.eye(2, device=x.device, dtype=x.dtype)
        outer = x.unsqueeze(-1) * x.unsqueeze(-2)
        return eye / r2.unsqueeze(-1) - 2.0 * outer / r2.square().unsqueeze(-1)

    def chart_domain(x: torch.Tensor) -> torch.Tensor:
        return torch.ones(x.shape[:-1], device=x.device, dtype=torch.bool)

    def overlap_domain(x: torch.Tensor) -> torch.Tensor:
        minimum = math.sqrt(torch.finfo(x.dtype).tiny)
        maximum = torch.finfo(x.dtype).max ** 0.25 / math.sqrt(2.0)
        scale = x.abs().amax(dim=-1)
        return (scale > minimum) & (scale < maximum)

    chart_a = Chart(
        0,
        2,
        None,
        metric_a,
        transitions={
            1: Transition(transition_a_to_b, overlap_domain, transition_jacobian)
        },
        domain=chart_domain,
    )
    chart_b = Chart(
        1,
        2,
        None,
        metric_b,
        transitions={
            0: Transition(transition_b_to_a, overlap_domain, transition_jacobian)
        },
        domain=chart_domain,
    )

    chart_a._defer_trial_validation = True
    chart_b._defer_trial_validation = True
    return Atlas([chart_a, chart_b], reference_chart_id=0)


def Torus2D(R: float = 2.0, r: float = 1.0) -> AnalyticMetric:
    """2D Torus T^2 in angle coordinates (theta, phi) in (-pi, pi]^2.

    Metric in (theta, phi) coordinates:
    G = diag((R + r * cos(phi))^2, r^2)
    """

    if not math.isfinite(R) or not math.isfinite(r) or not R > r > 0.0:
        raise ValueError("torus radii must satisfy finite R > r > 0")

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
        inv_g11 = 1.0 / ((R + r * torch.cos(phi)) ** 2)
        inv_g22 = torch.full_like(phi, 1.0 / (r**2))
        zero = torch.zeros_like(phi)

        row1 = torch.stack([inv_g11, zero], dim=-1)
        row2 = torch.stack([zero, inv_g22], dim=-1)
        return torch.stack([row1, row2], dim=-2)

    def sqrt_det_fn(x: torch.Tensor) -> torch.Tensor:
        phi = x[..., 1]
        return r * (R + r * torch.cos(phi))

    def log_volume_gradient_fn(x: torch.Tensor) -> torch.Tensor:
        phi = x[..., 1]
        grad_phi = -r * torch.sin(phi) / (R + r * torch.cos(phi))
        return torch.stack((torch.zeros_like(phi), grad_phi), dim=-1)

    def canonicalize_fn(x: torch.Tensor) -> torch.Tensor:
        return torch.remainder(x + math.pi, 2.0 * math.pi) - math.pi

    metric = AnalyticMetric(
        2,
        metric_fn,
        inverse_fn,
        sqrt_det_fn,
        canonicalize_fn=canonicalize_fn,
        log_volume_gradient_fn=log_volume_gradient_fn,
    )
    metric.coordinate_topology = "angles modulo 2*pi"
    metric.default_base_distribution = UniformAngleCoordinateBase(2)
    return metric


def PoincareDisk(dim: int = 2) -> AnalyticMetric:
    """Poincaré disk model of hyperbolic space H^d on ||x|| < 1.

    g_ij(x) = [4 / (1 - ||x||^2)^2] * delta_ij
    """

    if dim < 1:
        raise ValueError("Poincare disk dimension must be positive")

    def metric_fn(x: torch.Tensor) -> torch.Tensor:
        r2 = (x * x).sum(dim=-1, keepdim=True)
        lam = 4.0 / ((1.0 - r2) ** 2)
        eye = torch.eye(dim, device=x.device, dtype=x.dtype)
        return lam.unsqueeze(-1) * eye

    def inverse_fn(x: torch.Tensor) -> torch.Tensor:
        r2 = (x * x).sum(dim=-1, keepdim=True)
        lam_inv = ((1.0 - r2) ** 2) / 4.0
        eye = torch.eye(dim, device=x.device, dtype=x.dtype)
        return lam_inv.unsqueeze(-1) * eye

    def sqrt_det_fn(x: torch.Tensor) -> torch.Tensor:
        r2 = (x * x).sum(dim=-1)
        return (2.0 / (1.0 - r2)) ** dim

    def log_volume_gradient_fn(x: torch.Tensor) -> torch.Tensor:
        r2 = (x * x).sum(dim=-1, keepdim=True)
        return (2.0 * dim / (1.0 - r2)) * x

    def domain_fn(x: torch.Tensor) -> torch.Tensor:
        return x.square().sum(dim=-1) < 1.0

    metric = AnalyticMetric(
        dim,
        metric_fn,
        inverse_fn,
        sqrt_det_fn,
        domain_fn=domain_fn,
        log_volume_gradient_fn=log_volume_gradient_fn,
    )
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
