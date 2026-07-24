"""Normalization and chart identities for density relative to dV_g."""

from __future__ import annotations

import math

import torch

from geomflow.torch import EuclideanSpace, ManifoldCNF, SphereStereographicMetric


DTYPE = torch.float64


def _standard_normal_log_density(x: torch.Tensor) -> torch.Tensor:
    return -0.5 * (math.log(2.0 * math.pi) + x.square())


def test_coordinate_gaussian_converts_to_scaled_metric_volume_density() -> None:
    """MATH-250: rho=q_coord/c^d for the standard identity g=c^2 I."""
    scale, dimension = 2.5, 3
    point = torch.tensor([[0.2, -0.4, 1.1]], dtype=DTYPE)
    log_q = -0.5 * (
        dimension * math.log(2.0 * math.pi) + point.square().sum(dim=-1)
    )
    log_rho = log_q - dimension * math.log(scale)
    torch.testing.assert_close(
        torch.exp(log_rho) * scale**dimension, torch.exp(log_q)
    )


def test_scaled_metric_volume_density_integrates_to_unit_mass() -> None:
    """MATH-251: numerically integrate rho dV_g independently in one dimension."""
    scale = 3.0
    x = torch.linspace(-9.0, 9.0, 200_001, dtype=DTYPE)
    q_coord = torch.exp(_standard_normal_log_density(x))
    rho = q_coord / scale
    mass = torch.trapezoid(rho * scale, x)
    torch.testing.assert_close(mass, torch.tensor(1.0, dtype=DTYPE), atol=2e-15, rtol=0.0)


def test_euclidean_zero_flow_samples_match_log_probability() -> None:
    """MATH-252: zero-flow sample and log_prob represent one normalized Gaussian."""
    model = ManifoldCNF(EuclideanSpace(2), hidden_dim=4, n_layers=1, dt=0.2)
    for parameter in model.vf.parameters():
        parameter.data.zero_()

    torch.manual_seed(42)
    samples, chart = model.sample(32, device=torch.device("cpu"))
    expected = -0.5 * (
        2.0 * math.log(2.0 * math.pi) + samples.square().sum(dim=-1)
    )
    torch.testing.assert_close(model.log_prob(samples), expected)
    assert chart is None


def test_sphere_volume_density_is_scalar_across_stereographic_transition() -> None:
    """MATH-253--255: separate Riemannian and coordinate-density transitions."""
    metric = SphereStereographicMetric(2)
    x = torch.tensor([[0.6, -1.2], [1.3, 0.7]], dtype=DTYPE)
    radius_squared = x.square().sum(-1, keepdim=True)
    y = x / radius_squared

    log_q_x = -0.5 * (
        2.0 * math.log(2.0 * math.pi) + x.square().sum(dim=-1)
    )
    # For inversion in two dimensions, |det(dy/dx)|=||x||^-4.
    log_abs_det_dy_dx = -2.0 * torch.log(radius_squared[..., 0])
    log_q_y = log_q_x - log_abs_det_dy_dx

    log_rho_x = log_q_x - torch.log(metric.sqrt_det(x))
    log_rho_y = log_q_y - torch.log(metric.sqrt_det(y))
    torch.testing.assert_close(log_rho_y, log_rho_x, rtol=2e-14, atol=2e-14)
    torch.testing.assert_close(log_q_y - log_q_x, -log_abs_det_dy_dx)
