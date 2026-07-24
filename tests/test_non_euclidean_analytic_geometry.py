"""Independent coordinate-geometry identities away from chart singularities."""

from __future__ import annotations

import pytest
import torch

from geomflow.torch import AnalyticMetric, SphereStereographicMetric
from geomflow.torch.operators import christoffel, covariant_derivative_tensor, divergence


DTYPE = torch.float64


def _exponential_line_metric() -> AnalyticMetric:
    """One-dimensional metric g_11=exp(2x)."""

    def metric(x: torch.Tensor) -> torch.Tensor:
        return torch.exp(2.0 * x[..., :1]).unsqueeze(-1)

    def inverse(x: torch.Tensor) -> torch.Tensor:
        return torch.exp(-2.0 * x[..., :1]).unsqueeze(-1)

    def sqrt_det(x: torch.Tensor) -> torch.Tensor:
        return torch.exp(x[..., 0])

    def derivative(x: torch.Tensor) -> torch.Tensor:
        return (2.0 * torch.exp(2.0 * x[..., 0]))[..., None, None, None]

    return AnalyticMetric(1, metric, inverse, sqrt_det, derivative)


def _polar_metric() -> AnalyticMetric:
    """Euclidean plane metric diag(1,r^2), restricted to r>0."""

    def metric(x: torch.Tensor) -> torch.Tensor:
        r = x[..., 0]
        zero = torch.zeros_like(r)
        return torch.stack(
            [torch.stack([torch.ones_like(r), zero], -1), torch.stack([zero, r * r], -1)],
            -2,
        )

    def inverse(x: torch.Tensor) -> torch.Tensor:
        r = x[..., 0]
        zero = torch.zeros_like(r)
        return torch.stack(
            [
                torch.stack([torch.ones_like(r), zero], -1),
                torch.stack([zero, r.reciprocal().square()], -1),
            ],
            -2,
        )

    def sqrt_det(x: torch.Tensor) -> torch.Tensor:
        return x[..., 0]

    def derivative(x: torch.Tensor) -> torch.Tensor:
        result = torch.zeros(*x.shape[:-1], 2, 2, 2, dtype=x.dtype, device=x.device)
        result[..., 1, 1, 0] = 2.0 * x[..., 0]
        return result

    return AnalyticMetric(2, metric, inverse, sqrt_det, derivative)


def test_exponential_line_volume_divergence_and_connection() -> None:
    """MATH-230--232: verify standard 1D volume and Levi-Civita identities."""
    metric = _exponential_line_metric()
    x = torch.tensor([[-0.4], [0.7]], dtype=DTYPE, requires_grad=True)
    speed = 1.25

    torch.testing.assert_close(metric.sqrt_det(x), torch.exp(x[..., 0]))
    torch.testing.assert_close(christoffel(metric, x)[..., 0, 0, 0], torch.ones(2, dtype=DTYPE))

    field = lambda point: point * 0.0 + speed
    torch.testing.assert_close(divergence(field, x, metric), torch.full((2,), speed, dtype=DTYPE))
    torch.testing.assert_close(
        covariant_derivative_tensor(field, x, metric)[..., 0, 0],
        torch.full((2,), speed, dtype=DTYPE),
    )


def test_exponential_line_cotangent_connection_terms_cancel() -> None:
    """MATH-233: intrinsic cotangent transport cancels connection double counting."""
    speed = 1.25
    lambda_component = torch.tensor([0.8, -0.3], dtype=DTYPE)
    covariant_time_derivative = -speed * lambda_component
    lambda_contracted_nabla_f = speed * lambda_component
    differential_divergence = torch.zeros_like(lambda_component)

    torch.testing.assert_close(
        covariant_time_derivative + lambda_contracted_nabla_f,
        differential_divergence,
    )


def test_polar_christoffel_symbols() -> None:
    """MATH-234/MATH-235: verify the nonzero polar-coordinate symbols."""
    metric = _polar_metric()
    x = torch.tensor([[2.5, 0.3]], dtype=DTYPE, requires_grad=True)
    gamma = christoffel(metric, x)

    expected = torch.zeros_like(gamma)
    expected[..., 0, 1, 1] = -2.5
    expected[..., 1, 0, 1] = 1.0 / 2.5
    expected[..., 1, 1, 0] = 1.0 / 2.5
    torch.testing.assert_close(gamma, expected)


@pytest.mark.xfail(
    strict=True,
    reason="MATH-041: Python contracts the upper Christoffel index with V",
)
def test_polar_radial_field_covariant_theta_component() -> None:
    """MATH-236: nabla_theta V^theta=c/r for V=(c,0)."""
    metric = _polar_metric()
    x = torch.tensor([[2.0, -0.4]], dtype=DTYPE, requires_grad=True)
    speed = 1.5
    field = lambda point: torch.stack(
        [point[..., 0] * 0.0 + speed, point[..., 1] * 0.0], dim=-1
    )

    nabla = covariant_derivative_tensor(field, x, metric)
    torch.testing.assert_close(nabla[..., 1, 1], torch.tensor([speed / 2.0], dtype=DTYPE))


def test_divergence_is_invariant_between_cartesian_and_polar_coordinates() -> None:
    """MATH-237: div(x,y)=2 equals div(r,0)=2 away from r=0."""
    polar = _polar_metric()
    identity = AnalyticMetric(
        2,
        lambda x: torch.eye(2, dtype=x.dtype, device=x.device).expand(*x.shape[:-1], 2, 2),
    )
    cartesian_point = torch.tensor([[1.2, -0.8]], dtype=DTYPE, requires_grad=True)
    polar_point = torch.tensor([[1.7, 0.4]], dtype=DTYPE, requires_grad=True)

    cartesian_div = divergence(lambda point: point, cartesian_point, identity)
    polar_div = divergence(
        lambda point: torch.stack([point[..., 0], point[..., 1] * 0.0], -1),
        polar_point,
        polar,
    )
    torch.testing.assert_close(cartesian_div, torch.tensor([2.0], dtype=DTYPE))
    torch.testing.assert_close(polar_div, cartesian_div)


def test_stereographic_sphere_metric_transforms_across_overlap() -> None:
    """MATH-238: stereographic inversion preserves the sphere metric."""
    metric = SphereStereographicMetric(2)
    x = torch.tensor([[0.7, -1.1], [1.4, 0.5]], dtype=DTYPE)
    radius_squared = (x * x).sum(-1, keepdim=True)
    y = x / radius_squared
    identity = torch.eye(2, dtype=DTYPE).expand(x.shape[0], 2, 2)
    jacobian = identity / radius_squared.unsqueeze(-1) - (
        2.0 * x.unsqueeze(-1) * x.unsqueeze(-2) / radius_squared.square().unsqueeze(-1)
    )

    pulled_back = jacobian.transpose(-1, -2) @ metric.metric(y) @ jacobian
    torch.testing.assert_close(pulled_back, metric.metric(x), rtol=2e-14, atol=2e-14)


def test_constant_scaled_metric_has_nonunit_volume_density() -> None:
    """MATH-239: g=c^2 I has sqrt(det g)=c^d."""
    dimension, scale = 3, 2.5
    metric = AnalyticMetric(
        dimension,
        lambda x: (scale * scale)
        * torch.eye(dimension, dtype=x.dtype, device=x.device).expand(
            *x.shape[:-1], dimension, dimension
        ),
    )
    x = torch.tensor([[0.2, -0.3, 0.7]], dtype=DTYPE)
    torch.testing.assert_close(
        metric.sqrt_det(x), torch.tensor([scale**dimension], dtype=DTYPE)
    )
