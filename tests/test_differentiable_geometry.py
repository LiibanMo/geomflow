"""Phase 4 analytic tests for public differentiable geometry operators."""

from __future__ import annotations

import pytest
import torch

from geomflow.torch import (
    AnalyticMetric,
    EuclideanSpace,
    InducedMetric,
    batched_jacobian,
    christoffel,
    covariant_derivative_tensor,
    divergence,
    gradient,
)


DTYPE = torch.float64


def _polar_metric() -> AnalyticMetric:
    def metric(x: torch.Tensor) -> torch.Tensor:
        r = x[..., 0]
        zero = torch.zeros_like(r)
        one = torch.ones_like(r)
        return torch.stack(
            [torch.stack([one, zero], -1), torch.stack([zero, r.square()], -1)],
            -2,
        )

    return AnalyticMetric(2, metric)


def test_batched_jacobian_is_rectangular_batched_and_graph_preserving() -> None:
    x = torch.tensor(
        [[[0.2, -0.4], [0.7, 0.3]], [[-0.5, 0.8], [1.1, -0.2]]],
        dtype=DTYPE,
        requires_grad=True,
    )

    def immersion(point: torch.Tensor) -> torch.Tensor:
        u, v = point.unbind(-1)
        return torch.stack([u.square(), u * v, torch.sin(v)], -1)

    jacobian = batched_jacobian(immersion, x)
    expected = torch.zeros(*x.shape[:-1], 3, 2, dtype=DTYPE)
    expected[..., 0, 0] = 2.0 * x[..., 0]
    expected[..., 1, 0] = x[..., 1]
    expected[..., 1, 1] = x[..., 0]
    expected[..., 2, 1] = torch.cos(x[..., 1])
    torch.testing.assert_close(jacobian, expected)

    second = torch.autograd.grad(jacobian[..., 0, 0].sum(), x)[0]
    expected_second = torch.zeros_like(x)
    expected_second[..., 0] = 2.0
    torch.testing.assert_close(second, expected_second)


def test_batched_jacobian_rejects_non_pointwise_output_shape() -> None:
    x = torch.ones(2, 3, dtype=DTYPE, requires_grad=True)
    with pytest.raises(ValueError, match="leading batch dimensions"):
        batched_jacobian(lambda point: point.sum(0), x)


def test_batched_jacobian_value_does_not_require_grad_enabled_input() -> None:
    x = torch.tensor([[0.2, -0.4]], dtype=DTYPE)
    jacobian = batched_jacobian(lambda point: point.square(), x)
    torch.testing.assert_close(jacobian, torch.diag_embed(2.0 * x))


def test_metric_fallback_preserves_second_derivatives_and_partial_constants() -> None:
    x = torch.tensor([[0.4]], dtype=DTYPE, requires_grad=True)
    metric = AnalyticMetric(1, lambda point: (1.0 + point.square()).unsqueeze(-1))

    derivative = metric.derivative(x)
    torch.testing.assert_close(derivative[..., 0, 0, 0], 2.0 * x[..., 0])
    second = torch.autograd.grad(derivative.sum(), x)[0]
    torch.testing.assert_close(second, torch.full_like(x, 2.0))

    gamma = christoffel(metric, x)[..., 0, 0, 0]
    expected_gamma = x[..., 0] / (1.0 + x[..., 0].square())
    torch.testing.assert_close(gamma, expected_gamma)
    gamma_derivative = torch.autograd.grad(gamma.sum(), x)[0][..., 0]
    expected_gamma_derivative = (1.0 - x[..., 0].square()) / (
        1.0 + x[..., 0].square()
    ).square()
    torch.testing.assert_close(gamma_derivative, expected_gamma_derivative)

    def partial_metric(point: torch.Tensor) -> torch.Tensor:
        u = point[..., 0]
        zero = torch.zeros_like(u)
        return torch.stack(
            [
                torch.stack([1.0 + u.square(), zero], -1),
                torch.stack([zero, torch.full_like(u, 3.0)], -1),
            ],
            -2,
        )

    point = torch.tensor([[0.2, -0.7]], dtype=DTYPE, requires_grad=True)
    partial = AnalyticMetric(2, partial_metric).derivative(point)
    expected = torch.zeros_like(partial)
    expected[..., 0, 0, 0] = 0.4
    torch.testing.assert_close(partial, expected)


def test_fallback_and_analytic_metric_derivatives_agree() -> None:
    def metric_fn(x: torch.Tensor) -> torch.Tensor:
        return (1.0 + x.square()).unsqueeze(-1)

    def derivative_fn(x: torch.Tensor) -> torch.Tensor:
        return (2.0 * x).unsqueeze(-1).unsqueeze(-1)

    x = torch.tensor([[-0.6], [0.3]], dtype=DTYPE, requires_grad=True)
    fallback = AnalyticMetric(1, metric_fn)
    analytic = AnalyticMetric(1, metric_fn, derivative_fn=derivative_fn)
    torch.testing.assert_close(fallback.derivative(x), analytic.derivative(x))


def test_christoffel_symmetry_and_covariant_derivative_chart_covariance() -> None:
    metric = _polar_metric()
    x = torch.tensor([[1.7, -0.3], [2.2, 0.5]], dtype=DTYPE, requires_grad=True)
    gamma = christoffel(metric, x)
    torch.testing.assert_close(gamma, gamma.transpose(-1, -2))

    radial_position = lambda point: torch.stack(
        [point[..., 0], torch.zeros_like(point[..., 1])], -1
    )
    polar_nabla = covariant_derivative_tensor(radial_position, x, metric)
    expected = torch.eye(2, dtype=DTYPE).expand(x.shape[0], 2, 2)
    torch.testing.assert_close(polar_nabla, expected)


def test_constant_and_dependent_operator_edge_cases() -> None:
    x = torch.tensor([[0.2, -0.5], [0.8, 0.4]], dtype=DTYPE, requires_grad=True)
    euclidean = EuclideanSpace(2)
    constant_vector = torch.tensor([1.5, -0.2], dtype=DTYPE)

    vf = lambda point: constant_vector.expand_as(point)
    scalar = lambda point: torch.full(point.shape[:-1], 2.5, dtype=point.dtype)
    torch.testing.assert_close(divergence(vf, x, euclidean), torch.zeros(2, dtype=DTYPE))
    torch.testing.assert_close(gradient(scalar, x, euclidean), torch.zeros_like(x))
    torch.testing.assert_close(
        covariant_derivative_tensor(vf, x, euclidean),
        torch.zeros(2, 2, 2, dtype=DTYPE),
    )

    polar = _polar_metric()
    polar_divergence = divergence(vf, x, polar)
    torch.testing.assert_close(polar_divergence, constant_vector[0] / x[..., 0])
    trace_nabla = covariant_derivative_tensor(vf, x, polar).diagonal(
        dim1=-2, dim2=-1
    ).sum(-1)
    torch.testing.assert_close(polar_divergence, trace_nabla)


def test_induced_parabola_metric_volume_and_divergence() -> None:
    immersion = lambda x: torch.stack([x[..., 0], x[..., 0].square()], -1)
    metric = InducedMetric(1, immersion, debug=True)
    x = torch.tensor([[-0.6], [0.3]], dtype=DTYPE, requires_grad=True)

    expected_metric = 1.0 + 4.0 * x[..., 0].square()
    torch.testing.assert_close(metric.metric(x)[..., 0, 0], expected_metric)
    torch.testing.assert_close(metric.derivative(x)[..., 0, 0, 0], 8.0 * x[..., 0])
    expected_divergence = 4.0 * x[..., 0] / expected_metric
    constant_field = lambda point: torch.ones_like(point)
    torch.testing.assert_close(divergence(constant_field, x, metric), expected_divergence)


def test_paraboloid_geometry_through_second_derivatives() -> None:
    def immersion(x: torch.Tensor) -> torch.Tensor:
        u, v = x.unbind(-1)
        return torch.stack([u, v, u.square() + v.square()], -1)

    metric = InducedMetric(2, immersion)
    x = torch.tensor([[0.2, -0.3], [-0.4, 0.1]], dtype=DTYPE, requires_grad=True)
    identity = torch.eye(2, dtype=DTYPE).expand(x.shape[0], 2, 2)
    expected_metric = identity + 4.0 * x.unsqueeze(-1) * x.unsqueeze(-2)
    torch.testing.assert_close(metric.metric(x), expected_metric)

    gamma = christoffel(metric, x)
    expected_gamma = torch.zeros_like(gamma)
    factor = 4.0 * x / (1.0 + 4.0 * x.square().sum(-1, keepdim=True))
    expected_gamma[..., :, 0, 0] = factor
    expected_gamma[..., :, 1, 1] = factor
    torch.testing.assert_close(gamma, expected_gamma)

    constant_u = lambda point: torch.stack(
        [torch.ones_like(point[..., 0]), torch.zeros_like(point[..., 1])], -1
    )
    expected_divergence = 4.0 * x[..., 0] / (1.0 + 4.0 * x.square().sum(-1))
    torch.testing.assert_close(divergence(constant_u, x, metric), expected_divergence)

    metric_derivative = metric.derivative(x)
    second = torch.autograd.grad(metric_derivative.sum(), x)[0]
    torch.testing.assert_close(second, torch.full_like(x, 16.0))


def test_induced_metric_debug_rejects_singular_immersion() -> None:
    metric = InducedMetric(2, lambda x: x[..., :1], debug=True)
    x = torch.ones(2, 2, dtype=DTYPE, requires_grad=True)
    with pytest.raises(ValueError, match="full column rank"):
        metric.metric(x)
