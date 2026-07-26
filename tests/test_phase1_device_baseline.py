"""Phase 1 analytic and device baselines."""

from __future__ import annotations

import torch

from conftest import supported_device_dtype_cases
from geomflow.torch import (
    AnalyticMetric,
    ManifoldVectorField,
    christoffel,
    cnf_nll,
    covariant_derivative_tensor,
    gradient,
    integrate_rk4,
)


def _constant_metric(dim: int, scale: float) -> AnalyticMetric:
    def metric(x: torch.Tensor) -> torch.Tensor:
        eye = torch.eye(dim, dtype=x.dtype, device=x.device)
        return scale * eye.expand(*x.shape[:-1], dim, dim)

    def inverse(x: torch.Tensor) -> torch.Tensor:
        eye = torch.eye(dim, dtype=x.dtype, device=x.device)
        return (eye / scale).expand(*x.shape[:-1], dim, dim)

    def sqrt_det(x: torch.Tensor) -> torch.Tensor:
        return x.new_full(x.shape[:-1], scale ** (dim / 2.0))

    def derivative(x: torch.Tensor) -> torch.Tensor:
        return x.new_zeros(*x.shape[:-1], dim, dim, dim)

    return AnalyticMetric(dim, metric, inverse, sqrt_det, derivative)


@torch.no_grad()
def test_constant_metric_closed_form_values() -> None:
    x = torch.tensor([[0.2, -0.4], [0.7, 0.1]], dtype=torch.float64)
    metric = _constant_metric(2, 3.0)
    eye = torch.eye(2, dtype=x.dtype).expand(2, 2, 2)

    torch.testing.assert_close(metric.metric(x), 3.0 * eye)
    torch.testing.assert_close(metric.inverse(x), eye / 3.0)
    torch.testing.assert_close(metric.sqrt_det(x), x.new_full((2,), 3.0))
    torch.testing.assert_close(metric.derivative(x), x.new_zeros(2, 2, 2, 2))


def test_constant_metric_connection_gradient_and_covariant_derivative() -> None:
    x = torch.tensor(
        [[0.2, -0.4], [0.7, 0.1]], dtype=torch.float64, requires_grad=True
    )
    metric = _constant_metric(2, 3.0)

    torch.testing.assert_close(christoffel(metric, x), x.new_zeros(2, 2, 2, 2))
    actual_gradient = gradient(lambda value: value.square().sum(dim=-1), x, metric)
    torch.testing.assert_close(actual_gradient, (2.0 / 3.0) * x)
    derivative = covariant_derivative_tensor(lambda value: 2.0 * value, x, metric)
    expected = 2.0 * torch.eye(2, dtype=x.dtype).expand(2, 2, 2)
    torch.testing.assert_close(derivative, expected)


@torch.no_grad()
def test_metric_values_preserve_device_and_dtype(device_dtype) -> None:
    device, dtype = device_dtype
    x = torch.randn(3, 2, device=device, dtype=dtype)
    metric = _constant_metric(2, 2.0)
    for result in (
        metric.metric(x),
        metric.inverse(x),
        metric.sqrt_det(x),
        metric.derivative(x),
    ):
        assert result.device == x.device
        assert result.dtype == x.dtype


def test_zero_field_integration_preserves_device_and_dtype(device_dtype) -> None:
    device, dtype = device_dtype
    x = torch.randn(4, 2, device=device, dtype=dtype)
    metric = _constant_metric(2, 1.0)

    class ZeroField(torch.nn.Module):
        def forward(self, t: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
            return value * 0.0

    result = integrate_rk4(ZeroField(), metric, x, 0.0, 0.2, 0.1)
    assert result.x_final.device == x.device
    assert result.x_final.dtype == x.dtype
    assert result.divergence_integral.device == x.device
    assert result.divergence_integral.dtype == x.dtype


def test_single_chart_backward_preserves_parameter_device_and_dtype(device_dtype) -> None:
    device, dtype = device_dtype
    metric = _constant_metric(2, 1.0)
    field = ManifoldVectorField(2, hidden_dim=4, n_layers=1).to(
        device=device, dtype=dtype
    )
    x = torch.randn(3, 2, device=device, dtype=dtype)
    loss = cnf_nll(field, metric, x, dt=0.5)
    loss.backward()

    assert loss.device == x.device
    assert loss.dtype == x.dtype
    for name, parameter in field.named_parameters():
        assert parameter.grad is not None, name
        assert parameter.grad.device == parameter.device, name
        assert parameter.grad.dtype == parameter.dtype, name
        assert torch.isfinite(parameter.grad).all(), name


def test_device_dtype_matrix_declares_cpu_reference_cases() -> None:
    ids = {parameter.id for parameter in supported_device_dtype_cases()}
    assert {"cpu-float32", "cpu-float64"} <= ids
