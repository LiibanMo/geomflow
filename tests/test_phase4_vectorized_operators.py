"""Correctness gates for vectorized differential geometry operators."""

from __future__ import annotations

import pytest
import torch

from geomflow.torch import (
    AnalyticMetric,
    batched_jacobian,
    christoffel,
    covariant_derivative_tensor,
    divergence,
    gradient,
)


def _metric(dtype: torch.dtype, device: torch.device) -> AnalyticMetric:
    def metric_fn(x: torch.Tensor) -> torch.Tensor:
        u, v = x.unbind(-1)
        zero = torch.zeros_like(u)
        return torch.stack(
            (
                torch.stack((1.0 + u.square(), zero), -1),
                torch.stack((zero, 2.0 + v.square()), -1),
            ),
            -2,
        )

    return AnalyticMetric(2, metric_fn)


def _loop_jacobian(fn: callable, x: torch.Tensor) -> torch.Tensor:
    rows = []
    y = fn(x)
    for component in y.unbind(-1):
        rows.append(
            torch.autograd.grad(
                component.sum(), x, create_graph=True, retain_graph=True
            )[0]
        )
    return torch.stack(rows, -2)


@pytest.mark.parametrize("shape", [(2,), (1, 2), (2, 3, 2), (0, 2)])
def test_batched_jacobian_shapes_and_loop_oracle(shape: tuple[int, ...]) -> None:
    x = torch.randn(shape, dtype=torch.float64, requires_grad=True)

    def fn(point: torch.Tensor) -> torch.Tensor:
        u, v = point.unbind(-1)
        return torch.stack((u.square(), u * v, v.sin()), -1)

    actual = batched_jacobian(fn, x)
    assert actual.shape == shape[:-1] + (3, 2)
    if x.numel():
        torch.testing.assert_close(actual, _loop_jacobian(fn, x))


def test_batched_jacobian_accepts_noncontiguous_input() -> None:
    base = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
    x = base.transpose(0, 1)
    assert not x.is_contiguous()
    torch.testing.assert_close(
        batched_jacobian(lambda point: point.square(), x),
        torch.diag_embed(2.0 * x),
    )


def test_operator_gradcheck_and_gradgradcheck() -> None:
    metric = _metric(torch.float64, torch.device("cpu"))
    vf = lambda point: torch.stack(
        (point[..., 0].square() * point[..., 1], point[..., 0].sin()), -1
    )
    fn = lambda point: torch.cat(
        (
            christoffel(metric, point).reshape(-1),
            divergence(vf, point, metric).reshape(-1),
            gradient(lambda value: value.square().sum(-1), point, metric).reshape(-1),
            covariant_derivative_tensor(vf, point, metric).reshape(-1),
        )
    )
    x = torch.tensor([[0.2, -0.3]], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(fn, (x,), fast_mode=True)
    assert torch.autograd.gradgradcheck(fn, (x,), fast_mode=True)


def test_hutchinson_is_explicit_reproducible_and_unbiased() -> None:
    metric = _metric(torch.float64, torch.device("cpu"))
    x = torch.randn(2048, 2, dtype=torch.float64, requires_grad=True)
    vf = lambda point: torch.stack(
        (point[..., 0] + point[..., 1], point[..., 0] - 2.0 * point[..., 1]), -1
    )
    first = torch.Generator().manual_seed(17)
    second = torch.Generator().manual_seed(17)
    estimate = divergence(
        vf, x, metric, divergence_mode="hutchinson", generator=first
    )
    repeated = divergence(
        vf, x, metric, divergence_mode="hutchinson", generator=second
    )
    torch.testing.assert_close(estimate, repeated)
    exact = divergence(vf, x, metric)
    assert (estimate - exact).mean().abs() < 0.2
    with pytest.raises(ValueError, match="divergence_mode"):
        divergence(vf, x, metric, divergence_mode="silent_approximation")


def test_vectorized_operators_preserve_device_and_dtype(device_dtype) -> None:
    device, dtype = device_dtype
    metric = _metric(dtype, device)
    x = torch.tensor([[0.2, -0.3], [0.4, 0.1]], device=device, dtype=dtype)
    x.requires_grad_(True)
    vf = lambda point: torch.stack(
        (point[..., 0].square(), point[..., 0] * point[..., 1]), -1
    )
    outputs = (
        batched_jacobian(vf, x),
        metric.derivative(x),
        christoffel(metric, x),
        divergence(vf, x, metric),
        gradient(lambda point: point.square().sum(-1), x, metric),
        covariant_derivative_tensor(vf, x, metric),
    )
    for output in outputs:
        assert output.device == x.device
        assert output.dtype == x.dtype
        assert torch.isfinite(output).all()
    loss = sum(output.sum() for output in outputs)
    first = torch.autograd.grad(loss, x, create_graph=True)[0]
    second = torch.autograd.grad(first.sum(), x)[0]
    assert first.device == second.device == x.device
    assert first.dtype == second.dtype == x.dtype
