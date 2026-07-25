"""Finite-difference and analytic gradient oracles for complete CNF objectives."""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from geomflow.torch import (
    AnalyticMetric,
    EuclideanSpace,
    cnf_nll,
    integrate_rk4,
    intrinsic_adjoint_nll,
)

from analytic_references import central_difference, observed_order


DTYPE = torch.float64


class ScalarLinearField(nn.Module):
    def __init__(self, coefficient: float) -> None:
        super().__init__()
        self.coefficient = nn.Parameter(torch.tensor(coefficient, dtype=DTYPE))

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        del t
        return self.coefficient * x


class DiagonalField(nn.Module):
    def __init__(self, diagonal: torch.Tensor) -> None:
        super().__init__()
        self.diagonal = nn.Parameter(diagonal.clone())
        self.inactive = nn.Parameter(torch.tensor(0.4, dtype=diagonal.dtype))

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        del t
        return self.diagonal * x + self.inactive * x * 0.0


class ConstantParameterizedField(nn.Module):
    def __init__(self, speed: float) -> None:
        super().__init__()
        self.speed = nn.Parameter(torch.tensor(speed, dtype=DTYPE))

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        del t
        return self.speed.expand_as(x) + x * 0.0


def _linear_nll(coefficient: float, data: torch.Tensor, duration: float) -> float:
    base = data * math.exp(-coefficient * duration)
    values = 0.5 * base.square().sum(dim=-1) + coefficient * duration * data.shape[-1]
    return values.mean().item()


def test_complete_nll_parameter_gradient_matches_independent_finite_difference() -> None:
    """MATH-260: report the NLL quantity, reverse direction, step, and expected sign."""
    data = torch.tensor([[0.4], [1.1]], dtype=DTYPE)
    coefficient, duration, solver_step = 0.3, 1.0, 0.025
    field = ScalarLinearField(coefficient)
    loss = cnf_nll(field, EuclideanSpace(1), data, dt=solver_step, t0=0.0, t1=duration)
    (actual,) = torch.autograd.grad(loss, (field.coefficient,))
    expected = central_difference(
        lambda value: _linear_nll(value, data, duration), coefficient, step=1e-6
    )
    assert expected > 0.0
    torch.testing.assert_close(
        actual,
        torch.tensor(expected, dtype=DTYPE),
        rtol=2e-8,
        atol=2e-10,
        msg=lambda message: (
            f"complete NLL parameter gradient; direction=data->base; "
            f"dt={solver_step}; expected positive\n{message}"
        ),
    )


def test_complete_nll_input_gradient_matches_analytic_cotangent() -> None:
    """MATH-261: input gradients match finite differences and the exact cotangent."""
    data = torch.tensor([[0.4], [1.1]], dtype=DTYPE, requires_grad=True)
    coefficient, duration = 0.3, 1.0
    field = ScalarLinearField(coefficient)
    loss = cnf_nll(field, EuclideanSpace(1), data, dt=0.025, t0=0.0, t1=duration)
    (actual,) = torch.autograd.grad(loss, (data,))
    expected = data.detach() * math.exp(-2.0 * coefficient * duration) / data.shape[0]
    finite_difference = torch.empty_like(data)
    for sample in range(data.shape[0]):
        finite_difference[sample, 0] = central_difference(
            lambda value: _linear_nll(
                coefficient,
                data.detach().index_put(
                    (torch.tensor([sample]), torch.tensor([0])),
                    torch.tensor([value], dtype=DTYPE),
                ),
                duration,
            ),
            data[sample, 0].item(),
        )
    torch.testing.assert_close(actual, expected, rtol=2e-8, atol=2e-10)
    torch.testing.assert_close(actual, finite_difference, rtol=2e-8, atol=2e-10)


def test_every_trainable_parameter_gradient_is_checked() -> None:
    """MATH-263/MATH-264: check active entries and an explicit mathematical zero."""
    data = torch.tensor([[0.4, 0.8], [1.1, -0.3]], dtype=DTYPE)
    diagonal = torch.tensor([0.3, -0.2], dtype=DTYPE)
    field = DiagonalField(diagonal)
    loss = cnf_nll(field, EuclideanSpace(2), data, dt=0.025)
    actual_diagonal, actual_inactive = torch.autograd.grad(
        loss, (field.diagonal, field.inactive)
    )
    base = data * torch.exp(-diagonal)
    expected_diagonal = (1.0 - base.square()).mean(dim=0)

    torch.testing.assert_close(actual_diagonal, expected_diagonal, rtol=2e-8, atol=2e-10)
    torch.testing.assert_close(actual_inactive, torch.zeros_like(actual_inactive))


def test_mathematically_inactive_parameter_gets_zero_tensor_gradient() -> None:
    """MATH-264: a connected parameter with zero contribution returns zero, not None."""
    data = torch.tensor([[0.4, 0.8]], dtype=DTYPE)
    field = DiagonalField(torch.tensor([0.3, -0.2], dtype=DTYPE))
    result = integrate_rk4(
        field, EuclideanSpace(2), data, 0.0, 0.7, dt=0.1, compute_divergence=False
    )
    (gradient,) = torch.autograd.grad(result.x_final.sum(), (field.inactive,))
    torch.testing.assert_close(gradient, torch.zeros_like(gradient))


@pytest.mark.parametrize("ta,tb", [(0.0, 1.0), (1.0, 0.0)])
def test_state_parameter_gradient_in_forward_and_reverse_time(ta: float, tb: float) -> None:
    """MATH-265/MATH-266: state gradients honor orientation and remainder steps."""
    x = torch.tensor([[0.7]], dtype=DTYPE)
    coefficient = 0.4
    field = ScalarLinearField(coefficient)
    result = integrate_rk4(field, EuclideanSpace(1), x, ta, tb, dt=0.3, compute_divergence=False)
    (actual,) = torch.autograd.grad(result.x_final.sum(), (field.coefficient,))
    duration = tb - ta
    expected = (duration * x * math.exp(coefficient * duration)).sum()
    torch.testing.assert_close(actual, expected, rtol=8e-5, atol=2e-7)


@pytest.mark.slow
@pytest.mark.convergence
def test_direct_state_gradient_has_fourth_order_convergence() -> None:
    """MATH-273: measure direct parameter-gradient order at h, h/2, h/4."""
    x = torch.tensor([[0.9]], dtype=DTYPE)
    coefficient = 1.1
    exact = coefficient * 0.0 + x.item() * math.exp(coefficient)
    errors: list[float] = []
    for step in (0.2, 0.1, 0.05):
        field = ScalarLinearField(coefficient)
        result = integrate_rk4(
            field, EuclideanSpace(1), x, 0.0, 1.0, dt=step, compute_divergence=False
        )
        (actual,) = torch.autograd.grad(result.x_final.sum(), (field.coefficient,))
        errors.append(abs(actual.item() - exact))

    orders = observed_order(errors)
    assert min(orders) > 3.7, f"direct state-gradient order: errors={errors}, orders={orders}"


@pytest.mark.slow
@pytest.mark.convergence
def test_intrinsic_adjoint_input_gradient_convergence_is_measured_separately() -> None:
    """MATH-274: estimate adjoint input-gradient order independently of direct autograd."""
    data_value, coefficient = 0.9, 0.7
    exact = data_value * math.exp(-2.0 * coefficient)
    errors: list[float] = []
    for step in (0.2, 0.1, 0.05):
        data = torch.tensor([[data_value]], dtype=DTYPE, requires_grad=True)
        field = ScalarLinearField(coefficient)
        loss = intrinsic_adjoint_nll(
            field, EuclideanSpace(1), data, step, 0.0, 1.0
        )
        (actual,) = torch.autograd.grad(loss, (data,))
        errors.append(abs(actual.item() - exact))

    orders = observed_order(errors)
    assert min(orders) > 3.5, f"adjoint input-gradient order: errors={errors}, orders={orders}"


def test_cotangent_gradient_transforms_by_chart_pullback() -> None:
    """MATH-268: coordinate gradients of one scalar obey the cotangent law."""
    x = torch.tensor([[0.7, -1.1]], dtype=DTYPE, requires_grad=True)
    radius_squared = x.square().sum(-1, keepdim=True)
    y = (x / radius_squared).detach().requires_grad_(True)

    scalar_x = (x[..., 0] + 2.0 * x[..., 1]).sum()
    inverse_radius_squared = y.square().sum(-1, keepdim=True)
    x_from_y = y / inverse_radius_squared
    scalar_y = (x_from_y[..., 0] + 2.0 * x_from_y[..., 1]).sum()
    (covector_x,) = torch.autograd.grad(scalar_x, (x,))
    (covector_y,) = torch.autograd.grad(scalar_y, (y,))

    identity = torch.eye(2, dtype=DTYPE).unsqueeze(0)
    jacobian_x_by_y = identity / inverse_radius_squared.unsqueeze(-1) - (
        2.0 * y.unsqueeze(-1) * y.unsqueeze(-2) / inverse_radius_squared.square().unsqueeze(-1)
    )
    expected_y = (covector_x.unsqueeze(-2) @ jacobian_x_by_y).squeeze(-2)
    torch.testing.assert_close(covector_y, expected_y, rtol=2e-14, atol=2e-14)


def test_complete_nll_parameter_gradient_under_nonconstant_metric() -> None:
    """MATH-267: g=exp(2x) gives div(theta)=theta for a constant field."""
    metric = AnalyticMetric(
        1,
        lambda x: torch.exp(2.0 * x[..., :1]).unsqueeze(-1),
        inverse_fn=lambda x: torch.exp(-2.0 * x[..., :1]).unsqueeze(-1),
        sqrt_det_fn=lambda x: torch.exp(x[..., 0]),
    )
    data = torch.tensor([[0.8]], dtype=DTYPE)
    field = ConstantParameterizedField(0.25)
    duration = 0.9
    loss = cnf_nll(field, metric, data, dt=0.03, t0=0.0, t1=duration)
    (actual,) = torch.autograd.grad(loss, (field.speed,))
    base = data - field.speed.detach() * duration
    # log rho_base = log q_coord - x for sqrt(det g)=exp(x).
    expected = -duration * base.sum()
    torch.testing.assert_close(actual, expected, rtol=2e-8, atol=2e-10)
