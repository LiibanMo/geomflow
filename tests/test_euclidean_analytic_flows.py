"""Analytic Euclidean flow oracles for Mohamud's density equation."""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from geomflow.torch import EuclideanSpace, integrate_rk4

from analytic_references import (
    linear_divergence_integral,
    linear_log_density_change,
    observed_order,
    quadratic_flow_quantities,
    scalar_linear_state,
    time_linear_state,
)


DTYPE = torch.float64
DEVICE = torch.device("cpu")


class ZeroField(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0, dtype=DTYPE, device=DEVICE))

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        del t
        return self.scale * x * 0.0


class ConstantField(nn.Module):
    def __init__(self, velocity: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("velocity", velocity)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        del t
        return self.velocity.expand_as(x) + x * 0.0


class MatrixField(nn.Module):
    def __init__(self, matrix: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("matrix", matrix)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        del t
        return x @ self.matrix.T


class ScalarLinearField(nn.Module):
    def __init__(self, coefficient: float) -> None:
        super().__init__()
        self.coefficient = nn.Parameter(
            torch.tensor(coefficient, dtype=DTYPE, device=DEVICE)
        )

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        del t
        return self.coefficient * x


class TimeLinearField(nn.Module):
    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return t.unsqueeze(-1) * x


class TimeQuarticField(nn.Module):
    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return t.pow(4).unsqueeze(-1) * x


class QuadraticField(nn.Module):
    def __init__(self, theta: float) -> None:
        super().__init__()
        self.theta = nn.Parameter(torch.tensor(theta, dtype=DTYPE, device=DEVICE))

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        del t
        return self.theta * x.square()


def _integrate(field: nn.Module, x: torch.Tensor, ta: float, tb: float, dt: float):
    return integrate_rk4(field, EuclideanSpace(x.shape[-1]), x, ta, tb, dt)


def test_zero_field_preserves_augmented_state_and_has_zero_parameter_gradient() -> None:
    """MATH-210: zero field has zero state, density, and parameter variations."""
    x = torch.tensor([[1.5, -0.25], [-2.0, 3.0]], dtype=DTYPE, device=DEVICE)
    field = ZeroField()

    result = _integrate(field, x, -0.4, 0.7, 0.13)
    torch.testing.assert_close(result.x_final, x, rtol=0.0, atol=0.0)
    torch.testing.assert_close(result.log_det, torch.zeros(2, dtype=DTYPE))

    (gradient,) = torch.autograd.grad(
        result.x_final.sum() + result.log_det.sum(), (field.scale,)
    )
    torch.testing.assert_close(gradient, torch.zeros_like(gradient))


def test_constant_field_has_linear_state_zero_divergence_and_endpoint_gradient() -> None:
    """MATH-211: a coordinate-constant field translates without volume change."""
    x = torch.tensor([[0.2, -0.5]], dtype=DTYPE, device=DEVICE, requires_grad=True)
    velocity = torch.tensor([1.25, -0.75], dtype=DTYPE, device=DEVICE)

    result = _integrate(ConstantField(velocity), x, 0.2, 1.4, 0.17)
    expected = x.detach() + (1.4 - 0.2) * velocity
    torch.testing.assert_close(result.x_final, expected, rtol=0.0, atol=2e-15)
    torch.testing.assert_close(result.log_det, torch.zeros(1, dtype=DTYPE))

    (gradient,) = torch.autograd.grad(result.x_final.sum(), (x,))
    torch.testing.assert_close(gradient, torch.ones_like(x))


def test_scalar_linear_flow_state_divergence_and_density_identities() -> None:
    """MATH-212--215: verify Mohamud Section 3.2.1 for x_dot=a*x."""
    x = torch.tensor([[0.4], [-1.2]], dtype=DTYPE, device=DEVICE)
    a, ta, tb = 0.7, -0.25, 0.9

    result = _integrate(MatrixField(torch.tensor([[a]], dtype=DTYPE)), x, ta, tb, 0.01)
    torch.testing.assert_close(
        result.x_final, scalar_linear_state(x, a, ta, tb), rtol=2e-10, atol=2e-12
    )
    expected_divergence = linear_divergence_integral(a, ta, tb)
    torch.testing.assert_close(
        result.log_det,
        torch.full_like(result.log_det, expected_divergence),
        rtol=2e-14,
        atol=2e-14,
    )
    assert linear_log_density_change(a, ta, tb) == -expected_divergence


def test_cross_backend_linear_flow_and_full_objective_fixture() -> None:
    """MATH-1214/MATH-1215: shared Python/C++ state, density, and gradient fixture."""
    x = torch.tensor([[0.8]], dtype=DTYPE, device=DEVICE)
    a, ta, tb, dt = 0.7, -0.25, 0.9, 0.3
    field = ScalarLinearField(a)

    result = _integrate(field, x, ta, tb, dt)
    duration = tb - ta
    expected_state = x * math.exp(a * duration)
    expected_divergence = a * duration
    expected_gradient = duration * expected_state.square().sum() + duration

    torch.testing.assert_close(result.x_final, expected_state, rtol=2e-4, atol=2e-6)
    torch.testing.assert_close(
        result.divergence_integral,
        torch.full_like(result.divergence_integral, expected_divergence),
        rtol=2e-14,
        atol=2e-14,
    )
    torch.testing.assert_close(
        result.flow_log_abs_det_jacobian, result.divergence_integral
    )
    torch.testing.assert_close(result.log_density_change, -result.divergence_integral)

    objective = 0.5 * result.x_final.square().sum() + result.divergence_integral.sum()
    (gradient,) = torch.autograd.grad(objective, (field.coefficient,))
    torch.testing.assert_close(gradient, expected_gradient, rtol=3e-4, atol=3e-6)


def test_scalar_linear_reverse_interval_reverses_divergence_sign() -> None:
    """MATH-216: the oriented integral in Section 3.2.1 negates on reversal."""
    x = torch.tensor([[0.8]], dtype=DTYPE, device=DEVICE)
    a, ta, tb = -0.35, 1.2, -0.3

    result = _integrate(MatrixField(torch.tensor([[a]], dtype=DTYPE)), x, ta, tb, 0.02)
    expected = linear_divergence_integral(a, ta, tb)
    torch.testing.assert_close(result.log_det, torch.full_like(result.log_det, expected))


def test_diagonal_field_divergence_is_matrix_trace() -> None:
    """MATH-218: diagonal linear divergence equals trace(A)."""
    diagonal = torch.tensor([0.5, -0.2, 1.1], dtype=DTYPE, device=DEVICE)
    x = torch.tensor([[0.3, -0.7, 1.4]], dtype=DTYPE, device=DEVICE)
    ta, tb = 0.1, 0.8

    result = _integrate(MatrixField(torch.diag(diagonal)), x, ta, tb, 0.01)
    expected_state = x * torch.exp(diagonal * (tb - ta))
    expected_divergence = diagonal.sum().item() * (tb - ta)
    torch.testing.assert_close(result.x_final, expected_state, rtol=2e-10, atol=2e-12)
    torch.testing.assert_close(
        result.log_det, torch.full_like(result.log_det, expected_divergence)
    )


def test_rotation_has_zero_divergence_and_nontrivial_state() -> None:
    """MATH-219: planar rotation preserves volume while moving the state."""
    omega, duration = 1.3, 0.75
    matrix = torch.tensor([[0.0, -omega], [omega, 0.0]], dtype=DTYPE, device=DEVICE)
    x = torch.tensor([[0.6, -0.4]], dtype=DTYPE, device=DEVICE)

    result = _integrate(MatrixField(matrix), x, 0.0, duration, 0.005)
    angle = omega * duration
    rotation = torch.tensor(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
        dtype=DTYPE,
        device=DEVICE,
    )
    torch.testing.assert_close(result.x_final, x @ rotation.T, rtol=2e-10, atol=2e-12)
    torch.testing.assert_close(result.log_det, torch.zeros_like(result.log_det))


def test_time_linear_density_uses_matching_time_and_state() -> None:
    """MATH-220: x_dot=t*x exposes old-time/new-state divergence quadrature."""
    x = torch.tensor([[0.75]], dtype=DTYPE, device=DEVICE)
    ta, tb = 0.0, 1.0

    result = _integrate(TimeLinearField(), x, ta, tb, 0.1)
    torch.testing.assert_close(result.x_final, time_linear_state(x, ta, tb), rtol=2e-6, atol=1e-8)
    torch.testing.assert_close(
        result.log_det, torch.full_like(result.log_det, 0.5 * (tb * tb - ta * ta))
    )


def test_quadratic_flow_state_parameter_derivative() -> None:
    """MATH-217/MATH-221: exact state variation for x_dot=theta*x^2."""
    x = torch.tensor([[0.4]], dtype=DTYPE, device=DEVICE)
    field = QuadraticField(0.3)
    result = _integrate(field, x, 0.0, 0.8, 0.001)
    exact_state, _, exact_state_derivative, _ = quadratic_flow_quantities(
        x, field.theta.detach(), 0.8
    )

    torch.testing.assert_close(result.x_final, exact_state, rtol=2e-13, atol=2e-14)
    (state_derivative,) = torch.autograd.grad(result.x_final.sum(), (field.theta,), retain_graph=True)
    torch.testing.assert_close(
        state_derivative, exact_state_derivative.sum(), rtol=2e-12, atol=2e-13
    )


def test_quadratic_flow_density_parameter_derivative() -> None:
    """MATH-217/MATH-221: density variation includes theta dependence through x(t)."""
    x = torch.tensor([[0.4]], dtype=DTYPE, device=DEVICE)
    field = QuadraticField(0.3)
    result = _integrate(field, x, 0.0, 0.8, 0.001)
    _, exact_divergence, _, exact_divergence_derivative = quadratic_flow_quantities(
        x, field.theta.detach(), 0.8
    )

    torch.testing.assert_close(result.log_det, exact_divergence[..., 0], rtol=8e-4, atol=2e-7)
    (density_derivative,) = torch.autograd.grad(result.log_det.sum(), (field.theta,))
    torch.testing.assert_close(
        density_derivative, exact_divergence_derivative.sum(), rtol=8e-4, atol=2e-7
    )


def test_linear_flow_complete_nll_parameter_derivative() -> None:
    """MATH-217: fixed-data NLL derivative follows the Phase 0 analytic oracle."""
    data = torch.tensor([[0.8]], dtype=DTYPE, device=DEVICE)
    field = ScalarLinearField(0.35)
    duration = 0.9
    result = _integrate(field, data, duration, 0.0, 0.03)
    nll = 0.5 * result.x_final.square().sum() - result.log_det.sum()
    (derivative,) = torch.autograd.grad(nll, (field.coefficient,))

    base_state = data * torch.exp(-field.coefficient.detach() * duration)
    expected = duration * (1.0 - base_state.square()).sum()
    torch.testing.assert_close(derivative, expected, rtol=2e-8, atol=2e-10)


@pytest.mark.slow
@pytest.mark.convergence
def test_rk4_state_convergence_order_is_measured_from_refinements() -> None:
    """MATH-270/MATH-272/MATH-275: estimate state order from h, h/2, h/4."""
    x = torch.tensor([[0.9]], dtype=DTYPE, device=DEVICE)
    a, ta, tb = 1.1, 0.0, 1.0
    exact = scalar_linear_state(x, a, ta, tb)
    errors = []
    for step in (0.2, 0.1, 0.05):
        result = _integrate(MatrixField(torch.tensor([[a]], dtype=DTYPE)), x, ta, tb, step)
        errors.append(torch.max(torch.abs(result.x_final - exact)).item())

    orders = observed_order(errors)
    assert min(orders) > 3.8, f"state RK order too low: errors={errors}, orders={orders}"


@pytest.mark.slow
@pytest.mark.convergence
def test_density_convergence_order_is_measured_from_refinements() -> None:
    """MATH-271/MATH-272: augmented density integration must attain RK4 order."""
    x = torch.tensor([[0.9]], dtype=DTYPE, device=DEVICE)
    exact = 0.2
    errors = []
    for step in (0.2, 0.1, 0.05):
        result = _integrate(TimeQuarticField(), x, 0.0, 1.0, step)
        errors.append(abs(result.log_det.item() - exact))

    orders = observed_order(errors)
    assert min(orders) > 3.8, f"density RK order too low: errors={errors}, orders={orders}"


@pytest.mark.parametrize("dt", [0.0, -0.1, float("inf"), float("nan")])
def test_integrator_rejects_invalid_step_magnitudes(dt: float) -> None:
    """MATH-410: dt is a finite positive magnitude under the Phase 0 contract."""
    x = torch.tensor([[0.5]], dtype=DTYPE)
    with pytest.raises(ValueError, match="finite positive"):
        _integrate(ZeroField(), x, 0.0, 1.0, dt)


def test_zero_interval_and_trajectory_include_augmented_state() -> None:
    """MATH-408/MATH-416: exact endpoints retain time, state, and integral."""
    x = torch.tensor([[0.5]], dtype=DTYPE)
    field = ScalarLinearField(0.4)
    zero = integrate_rk4(field, EuclideanSpace(1), x, 0.3, 0.3, 0.1, True)
    torch.testing.assert_close(zero.x_final, x)
    torch.testing.assert_close(zero.divergence_integral, torch.zeros(1, dtype=DTYPE))
    assert zero.trajectory[0][0] == 0.3

    result = integrate_rk4(field, EuclideanSpace(1), x, 0.0, 1.0, 0.3, True)
    assert [entry[0] for entry in result.trajectory] == pytest.approx(
        [0.0, 0.3, 0.6, 0.9, 1.0]
    )
    torch.testing.assert_close(
        result.trajectory[-1][2], result.divergence_integral
    )
    torch.testing.assert_close(
        result.log_density_change, -result.flow_log_abs_det_jacobian
    )
