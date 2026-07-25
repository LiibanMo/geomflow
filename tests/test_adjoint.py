"""Phase 6 oracles for Mohamud's Python intrinsic adjoint."""

from __future__ import annotations

import copy
import math

import pytest
import torch
from torch import nn

from geomflow.torch import (
    AnalyticMetric,
    EuclideanSpace,
    ManifoldVectorField,
    StandardNormalCoordinateBase,
    UniformAngleCoordinateBase,
    cnf_nll,
    intrinsic_adjoint_nll,
)

from analytic_references import central_difference, observed_order


DTYPE = torch.float64


class LinearField(nn.Module):
    def __init__(self, coefficient: float) -> None:
        super().__init__()
        self.coefficient = nn.Parameter(torch.tensor(coefficient, dtype=DTYPE))
        self.inactive = nn.Parameter(torch.tensor(0.7, dtype=DTYPE))

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        del t
        return self.coefficient * x + self.inactive * x * 0.0


class QuadraticField(nn.Module):
    def __init__(self, coefficient: float) -> None:
        super().__init__()
        self.coefficient = nn.Parameter(torch.tensor(coefficient, dtype=DTYPE))

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        del t
        return self.coefficient * x.square()


class AffineField(nn.Module):
    def __init__(self, slope: float, offset: float) -> None:
        super().__init__()
        self.slope = nn.Parameter(torch.tensor(slope, dtype=DTYPE))
        self.offset = nn.Parameter(torch.tensor(offset, dtype=DTYPE))

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        del t
        return self.slope * x + self.offset


class ZeroField(nn.Module):
    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        del t
        return torch.zeros_like(x)


class ConstantField(nn.Module):
    def __init__(self, speed: float) -> None:
        super().__init__()
        self.speed = nn.Parameter(torch.tensor(speed, dtype=DTYPE))

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        del t
        return self.speed.expand_as(x)


class CompactUniformVolumeBase:
    dim = 1

    def sample(self, sample_shape, *, device, dtype):
        return 2.0 * torch.rand(*sample_shape, 1, device=device, dtype=dtype) - 1.0

    def log_prob_volume(self, x: torch.Tensor, metric) -> torch.Tensor:
        del metric
        if not self.contains(x).all():
            raise ValueError("outside compact support")
        return torch.full(x.shape[:-1], -math.log(2.0), dtype=x.dtype, device=x.device)

    def contains(self, x: torch.Tensor) -> torch.Tensor:
        return (x.abs() < 1.0).all(dim=-1)


class ScaledNormalVolumeBase:
    dim = 1

    def sample(self, sample_shape, *, device, dtype):
        return 2.0 * torch.randn(*sample_shape, 1, device=device, dtype=dtype)

    def log_prob_volume(self, x: torch.Tensor, metric) -> torch.Tensor:
        del metric
        return -0.5 * (math.log(2.0 * math.pi) + (x[..., 0] / 2.0).square())

    def contains(self, x: torch.Tensor) -> torch.Tensor:
        return torch.isfinite(x).all(dim=-1)

    def __bool__(self) -> bool:
        return False


class LogChartField(nn.Module):
    def __init__(self, coefficient: float) -> None:
        super().__init__()
        self.coefficient = nn.Parameter(torch.tensor(coefficient, dtype=DTYPE))

    def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        del t
        return self.coefficient * y * torch.log(y)


class LogNormalVolumeBase:
    dim = 1

    def sample(self, sample_shape, *, device, dtype):
        return torch.exp(torch.randn(*sample_shape, 1, device=device, dtype=dtype))

    def log_prob_volume(self, y: torch.Tensor, metric) -> torch.Tensor:
        del metric
        return -0.5 * (math.log(2.0 * math.pi) + torch.log(y[..., 0]).square())

    def contains(self, y: torch.Tensor) -> torch.Tensor:
        return torch.isfinite(y).all(dim=-1) & (y > 0.0).all(dim=-1)


class BufferedSharedField(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        shared = nn.Parameter(torch.tensor(0.2, dtype=DTYPE))
        self.left = shared
        self.right = shared
        self.register_buffer("scale", torch.tensor(0.5, dtype=DTYPE))

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        del t
        return (self.left + self.right) * self.scale * x


def _gradients(loss: torch.Tensor, field: nn.Module, data: torch.Tensor):
    return torch.autograd.grad(loss, (*field.parameters(), data))


def test_custom_forward_exactly_matches_corrected_direct_forward() -> None:
    """MATH-720/MATH-750: custom and direct paths execute the same augmented RK4."""
    data = torch.tensor([[0.35], [0.8]], dtype=DTYPE)
    field = QuadraticField(0.2)

    direct = cnf_nll(field, EuclideanSpace(1), data, dt=0.3, t1=0.8)
    adjoint = intrinsic_adjoint_nll(field, EuclideanSpace(1), data, dt=0.3, t1=0.8)

    torch.testing.assert_close(adjoint, direct, rtol=0.0, atol=0.0)


def test_input_and_every_parameter_gradient_match_direct_autograd() -> None:
    """MATH-710--716/MATH-751--752: ordered VJPs include explicit zeros."""
    direct_data = torch.tensor([[0.4], [0.9]], dtype=DTYPE, requires_grad=True)
    adjoint_data = direct_data.detach().clone().requires_grad_(True)
    direct_field = LinearField(0.3)
    adjoint_field = LinearField(0.3)

    direct = _gradients(
        cnf_nll(direct_field, EuclideanSpace(1), direct_data, dt=0.2),
        direct_field,
        direct_data,
    )
    adjoint = _gradients(
        intrinsic_adjoint_nll(adjoint_field, EuclideanSpace(1), adjoint_data, dt=0.2),
        adjoint_field,
        adjoint_data,
    )

    for actual, expected in zip(adjoint, direct):
        torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(adjoint[1], torch.zeros_like(adjoint[1]))


def test_every_vector_field_layer_gradient_matches_direct_autograd() -> None:
    """MATH-744/MATH-752: every trainable MLP tensor receives its ordered VJP."""
    torch.manual_seed(17)
    direct_field = ManifoldVectorField(2, hidden_dim=4, n_layers=2).to(DTYPE)
    adjoint_field = copy.deepcopy(direct_field)
    direct_data = torch.tensor([[0.2, -0.4], [0.7, 0.3]], dtype=DTYPE)
    adjoint_data = direct_data.clone()

    direct_loss = cnf_nll(direct_field, EuclideanSpace(2), direct_data, dt=0.5)
    adjoint_loss = intrinsic_adjoint_nll(
        adjoint_field, EuclideanSpace(2), adjoint_data, dt=0.5
    )
    direct_gradients = torch.autograd.grad(direct_loss, tuple(direct_field.parameters()))
    adjoint_gradients = torch.autograd.grad(
        adjoint_loss, tuple(adjoint_field.parameters())
    )

    assert len(adjoint_gradients) == len(tuple(adjoint_field.parameters()))
    for actual, expected in zip(adjoint_gradients, direct_gradients):
        torch.testing.assert_close(actual, expected, rtol=2e-11, atol=2e-11)


def test_shared_parameter_and_buffer_are_preserved() -> None:
    """MATH-711--713: functional replay keeps tied parameters and buffers."""
    data = torch.tensor([[0.7]], dtype=DTYPE, requires_grad=True)
    field = BufferedSharedField()
    assert len(tuple(field.parameters())) == 1

    direct_loss = cnf_nll(field, EuclideanSpace(1), data, dt=0.25)
    direct = torch.autograd.grad(direct_loss, (field.left, data), retain_graph=True)
    adjoint_loss = intrinsic_adjoint_nll(field, EuclideanSpace(1), data, dt=0.25)
    field.scale.fill_(1.5)
    adjoint = torch.autograd.grad(adjoint_loss, (field.left, data))

    torch.testing.assert_close(adjoint_loss, direct_loss, rtol=0.0, atol=0.0)
    for actual, expected in zip(adjoint, direct):
        torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-12)


def test_parameter_gradient_matches_independent_finite_difference() -> None:
    """MATH-740--744/MATH-753--755: direct and state-mediated terms are present."""
    data = torch.tensor([[0.35], [0.8]], dtype=DTYPE)
    field = QuadraticField(0.2)
    loss = intrinsic_adjoint_nll(field, EuclideanSpace(1), data, dt=0.025, t1=0.9)
    (actual,) = torch.autograd.grad(loss, (field.coefficient,))

    expected = central_difference(
        lambda value: cnf_nll(
            QuadraticField(value), EuclideanSpace(1), data, dt=0.025, t1=0.9
        ).item(),
        0.2,
    )
    torch.testing.assert_close(actual, torch.tensor(expected, dtype=DTYPE), rtol=3e-7, atol=3e-9)


def test_every_tiny_model_parameter_matches_independent_finite_difference() -> None:
    """MATH-753: finite differences independently cover each tiny-model parameter."""
    data = torch.tensor([[0.25], [0.75]], dtype=DTYPE)
    field = AffineField(0.2, -0.1)
    loss = intrinsic_adjoint_nll(field, EuclideanSpace(1), data, dt=0.05, t1=0.7)
    actual = torch.autograd.grad(loss, (field.slope, field.offset))

    values = (0.2, -0.1)
    for index, gradient in enumerate(actual):
        expected = central_difference(
            lambda value: cnf_nll(
                AffineField(*(values[:index] + (value,) + values[index + 1 :])),
                EuclideanSpace(1),
                data,
                dt=0.05,
                t1=0.7,
            ).item(),
            values[index],
        )
        torch.testing.assert_close(
            gradient, torch.tensor(expected, dtype=DTYPE), rtol=3e-7, atol=3e-9
        )


def test_non_euclidean_cotangent_matches_direct_autograd() -> None:
    """MATH-734/MATH-737/MATH-756: volume and divergence derivatives stay intrinsic."""
    metric = AnalyticMetric(
        1,
        lambda x: torch.exp(2.0 * x[..., :1]).unsqueeze(-1),
        inverse_fn=lambda x: torch.exp(-2.0 * x[..., :1]).unsqueeze(-1),
        sqrt_det_fn=lambda x: torch.exp(x[..., 0]),
    )
    direct_data = torch.tensor([[0.8]], dtype=DTYPE, requires_grad=True)
    adjoint_data = direct_data.detach().clone().requires_grad_(True)
    direct_field = LinearField(0.25)
    adjoint_field = LinearField(0.25)

    direct = _gradients(cnf_nll(direct_field, metric, direct_data, dt=0.03, t1=0.9), direct_field, direct_data)
    adjoint = _gradients(
        intrinsic_adjoint_nll(adjoint_field, metric, adjoint_data, dt=0.03, t1=0.9),
        adjoint_field,
        adjoint_data,
    )
    for actual, expected in zip(adjoint, direct):
        torch.testing.assert_close(actual, expected, rtol=2e-11, atol=2e-11)


def test_adjoint_cotangent_obeys_chart_pullback() -> None:
    """MATH-737: y=2x coordinates produce dL/dy=(dx/dy)dL/dx."""
    x_data = torch.tensor([[0.4], [0.9]], dtype=DTYPE, requires_grad=True)
    y_data = (2.0 * x_data.detach()).requires_grad_(True)
    x_field = LinearField(0.25)
    y_field = LinearField(0.25)
    y_metric = AnalyticMetric(
        1,
        lambda y: torch.full((*y.shape[:-1], 1, 1), 0.25, dtype=y.dtype, device=y.device),
        inverse_fn=lambda y: torch.full(
            (*y.shape[:-1], 1, 1), 4.0, dtype=y.dtype, device=y.device
        ),
        sqrt_det_fn=lambda y: torch.full(
            y.shape[:-1], 0.5, dtype=y.dtype, device=y.device
        ),
    )

    x_loss = intrinsic_adjoint_nll(x_field, EuclideanSpace(1), x_data, dt=0.2)
    y_loss = intrinsic_adjoint_nll(
        y_field,
        y_metric,
        y_data,
        dt=0.2,
        base_distribution=ScaledNormalVolumeBase(),
    )
    x_covector, x_parameter = torch.autograd.grad(
        x_loss, (x_data, x_field.coefficient)
    )
    y_covector, y_parameter = torch.autograd.grad(
        y_loss, (y_data, y_field.coefficient)
    )

    torch.testing.assert_close(y_loss, x_loss, rtol=2e-13, atol=2e-13)
    torch.testing.assert_close(y_covector, 0.5 * x_covector, rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(y_parameter, x_parameter, rtol=2e-12, atol=2e-12)


def test_nonlinear_chart_cotangent_covariance_converges_at_rk4_order() -> None:
    """MATH-737: nonlinear-chart covariance holds to the solver's error order."""
    y_metric = AnalyticMetric(
        1,
        lambda y: y.reciprocal().square().unsqueeze(-1),
        inverse_fn=lambda y: y.square().unsqueeze(-1),
        sqrt_det_fn=lambda y: y[..., 0].reciprocal(),
    )
    errors: list[float] = []
    for step in (0.2, 0.1, 0.05):
        x_data = torch.tensor([[0.3], [0.7]], dtype=DTYPE, requires_grad=True)
        y_data = torch.exp(x_data.detach()).requires_grad_(True)
        x_field = LinearField(0.35)
        y_field = LogChartField(0.35)

        x_loss = intrinsic_adjoint_nll(
            x_field, EuclideanSpace(1), x_data, dt=step, t1=0.8
        )
        y_loss = intrinsic_adjoint_nll(
            y_field,
            y_metric,
            y_data,
            dt=step,
            t1=0.8,
            base_distribution=LogNormalVolumeBase(),
        )
        x_covector, x_parameter = torch.autograd.grad(
            x_loss, (x_data, x_field.coefficient)
        )
        y_covector, y_parameter = torch.autograd.grad(
            y_loss, (y_data, y_field.coefficient)
        )
        cotangent_error = (y_covector - x_covector / y_data).abs().max()
        parameter_error = (y_parameter - x_parameter).abs()
        loss_error = (y_loss - x_loss).abs()
        errors.append(max(cotangent_error, parameter_error, loss_error).item())

    orders = observed_order(errors)
    assert min(orders) > 3.5, f"chart covariance: errors={errors}, orders={orders}"


@pytest.mark.parametrize("t0,t1", [(0.0, 1.0), (1.0, 0.0)])
def test_orientation_and_remainder_step_match_direct_autograd(t0: float, t1: float) -> None:
    """MATH-721--725/MATH-736/MATH-757: replay uses signed accepted steps exactly."""
    direct_data = torch.tensor([[0.6]], dtype=DTYPE, requires_grad=True)
    adjoint_data = direct_data.detach().clone().requires_grad_(True)
    direct_field = QuadraticField(0.12)
    adjoint_field = QuadraticField(0.12)

    direct_loss = cnf_nll(direct_field, EuclideanSpace(1), direct_data, dt=0.3, t0=t0, t1=t1)
    adjoint_loss = intrinsic_adjoint_nll(
        adjoint_field, EuclideanSpace(1), adjoint_data, dt=0.3, t0=t0, t1=t1
    )
    direct = _gradients(direct_loss, direct_field, direct_data)
    adjoint = _gradients(adjoint_loss, adjoint_field, adjoint_data)

    torch.testing.assert_close(adjoint_loss, direct_loss, rtol=0.0, atol=0.0)
    for actual, expected in zip(adjoint, direct):
        torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-12)


def test_mean_reduction_scales_per_sample_cotangents() -> None:
    """MATH-758: duplicating a batch preserves loss and parameter gradient."""
    data = torch.tensor([[0.4], [0.9]], dtype=DTYPE, requires_grad=True)
    duplicated = data.detach().repeat(2, 1).requires_grad_(True)
    field = LinearField(0.3)
    duplicate_field = LinearField(0.3)

    loss = intrinsic_adjoint_nll(field, EuclideanSpace(1), data, dt=0.2)
    duplicate_loss = intrinsic_adjoint_nll(
        duplicate_field, EuclideanSpace(1), duplicated, dt=0.2
    )
    parameter_gradient, data_gradient = torch.autograd.grad(loss, (field.coefficient, data))
    duplicate_parameter_gradient, duplicate_data_gradient = torch.autograd.grad(
        duplicate_loss, (duplicate_field.coefficient, duplicated)
    )

    torch.testing.assert_close(duplicate_loss, loss)
    torch.testing.assert_close(duplicate_parameter_gradient, parameter_gradient)
    torch.testing.assert_close(duplicate_data_gradient[:2] * 2.0, data_gradient)


def test_higher_order_gradients_are_explicitly_unsupported() -> None:
    """MATH-759: the discrete intrinsic adjoint is first-order only."""
    data = torch.tensor([[0.7]], dtype=DTYPE, requires_grad=True)
    field = LinearField(0.2)
    loss = intrinsic_adjoint_nll(field, EuclideanSpace(1), data, dt=0.2)
    (gradient,) = torch.autograd.grad(loss, (field.coefficient,), create_graph=True)
    assert not gradient.requires_grad


def test_disconnected_constant_objective_returns_zero_input_cotangent() -> None:
    """MATH-716: a truly graph-independent objective materializes zero."""
    data = torch.tensor([[0.3]], dtype=DTYPE, requires_grad=True)
    loss = intrinsic_adjoint_nll(
        ZeroField(),
        EuclideanSpace(1),
        data,
        base_distribution=UniformAngleCoordinateBase(1),
    )
    (gradient,) = torch.autograd.grad(loss, (data,))
    torch.testing.assert_close(gradient, torch.zeros_like(data))


def test_base_dependency_probe_uses_replayed_base_endpoint() -> None:
    """MATH-750: compact support is checked at the base endpoint, not data."""
    data = torch.tensor([[1.2]], dtype=DTYPE, requires_grad=True)
    field = ConstantField(0.5)
    base = CompactUniformVolumeBase()
    direct = cnf_nll(field, EuclideanSpace(1), data, dt=0.25, base_distribution=base)
    adjoint = intrinsic_adjoint_nll(
        field, EuclideanSpace(1), data, dt=0.25, base_distribution=base
    )
    torch.testing.assert_close(adjoint, direct, rtol=0.0, atol=0.0)


def test_trainable_geometry_and_base_are_rejected() -> None:
    """MATH-745--746: the supported adjoint scope requires fixed configuration."""
    data = torch.tensor([[0.7]], dtype=DTYPE)
    field = LinearField(0.2)
    metric = EuclideanSpace(1)
    metric.nested_state = {
        "scales": [torch.tensor(1.0, dtype=DTYPE, requires_grad=True)]
    }
    with pytest.raises(ValueError, match="trainable metric"):
        intrinsic_adjoint_nll(field, metric, data)

    base = StandardNormalCoordinateBase(1)
    base.nested_state = {
        "scales": [torch.tensor(1.0, dtype=DTYPE, requires_grad=True)]
    }
    with pytest.raises(ValueError, match="trainable base-distribution"):
        intrinsic_adjoint_nll(field, EuclideanSpace(1), data, base_distribution=base)


def test_hidden_trainable_metric_dependency_is_rejected() -> None:
    """MATH-745: runtime probing catches dependencies hidden from attributes."""
    scale = torch.tensor(1.0, dtype=DTYPE, requires_grad=True)

    def metric_fn(x: torch.Tensor) -> torch.Tensor:
        return (scale * torch.ones_like(x)).unsqueeze(-1)

    metric = AnalyticMetric(1, metric_fn)
    with pytest.raises(ValueError, match="trainable metric"):
        intrinsic_adjoint_nll(LinearField(0.2), metric, torch.tensor([[0.7]], dtype=DTYPE))


def test_stage_local_trainable_metric_dependency_is_rejected() -> None:
    """MATH-745: fixed-geometry validation runs at every accepted RK stage."""
    scale = torch.tensor(0.2, dtype=DTYPE, requires_grad=True)

    def metric_value(x: torch.Tensor) -> torch.Tensor:
        if bool(((x > 0.2) & (x < 0.8)).all()):
            return torch.exp(2.0 * scale * x).unsqueeze(-1)
        return torch.ones_like(x).unsqueeze(-1)

    metric = AnalyticMetric(1, metric_value)
    data = torch.tensor([[1.0]], dtype=DTYPE)
    with pytest.raises(ValueError, match="trainable metric"):
        intrinsic_adjoint_nll(ConstantField(1.0), metric, data, dt=1.0)
