"""Phase 5 fixed-step solver correctness and device contracts."""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from geomflow.torch import (
    Atlas,
    Chart,
    EuclideanSpace,
    Transition,
    integrate_multichart,
    integrate_rk4,
)
from geomflow.torch._schedule import FixedStepSchedule


class LinearField(nn.Module):
    def __init__(self, coefficient: float, device: torch.device) -> None:
        super().__init__()
        self.coefficient = nn.Parameter(
            torch.tensor(coefficient, dtype=torch.float64, device=device)
        )

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        del t
        return self.coefficient * x


class MultiLinearField(nn.Module):
    def forward(
        self, t: torch.Tensor, x: torch.Tensor, chart_id: int
    ) -> torch.Tensor:
        del t, chart_id
        return 0.5 * x


def _cuda_threshold_atlas(limit: float, *, covered: bool = True) -> Atlas:
    finite = lambda x: torch.isfinite(x).all(dim=-1)
    overlap = (
        (lambda x: x[..., 0] <= limit)
        if covered
        else (lambda x: x[..., 0] < 0)
    )
    chart0 = Chart(
        0,
        1,
        None,
        EuclideanSpace(1),
        transitions={1: Transition(lambda x: x, overlap)},
        domain=lambda x: x[..., 0] <= limit,
    )
    chart1 = Chart(1, 1, None, EuclideanSpace(1), domain=finite)
    return Atlas([chart0, chart1], 0)


@pytest.mark.parametrize(
    ("t0", "t1", "dt", "expected"),
    [
        (0.0, 0.01, 1.0, [0.0, 0.01]),
        (0.0, 1.0, 0.3, [0.0, 0.3, 0.6, 0.9, 1.0]),
        (1.0, -0.1, 0.4, [1.0, 0.6, 0.2, -0.1]),
        (0.4, 0.4, 0.1, [0.4]),
    ],
)
def test_scalar_schedule_has_exact_bounded_endpoints(
    t0: float, t1: float, dt: float, expected: list[float]
) -> None:
    schedule = FixedStepSchedule(t0, t1, dt)
    actual = [schedule.t0, *(step.end for step in schedule)]
    assert actual == pytest.approx(expected, abs=1e-15)
    if len(schedule):
        assert actual[-1] == t1


def test_scalar_schedule_is_lazy_for_large_step_counts() -> None:
    schedule = FixedStepSchedule(0.0, 1.0, 1e-7)
    assert len(schedule) == 10_000_000
    iterator = iter(schedule)
    assert next(iterator).size == pytest.approx(1e-7)


@pytest.mark.parametrize("dt", [0.0, -1.0, math.inf, -math.inf, math.nan])
def test_both_schedule_and_solver_reject_invalid_dt(dt: float) -> None:
    with pytest.raises(ValueError, match="finite positive"):
        FixedStepSchedule(0.0, 1.0, dt)


def test_zero_interval_and_checkpoint_semantics() -> None:
    field = LinearField(0.2, torch.device("cpu"))
    x = torch.tensor([[2.0]], dtype=torch.float64, requires_grad=True)
    result = integrate_rk4(
        field,
        EuclideanSpace(1),
        x,
        0.5,
        0.5,
        0.1,
        track_trajectory=True,
        checkpoint_interval=4,
        detach_trajectory=True,
    )
    assert len(result.trajectory) == 1
    assert result.trajectory[0][0] == 0.5
    assert not result.trajectory[0][1].requires_grad
    assert result.trajectory_checkpoint_interval == 4
    assert result.trajectory_is_detached
    torch.testing.assert_close(result.x_final, x)
    torch.testing.assert_close(
        result.divergence_integral, torch.zeros(1, dtype=x.dtype)
    )


def test_checkpoint_interval_keeps_initial_periodic_and_final_states() -> None:
    field = LinearField(0.0, torch.device("cpu"))
    x = torch.tensor([[1.0]], dtype=torch.float64)
    result = integrate_rk4(
        field,
        EuclideanSpace(1),
        x,
        0.0,
        1.0,
        0.1,
        track_trajectory=True,
        checkpoint_interval=3,
    )
    assert [time for time, _, _ in result.trajectory] == pytest.approx(
        [0.0, 0.3, 0.6, 0.9, 1.0]
    )


def test_augmented_rk4_state_and_density_have_fourth_order_convergence() -> None:
    coefficient = 0.7
    exact_state = math.exp(coefficient)

    def error(dt: float) -> tuple[float, float]:
        field = LinearField(coefficient, torch.device("cpu"))
        x = torch.ones((1, 1), dtype=torch.float64)
        result = integrate_rk4(field, EuclideanSpace(1), x, 0.0, 1.0, dt)
        return (
            abs(result.x_final.item() - exact_state),
            abs(result.divergence_integral.item() - coefficient),
        )

    coarse = error(0.2)
    fine = error(0.1)
    assert coarse[0] / fine[0] > 12.0
    # Constant divergence is integrated exactly at every RK stage.
    assert coarse[1] < 1e-14 and fine[1] < 1e-14


@pytest.mark.parametrize(
    "device",
    [torch.device("cpu")]
    + ([torch.device("cuda")] if torch.cuda.is_available() else []),
)
def test_state_only_inference_has_no_graph_and_stays_on_device(
    device: torch.device,
) -> None:
    field = LinearField(0.2, device)
    x = torch.ones((8, 1), dtype=torch.float64, device=device)
    with torch.inference_mode():
        result = integrate_rk4(
            field,
            EuclideanSpace(1),
            x,
            1.0,
            -0.25,
            0.13,
            compute_divergence=False,
        )
    assert result.x_final.device.type == device.type
    assert result.divergence_integral.device.type == device.type
    assert not result.x_final.requires_grad
    assert result.x_final.grad_fn is None
    assert result.trajectory == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("limit", [0.11, 10.0])
def test_cuda_multichart_switching_and_no_switch_stay_device_native(
    limit: float,
) -> None:
    device = torch.device("cuda")
    x = torch.tensor([[0.1]], dtype=torch.float64, device=device, requires_grad=True)
    result = integrate_multichart(
        MultiLinearField(),
        _cuda_threshold_atlas(limit),
        x,
        0,
        0.0,
        0.4,
        0.2,
        checkpoint_interval=2,
    )
    assert result.x_final.device.type == device.type
    assert result.divergence_integral.device.type == device.type
    assert len(result.transition_events) == (1 if limit < 1.0 else 0)
    assert result.operations == []
    (gradient,) = torch.autograd.grad(
        result.x_final.sum() + result.divergence_integral.sum(), x
    )
    assert gradient.device.type == device.type


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_uncovered_rejection_does_not_mutate_input() -> None:
    device = torch.device("cuda")
    x = torch.tensor([[0.1]], dtype=torch.float64, device=device)
    original = x.clone()
    with pytest.raises(RuntimeError, match="overlap|leaves chart"):
        integrate_multichart(
            MultiLinearField(),
            _cuda_threshold_atlas(0.11, covered=False),
            x,
            0,
            0.0,
            0.4,
            0.2,
            max_subdivisions=12,
        )
    torch.testing.assert_close(x, original)
