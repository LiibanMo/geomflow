"""Single-chart ODE integration for Riemannian CNF flows."""

from __future__ import annotations

from typing import Callable

import torch

from ._utils import (
    validate_autocast_disabled,
    validate_supported_floating_tensor,
    validate_tensor_module_compatibility,
)
from ._schedule import FixedStepSchedule, checkpoint_due, validate_checkpoint_interval
from .analytic_metric import AnalyticMetric
from .operators import divergence
from .vector_field import ManifoldVectorField


class FlowResult:
    """Result of an augmented state/divergence integration.

    ``divergence_integral`` and ``flow_log_abs_det_jacobian`` are the signed
    integral of ``div_g f``. ``log_density_change`` is its negative, as in
    Mohamud's manifold volume-form density equation.
    """

    def __init__(
        self,
        x_final: torch.Tensor,
        divergence_integral: torch.Tensor,
        trajectory: list[tuple[float, torch.Tensor, torch.Tensor]],
        trajectory_checkpoint_interval: int = 1,
        trajectory_is_detached: bool = False,
    ) -> None:
        self.x_final = x_final
        self.divergence_integral = divergence_integral
        self.trajectory = trajectory
        self.trajectory_checkpoint_interval = trajectory_checkpoint_interval
        self.trajectory_is_detached = trajectory_is_detached

    @property
    def flow_log_abs_det_jacobian(self) -> torch.Tensor:
        return self.divergence_integral

    @property
    def log_density_change(self) -> torch.Tensor:
        return -self.divergence_integral

    @property
    def log_det(self) -> torch.Tensor:
        """Deprecated migration alias for ``divergence_integral``."""
        return self.divergence_integral


def _augmented_rk4_step(
    vf: ManifoldVectorField,
    metric: AnalyticMetric,
    x: torch.Tensor,
    time: float,
    step_size: float,
    *,
    compute_divergence: bool = True,
    stage_callback: Callable[[float, torch.Tensor], None] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance one intrinsic augmented RK4 step."""
    build_graph = torch.is_grad_enabled()

    def augmented_rhs(
        stage_time: float, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        metric.validate_points(state)
        if stage_callback is not None:
            stage_callback(stage_time, state)
        time_tensor = torch.full(
            state.shape[:-1], stage_time, device=state.device, dtype=state.dtype
        )
        field_value = vf(time_tensor, state)
        if not compute_divergence:
            return field_value, state.new_zeros(state.shape[:-1])

        with torch.enable_grad():
            divergence_state = state
            if not divergence_state.requires_grad:
                divergence_state = divergence_state.clone().requires_grad_(True)

            def field_at_state(value: torch.Tensor) -> torch.Tensor:
                value_time = torch.full(
                    value.shape[:-1], stage_time, device=value.device, dtype=value.dtype
                )
                return vf(value_time, value)

            divergence_value = divergence(field_at_state, divergence_state, metric)
            if not build_graph:
                divergence_value = divergence_value.detach()
        return field_value, divergence_value

    half_h = step_size / 2.0
    k1_x, k1_i = augmented_rhs(time, x)
    k2_x, k2_i = augmented_rhs(time + half_h, x + half_h * k1_x)
    k3_x, k3_i = augmented_rhs(time + half_h, x + half_h * k2_x)
    k4_x, k4_i = augmented_rhs(time + step_size, x + step_size * k3_x)
    x_next = metric.canonicalize(
        x + (step_size / 6.0) * (k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x)
    )
    integral_increment = (step_size / 6.0) * (
        k1_i + 2.0 * k2_i + 2.0 * k3_i + k4_i
    )
    return x_next, integral_increment


def integrate_rk4(
    vf: ManifoldVectorField,
    metric: AnalyticMetric,
    x0: torch.Tensor,
    t0: float,
    t1: float,
    dt: float,
    track_trajectory: bool = False,
    compute_divergence: bool = True,
    stage_callback: Callable[[float, torch.Tensor], None] | None = None,
    checkpoint_interval: int = 1,
    detach_trajectory: bool = False,
) -> FlowResult:
    """Integrate ``x_dot=f`` and ``I_dot=div_g f`` with augmented RK4.

    ``dt`` is a finite positive magnitude. The interval determines each
    step's sign. Trajectory entries are ``(time, state, divergence_integral)``.
    By default they retain autograd history. ``detach_trajectory=True`` stores
    replay-only checkpoints, and ``checkpoint_interval`` controls their spacing.
    """
    if x0.dim() < 1:
        raise ValueError("x0 must have shape (..., dim); got 0-d tensor")
    validate_autocast_disabled("integrate_rk4")
    validate_supported_floating_tensor(x0, "integrate_rk4")
    validate_tensor_module_compatibility(x0, vf, "integrate_rk4")
    schedule = FixedStepSchedule(t0, t1, dt)
    checkpoint_interval = validate_checkpoint_interval(checkpoint_interval)

    x = metric.canonicalize(x0.clone())
    integral = torch.zeros(x0.shape[:-1], device=x0.device, dtype=x0.dtype)
    trajectory: list[tuple[float, torch.Tensor, torch.Tensor]] = []
    def checkpoint(time: float) -> None:
        state = x.detach() if detach_trajectory else x.clone()
        density = integral.detach() if detach_trajectory else integral.clone()
        trajectory.append((time, state, density))

    if track_trajectory:
        checkpoint(schedule.t0)

    for step in schedule:
        x, integral_increment = _augmented_rk4_step(
            vf,
            metric,
            x,
            step.start,
            step.size,
            compute_divergence=compute_divergence,
            stage_callback=stage_callback,
        )
        integral = integral + integral_increment
        if track_trajectory and checkpoint_due(
            step.index + 1, len(schedule), checkpoint_interval
        ):
            checkpoint(step.end)

    return FlowResult(
        x, integral, trajectory, checkpoint_interval, detach_trajectory
    )
