"""Single-chart ODE integration for Riemannian CNF flows."""

from __future__ import annotations

import math
from typing import Callable

import torch

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
    ) -> None:
        self.x_final = x_final
        self.divergence_integral = divergence_integral
        self.trajectory = trajectory

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
) -> FlowResult:
    """Integrate ``x_dot=f`` and ``I_dot=div_g f`` with augmented RK4.

    ``dt`` is a finite positive magnitude. The interval determines each
    step's sign. Trajectory entries are ``(time, state, divergence_integral)``.
    """
    if x0.dim() < 1:
        raise ValueError("x0 must have shape (..., dim); got 0-d tensor")
    if not x0.is_floating_point():
        raise TypeError("x0 must have a floating-point dtype")
    if not math.isfinite(float(t0)) or not math.isfinite(float(t1)):
        raise ValueError("t0 and t1 must be finite")
    if not math.isfinite(float(dt)) or dt <= 0.0:
        raise ValueError("dt must be a finite positive step magnitude")

    x = metric.canonicalize(x0.clone())
    integral = torch.zeros(x0.shape[:-1], device=x0.device, dtype=x0.dtype)
    trajectory: list[tuple[float, torch.Tensor, torch.Tensor]] = []
    if track_trajectory:
        trajectory.append((float(t0), x.clone(), integral.clone()))

    duration = abs(float(t1) - float(t0))
    if duration == 0.0:
        return FlowResult(x, integral, trajectory)

    step_magnitude = float(dt)
    n_steps = math.ceil(duration / step_magnitude)
    direction = 1.0 if t1 > t0 else -1.0
    t = float(t0)

    def augmented_rhs(
        time: float, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        metric.validate_points(state)
        if stage_callback is not None:
            stage_callback(time, state)
        time_tensor = torch.full(
            state.shape[:-1], time, device=state.device, dtype=state.dtype
        )
        field_value = vf(time_tensor, state)
        if not compute_divergence:
            return field_value, torch.zeros(
                state.shape[:-1], device=state.device, dtype=state.dtype
            )

        with torch.enable_grad():
            divergence_state = state
            if not divergence_state.requires_grad:
                divergence_state = divergence_state.clone().requires_grad_(True)

            def field_at_state(value: torch.Tensor) -> torch.Tensor:
                stage_time = torch.full(
                    value.shape[:-1], time, device=value.device, dtype=value.dtype
                )
                return vf(stage_time, value)

            divergence_value = divergence(
                field_at_state, divergence_state, metric
            )
        return field_value, divergence_value

    for step_index in range(n_steps):
        remaining = abs(float(t1) - t)
        h = direction * min(step_magnitude, remaining)
        if step_index == n_steps - 1:
            h = float(t1) - t
        half_h = h / 2.0

        k1_x, k1_i = augmented_rhs(t, x)
        k2_x, k2_i = augmented_rhs(t + half_h, x + half_h * k1_x)
        k3_x, k3_i = augmented_rhs(t + half_h, x + half_h * k2_x)
        k4_x, k4_i = augmented_rhs(t + h, x + h * k3_x)

        x = metric.canonicalize(
            x + (h / 6.0) * (k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x)
        )
        integral = integral + (h / 6.0) * (
            k1_i + 2.0 * k2_i + 2.0 * k3_i + k4_i
        )
        t = float(t1) if step_index == n_steps - 1 else t + h

        if track_trajectory:
            trajectory.append((t, x.clone(), integral.clone()))

    return FlowResult(x, integral, trajectory)
