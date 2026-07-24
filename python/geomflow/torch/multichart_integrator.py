"""Multi-chart integration helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .atlas import Atlas
from .base_distribution import AtlasBaseDistribution, StandardNormalCoordinateBase
from .multichart import MultiChartVectorField
from .operators import divergence


@dataclass
class ChartTransitionEvent:
    """An accepted coordinate transition at an exact solver time."""

    time: float
    source_chart: int
    target_chart: int
    source_coordinates: torch.Tensor
    target_coordinates: torch.Tensor


class MultiChartFlowResult:
    """Result of augmented flow integration across an atlas."""

    def __init__(
        self,
        x_final: torch.Tensor,
        chart_final: int,
        divergence_integral: torch.Tensor,
        trajectory: list[tuple[float, int, torch.Tensor, torch.Tensor]],
        transition_events: list[ChartTransitionEvent],
    ) -> None:
        self.x_final = x_final
        self.chart_final = chart_final
        self.divergence_integral = divergence_integral
        self.trajectory = trajectory
        self.transition_events = transition_events

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


def integrate_multichart(
    vf: MultiChartVectorField,
    atlas: Atlas,
    x0: torch.Tensor,
    start_chart: int,
    t0: float,
    t1: float,
    dt: float,
    track_trajectory: bool = False,
    compute_divergence: bool = True,
) -> MultiChartFlowResult:
    """Integrate ``x_dot=f`` and ``I_dot=div_g f`` with chart switching.

    All RK stages for an attempted step use its source chart. A rejected step
    commits neither state nor divergence. Riemannian density is scalar, so an
    accepted chart transition introduces no Jacobian jump in ``I``.
    """
    if x0.dim() < 2:
        raise ValueError(
            "integrate_multichart expects x0 of shape (batch, dim); "
            f"got shape {tuple(x0.shape)}"
        )
    if not x0.is_floating_point():
        raise TypeError("x0 must have a floating-point dtype")
    if start_chart not in atlas.charts:
        raise ValueError(f"unknown start chart {start_chart}")
    if not math.isfinite(float(t0)) or not math.isfinite(float(t1)):
        raise ValueError("t0 and t1 must be finite")
    if not math.isfinite(float(dt)) or dt <= 0.0:
        raise ValueError("dt must be a finite positive step magnitude")

    current_chart = start_chart
    x = x0.clone()
    integral = torch.zeros(x0.shape[:-1], device=x0.device, dtype=x0.dtype)
    trajectory: list[tuple[float, int, torch.Tensor, torch.Tensor]] = []
    transition_events: list[ChartTransitionEvent] = []
    t = float(t0)
    if track_trajectory:
        trajectory.append((t, current_chart, x.clone(), integral.clone()))

    duration = abs(float(t1) - float(t0))
    if duration == 0.0:
        return MultiChartFlowResult(
            x, current_chart, integral, trajectory, transition_events
        )

    direction = 1.0 if t1 > t0 else -1.0
    base_dt = float(dt)
    current_dt = base_dt
    min_dt = base_dt * 1e-4
    max_attempts = 20 * math.ceil(duration / base_dt) + 10

    def augmented_rhs(
        time: float, state: torch.Tensor, chart_id: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        time_tensor = torch.full(
            state.shape[:-1], time, device=state.device, dtype=state.dtype
        )
        field_value = vf(time_tensor, state, chart_id)
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
                return vf(stage_time, value, chart_id)

            divergence_value = divergence(
                field_at_state,
                divergence_state,
                atlas[chart_id].analytic_metric,
            )
        return field_value, divergence_value

    for _ in range(max_attempts):
        if t == float(t1):
            break

        remaining = abs(float(t1) - t)
        h = direction * min(current_dt, remaining)
        half_h = h / 2.0
        source_chart = current_chart

        k1_x, k1_i = augmented_rhs(t, x, source_chart)
        k2_x, k2_i = augmented_rhs(
            t + half_h, x + half_h * k1_x, source_chart
        )
        k3_x, k3_i = augmented_rhs(
            t + half_h, x + half_h * k2_x, source_chart
        )
        k4_x, k4_i = augmented_rhs(t + h, x + h * k3_x, source_chart)
        proposed_x = x + (h / 6.0) * (
            k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x
        )
        proposed_integral = integral + (h / 6.0) * (
            k1_i + 2.0 * k2_i + 2.0 * k3_i + k4_i
        )

        target_chart = source_chart
        accepted_x = proposed_x
        if not atlas[source_chart].is_inside(proposed_x).all():
            try:
                target_chart, accepted_x = atlas.best_chart(proposed_x, source_chart)
            except ValueError:
                current_dt *= 0.5
                if current_dt < min_dt:
                    raise RuntimeError(
                        "integrate_multichart: no chart covers the trajectory "
                        f"near t={t:.6g}; step fell below {min_dt:.6g}"
                    )
                continue

        next_t = float(t1) if remaining <= current_dt else t + h
        x = accepted_x
        integral = proposed_integral
        t = next_t
        current_chart = target_chart
        current_dt = base_dt

        if target_chart != source_chart:
            transition_events.append(
                ChartTransitionEvent(
                    t,
                    source_chart,
                    target_chart,
                    proposed_x.clone(),
                    accepted_x.clone(),
                )
            )
        if track_trajectory:
            trajectory.append((t, current_chart, x.clone(), integral.clone()))
    else:
        raise RuntimeError(
            f"integrate_multichart exceeded {max_attempts} bounded step attempts"
        )

    return MultiChartFlowResult(
        x, current_chart, integral, trajectory, transition_events
    )


def cnf_log_prob_multichart(
    vf: MultiChartVectorField,
    atlas: Atlas,
    x_data: torch.Tensor,
    start_chart: int,
    dt: float = 0.05,
    t0: float = 0.0,
    t1: float = 1.0,
    base_distribution: AtlasBaseDistribution | None = None,
) -> torch.Tensor:
    """Per-sample log density using Mohamud's signed divergence integral."""
    result = integrate_multichart(
        vf,
        atlas,
        x_data,
        start_chart,
        t1,
        t0,
        dt,
        track_trajectory=False,
        compute_divergence=True,
    )
    base = base_distribution or AtlasBaseDistribution(
        StandardNormalCoordinateBase(atlas[atlas.reference_chart_id].dim),
        atlas.reference_chart_id,
    )
    return (
        base.log_prob_volume(result.x_final, atlas, result.chart_final)
        + result.divergence_integral
    )


def cnf_nll_multichart(
    vf: MultiChartVectorField,
    atlas: Atlas,
    x_data: torch.Tensor,
    start_chart: int,
    dt: float = 0.05,
    t0: float = 0.0,
    t1: float = 1.0,
    base_distribution: AtlasBaseDistribution | None = None,
) -> torch.Tensor:
    """Mean NLL using Mohamud's signed Riemannian divergence integral."""
    return -cnf_log_prob_multichart(
        vf, atlas, x_data, start_chart, dt, t0, t1, base_distribution
    ).mean()
