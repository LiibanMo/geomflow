"""Multi-chart integration helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from ._utils import validate_autocast_disabled, validate_supported_floating_tensor

from ._schedule import FixedStepSchedule, checkpoint_due, validate_checkpoint_interval
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
    transition_jacobian: torch.Tensor


@dataclass
class AcceptedChartSegment:
    """One accepted RK4 segment in a single coordinate chart."""

    t_start: float
    t_end: float
    chart_id: int
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
        operations: list[AcceptedChartSegment | ChartTransitionEvent] | None = None,
        trajectory_checkpoint_interval: int = 1,
        trajectory_is_detached: bool = False,
    ) -> None:
        self.x_final = x_final
        self.chart_final = chart_final
        self.divergence_integral = divergence_integral
        self.trajectory = trajectory
        self.transition_events = transition_events
        self.operations = operations or []
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


def replay_transition_pullbacks(
    transition_events: list[ChartTransitionEvent],
    terminal_covector: torch.Tensor,
) -> torch.Tensor:
    """Replay recorded coordinate-event pullbacks in reverse event order.

    This handles only instantaneous coordinate changes. Segment cotangent
    dynamics remain those of the selected direct-autograd or adjoint solver.
    """
    covector = terminal_covector
    for event in reversed(transition_events):
        covector = torch.matmul(
            event.transition_jacobian.transpose(-1, -2),
            covector.unsqueeze(-1),
        ).squeeze(-1)
    return covector


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
    checkpoint_interval: int = 1,
    detach_trajectory: bool = False,
    min_step: float | None = None,
    max_subdivisions: int = 20,
    record_operations: bool = False,
) -> MultiChartFlowResult:
    """Integrate ``x_dot=f`` and ``I_dot=div_g f`` with chart switching.

    One chart id applies to the complete batch. Every RK stage is checked before
    field evaluation. Unsafe intervals are bisected to a source-valid point in
    a declared overlap, then their remainder is integrated in the target chart.
    Riemannian density is scalar, so transitions introduce no jump in ``I``.
    ``record_operations`` opts into full accepted-segment retention for replay;
    normal direct-autograd integration stores only transition events.
    """
    if x0.dim() < 2:
        raise ValueError(
            "integrate_multichart expects x0 of shape (batch, dim); "
            f"got shape {tuple(x0.shape)}"
        )
    validate_autocast_disabled("integrate_multichart")
    validate_supported_floating_tensor(x0, "integrate_multichart")
    if start_chart not in atlas.charts:
        raise ValueError(f"unknown start chart {start_chart}")
    schedule = FixedStepSchedule(t0, t1, dt)
    checkpoint_interval = validate_checkpoint_interval(checkpoint_interval)
    if isinstance(max_subdivisions, bool) or not isinstance(max_subdivisions, int):
        raise ValueError("max_subdivisions must be a positive integer")
    if max_subdivisions <= 0:
        raise ValueError("max_subdivisions must be a positive integer")
    minimum_step = schedule.dt * 1e-4 if min_step is None else float(min_step)
    if not math.isfinite(minimum_step) or minimum_step <= 0.0:
        raise ValueError("min_step must be a finite positive magnitude")

    current_chart = start_chart
    x = x0.clone()
    integral = torch.zeros(x0.shape[:-1], device=x0.device, dtype=x0.dtype)
    trajectory: list[tuple[float, int, torch.Tensor, torch.Tensor]] = []
    transition_events: list[ChartTransitionEvent] = []
    operations: list[AcceptedChartSegment | ChartTransitionEvent] = []
    def checkpoint(time: float, chart_id: int) -> None:
        state = x.detach() if detach_trajectory else x.clone()
        density = integral.detach() if detach_trajectory else integral.clone()
        trajectory.append((time, chart_id, state, density))

    if track_trajectory:
        checkpoint(schedule.t0, current_chart)

    build_graph = torch.is_grad_enabled()

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
            if not build_graph:
                divergence_value = divergence_value.detach()
        return field_value, divergence_value

    def chart_contains(chart_id: int, state: torch.Tensor) -> torch.Tensor:
        chart = atlas[chart_id]
        predicate = getattr(chart, "contains", chart.is_inside)
        return predicate(state)

    def rk4_trial(
        time: float,
        state: torch.Tensor,
        integral_state: torch.Tensor,
        h: float,
        chart_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if not chart_contains(chart_id, state).all():
            return None
        half_h = h / 2.0
        k1_x, k1_i = augmented_rhs(time, state, chart_id)
        stage2 = state + half_h * k1_x
        if not chart_contains(chart_id, stage2).all():
            return None
        k2_x, k2_i = augmented_rhs(time + half_h, stage2, chart_id)
        stage3 = state + half_h * k2_x
        if not chart_contains(chart_id, stage3).all():
            return None
        k3_x, k3_i = augmented_rhs(time + half_h, stage3, chart_id)
        stage4 = state + h * k3_x
        if not chart_contains(chart_id, stage4).all():
            return None
        k4_x, k4_i = augmented_rhs(time + h, stage4, chart_id)
        proposed_x = state + (h / 6.0) * (
            k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x
        )
        if not chart_contains(chart_id, proposed_x).all():
            return None
        proposed_integral = integral_state + (h / 6.0) * (
            k1_i + 2.0 * k2_i + 2.0 * k3_i + k4_i
        )
        return proposed_x, proposed_integral

    max_segments_per_step = max_subdivisions + 1
    for scheduled_step in schedule:
        t = scheduled_step.start
        nominal_end = scheduled_step.end
        for _segment_index in range(max_segments_per_step):
            if t == nominal_end:
                break
            source_chart = current_chart
            h = nominal_end - t
            source_x = x
            trial = rk4_trial(t, x, integral, h, source_chart)
            if trial is not None:
                x, integral = trial
                if record_operations:
                    operations.append(
                        AcceptedChartSegment(
                            t, nominal_end, source_chart, source_x.clone(), x.clone()
                        )
                    )
                t = nominal_end
                break

            lo = 0.0
            hi = h
            safe_trial: tuple[torch.Tensor, torch.Tensor] | None = None
            for _attempt in range(max_subdivisions):
                if abs(hi - lo) <= minimum_step:
                    break
                mid = (lo + hi) / 2.0
                candidate = rk4_trial(t, x, integral, mid, source_chart)
                if candidate is None:
                    hi = mid
                else:
                    lo = mid
                    safe_trial = candidate
            if safe_trial is None or abs(lo) <= minimum_step:
                raise RuntimeError(
                    f"trajectory leaves chart {source_chart} near t={t:.6g} "
                    "before a valid overlap transition"
                )

            event_time = t + lo
            event_source, event_integral = safe_trial
            target_chart = None
            event_target = None
            source = atlas[source_chart]
            if hasattr(source, "transitions"):
                for candidate_id in sorted(source.transitions):
                    if not source.can_transition_to(candidate_id, event_source).all():
                        continue
                    mapped = source.transition_to(candidate_id, event_source)
                    if chart_contains(candidate_id, mapped).all():
                        target_chart = candidate_id
                        event_target = mapped
                        break
            else:
                try:
                    target_chart, event_target = atlas.best_chart(
                        event_source, source_chart
                    )
                except ValueError:
                    pass
            if target_chart is None or event_target is None:
                raise RuntimeError(
                    f"no declared overlap covers chart {source_chart} "
                    f"near t={event_time:.6g}"
                )

            if record_operations:
                operations.append(
                    AcceptedChartSegment(
                        t,
                        event_time,
                        source_chart,
                        source_x.clone(),
                        event_source.clone(),
                    )
                )
            if hasattr(source, "jacobian"):
                jacobian = source.jacobian(target_chart, event_source)
            else:
                eye = torch.eye(x.shape[-1], device=x.device, dtype=x.dtype)
                jacobian = eye.expand(*x.shape[:-1], -1, -1)
            event = ChartTransitionEvent(
                event_time,
                source_chart,
                target_chart,
                event_source.clone(),
                event_target.clone(),
                jacobian.clone(),
            )
            transition_events.append(event)
            if record_operations:
                operations.append(event)
            x = event_target
            integral = event_integral
            t = event_time
            current_chart = target_chart
            if track_trajectory:
                source_state = (
                    event_source.detach() if detach_trajectory else event_source.clone()
                )
                target_state = x.detach() if detach_trajectory else x.clone()
                density = integral.detach() if detach_trajectory else integral.clone()
                trajectory.append((t, source_chart, source_state, density))
                trajectory.append((t, current_chart, target_state, density))
        else:
            raise RuntimeError(
                "integrate_multichart exceeded max_subdivisions for one fixed step"
            )
        if t != nominal_end:
            raise RuntimeError(
                "integrate_multichart exceeded max_subdivisions for one fixed step"
            )
        if track_trajectory and checkpoint_due(
            scheduled_step.index + 1, len(schedule), checkpoint_interval
        ):
            checkpoint(t, current_chart)

    return MultiChartFlowResult(
        x,
        current_chart,
        integral,
        trajectory,
        transition_events,
        operations,
        checkpoint_interval,
        detach_trajectory,
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
