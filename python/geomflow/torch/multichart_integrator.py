"""Multi-chart integration helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from ._utils import (
    validate_autocast_disabled,
    validate_supported_floating_tensor,
    validate_tensor_module_compatibility,
)

from ._schedule import FixedStepSchedule, checkpoint_due, validate_checkpoint_interval
from .analytic_metric import AnalyticMetric
from .atlas import Atlas, Chart
from .base_distribution import AtlasBaseDistribution, StandardNormalCoordinateBase
from .multichart import MultiChartVectorField
from .operators import _divergence_from_value

_BASE_CHART_CONTAINS = Chart.contains
_BASE_METRIC_CONTAINS = AnalyticMetric.contains


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


@dataclass
class MultiChartStatistics:
    """Optional structural counters for benchmark and profiler assertions."""

    chart_predicate_count: int = 0
    scalar_decision_count: int = 0
    rk_trial_count: int = 0
    accepted_trial_count: int = 0
    rejected_trial_count: int = 0
    wasted_rk_stage_count: int = 0
    field_call_count: int = 0
    transition_predicate_count: int = 0
    transition_map_count: int = 0
    transition_jacobian_count: int = 0
    transition_event_count: int = 0


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
        statistics: MultiChartStatistics | None = None,
        trajectory_checkpoint_interval: int = 1,
        trajectory_is_detached: bool = False,
        execution_backend: str = "component-gradient-eager",
        fallback_reason: str | None = None,
    ) -> None:
        self.x_final = x_final
        self.chart_final = chart_final
        self.divergence_integral = divergence_integral
        self.trajectory = trajectory
        self.transition_events = transition_events
        self.operations = operations or []
        self.statistics = statistics
        self.trajectory_checkpoint_interval = trajectory_checkpoint_interval
        self.trajectory_is_detached = trajectory_is_detached
        self._execution_backend = execution_backend
        self._fallback_reason = fallback_reason

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
    record_statistics: bool = False,
    compile: bool | None = None,
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
    validate_tensor_module_compatibility(x0, vf, "integrate_multichart")
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
    builtin_solver_charts = (
        type(atlas) is Atlas
        and getattr(atlas, "_solver_kind", None) == "sphere-2d-stereographic"
        and all(
            type(chart) is Chart
            and type(chart).contains is _BASE_CHART_CONTAINS
            and chart._domain is getattr(chart, "_solver_domain_fn", None)
            and "contains" not in chart.__dict__
            and chart.analytic_metric._supports_tensor_solver()
            and type(chart.analytic_metric).contains is _BASE_METRIC_CONTAINS
            and "contains" not in chart.analytic_metric.__dict__
            for chart in atlas.charts.values()
        )
    )
    tensor_eligible = (
        compute_divergence
        and not track_trajectory
        and not record_operations
        and not record_statistics
        and type(vf) is MultiChartVectorField
        and builtin_solver_charts
        and all(
            vf._supports_tensor_value_and_trace(chart_id)
            for chart_id in atlas.charts
        )
    )
    prevalidated_domains = (
        {
            chart_id: (chart._domain, chart.analytic_metric._domain_fn)
            for chart_id, chart in atlas.charts.items()
        }
        if builtin_solver_charts
        else {}
    )
    prevalidated_metrics = (
        {
            chart_id: chart.analytic_metric
            for chart_id, chart in atlas.charts.items()
        }
        if builtin_solver_charts
        else {}
    )
    execution_backend = (
        "tensor-eager" if tensor_eligible else "component-gradient-eager"
    )
    fallback_reason = None

    current_chart = start_chart
    x = x0.clone()
    integral = torch.zeros(x0.shape[:-1], device=x0.device, dtype=x0.dtype)
    trajectory: list[tuple[float, int, torch.Tensor, torch.Tensor]] = []
    transition_events: list[ChartTransitionEvent] = []
    operations: list[AcceptedChartSegment | ChartTransitionEvent] = []
    statistics = MultiChartStatistics() if record_statistics else None

    def count(name: str, amount: int = 1) -> None:
        if statistics is not None:
            setattr(statistics, name, getattr(statistics, name) + amount)

    def decide(mask: torch.Tensor) -> bool:
        count("scalar_decision_count")
        return bool(mask.all())

    def checkpoint(time: float, chart_id: int) -> None:
        state = x.detach() if detach_trajectory else x.clone()
        density = integral.detach() if detach_trajectory else integral.clone()
        trajectory.append((time, chart_id, state, density))

    if track_trajectory:
        checkpoint(schedule.t0, current_chart)

    build_graph = torch.is_grad_enabled() and not torch.is_inference_mode_enabled()
    field_call = vf._solver_forward if type(vf) is MultiChartVectorField else vf
    tensor_heads = (
        {chart_id: vf.head(chart_id) for chart_id in atlas.charts}
        if tensor_eligible
        else {}
    )

    def augmented_rhs(
        time: float, state: torch.Tensor, chart_id: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        count("field_call_count")
        if not compute_divergence:
            time_tensor = state.new_full(state.shape[:-1], time)
            field_value = field_call(time_tensor, state, chart_id)
            return field_value, state.new_zeros(state.shape[:-1])

        if tensor_eligible:
            time_tensor = state.new_full(state.shape[:-1], time)
            field_value, trace = tensor_heads[
                chart_id
            ]._tensor_value_and_trace_unchecked(
                time_tensor, state
            )
            volume_gradient = prevalidated_metrics[
                chart_id
            ]._tensor_log_volume_gradient_unchecked(state)
            return field_value, trace + (field_value * volume_gradient).sum(-1)

        with torch.inference_mode(False), torch.enable_grad():
            divergence_state = state
            if divergence_state.is_inference() or not divergence_state.requires_grad:
                divergence_state = divergence_state.clone().requires_grad_(True)
            time_tensor = divergence_state.new_full(divergence_state.shape[:-1], time)
            field_value = field_call(time_tensor, divergence_state, chart_id)
            metric = (
                prevalidated_metrics[chart_id]
                if chart_id in prevalidated_metrics
                else atlas[chart_id].analytic_metric
            )
            divergence_value = _divergence_from_value(
                field_value,
                divergence_state,
                metric,
            )
            if not build_graph:
                field_value = field_value.detach()
                divergence_value = divergence_value.detach()
        return field_value, divergence_value

    def chart_contains(chart_id: int, state: torch.Tensor) -> torch.Tensor:
        count("chart_predicate_count")
        domains = prevalidated_domains.get(chart_id)
        if domains is not None:
            chart_domain, metric_domain = domains
            mask = torch.isfinite(state).all(dim=-1)
            if chart_domain is not None:
                mask = mask & chart_domain(state)
            if metric_domain is not None:
                mask = mask & metric_domain(state)
            return mask
        chart = atlas[chart_id]
        predicate = getattr(chart, "contains", chart.is_inside)
        return predicate(state) & chart.analytic_metric.contains(state)

    if not decide(chart_contains(current_chart, x)):
        raise RuntimeError(f"initial state is outside chart {current_chart}")

    request_compilation = compile is True or (
        compile is None and x0.device.type == "cuda" and tensor_eligible
    )
    if request_compilation and not tensor_eligible:
        from .compilation import _warn_fallback

        fallback_reason = "field, atlas, or options are not eligible for tensor compilation"
        _warn_fallback(fallback_reason)
    elif request_compilation:
        from .compilation import _integrate_rk4_compiled

        chart = atlas[current_chart]
        compiled, compiler_failure = _integrate_rk4_compiled(
            vf.head(current_chart),
            chart.analytic_metric,
            x,
            schedule,
            compute_divergence=compute_divergence,
            unsupported_reason=None,
            warn_fallback=compile is True,
        )
        if compiled is not None:
            compiled_x, compiled_integral = compiled
            if decide(chart_contains(current_chart, compiled_x)):
                return MultiChartFlowResult(
                    compiled_x,
                    current_chart,
                    compiled_integral,
                    [],
                    [],
                    execution_backend="inductor",
                )
            fallback_reason = "compiled result left the active chart"
        else:
            fallback_reason = compiler_failure

    def rk4_trial(
        time: float,
        state: torch.Tensor,
        integral_state: torch.Tensor,
        h: float,
        chart_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        count("rk_trial_count")
        deferred = bool(getattr(atlas[chart_id], "_defer_trial_validation", False))
        half_h = h / 2.0
        k1_x, k1_i = augmented_rhs(time, state, chart_id)
        stage2 = state + half_h * k1_x
        stage2_valid = chart_contains(chart_id, stage2)
        if not deferred and not decide(stage2_valid):
            count("rejected_trial_count")
            count("wasted_rk_stage_count")
            return None
        k2_x, k2_i = augmented_rhs(time + half_h, stage2, chart_id)
        stage3 = state + half_h * k2_x
        stage3_valid = chart_contains(chart_id, stage3)
        if not deferred and not decide(stage3_valid):
            count("rejected_trial_count")
            count("wasted_rk_stage_count", 2)
            return None
        k3_x, k3_i = augmented_rhs(time + half_h, stage3, chart_id)
        stage4 = state + h * k3_x
        stage4_valid = chart_contains(chart_id, stage4)
        if not deferred and not decide(stage4_valid):
            count("rejected_trial_count")
            count("wasted_rk_stage_count", 3)
            return None
        k4_x, k4_i = augmented_rhs(time + h, stage4, chart_id)
        proposed_x = state + (h / 6.0) * (k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x)
        proposed_valid = chart_contains(chart_id, proposed_x)
        valid = stage2_valid & stage3_valid & stage4_valid & proposed_valid
        if not decide(valid if deferred else proposed_valid):
            count("rejected_trial_count")
            count("wasted_rk_stage_count", 4)
            return None
        proposed_integral = integral_state + (h / 6.0) * (
            k1_i + 2.0 * k2_i + 2.0 * k3_i + k4_i
        )
        count("accepted_trial_count")
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
                    count("transition_predicate_count")
                    transition_domain = (
                        source._validate_mask(
                            "transition domain",
                            source.transitions[candidate_id].source_domain(
                                event_source
                            ),
                            event_source,
                        )
                        if type(source) is Chart
                        else source.can_transition_to(candidate_id, event_source)
                    )
                    if not decide(transition_domain):
                        continue
                    transition = (
                        source._transition_unchecked
                        if type(source) is Chart
                        else source.transition_to
                    )
                    count("transition_map_count")
                    mapped = transition(candidate_id, event_source)
                    if decide(chart_contains(candidate_id, mapped)):
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
                jacobian_fn = (
                    source._jacobian_unchecked
                    if type(source) is Chart
                    else source.jacobian
                )
                jacobian = jacobian_fn(target_chart, event_source)
                count("transition_jacobian_count")
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
            count("transition_event_count")
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
        statistics,
        checkpoint_interval,
        detach_trajectory,
        execution_backend,
        fallback_reason,
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
