"""Multi-chart integration helpers."""

from __future__ import annotations

import torch

from .analytic_metric import AnalyticMetric
from .atlas import Atlas
from .multichart import MultiChartVectorField
from .operators import divergence
from .integrator import _base_log_prob


class MultiChartFlowResult:
    """Result of integrating a flow across an atlas of charts."""

    x_final: torch.Tensor
    chart_final: int
    log_det: torch.Tensor
    trajectory: list  # of (t, chart_id, x)


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
    """Integrate with dynamic chart switching.

    1. Start in ``start_chart`` at position ``x0`` (in that chart's coords).
    2. At each accepted step, check if the proposed point still lies inside
       the current chart's k‑NN validity ball.
    3. If not, switch to the best valid overlapping chart via the user
       transition map and continue.

    Parameters
    ----------
    vf : MultiChartVectorField
    atlas : Atlas
    x0 : Tensor
        ``(..., dim)`` in chart *start_chart* coordinates.
    start_chart : int
    t0, t1 : float
        Integration times.
    dt : float
        Fixed step size (including sign if t1 < t0).
    track_trajectory : bool
    compute_divergence : bool

    Returns
    -------
    MultiChartFlowResult

    Notes
    -----
    Operates on one chart per *entire* batch at a time: if the batch is
    split across multiple charts after a step, all proposed points must
    be covered by a single chart (found via :meth:`Atlas.best_chart`)
    before the step is accepted. This keeps chart-switch semantics
    well-defined for batched integration.
    """
    if x0.dim() < 2:
        raise ValueError(
            "integrate_multichart expects x0 of shape (batch, dim); "
            f"got shape {tuple(x0.shape)}"
        )

    current_chart = start_chart
    x = x0
    log_det = torch.zeros(x0.shape[:-1], device=x0.device, dtype=x0.dtype)
    trajectory: list[tuple[float, int, torch.Tensor]] = []
    t = t0

    forward = t1 > t0
    sign = 1.0 if forward else -1.0
    base_dt = abs(dt)
    cur_dt = base_dt
    min_dt = base_dt * 1e-4
    max_steps = int(20 * abs(t1 - t0) / base_dt) + 10
    n_steps = 0

    if track_trajectory:
        trajectory.append((t, current_chart, x.clone()))

    while (forward and t < t1) or (not forward and t > t1):
        n_steps += 1
        if n_steps > max_steps:
            raise RuntimeError(
                "integrate_multichart exceeded max_steps "
                f"({max_steps}); check chart coverage/transitions for gaps."
            )

        h = sign * cur_dt
        if forward and t + h > t1:
            h = t1 - t
        elif not forward and t + h < t1:
            h = t1 - t
        half_h = h / 2.0

        def _f(t_: float, x_t_: torch.Tensor, cid: int) -> torch.Tensor:
            return vf(
                torch.full(x_t_.shape[:-1], t_, device=x_t_.device, dtype=x_t_.dtype),
                x_t_,
                cid,
            )

        # RK4 stages in current chart
        k1 = _f(t, x, current_chart)
        k2 = _f(t + half_h, x + half_h * k1, current_chart)
        k3 = _f(t + half_h, x + half_h * k2, current_chart)
        k4 = _f(t + h, x + h * k3, current_chart)
        x_proposed = x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        chart_obj = atlas[current_chart]

        if chart_obj.is_inside(x_proposed).all():
            # Accept step in current chart; restore step size on success.
            x = x_proposed
            t = t + h
            cur_dt = base_dt
            if compute_divergence:
                x_grad = x.detach().requires_grad_(True)
                div_val = divergence(
                    lambda x_: _f(t, x_, current_chart),
                    x_grad,
                    chart_obj.analytic_metric,
                )
                log_det = log_det + div_val * abs(h)
        else:
            # Try to switch the whole batch to a single overlapping chart.
            try:
                new_cid, x_mapped = atlas.best_chart(x_proposed, current_chart)
            except ValueError:
                # No single chart covers the proposed point(s): shrink the
                # step and retry, down to a minimum step size.
                cur_dt = cur_dt * 0.5
                if cur_dt < min_dt:
                    raise RuntimeError(
                        "integrate_multichart: step size collapsed below "
                        f"minimum ({min_dt}); no chart covers the "
                        "trajectory near t={:.4f}. Check atlas coverage/"
                        "transition maps for gaps.".format(t)
                    )
                continue

            current_chart = new_cid
            x = x_mapped
            t = t + h
            cur_dt = base_dt
            if compute_divergence:
                x_grad = x.detach().requires_grad_(True)
                div_val = divergence(
                    lambda x_: _f(t, x_, current_chart),
                    x_grad,
                    atlas[current_chart].analytic_metric,
                )
                log_det = log_det + div_val * abs(h)

        if track_trajectory:
            trajectory.append((t, current_chart, x.clone()))

    result = MultiChartFlowResult()
    result.x_final = x
    result.chart_final = current_chart
    result.log_det = log_det
    result.trajectory = trajectory
    return result


def cnf_nll_multichart(
    vf: MultiChartVectorField,
    atlas: Atlas,
    x_data: torch.Tensor,
    start_chart: int,
    dt: float = 0.05,
    t0: float = 0.0,
    t1: float = 1.0,
) -> torch.Tensor:
    """Mean NLL for data in an arbitrary chart, using standard autograd."""
    result = integrate_multichart(
        vf, atlas, x_data, start_chart, t1, t0, dt,
        track_trajectory=False,
        compute_divergence=True,
    )
    # Base density is evaluated in the reference chart.
    # If final chart != reference, transition the base point.
    x0 = result.x_final
    chart0 = result.chart_final
    if chart0 != atlas.reference_chart_id:
        x0 = atlas[chart0].transition_to(atlas.reference_chart_id, x0)
    return -(_base_log_prob(x0) + result.log_det).mean()
