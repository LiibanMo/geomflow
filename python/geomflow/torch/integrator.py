"""Single-chart ODE integrator for Riemannian CNF forward/backward passes."""

from __future__ import annotations

from typing import Optional

import torch

from .analytic_metric import AnalyticMetric
from .operators import divergence
from .vector_field import ManifoldVectorField


class FlowResult:
    """Result of integrating a flow on the manifold."""

    def __init__(self, x_final: torch.Tensor, log_det: torch.Tensor, trajectory: list):
        self.x_final = x_final
        self.log_det = log_det
        self.trajectory = trajectory  # list of (t, x) tuples


def integrate_rk4(
    vf: ManifoldVectorField,
    metric: AnalyticMetric,
    x0: torch.Tensor,
    t0: float,
    t1: float,
    dt: float,
    track_trajectory: bool = False,
    compute_divergence: bool = True,
) -> FlowResult:
    """Integrate ``dx/dt = f(t, x)`` with RK4, accumulating ``-∫ div f dt``.

    Parameters
    ----------
    vf : ManifoldVectorField
        The learned vector field.
    metric : AnalyticMetric
        The analytic metric object.
    x0 : Tensor
        Initial point(s), shape ``(..., dim)``.
    t0 : float
        Start time.
    t1 : float
        End time.
    dt : float
        Step size (absolute value; sign is inferred from t0→t1).
    track_trajectory : bool
        If True, record every ``(t, x)`` for adjoint replay.
    compute_divergence : bool
        If True, accumulate the divergence integral.

    Returns
    -------
    FlowResult
        ``x_final``, ``log_det`` (∫ div f dt from t0 to t1 with appropriate sign),
        and optionally ``trajectory``.
    """
    if x0.dim() < 1:
        raise ValueError("x0 must have shape (..., dim); got 0-d tensor")

    x = x0.clone()
    log_det = torch.zeros(x0.shape[:-1], device=x0.device, dtype=x0.dtype)
    trajectory: list[tuple[float, torch.Tensor]] = []
    t = t0

    forward = t1 > t0
    sign = 1.0 if forward else -1.0
    h = sign * abs(dt)

    if track_trajectory:
        trajectory.append((t, x.clone()))

    while (forward and t < t1) or (not forward and t > t1):
        h = sign * abs(dt)
        if forward and t + h > t1:
            h = t1 - t
        elif not forward and t + h < t1:
            h = t1 - t

        half_h = h / 2.0

        # --- RK4 step ---
        def _f(t_: float, x_: torch.Tensor) -> torch.Tensor:
            tm = torch.full(
                x_.shape[:-1], t_, device=x_.device, dtype=x_.dtype
            )
            return vf(tm, x_)

        k1 = _f(t, x)
        k2 = _f(t + half_h, x + half_h * k1)
        k3 = _f(t + half_h, x + half_h * k2)
        k4 = _f(t + h, x + h * k3)
        x = x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        # --- divergence accumulation ---
        if compute_divergence:
            with torch.enable_grad():
                x_detached = x.detach().requires_grad_(True)

                def _vf_wrapper(x_: torch.Tensor) -> torch.Tensor:
                    tm = torch.full(x_.shape[:-1], t, device=x_.device, dtype=x_.dtype)
                    return vf(tm, x_)

                div_val = divergence(_vf_wrapper, x_detached, metric)
            # The instantaneous-change-of-variables formula says:
            #   d log p(z(t))/dt = -div f(z(t), t)
            # So we ADD +div * h to get +∫ div dt (since log_det tracks this).
            log_det = log_det + div_val * abs(h)

        t = t + h

        if track_trajectory:
            trajectory.append((t, x.clone()))

    return FlowResult(x, log_det, trajectory)
