"""CNF loss for a Riemannian manifold with user-provided analytic metric.

The custom intrinsic adjoint Function is experimental; the public
``cnf_nll`` uses the robust standard-autograd path through the solver.
"""

from __future__ import annotations

import torch

from .analytic_metric import AnalyticMetric
from .base_distribution import (
    BaseDistribution,
    StandardNormalCoordinateBase,
    validate_base_distribution,
)
from .integrator import integrate_rk4
from .operators import divergence
from .vector_field import ManifoldVectorField, lipschitz_regularizer, weight_decay_loss


def cnf_nll(
    vf: ManifoldVectorField,
    metric: AnalyticMetric,
    x_data: torch.Tensor,
    dt: float = 0.05,
    t0: float = 0.0,
    t1: float = 1.0,
    lipschitz_weight: float = 0.0,
    weight_decay_weight: float = 0.0,
    base_distribution: BaseDistribution | None = None,
) -> torch.Tensor:
    """Negative log-likelihood for a CNF on a Riemannian manifold.

    Integrates from ``t1`` (data) backward to ``t0`` (base) and returns
    the mean NLL by standard reverse-mode autodifferentiation through
    the solver.  Optionally adds Lipschitz / weight-decay regularization
    terms directly into the returned scalar so a single ``.backward()``
    updates the vector field with all regularizers active.

    Parameters
    ----------
    vf : ManifoldVectorField
        The learned vector field.
    metric : AnalyticMetric
        User-provided analytic metric of the manifold.  Not trainable.
    x_data : Tensor
        Data point(s), shape ``(batch, dim)``.
    dt : float
        ODE step size.
    t0 : float
        Base time (default 0).
    t1 : float
        Data time (default 1).
    lipschitz_weight : float
        If > 0, add ``lipschitz_weight * lipschitz_regularizer(vf, x_data, t=(t0+t1)/2)``.
    weight_decay_weight : float
        If > 0, add ``weight_decay_weight * weight_decay_loss(vf)``.

    Returns
    -------
    nll : Tensor
        Scalar loss (NLL plus any active regularizers); call ``.backward()``
        to fill vector-field parameter gradients.
    """
    base = base_distribution or getattr(
        metric, "default_base_distribution", StandardNormalCoordinateBase(metric.dim)
    )
    validate_base_distribution(base, metric.dim)
    result = integrate_rk4(vf, metric, x_data, t1, t0, dt, track_trajectory=False)
    loss = -(
        base.log_prob_volume(result.x_final, metric)
        + result.divergence_integral
    ).mean()

    if lipschitz_weight > 0.0:
        loss = loss + lipschitz_weight * lipschitz_regularizer(
            vf, x_data, t=(t0 + t1) / 2.0
        )
    if weight_decay_weight > 0.0:
        loss = loss + weight_decay_weight * weight_decay_loss(vf)
    return loss


class IntrinsicAdjointFunction(torch.autograd.Function):
    """Custom PyTorch autograd Function implementing Mohamud's Theorem 3.7 intrinsic adjoint ODE.

    Theorem 3.7 (Mohamud):
    The intrinsic backward adjoint equation for Riemannian CNF log-density gradient is:
        lambda_dot(t) = - J_f(t, x(t))^T * lambda(t) + grad_x(div_g f(t, x(t)))

    No Whitney embedding R^(2n+1) ambient space is invoked.
    """

    @staticmethod
    def forward(ctx, x_data, vf, metric, dt=0.05, t0=0.0, t1=1.0):
        ctx.dt = dt
        ctx.t0 = t0
        ctx.t1 = t1
        ctx.vf = vf
        ctx.metric = metric

        with torch.no_grad():
            res = integrate_rk4(vf, metric, x_data, t0=t1, t1=t0, dt=dt, track_trajectory=True)

        ctx.save_for_backward(x_data, res.x_final, res.divergence_integral)
        ctx.trajectory = res.trajectory
        base = StandardNormalCoordinateBase(metric.dim)
        return -(
            base.log_prob_volume(res.x_final, metric)
            + res.divergence_integral
        ).mean()

    @staticmethod
    def backward(ctx, grad_output):
        x_data, x_final, divergence_integral = ctx.saved_tensors
        dt = ctx.dt
        vf = ctx.vf
        metric = ctx.metric

        x_cur = x_final.clone().detach().requires_grad_(True)
        adj_x = x_cur.clone()

        trajectory = list(reversed(ctx.trajectory))
        for i in range(len(trajectory) - 1):
            t_curr, x_t = trajectory[i][0], trajectory[i][1]
            t_prev = trajectory[i + 1][0]
            h = t_prev - t_curr

            with torch.enable_grad():
                x_step = x_t.clone().detach().requires_grad_(True)

                def _adj_rhs(t_val, l_val):
                    t_s = torch.full((x_step.shape[0],), t_val, device=x_step.device)
                    div_val = divergence(lambda x_: vf(t_s, x_), x_step, metric)
                    (grad_div,) = torch.autograd.grad(div_val.sum(), x_step, retain_graph=True)
                    f_val = vf(t_s, x_step)
                    v_prod = (f_val * l_val).sum()
                    (Jf_T_l,) = torch.autograd.grad(v_prod, x_step, retain_graph=True)
                    return -Jf_T_l + grad_div

                k1 = _adj_rhs(t_curr, adj_x)
                k2 = _adj_rhs(t_curr + 0.5 * h, adj_x + 0.5 * h * k1)
                k3 = _adj_rhs(t_curr + 0.5 * h, adj_x + 0.5 * h * k2)
                k4 = _adj_rhs(t_curr + h, adj_x + h * k3)
                adj_x = adj_x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        grad_x_data = (grad_output / x_data.shape[0]) * adj_x
        return grad_x_data, None, None, None, None, None
