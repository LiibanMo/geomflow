"""CNF losses and Mohamud's intrinsic discrete adjoint."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.func import functional_call

from .analytic_metric import AnalyticMetric
from .base_distribution import (
    BaseDistribution,
    StandardNormalCoordinateBase,
    validate_base_distribution,
)
from .integrator import integrate_rk4
from .vector_field import (
    ManifoldVectorField,
    coordinate_jacobian_regularizer,
    weight_decay_loss,
)


def cnf_log_prob(
    vf: ManifoldVectorField,
    metric: AnalyticMetric,
    x_data: torch.Tensor,
    dt: float = 0.05,
    t0: float = 0.0,
    t1: float = 1.0,
    base_distribution: BaseDistribution | None = None,
) -> torch.Tensor:
    """Per-sample Riemannian log density from the direct-autograd solver."""
    base = base_distribution or getattr(
        metric, "default_base_distribution", StandardNormalCoordinateBase(metric.dim)
    )
    validate_base_distribution(base, metric.dim)
    result = integrate_rk4(vf, metric, x_data, t1, t0, dt, track_trajectory=False)
    return (
        base.log_prob_volume(result.x_final, metric)
        + result.divergence_integral
    )


@dataclass(frozen=True)
class CNFLossTerms:
    """Separated mathematical NLL and weighted training penalties."""

    nll: torch.Tensor
    lipschitz_penalty: torch.Tensor
    weight_decay_penalty: torch.Tensor

    @property
    def total(self) -> torch.Tensor:
        return self.nll + self.lipschitz_penalty + self.weight_decay_penalty


def cnf_loss_terms(
    vf: ManifoldVectorField,
    metric: AnalyticMetric,
    x_data: torch.Tensor,
    dt: float = 0.05,
    t0: float = 0.0,
    t1: float = 1.0,
    lipschitz_weight: float = 0.0,
    weight_decay_weight: float = 0.0,
    base_distribution: BaseDistribution | None = None,
) -> CNFLossTerms:
    """Return mean NLL and regularizers as distinct diagnostic quantities."""
    nll = -cnf_log_prob(
        vf, metric, x_data, dt, t0, t1, base_distribution
    ).mean()
    zero = nll.new_zeros(())
    lipschitz_penalty = zero
    weight_decay_penalty = zero
    if lipschitz_weight > 0.0:
        lipschitz_penalty = lipschitz_weight * coordinate_jacobian_regularizer(
            vf, x_data, t=(t0 + t1) / 2.0
        )
    if weight_decay_weight > 0.0:
        weight_decay_penalty = weight_decay_weight * weight_decay_loss(vf)
    return CNFLossTerms(nll, lipschitz_penalty, weight_decay_penalty)


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
    return cnf_loss_terms(
        vf,
        metric,
        x_data,
        dt,
        t0,
        t1,
        lipschitz_weight,
        weight_decay_weight,
        base_distribution,
    ).total


class IntrinsicAdjointFunction(torch.autograd.Function):
    """Exact discrete adjoint of the intrinsic augmented RK4 computation.

    This is the solver-discretized form of Liiban Mohamud's Theorem 3.7.
    Reverse-mode VJPs through every accepted RK stage implement the coordinate
    cotangent equation ``lambda_dot + lambda_i partial_j f^i = partial_j div_g f``.
    VJPs with respect to the explicit parameter inputs include both
    ``delta_theta div_g f`` and the approved negative cotangent pairing from
    the theorem's proof. No ambient-space embedding is used.

    Use :func:`intrinsic_adjoint_nll`; direct calls to ``apply`` require the
    internal configuration arguments and explicit parameter tensors.
    Higher-order derivatives through this custom backward are unsupported.
    """

    @staticmethod
    def forward(
        ctx,
        x_data,
        vf,
        metric,
        base_distribution,
        dt,
        t0,
        t1,
        parameter_names,
        *parameters,
    ):
        ctx.dt = dt
        ctx.t0 = t0
        ctx.t1 = t1
        ctx.vf = vf
        ctx.metric = metric
        ctx.base_distribution = base_distribution
        ctx.parameter_names = parameter_names
        named_buffers = tuple(vf.named_buffers())
        ctx.buffer_names = tuple(name for name, _ in named_buffers)
        buffers = tuple(buffer.detach().clone() for _, buffer in named_buffers)
        ctx.parameter_count = len(parameters)

        with torch.no_grad():
            field = _functional_field(
                vf,
                parameter_names,
                parameters,
                ctx.buffer_names,
                tuple(buffer.clone() for buffer in buffers),
            )
            res = integrate_rk4(
                field,
                metric,
                x_data,
                t0=t1,
                t1=t0,
                dt=dt,
                track_trajectory=False,
                stage_callback=lambda _time, state: _reject_differentiable_metric(
                    metric, state
                ),
            )
            _reject_differentiable_metric(metric, res.x_final)
            _reject_differentiable_base(
                base_distribution, metric, res.x_final
            )
            loss = -(
                base_distribution.log_prob_volume(res.x_final, metric)
                + res.divergence_integral
            ).mean()

        ctx.save_for_backward(x_data, *parameters, *buffers)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        saved_x, *saved_values = ctx.saved_tensors
        saved_parameters = saved_values[: ctx.parameter_count]
        saved_buffers = saved_values[ctx.parameter_count :]
        with torch.enable_grad():
            replay_x = saved_x.detach().requires_grad_(True)
            replay_parameters = tuple(
                parameter.detach().requires_grad_(True)
                for parameter in saved_parameters
            )
            field = _functional_field(
                ctx.vf,
                ctx.parameter_names,
                replay_parameters,
                ctx.buffer_names,
                tuple(buffer.clone() for buffer in saved_buffers),
            )
            result = integrate_rk4(
                field,
                ctx.metric,
                replay_x,
                t0=ctx.t1,
                t1=ctx.t0,
                dt=ctx.dt,
                track_trajectory=False,
            )
            replay_loss = -(
                ctx.base_distribution.log_prob_volume(
                    result.x_final, ctx.metric
                )
                + result.divergence_integral
            ).mean()
            replay_inputs = (replay_x, *replay_parameters)
            if replay_loss.requires_grad:
                gradients = torch.autograd.grad(
                    replay_loss,
                    replay_inputs,
                    grad_outputs=grad_output.detach(),
                    allow_unused=True,
                )
            else:
                gradients = (None,) * len(replay_inputs)

        materialized = tuple(
            torch.zeros_like(value) if gradient is None else gradient
            for value, gradient in zip(replay_inputs, gradients)
        )
        grad_x, *parameter_gradients = materialized
        return (
            grad_x,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            *parameter_gradients,
        )


def _functional_field(
    vf: nn.Module,
    parameter_names: tuple[str, ...],
    parameters: tuple[torch.Tensor, ...],
    buffer_names: tuple[str, ...],
    buffers: tuple[torch.Tensor, ...],
):
    state_map = dict(zip(parameter_names, parameters, strict=True))
    state_map.update(zip(buffer_names, buffers, strict=True))

    def field(time: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return functional_call(vf, state_map, (time, state), strict=False)

    return field


def _contains_trainable_tensor(
    value: object, seen: set[int] | None = None
) -> bool:
    if seen is None:
        seen = set()
    identity = id(value)
    if identity in seen:
        return False
    seen.add(identity)
    if isinstance(value, torch.Tensor):
        return value.requires_grad
    if isinstance(value, nn.Module):
        return any(
            tensor.requires_grad
            for tensor in (*value.parameters(), *value.buffers())
        )
    if isinstance(value, dict):
        return any(_contains_trainable_tensor(item, seen) for item in value.values())
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(_contains_trainable_tensor(item, seen) for item in value)
    closure = getattr(value, "__closure__", None)
    if closure is not None:
        return any(
            _contains_trainable_tensor(cell.cell_contents, seen) for cell in closure
        )
    attributes = getattr(value, "__dict__", None)
    if attributes is not None:
        return any(
            _contains_trainable_tensor(attribute, seen)
            for attribute in attributes.values()
        )
    return False


def _reject_trainable_configuration(value: object, name: str) -> None:
    if _contains_trainable_tensor(value):
        raise ValueError(f"trainable {name} tensors are not supported")


def _reject_differentiable_metric(
    metric: AnalyticMetric, x_data: torch.Tensor
) -> None:
    probe = x_data.detach()
    with torch.enable_grad():
        metric_outputs = (
            metric.metric(probe),
            metric.inverse(probe),
            metric.sqrt_det(probe),
        )
        if any(output.requires_grad for output in metric_outputs):
            raise ValueError("trainable metric tensors are not supported")


def _reject_differentiable_base(
    base: BaseDistribution, metric: AnalyticMetric, base_point: torch.Tensor
) -> None:
    with torch.enable_grad():
        if base.log_prob_volume(base_point.detach(), metric).requires_grad:
            raise ValueError("trainable base-distribution tensors are not supported")


def intrinsic_adjoint_nll(
    vf: nn.Module,
    metric: AnalyticMetric,
    x_data: torch.Tensor,
    dt: float = 0.05,
    t0: float = 0.0,
    t1: float = 1.0,
    base_distribution: BaseDistribution | None = None,
) -> torch.Tensor:
    """Mean CNF NLL with Mohamud's intrinsic first-order discrete adjoint.

    The forward value is exactly :func:`cnf_nll` without regularizers. The
    vector-field parameters are explicit custom-autograd inputs in stable
    ``named_parameters()`` order. Geometry and base-distribution parameters
    are fixed in this first supported scope.
    """
    base = base_distribution
    if base is None:
        base = getattr(
            metric,
            "default_base_distribution",
            StandardNormalCoordinateBase(metric.dim),
        )
    validate_base_distribution(base, metric.dim)
    _reject_trainable_configuration(metric, "metric")
    _reject_trainable_configuration(base, "base-distribution")
    _reject_differentiable_metric(metric, x_data)
    named_parameters = tuple(
        (name, parameter)
        for name, parameter in vf.named_parameters()
        if parameter.requires_grad
    )
    parameter_names = tuple(name for name, _ in named_parameters)
    parameters = tuple(parameter for _, parameter in named_parameters)
    return IntrinsicAdjointFunction.apply(
        x_data,
        vf,
        metric,
        base,
        dt,
        t0,
        t1,
        parameter_names,
        *parameters,
    )
