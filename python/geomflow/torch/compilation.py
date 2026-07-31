"""Optional compiled execution for the single-chart direct-autograd solver."""

from __future__ import annotations

from collections import OrderedDict, namedtuple
from collections.abc import Callable
import warnings

import torch

from ._schedule import FixedStepSchedule
from .analytic_metric import AnalyticMetric
from .vector_field import ManifoldVectorField

_CACHE_LIMIT = 8
_CacheInfo = namedtuple("CompilationCacheInfo", "hits misses maxsize currsize")
_cache: OrderedDict[
    tuple[object, ...], Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
] = OrderedDict()
_failures: OrderedDict[tuple[object, ...], str] = OrderedDict()
_hits = 0
_misses = 0


def _with_exact_higher_order_fallback(
    compiled_solver: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    eager_solver: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    vf: ManifoldVectorField,
) -> Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    parameters = tuple(vf.parameters())

    class ExactHigherOrderBridge(torch.autograd.Function):
        @staticmethod
        def forward(ctx, compiled_x, compiled_integral, x0, *parameter_values):
            ctx.save_for_backward(x0, *parameter_values)
            return compiled_x.view_as(compiled_x), compiled_integral.view_as(
                compiled_integral
            )

        @staticmethod
        def backward(ctx, grad_x, grad_integral):
            saved = ctx.saved_tensors
            x0, *parameter_values = saved
            requested = [
                value for value in (x0, *parameter_values) if value.requires_grad
            ]
            with torch.enable_grad():
                eager_x, eager_integral = eager_solver(x0)
                gradients = torch.autograd.grad(
                    (eager_x, eager_integral),
                    requested,
                    (
                        torch.zeros_like(eager_x) if grad_x is None else grad_x,
                        (
                            torch.zeros_like(eager_integral)
                            if grad_integral is None
                            else grad_integral
                        ),
                    ),
                    create_graph=torch.is_grad_enabled(),
                    allow_unused=True,
                )
            gradient_by_id = {
                id(value): gradient for value, gradient in zip(requested, gradients)
            }
            direct = tuple(gradient_by_id.get(id(value)) for value in saved)
            return None, None, *direct

    def run(x0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            compiled_x, compiled_integral = compiled_solver(x0)
        return ExactHigherOrderBridge.apply(
            compiled_x.detach(), compiled_integral.detach(), x0, *parameters
        )

    return run


def compilation_cache_info() -> tuple[int, int, int, int]:
    """Return cache hits, misses, maximum size, and current size."""
    return _CacheInfo(_hits, _misses, _CACHE_LIMIT, len(_cache))


def clear_compilation_cache() -> None:
    """Discard compiled solver variants and reset cache statistics."""
    global _hits, _misses
    _cache.clear()
    _failures.clear()
    _hits = 0
    _misses = 0


def _warn_fallback(reason: str) -> None:
    warnings.warn(
        f"geomflow compiled solver unavailable ({reason}); using eager execution "
        "on the input device",
        RuntimeWarning,
        stacklevel=3,
    )


def _cache_key(
    vf: ManifoldVectorField,
    metric: AnalyticMetric,
    x: torch.Tensor,
    schedule: FixedStepSchedule,
    compute_divergence: bool,
) -> tuple[object, ...]:
    key = (
        vf,
        metric,
        x.device.type,
        x.device.index,
        x.dtype,
        schedule.t0,
        schedule.t1,
        schedule.dt,
        compute_divergence,
        torch.is_grad_enabled(),
    )
    return key + (tuple(x.shape),) if x.device.type == "cuda" else key


def _make_compiled_solver(
    vf: ManifoldVectorField,
    metric: AnalyticMetric,
    schedule: FixedStepSchedule,
    compute_divergence: bool,
) -> Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    steps = tuple(schedule)

    def valid_points(value: torch.Tensor) -> torch.Tensor:
        valid = torch.isfinite(value).all()
        if metric._solver_kind == "sphere-stereographic":
            limit = torch.finfo(value.dtype).max ** 0.25 / metric.dim**0.5
            valid = valid & (value.abs().amax(dim=-1) < limit).all()
        return valid

    def rhs(state: torch.Tensor, stage_time: float):
        time = state.new_full(state.shape[:-1], stage_time)
        if not compute_divergence:
            return vf._forward_unchecked(time, state), state.new_zeros(state.shape[:-1])
        value, trace = vf._tensor_value_and_trace_unchecked(time, state)
        volume_gradient = metric._tensor_log_volume_gradient_unchecked(state)
        return value, trace + (value * volume_gradient).sum(-1)

    def tensor_solver(x0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x0.clone()
        integral = torch.zeros(x0.shape[:-1], device=x0.device, dtype=x0.dtype)
        valid = valid_points(x)
        for step in steps:
            half_h = step.size / 2.0
            k1_x, k1_i = rhs(x, step.start)
            stage2 = x + half_h * k1_x
            valid = valid & valid_points(stage2)
            k2_x, k2_i = rhs(stage2, step.start + half_h)
            stage3 = x + half_h * k2_x
            valid = valid & valid_points(stage3)
            k3_x, k3_i = rhs(stage3, step.start + half_h)
            stage4 = x + step.size * k3_x
            valid = valid & valid_points(stage4)
            k4_x, k4_i = rhs(stage4, step.start + step.size)
            x = x + (step.size / 6.0) * (k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x)
            valid = valid & valid_points(x)
            integral = integral + (step.size / 6.0) * (
                k1_i + 2.0 * k2_i + 2.0 * k3_i + k4_i
            )
        invalid = torch.full_like(x, float("nan"))
        x = torch.where(valid, x, invalid)
        integral = torch.where(valid, integral, torch.full_like(integral, float("nan")))
        return x, integral

    device_type = next(vf.parameters()).device.type
    compile_options = {
        "backend": "inductor",
        "fullgraph": True,
        "dynamic": device_type != "cuda",
    }
    if device_type == "cuda":
        compile_options["mode"] = "reduce-overhead"
    compiled_solver = torch.compile(tensor_solver, **compile_options)
    return _with_exact_higher_order_fallback(compiled_solver, tensor_solver, vf)


def _integrate_rk4_compiled(
    vf: ManifoldVectorField,
    metric: AnalyticMetric,
    x0: torch.Tensor,
    schedule: FixedStepSchedule,
    *,
    compute_divergence: bool,
    unsupported_reason: str | None,
    warn_fallback: bool = True,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    global _hits, _misses
    if unsupported_reason is not None:
        if warn_fallback:
            _warn_fallback(unsupported_reason)
        return None
    if not hasattr(torch, "compile"):
        if warn_fallback:
            _warn_fallback("torch.compile is not provided by this PyTorch build")
        return None

    key = _cache_key(vf, metric, x0, schedule, compute_divergence)
    failed_reason = _failures.get(key)
    if failed_reason is not None:
        _failures.move_to_end(key)
        if warn_fallback:
            _warn_fallback(failed_reason)
        return None
    solver = _cache.get(key)
    if solver is None:
        _misses += 1
        try:
            solver = _make_compiled_solver(vf, metric, schedule, compute_divergence)
        except Exception as error:
            reason = f"{type(error).__name__}: {error}"
            _failures[key] = reason
            while len(_failures) > _CACHE_LIMIT:
                _failures.popitem(last=False)
            if warn_fallback:
                _warn_fallback(reason)
            return None
        _cache[key] = solver
        _cache.move_to_end(key)
        while len(_cache) > _CACHE_LIMIT:
            _cache.popitem(last=False)
    else:
        _hits += 1
        _cache.move_to_end(key)

    try:
        result = solver(x0)
        if not torch.isfinite(result[0]).all():
            _cache.pop(key, None)
            if warn_fallback:
                _warn_fallback("compiled stage validation failed")
            return None
        return result
    except Exception as error:
        _cache.pop(key, None)
        reason = f"{type(error).__name__}: {error}"
        _failures[key] = reason
        while len(_failures) > _CACHE_LIMIT:
            _failures.popitem(last=False)
        if warn_fallback:
            _warn_fallback(reason)
        return None
