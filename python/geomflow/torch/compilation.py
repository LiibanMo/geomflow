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


def _mark_cudagraph_step(value: torch.Tensor) -> None:
    compiler = getattr(torch, "compiler", None)
    marker = getattr(compiler, "cudagraph_mark_step_begin", None)
    if value.device.type == "cuda" and marker is not None:
        marker()


def _isolate_dynamo_code(function: Callable) -> Callable:
    # Nested solver closures otherwise share one Dynamo recompile budget.
    function.__code__ = function.__code__.replace()
    return function


def _with_exact_higher_order_fallback(
    compiled_solver: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    compiled_vjp: Callable[..., tuple[torch.Tensor, ...]],
    eager_solver: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    vf: ManifoldVectorField,
    *,
    mark_cudagraph_steps: bool = False,
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
            grad_x = torch.zeros_like(x0) if grad_x is None else grad_x
            grad_integral = (
                x0.new_zeros(x0.shape[:-1]) if grad_integral is None else grad_integral
            )
            if not torch.is_grad_enabled():
                gradients = compiled_vjp(
                    x0.detach(), *parameter_values, grad_x, grad_integral
                )
                direct = tuple(
                    gradient.clone() if needed and gradient is not None else None
                    for gradient, needed in zip(
                        gradients, ctx.needs_input_grad[2:], strict=True
                    )
                )
                return None, None, *direct

            requested = [
                value for value in (x0, *parameter_values) if value.requires_grad
            ]
            with torch.enable_grad():
                eager_x, eager_integral = eager_solver(x0, *parameter_values)
                gradients = torch.autograd.grad(
                    (eager_x, eager_integral),
                    requested,
                    (grad_x, grad_integral),
                    create_graph=True,
                    allow_unused=True,
                )
            gradient_by_id = {
                id(value): gradient for value, gradient in zip(requested, gradients)
            }
            direct = tuple(gradient_by_id.get(id(value)) for value in saved)
            return None, None, *direct

    def run(x0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            if mark_cudagraph_steps:
                _mark_cudagraph_step(x0)
            compiled_x, compiled_integral = compiled_solver(x0.detach(), *parameters)
            compiled_x = compiled_x.clone()
            compiled_integral = compiled_integral.clone()
        return ExactHigherOrderBridge.apply(
            compiled_x, compiled_integral, x0, *parameters
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
    structure = tuple(
        (
            name,
            type(module),
            id(module),
            tuple(
                (
                    parameter_name,
                    id(parameter),
                    tuple(parameter.shape),
                    parameter.device,
                    parameter.dtype,
                )
                for parameter_name, parameter in module.named_parameters(recurse=False)
            ),
            tuple(
                (
                    buffer_name,
                    id(buffer),
                    tuple(buffer.shape),
                    buffer.device,
                    buffer.dtype,
                )
                for buffer_name, buffer in module.named_buffers(recurse=False)
            ),
        )
        for name, module in vf.named_modules()
    )
    metric_structure = (
        type(metric),
        metric._solver_kind,
        metric.debug_validation,
        id(metric._metric_fn),
        id(metric._domain_fn),
        id(metric._canonicalize_fn),
        id(metric._log_volume_gradient_fn),
    )
    key = (
        vf,
        structure,
        metric,
        metric_structure,
        x.device.type,
        x.device.index,
        x.dtype,
        schedule.t0,
        schedule.t1,
        schedule.dt,
        compute_divergence,
        torch.is_grad_enabled(),
    )
    return key


def _make_compiled_solver(
    vf: ManifoldVectorField,
    metric: AnalyticMetric,
    schedule: FixedStepSchedule,
    compute_divergence: bool,
    compile_mode: str | None = None,
) -> Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    steps = tuple(schedule)
    parameters = tuple(vf.parameters())
    linear_layers = tuple(vf.net[::2])
    parameter_positions: list[tuple[int, int | None]] = []
    position_by_id = {
        id(parameter): index for index, parameter in enumerate(parameters)
    }
    for layer in linear_layers:
        parameter_positions.append(
            (
                position_by_id[id(layer.weight)],
                None if layer.bias is None else position_by_id[id(layer.bias)],
            )
        )

    def valid_points(value: torch.Tensor) -> torch.Tensor:
        valid = torch.isfinite(value).all()
        if metric._solver_kind == "sphere-stereographic":
            limit = torch.finfo(value.dtype).max ** 0.25 / metric.dim**0.5
            valid = valid & (value.abs().amax(dim=-1) < limit).all()
        return valid

    def value_and_trace(
        time: torch.Tensor,
        state: torch.Tensor,
        parameter_values: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if time.dim() == state.dim() - 1:
            time = time.unsqueeze(-1)
        value = torch.cat((time, state), dim=-1)
        identity = torch.eye(vf.dim, device=state.device, dtype=state.dtype).expand(
            *state.shape[:-1], vf.dim, vf.dim
        )
        tangent = torch.cat(
            (state.new_zeros(*state.shape[:-1], 1, vf.dim), identity), dim=-2
        )
        for index, (weight_position, bias_position) in enumerate(parameter_positions):
            weight = parameter_values[weight_position]
            bias = None if bias_position is None else parameter_values[bias_position]
            value = torch.nn.functional.linear(value, weight, bias)
            tangent = torch.einsum("oi,...ij->...oj", weight, tangent)
            if index + 1 != len(parameter_positions):
                sigmoid = torch.sigmoid(value)
                derivative = sigmoid * (1.0 + value * (1.0 - sigmoid))
                value = torch.nn.functional.silu(value)
                tangent = derivative.unsqueeze(-1) * tangent
        return value, tangent.diagonal(dim1=-2, dim2=-1).sum(-1)

    def rhs(
        state: torch.Tensor,
        stage_time: float,
        parameter_values: tuple[torch.Tensor, ...],
    ):
        time = state.new_full(state.shape[:-1], stage_time)
        if not compute_divergence:
            value, _ = value_and_trace(time, state, parameter_values)
            return value, state.new_zeros(state.shape[:-1])
        value, trace = value_and_trace(time, state, parameter_values)
        volume_gradient = metric._tensor_log_volume_gradient_unchecked(state)
        return value, trace + (value * volume_gradient).sum(-1)

    def tensor_solver(
        x0: torch.Tensor, *parameter_values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = x0.clone()
        integral = torch.zeros(x0.shape[:-1], device=x0.device, dtype=x0.dtype)
        valid = valid_points(x)
        for step in steps:
            half_h = step.size / 2.0
            k1_x, k1_i = rhs(x, step.start, parameter_values)
            stage2 = x + half_h * k1_x
            valid = valid & valid_points(stage2)
            k2_x, k2_i = rhs(stage2, step.start + half_h, parameter_values)
            stage3 = x + half_h * k2_x
            valid = valid & valid_points(stage3)
            k3_x, k3_i = rhs(stage3, step.start + half_h, parameter_values)
            stage4 = x + step.size * k3_x
            valid = valid & valid_points(stage4)
            k4_x, k4_i = rhs(stage4, step.start + step.size, parameter_values)
            x = x + (step.size / 6.0) * (k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x)
            valid = valid & valid_points(x)
            integral = integral + (step.size / 6.0) * (
                k1_i + 2.0 * k2_i + 2.0 * k3_i + k4_i
            )
        invalid = torch.full_like(x, float("nan"))
        x = torch.where(valid, x, invalid)
        integral = torch.where(valid, integral, torch.full_like(integral, float("nan")))
        return x, integral

    compile_options = {
        "backend": "inductor",
        "fullgraph": True,
        "dynamic": compile_mode != "reduce-overhead",
    }
    if compile_mode is not None and compile_mode != "default":
        compile_options["mode"] = compile_mode
    compiled_solver = torch.compile(
        _isolate_dynamo_code(tensor_solver), **compile_options
    )

    def scalar_objective(
        x0: torch.Tensor, *values_and_grad_outputs: torch.Tensor
    ) -> torch.Tensor:
        parameter_values = values_and_grad_outputs[: len(parameters)]
        grad_x, grad_integral = values_and_grad_outputs[len(parameters) :]
        output_x, output_integral = tensor_solver(x0, *parameter_values)
        return (output_x * grad_x).sum() + (output_integral * grad_integral).sum()

    vjp = torch.func.grad(scalar_objective, argnums=tuple(range(len(parameters) + 1)))
    compiled_vjp = torch.compile(_isolate_dynamo_code(vjp), **compile_options)
    return _with_exact_higher_order_fallback(
        compiled_solver,
        compiled_vjp,
        tensor_solver,
        vf,
        mark_cudagraph_steps=compile_mode == "reduce-overhead",
    )


def _integrate_rk4_compiled(
    vf: ManifoldVectorField,
    metric: AnalyticMetric,
    x0: torch.Tensor,
    schedule: FixedStepSchedule,
    *,
    compute_divergence: bool,
    unsupported_reason: str | None,
    warn_fallback: bool = True,
) -> tuple[tuple[torch.Tensor, torch.Tensor] | None, str | None]:
    global _hits, _misses
    if unsupported_reason is not None:
        if warn_fallback:
            _warn_fallback(unsupported_reason)
        return None, unsupported_reason
    if not hasattr(torch, "compile"):
        if warn_fallback:
            _warn_fallback("torch.compile is not provided by this PyTorch build")
        return None, "torch.compile is not provided by this PyTorch build"

    key = _cache_key(vf, metric, x0, schedule, compute_divergence)
    runtime_failure_key = key + ("runtime-shape", tuple(x0.shape))
    failed_key = key if key in _failures else runtime_failure_key
    failed_reason = _failures.get(failed_key)
    if failed_reason is not None:
        _failures.move_to_end(failed_key)
        if warn_fallback:
            _warn_fallback(failed_reason)
        return None, failed_reason
    solver = _cache.get(key)
    new_solver = solver is None
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
            return None, reason
    else:
        _hits += 1
        _cache.move_to_end(key)

    try:
        if new_solver and torch.is_grad_enabled():
            probe_x = x0.detach().requires_grad_(True)
            probe_x_final, probe_integral = solver(probe_x)
            requested = tuple(
                value
                for value in (probe_x, *vf.parameters())
                if value.requires_grad
            )
            probe_objective = probe_x_final.sum() + probe_integral.sum()
            if requested and probe_objective.requires_grad:
                torch.autograd.grad(
                    probe_objective,
                    requested,
                    allow_unused=True,
                )
        result = solver(x0)
        if not torch.isfinite(result[0]).all():
            _cache.pop(key, None)
            if warn_fallback:
                _warn_fallback("compiled stage validation failed")
            return None, "compiled stage validation failed"
        if new_solver:
            _cache[key] = solver
            _cache.move_to_end(key)
            while len(_cache) > _CACHE_LIMIT:
                _cache.popitem(last=False)
        return result, None
    except Exception as error:
        _cache.pop(key, None)
        reason = f"{type(error).__name__}: {error}"
        _failures[runtime_failure_key] = reason
        while len(_failures) > _CACHE_LIMIT:
            _failures.popitem(last=False)
        if warn_fallback:
            _warn_fallback(reason)
        return None, reason
