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
_hits = 0
_misses = 0


def compilation_cache_info() -> tuple[int, int, int, int]:
    """Return cache hits, misses, maximum size, and current size."""
    return _CacheInfo(_hits, _misses, _CACHE_LIMIT, len(_cache))


def clear_compilation_cache() -> None:
    """Discard compiled solver variants and reset cache statistics."""
    global _hits, _misses
    _cache.clear()
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
    return (
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


def _make_compiled_solver(
    vf: ManifoldVectorField,
    metric: AnalyticMetric,
    schedule: FixedStepSchedule,
    compute_divergence: bool,
) -> Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    from .integrator import _augmented_rk4_step

    def tensor_solver(x0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = metric.canonicalize(x0.clone())
        integral = torch.zeros(x0.shape[:-1], device=x0.device, dtype=x0.dtype)
        for step in schedule:
            x, increment = _augmented_rk4_step(
                vf,
                metric,
                x,
                step.start,
                step.size,
                compute_divergence=compute_divergence,
            )
            integral = integral + increment
        return x, integral

    def input_boundary(x0: torch.Tensor) -> torch.Tensor:
        return x0.clone()

    compiled_boundary = torch.compile(input_boundary, backend="eager", dynamic=True)

    def run(x0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return tensor_solver(compiled_boundary(x0))

    return run


def _integrate_rk4_compiled(
    vf: ManifoldVectorField,
    metric: AnalyticMetric,
    x0: torch.Tensor,
    schedule: FixedStepSchedule,
    *,
    compute_divergence: bool,
    unsupported_reason: str | None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    global _hits, _misses
    if unsupported_reason is not None:
        _warn_fallback(unsupported_reason)
        return None
    if not hasattr(torch, "compile"):
        _warn_fallback("torch.compile is not provided by this PyTorch build")
        return None

    key = _cache_key(vf, metric, x0, schedule, compute_divergence)
    solver = _cache.get(key)
    if solver is None:
        _misses += 1
        try:
            solver = _make_compiled_solver(vf, metric, schedule, compute_divergence)
        except Exception as error:
            _warn_fallback(f"{type(error).__name__}: {error}")
            return None
        _cache[key] = solver
        _cache.move_to_end(key)
        while len(_cache) > _CACHE_LIMIT:
            _cache.popitem(last=False)
    else:
        _hits += 1
        _cache.move_to_end(key)

    try:
        return solver(x0)
    except Exception as error:
        _cache.pop(key, None)
        _warn_fallback(f"{type(error).__name__}: {error}")
        return None
