from __future__ import annotations

import warnings

import pytest
import torch

import geomflow.torch.compilation as compilation_runtime
from conftest import requires_cuda
from geomflow.torch import (
    EuclideanSpace,
    ManifoldVectorField,
    clear_compilation_cache,
    compilation_cache_info,
    integrate_rk4,
)


def _field() -> ManifoldVectorField:
    torch.manual_seed(13)
    return ManifoldVectorField(2, 4, 1).double()


@pytest.mark.compilation
def test_compiled_forward_and_backward_match_eager() -> None:
    clear_compilation_cache()
    eager_field = _field()
    compiled_field = _field()
    x_eager = torch.randn(3, 2, dtype=torch.double, requires_grad=True)
    x_compiled = x_eager.detach().clone().requires_grad_(True)

    eager = integrate_rk4(eager_field, EuclideanSpace(2), x_eager, 0.0, 0.2, 0.1)
    compiled = integrate_rk4(
        compiled_field, EuclideanSpace(2), x_compiled, 0.0, 0.2, 0.1, compile=True
    )
    eager_loss = eager.x_final.square().sum() + eager.divergence_integral.sum()
    compiled_loss = compiled.x_final.square().sum() + compiled.divergence_integral.sum()
    eager_loss.backward()
    compiled_loss.backward()

    torch.testing.assert_close(compiled.x_final, eager.x_final)
    torch.testing.assert_close(compiled.divergence_integral, eager.divergence_integral)
    torch.testing.assert_close(x_compiled.grad, x_eager.grad)
    for eager_parameter, compiled_parameter in zip(
        eager_field.parameters(), compiled_field.parameters()
    ):
        torch.testing.assert_close(compiled_parameter.grad, eager_parameter.grad)


@pytest.mark.compilation
def test_dynamic_batch_reuses_one_compiled_variant() -> None:
    clear_compilation_cache()
    field = _field()
    metric = EuclideanSpace(2)
    for batch_size in (2, 7):
        result = integrate_rk4(
            field,
            metric,
            torch.randn(batch_size, 2, dtype=torch.double),
            0.0,
            0.1,
            0.1,
            compile=True,
        )
        assert result.x_final.shape == (batch_size, 2)
    info = compilation_cache_info()
    assert (info.misses, info.hits, info.currsize) == (1, 1, 1)


@pytest.mark.compilation
def test_compilation_cache_is_bounded_and_reuses_recent_entries(monkeypatch) -> None:
    clear_compilation_cache()
    builds = 0

    def fake_compile(vf, metric, schedule, compute_divergence):
        nonlocal builds
        builds += 1

        def solver(x):
            return x.clone(), x.new_zeros(x.shape[:-1])

        return solver

    monkeypatch.setattr(compilation_runtime, "_make_compiled_solver", fake_compile)
    fields = [_field() for _ in range(compilation_cache_info().maxsize + 1)]
    metric = EuclideanSpace(2)
    x = torch.zeros(1, 2, dtype=torch.double)
    for field in fields:
        integrate_rk4(field, metric, x, 0.0, 0.1, 0.1, compile=True)
    integrate_rk4(fields[-1], metric, x, 0.0, 0.1, 0.1, compile=True)

    info = compilation_cache_info()
    assert info.currsize == info.maxsize
    assert info.hits == 1
    assert builds == info.maxsize + 1


@pytest.mark.compilation
def test_callback_and_compiler_failure_warn_and_fall_back_eager(monkeypatch) -> None:
    clear_compilation_cache()
    field = _field()
    metric = EuclideanSpace(2)
    x = torch.randn(2, 2, dtype=torch.double)
    stages: list[float] = []
    with pytest.warns(RuntimeWarning, match="callbacks require eager"):
        callback_result = integrate_rk4(
            field,
            metric,
            x,
            0.0,
            0.1,
            0.1,
            stage_callback=lambda time, _: stages.append(time),
            compile=True,
        )
    assert len(stages) == 4
    assert callback_result.x_final.device == x.device

    def fail_compile(*args, **kwargs):
        raise RuntimeError("compiler unavailable")

    monkeypatch.setattr(compilation_runtime, "_make_compiled_solver", fail_compile)
    with pytest.warns(RuntimeWarning, match="compiler unavailable"):
        fallback = integrate_rk4(field, metric, x, 0.0, 0.1, 0.1, compile=True)
    with warnings.catch_warnings():
        eager = integrate_rk4(field, metric, x, 0.0, 0.1, 0.1)
    assert fallback.x_final.device == x.device
    torch.testing.assert_close(fallback.x_final, eager.x_final)


@pytest.mark.compilation
@pytest.mark.gpu
@pytest.mark.optional
@requires_cuda
def test_compiled_solver_uses_current_non_default_cuda_stream() -> None:
    clear_compilation_cache()
    device = torch.device("cuda")
    field = ManifoldVectorField(2, 4, 1).to(device)
    metric = EuclideanSpace(2)
    stream = torch.cuda.Stream(device=device)
    x = torch.randn(4, 2, device=device)
    with torch.cuda.stream(stream):
        result = integrate_rk4(
            field, metric, x, 0.0, 0.1, 0.1, compute_divergence=False, compile=True
        )
        event = torch.cuda.Event()
        event.record()
    torch.cuda.current_stream(device).wait_event(event)
    assert result.x_final.device.type == device.type
    assert result.x_final.device.index == torch.cuda.current_device()
    assert torch.isfinite(result.x_final).all()
