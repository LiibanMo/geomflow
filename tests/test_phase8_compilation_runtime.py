from __future__ import annotations

import warnings

import pytest
import torch

import geomflow.torch.compilation as compilation_runtime
from conftest import requires_cuda
from geomflow.torch import (
    EuclideanSpace,
    ManifoldVectorField,
    MultiChartVectorField,
    Sphere2DAtlas,
    clear_compilation_cache,
    compilation_cache_info,
    integrate_multichart,
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
def test_compiled_solver_recomputes_exact_graph_for_double_backward() -> None:
    clear_compilation_cache()
    eager_field = _field()
    compiled_field = _field()
    x = torch.randn(3, 2, dtype=torch.double)

    eager = integrate_rk4(
        eager_field, EuclideanSpace(2), x, 0.0, 0.1, 0.1, compile=False
    )
    compiled = integrate_rk4(
        compiled_field, EuclideanSpace(2), x, 0.0, 0.1, 0.1, compile=True
    )
    eager_first = torch.autograd.grad(
        eager.divergence_integral.sum(),
        tuple(eager_field.parameters()),
        create_graph=True,
    )
    compiled_first = torch.autograd.grad(
        compiled.divergence_integral.sum(),
        tuple(compiled_field.parameters()),
        create_graph=True,
    )
    eager_second = torch.autograd.grad(
        sum(value.square().sum() for value in eager_first),
        tuple(eager_field.parameters()),
        allow_unused=True,
    )
    compiled_second = torch.autograd.grad(
        sum(value.square().sum() for value in compiled_first),
        tuple(compiled_field.parameters()),
        allow_unused=True,
    )
    for actual, expected in zip(compiled_first, eager_first, strict=True):
        torch.testing.assert_close(actual, expected)
    for actual, expected in zip(compiled_second, eager_second, strict=True):
        if actual is None or expected is None:
            assert actual is expected
        else:
            torch.testing.assert_close(actual, expected)


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
def test_production_compiler_uses_dynamic_batch_policy(monkeypatch) -> None:
    options = []
    code_objects = []

    def fake_torch_compile(function, **kwargs):
        options.append(kwargs)
        code_objects.append(function.__code__)
        return function

    monkeypatch.setattr(torch, "compile", fake_torch_compile)
    compilation_runtime._make_compiled_solver(
        _field(),
        EuclideanSpace(2),
        compilation_runtime.FixedStepSchedule(0.0, 0.1, 0.1),
        True,
    )
    compilation_runtime._make_compiled_solver(
        _field(),
        EuclideanSpace(2),
        compilation_runtime.FixedStepSchedule(0.0, 0.1, 0.1),
        True,
    )

    assert len(options) == 4
    assert all(option["dynamic"] is True for option in options)
    assert all("mode" not in option for option in options)
    assert len({id(code) for code in code_objects}) == 4


@pytest.mark.compilation
def test_compiled_probe_preserves_real_input_requires_grad(monkeypatch) -> None:
    clear_compilation_cache()
    observed_requires_grad = []

    def fake_compile(vf, metric, schedule, compute_divergence):
        del metric, schedule, compute_divergence
        parameter_count = len(tuple(vf.parameters()))

        def compiled_solver(x, *_parameters):
            observed_requires_grad.append(x.requires_grad)
            return x.clone(), x.new_zeros(x.shape[:-1])

        def compiled_vjp(x, *values):
            grad_x = values[parameter_count]
            return (
                grad_x,
                *(torch.zeros_like(parameter) for parameter in vf.parameters()),
            )

        def eager_solver(x, *_parameters):
            return x.clone(), x.new_zeros(x.shape[:-1])

        return compilation_runtime._with_exact_higher_order_fallback(
            compiled_solver, compiled_vjp, eager_solver, vf
        )

    monkeypatch.setattr(compilation_runtime, "_make_compiled_solver", fake_compile)
    field = _field()
    result = integrate_rk4(
        field,
        EuclideanSpace(2),
        torch.randn(3, 2, dtype=torch.double),
        0.0,
        0.1,
        0.1,
        compile=True,
    )
    assert observed_requires_grad == [False, False]
    assert result._execution_backend == "inductor"


@pytest.mark.compilation
def test_compiled_no_switch_atlas_matches_tensor_eager() -> None:
    clear_compilation_cache()
    atlas = Sphere2DAtlas()
    eager_field = MultiChartVectorField(atlas, hidden_dim=4, n_layers=1).double()
    compiled_field = MultiChartVectorField(atlas, hidden_dim=4, n_layers=1).double()
    compiled_field.load_state_dict(eager_field.state_dict())
    x = 0.1 * torch.randn(3, 2, dtype=torch.double)

    eager = integrate_multichart(eager_field, atlas, x, 0, 0.0, 0.2, 0.1, compile=False)
    compiled = integrate_multichart(
        compiled_field, atlas, x, 0, 0.0, 0.2, 0.1, compile=True
    )
    torch.testing.assert_close(compiled.x_final, eager.x_final)
    torch.testing.assert_close(compiled.divergence_integral, eager.divergence_integral)
    assert compiled.chart_final == eager.chart_final == 0
    assert compiled._execution_backend == "inductor"


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
def test_compilation_cache_invalidates_structure_but_not_parameter_values(
    monkeypatch,
) -> None:
    clear_compilation_cache()
    builds = 0

    def fake_compile(vf, metric, schedule, compute_divergence):
        del metric, schedule, compute_divergence
        nonlocal builds
        builds += 1

        def solver(x):
            offset = next(vf.parameters()).sum()
            return x + offset, x.new_zeros(x.shape[:-1])

        return solver

    monkeypatch.setattr(compilation_runtime, "_make_compiled_solver", fake_compile)
    field = _field()
    metric = EuclideanSpace(2)
    x = torch.zeros(1, 2, dtype=torch.double)

    first = integrate_rk4(field, metric, x, 0.0, 0.1, 0.1, compile=True)
    with torch.no_grad():
        next(field.parameters()).add_(1.0)
    updated = integrate_rk4(field, metric, x, 0.0, 0.1, 0.1, compile=True)
    assert builds == 1
    assert not torch.equal(first.x_final, updated.x_final)

    field.net[-1] = torch.nn.Linear(4, 2, dtype=torch.double)
    integrate_rk4(field, metric, x, 0.0, 0.1, 0.1, compile=True)
    assert builds == 2


@pytest.mark.compilation
def test_compiled_bridge_owns_outputs_across_repeated_calls() -> None:
    field = _field()
    output_buffer = torch.empty(1, 2, dtype=torch.double)
    integral_buffer = torch.empty(1, dtype=torch.double)

    def compiled_solver(x, *_parameters):
        output_buffer.copy_(x)
        integral_buffer.copy_(x.sum(-1))
        return output_buffer, integral_buffer

    def compiled_vjp(x, *values):
        parameter_count = len(tuple(field.parameters()))
        grad_x = values[parameter_count]
        return (grad_x, *(torch.zeros_like(parameter) for parameter in field.parameters()))

    def eager_solver(x, *_parameters):
        return x, x.sum(-1)

    solver = compilation_runtime._with_exact_higher_order_fallback(
        compiled_solver, compiled_vjp, eager_solver, field
    )
    first_x, first_integral = solver(torch.tensor([[1.0, 2.0]], dtype=torch.double))
    solver(torch.tensor([[3.0, 4.0]], dtype=torch.double))

    torch.testing.assert_close(
        first_x, torch.tensor([[1.0, 2.0]], dtype=torch.double)
    )
    torch.testing.assert_close(first_integral, torch.tensor([3.0], dtype=torch.double))


@pytest.mark.compilation
def test_lazy_compiled_vjp_failure_falls_back_before_return(monkeypatch) -> None:
    clear_compilation_cache()

    class BrokenVjp(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            return x.clone()

        @staticmethod
        def backward(ctx, grad_output):
            raise RuntimeError("broken compiled vjp")

    def fake_compile(vf, metric, schedule, compute_divergence):
        del vf, metric, schedule, compute_divergence

        def solver(x):
            return BrokenVjp.apply(x), x.new_zeros(x.shape[:-1])

        return solver

    monkeypatch.setattr(compilation_runtime, "_make_compiled_solver", fake_compile)
    field = _field()
    x = torch.randn(2, 2, dtype=torch.double)
    with pytest.warns(RuntimeWarning, match="broken compiled vjp"):
        result = integrate_rk4(
            field, EuclideanSpace(2), x, 0.0, 0.1, 0.1, compile=True
        )
    assert result._execution_backend == "tensor-eager"
    assert result._fallback_reason == "RuntimeError: broken compiled vjp"
    assert compilation_cache_info().currsize == 0


@pytest.mark.compilation
def test_dynamic_runtime_failure_does_not_poison_other_batch_shapes(monkeypatch) -> None:
    clear_compilation_cache()

    def fake_compile(vf, metric, schedule, compute_divergence):
        del vf, metric, schedule, compute_divergence

        def solver(x):
            if x.shape[0] == 7:
                raise RuntimeError("shape-specific failure")
            return x.clone(), x.new_zeros(x.shape[:-1])

        return solver

    monkeypatch.setattr(compilation_runtime, "_make_compiled_solver", fake_compile)
    field = _field()
    metric = EuclideanSpace(2)
    with pytest.warns(RuntimeWarning, match="shape-specific failure"):
        failed = integrate_rk4(
            field,
            metric,
            torch.randn(7, 2, dtype=torch.double),
            0.0,
            0.1,
            0.1,
            compile=True,
        )
    passed = integrate_rk4(
        field,
        metric,
        torch.randn(2, 2, dtype=torch.double),
        0.0,
        0.1,
        0.1,
        compile=True,
    )

    assert "shape-specific failure" in failed._fallback_reason
    assert passed._execution_backend == "inductor"
    assert passed._fallback_reason is None


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
def test_cuda_auto_compilation_survives_worker_lifecycle() -> None:
    clear_compilation_cache()
    device = torch.device("cuda")
    field = ManifoldVectorField(2, 4, 1).to(device)
    metric = EuclideanSpace(2)

    for batch_size in (2, 7):
        x = torch.randn(batch_size, 2, device=device)
        result = integrate_rk4(field, metric, x, 0.0, 0.1, 0.1)
        result.divergence_integral.sum().backward()
        assert result._execution_backend == "inductor"
        assert result._fallback_reason is None
        field.zero_grad(set_to_none=True)

    with torch.no_grad():
        next(field.parameters()).add_(0.01)
    updated = integrate_rk4(
        field, metric, torch.randn(7, 2, device=device), 0.0, 0.1, 0.1
    )
    assert updated._execution_backend == "inductor"
    assert updated._fallback_reason is None
    info = compilation_cache_info()
    assert (info.misses, info.hits, info.currsize) == (1, 2, 1)


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
