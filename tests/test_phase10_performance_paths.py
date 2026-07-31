"""Structural correctness gates for Phase 10 hot-path remediation."""

from __future__ import annotations

import pytest
import torch

from geomflow.torch import (
    AnalyticMetric,
    Atlas,
    Chart,
    EuclideanSpace,
    ManifoldVectorField,
    MultiChartVectorField,
    Sphere2DAtlas,
    SphereStereographicMetric,
    divergence,
    integrate_multichart,
    integrate_rk4,
    overlap_consistency_loss,
)


class CountingField(ManifoldVectorField):
    def __init__(self) -> None:
        super().__init__(2, hidden_dim=4, n_layers=1)
        self.public_calls = 0
        self.unchecked_calls = 0

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        self.public_calls += 1
        return super().forward(t, x)

    def _forward_unchecked(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        self.unchecked_calls += 1
        return super()._forward_unchecked(t, x)


class CountingMultiField(MultiChartVectorField):
    def __init__(self, atlas) -> None:
        super().__init__(atlas, hidden_dim=4, n_layers=1)
        self.public_calls = 0
        self.unchecked_calls = 0

    def forward(self, t: torch.Tensor, x: torch.Tensor, chart_id: int) -> torch.Tensor:
        self.public_calls += 1
        return super().forward(t, x, chart_id)

    def _forward_unchecked(
        self, t: torch.Tensor, x: torch.Tensor, chart_id: int
    ) -> torch.Tensor:
        self.unchecked_calls += 1
        return super()._forward_unchecked(t, x, chart_id)


def test_rk4_uses_one_field_call_per_stage() -> None:
    field = ManifoldVectorField(2, hidden_dim=4, n_layers=1)
    calls = 0

    def count_call(_module, _inputs, _output) -> None:
        nonlocal calls
        calls += 1

    handle = field.net[-1].register_forward_hook(count_call)
    x = torch.randn(8, 2, requires_grad=True)

    result = integrate_rk4(field, EuclideanSpace(2), x, 0.0, 1.0, 0.25)
    loss = result.x_final.square().mean() + result.divergence_integral.mean()
    loss.backward()

    handle.remove()
    assert calls == 16
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(parameter.grad is not None for parameter in field.parameters())


@pytest.mark.parametrize("metric_factory", [EuclideanSpace, SphereStereographicMetric])
def test_tensor_value_and_trace_matches_exact_component_gradients(
    metric_factory,
) -> None:
    torch.manual_seed(4)
    field = ManifoldVectorField(2, hidden_dim=5, n_layers=2).double()
    metric = metric_factory(2)
    x = torch.randn(3, 2, dtype=torch.float64, requires_grad=True)
    time = torch.full((3,), 0.3, dtype=torch.float64)

    value, trace = field._tensor_value_and_trace_unchecked(time, x)
    actual = trace + (value * metric._tensor_log_volume_gradient_unchecked(x)).sum(-1)
    expected = divergence(lambda point: field(time, point), x, metric)
    torch.testing.assert_close(value, field(time, x), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)

    variables = (x, *field.parameters())
    actual_first = torch.autograd.grad(actual.sum(), variables, create_graph=True)
    expected_first = torch.autograd.grad(expected.sum(), variables, create_graph=True)
    for actual_value, expected_value in zip(actual_first, expected_first, strict=True):
        torch.testing.assert_close(actual_value, expected_value, rtol=1e-10, atol=1e-10)
    actual_second = torch.autograd.grad(
        sum(value.square().sum() for value in actual_first),
        variables,
        allow_unused=True,
    )
    expected_second = torch.autograd.grad(
        sum(value.square().sum() for value in expected_first),
        variables,
        allow_unused=True,
    )
    for actual_value, expected_value in zip(
        actual_second, expected_second, strict=True
    ):
        if actual_value is None or expected_value is None:
            assert actual_value is expected_value
        else:
            torch.testing.assert_close(
                actual_value, expected_value, rtol=1e-9, atol=1e-9
            )


def test_tensor_core_eligibility_is_narrow_and_solver_reports_backend() -> None:
    field = ManifoldVectorField(2, hidden_dim=4, n_layers=1)
    assert field._supports_tensor_value_and_trace()
    result = integrate_rk4(field, EuclideanSpace(2), torch.randn(3, 2), 0.0, 0.1, 0.1)
    assert result._execution_backend == "tensor-eager"
    assert result._fallback_reason is None

    handle = field.net[-1].register_forward_hook(lambda *_args: None)
    try:
        assert not field._supports_tensor_value_and_trace()
        fallback = integrate_rk4(
            field, EuclideanSpace(2), torch.randn(3, 2), 0.0, 0.1, 0.1
        )
    finally:
        handle.remove()
    assert fallback._execution_backend == "component-gradient-eager"
    assert not ManifoldVectorField(2, periodic=True)._supports_tensor_value_and_trace()
    assert not ManifoldVectorField(
        2, activation=torch.nn.Tanh
    )._supports_tensor_value_and_trace()


def test_solver_preserves_custom_forward_dispatch_and_hooks() -> None:
    field = CountingField()
    hook_calls = 0

    def count_hook(_module, _inputs, _output) -> None:
        nonlocal hook_calls
        hook_calls += 1

    field.register_forward_hook(count_hook)
    integrate_rk4(field, EuclideanSpace(2), torch.randn(4, 2), 0.0, 0.25, 0.25)
    assert field.public_calls == 4
    assert field.unchecked_calls == 4
    assert hook_calls == 4


def test_solver_preserves_global_module_forward_hooks() -> None:
    field = ManifoldVectorField(2, hidden_dim=4, n_layers=1)
    field_calls = 0

    def count_global(module, _inputs, output):
        nonlocal field_calls
        if module is field:
            field_calls += 1
        return output

    handle = torch.nn.modules.module.register_module_forward_hook(count_global)
    try:
        integrate_rk4(field, EuclideanSpace(2), torch.randn(4, 2), 0.0, 0.25, 0.25)
    finally:
        handle.remove()
    assert field_calls == 4


def test_solver_preserves_exact_class_local_execution_hooks() -> None:
    field = ManifoldVectorField(2, hidden_dim=4, n_layers=1)
    calls = {"pre": 0, "forward": 0, "backward_pre": 0, "backward": 0}

    def pre_hook(_module, _inputs) -> None:
        calls["pre"] += 1

    def forward_hook(_module, _inputs, _output) -> None:
        calls["forward"] += 1

    def backward_pre_hook(_module, _grad_output) -> None:
        calls["backward_pre"] += 1

    def backward_hook(_module, _grad_input, _grad_output) -> None:
        calls["backward"] += 1

    handles = [
        field.register_forward_pre_hook(pre_hook),
        field.register_forward_hook(forward_hook),
        field.register_full_backward_pre_hook(backward_pre_hook),
        field.register_full_backward_hook(backward_hook),
    ]
    try:
        x = torch.randn(4, 2, requires_grad=True)
        result = integrate_rk4(field, EuclideanSpace(2), x, 0.0, 0.25, 0.25)
        (result.x_final.sum() + result.divergence_integral.sum()).backward()
    finally:
        for handle in handles:
            handle.remove()

    assert calls["pre"] == 4
    assert calls["forward"] == 4
    assert calls["backward_pre"] > 0
    assert calls["backward"] > 0


@pytest.mark.compilation
def test_solver_preserves_module_compile_dispatch() -> None:
    if not hasattr(torch.nn.Module, "compile"):
        pytest.skip("torch.nn.Module.compile is unavailable")

    field = ManifoldVectorField(1, hidden_dim=4, n_layers=1)
    compilations = 0

    def backend(graph_module, _example_inputs):
        nonlocal compilations
        compilations += 1
        return graph_module.forward

    field.compile(backend=backend)
    integrate_rk4(
        field,
        EuclideanSpace(1),
        torch.zeros(3, 1),
        0.0,
        0.25,
        0.25,
        compute_divergence=False,
    )
    assert compilations > 0


def test_solver_reuses_stage_validation_for_generic_volume_callback() -> None:
    counts = {"domain": 0, "sqrt_det": 0}

    def metric_fn(x: torch.Tensor) -> torch.Tensor:
        return torch.eye(2, device=x.device, dtype=x.dtype).expand(*x.shape[:-1], 2, 2)

    def sqrt_det_fn(x: torch.Tensor) -> torch.Tensor:
        counts["sqrt_det"] += 1
        return torch.ones_like(x[..., 0])

    def domain_fn(x: torch.Tensor) -> torch.Tensor:
        counts["domain"] += 1
        return torch.ones_like(x[..., 0], dtype=torch.bool)

    metric = AnalyticMetric(2, metric_fn, sqrt_det_fn=sqrt_det_fn, domain_fn=domain_fn)
    field = CountingField()
    integrate_rk4(field, metric, torch.randn(4, 2), 0.0, 0.25, 0.25)

    assert counts == {"domain": 6, "sqrt_det": 4}
    assert field.unchecked_calls == 4


def test_multichart_rk4_uses_one_field_call_per_stage() -> None:
    atlas = Sphere2DAtlas()
    field = CountingMultiField(atlas)
    x = 0.1 * torch.randn(8, 2, requires_grad=True)

    result = integrate_multichart(
        field, atlas, x, 0, 0.0, 0.25, 0.25, record_statistics=True
    )
    (result.x_final.square().mean() + result.divergence_integral.mean()).backward()

    assert field.public_calls == 4
    assert field.unchecked_calls == 0
    assert all(parameter.grad is not None for parameter in field.head(0).parameters())
    assert result.statistics is not None
    assert result.statistics.field_call_count == 4
    assert result.statistics.rk_trial_count == 1
    assert result.statistics.accepted_trial_count == 1
    assert result.statistics.rejected_trial_count == 0
    assert result.statistics.chart_predicate_count == 5
    assert result.statistics.scalar_decision_count == 2


def test_multichart_tensor_core_reports_backend_and_hooks_force_fallback() -> None:
    atlas = Sphere2DAtlas()
    field = MultiChartVectorField(atlas, hidden_dim=4, n_layers=1)
    x = 0.1 * torch.randn(4, 2)
    result = integrate_multichart(field, atlas, x, 0, 0.0, 0.1, 0.1)
    assert result._execution_backend == "tensor-eager"

    handle = field.head(0).net[-1].register_forward_hook(lambda *_args: None)
    try:
        fallback = integrate_multichart(field, atlas, x, 0, 0.0, 0.1, 0.1)
    finally:
        handle.remove()
    assert fallback._execution_backend == "component-gradient-eager"

    original_domain = atlas[0]._domain
    atlas[0]._domain = lambda value: torch.ones_like(value[..., 0], dtype=torch.bool)
    try:
        changed_domain = integrate_multichart(field, atlas, x, 0, 0.0, 0.1, 0.1)
    finally:
        atlas[0]._domain = original_domain
    assert changed_domain._execution_backend == "component-gradient-eager"


def test_multichart_solver_preserves_exact_class_hooks() -> None:
    atlas = Sphere2DAtlas()
    field = MultiChartVectorField(atlas, hidden_dim=4, n_layers=1)
    calls = {"field": 0, "head": 0}

    def field_hook(_module, _inputs, _output) -> None:
        calls["field"] += 1

    def head_hook(_module, _inputs, _output) -> None:
        calls["head"] += 1

    handles = [
        field.register_forward_hook(field_hook),
        field.head(0).register_forward_hook(head_hook),
    ]
    try:
        integrate_multichart(
            field,
            atlas,
            0.1 * torch.randn(4, 2),
            0,
            0.0,
            0.25,
            0.25,
        )
    finally:
        for handle in handles:
            handle.remove()

    assert calls == {"field": 4, "head": 4}


def test_multichart_intersects_chart_and_metric_domains() -> None:
    metric = AnalyticMetric(
        1,
        lambda x: x.new_ones(*x.shape[:-1], 1, 1),
        sqrt_det_fn=lambda x: torch.ones_like(x[..., 0]),
        domain_fn=lambda x: x[..., 0] < 0.2,
    )
    chart = Chart(
        0,
        1,
        None,
        metric,
        domain=lambda x: torch.ones_like(x[..., 0], dtype=torch.bool),
    )

    class ConstantField(torch.nn.Module):
        def forward(self, t, x, chart_id):
            del t, chart_id
            return torch.ones_like(x)

    with torch.no_grad(), pytest.raises(RuntimeError, match="overlap|leaves chart"):
        integrate_multichart(
            ConstantField(),
            Atlas([chart], 0),
            torch.tensor([[0.1]]),
            0,
            0.0,
            0.2,
            0.2,
        )


def test_analytic_log_volume_path_matches_weighted_field_reference() -> None:
    analytic = SphereStereographicMetric(2)

    def metric_fn(x: torch.Tensor) -> torch.Tensor:
        r2 = x.square().sum(-1, keepdim=True)
        eye = torch.eye(2, device=x.device, dtype=x.dtype)
        return (4.0 / (1.0 + r2).square()).unsqueeze(-1) * eye

    def sqrt_det_fn(x: torch.Tensor) -> torch.Tensor:
        return (2.0 / (1.0 + x.square().sum(-1))).square()

    weighted = AnalyticMetric(2, metric_fn, sqrt_det_fn=sqrt_det_fn)
    x = torch.tensor([[0.2, -0.3], [-0.1, 0.4]], dtype=torch.float64)
    x.requires_grad_(True)

    def field(point: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            (point[..., 0].square() + point[..., 1], point[..., 0].sin()), -1
        )

    actual = divergence(field, x, analytic)
    expected = divergence(field, x, weighted)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    actual_grad = torch.autograd.grad(actual.sum(), x, create_graph=True)[0]
    expected_grad = torch.autograd.grad(expected.sum(), x, create_graph=True)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(
        torch.autograd.grad(actual_grad.sum(), x, retain_graph=True)[0],
        torch.autograd.grad(expected_grad.sum(), x)[0],
        rtol=1e-10,
        atol=1e-10,
    )


@pytest.mark.parametrize("use_log_volume_gradient", [False, True])
def test_divergence_preserves_public_metric_subclass_overrides(
    use_log_volume_gradient: bool,
) -> None:
    class PublicMetric(AnalyticMetric):
        def __init__(self) -> None:
            super().__init__(
                1,
                lambda x: x.new_ones(*x.shape[:-1], 1, 1),
            )
            self.sqrt_det_calls = 0
            self.log_volume_gradient_calls = 0

        @property
        def has_log_volume_gradient(self) -> bool:
            return use_log_volume_gradient

        def metric(self, x: torch.Tensor) -> torch.Tensor:
            self.validate_points(x)
            return torch.exp(2.0 * x[..., :1]).unsqueeze(-1)

        def sqrt_det(self, x: torch.Tensor) -> torch.Tensor:
            self.sqrt_det_calls += 1
            self.validate_points(x)
            return torch.exp(x[..., 0])

        def log_volume_gradient(self, x: torch.Tensor) -> torch.Tensor:
            self.log_volume_gradient_calls += 1
            self.validate_points(x)
            return torch.ones_like(x)

    metric = PublicMetric()
    x = torch.tensor([[-0.4], [0.7]], dtype=torch.float64, requires_grad=True)
    actual = divergence(lambda point: torch.ones_like(point), x, metric)

    torch.testing.assert_close(actual, torch.ones(2, dtype=torch.float64))
    if use_log_volume_gradient:
        assert metric.log_volume_gradient_calls == 1
        assert metric.sqrt_det_calls == 0
    else:
        assert metric.sqrt_det_calls == 1
        assert metric.log_volume_gradient_calls == 0


def test_exact_divergence_and_integrators_work_in_inference_mode() -> None:
    class LinearField(torch.nn.Module):
        def forward(self, t, x):
            del t
            return 2.0 * x

    class MultiLinearField(torch.nn.Module):
        def forward(self, t, x, chart_id):
            del t, chart_id
            return 2.0 * x

    metric = EuclideanSpace(1)
    chart = Chart(
        0,
        1,
        None,
        metric,
        domain=lambda x: torch.ones_like(x[..., 0], dtype=torch.bool),
    )
    x = torch.ones(2, 1, dtype=torch.float64)

    with torch.inference_mode():
        direct = divergence(lambda point: 2.0 * point, x, metric)
        single = integrate_rk4(LinearField(), metric, x, 0.0, 0.1, 0.1)
        multichart = integrate_multichart(
            MultiLinearField(), Atlas([chart], 0), x, 0, 0.0, 0.1, 0.1
        )

    torch.testing.assert_close(direct, torch.full((2,), 2.0, dtype=x.dtype))
    torch.testing.assert_close(
        single.divergence_integral, torch.full((2,), 0.2, dtype=x.dtype)
    )
    torch.testing.assert_close(
        multichart.divergence_integral, torch.full((2,), 0.2, dtype=x.dtype)
    )
    assert not direct.requires_grad
    assert not single.x_final.requires_grad
    assert not multichart.x_final.requires_grad


def test_chart_subclass_public_overrides_are_preserved() -> None:
    always = lambda x: torch.ones_like(x[..., 0], dtype=torch.bool)

    class TrackingChart(Chart):
        def __init__(self, *args, transition_offset=0.0, jacobian_scale=1.0, **kwargs):
            super().__init__(*args, **kwargs)
            self.transition_offset = transition_offset
            self.jacobian_scale = jacobian_scale
            self.transition_calls = 0
            self.jacobian_calls = 0

        def transition_to(self, target_id: int, x: torch.Tensor) -> torch.Tensor:
            self.transition_calls += 1
            return super().transition_to(target_id, x) + self.transition_offset

        def jacobian(self, target_id: int, x: torch.Tensor) -> torch.Tensor:
            self.jacobian_calls += 1
            return self.jacobian_scale * super().jacobian(target_id, x)

    selection_source = TrackingChart(
        0,
        1,
        None,
        EuclideanSpace(1),
        transitions={1: lambda x: x + 1.0},
        transition_domains={1: always},
        domain=always,
        transition_offset=100.0,
    )
    selection_atlas = Atlas(
        [selection_source, Chart(1, 1, None, EuclideanSpace(1), domain=always)],
        0,
    )
    selection = selection_atlas.find_chart(torch.zeros(1, 1), source_chart=0, prefer=1)
    torch.testing.assert_close(selection.coordinates, torch.tensor([[101.0]]))
    assert selection_source.transition_calls == 1

    switching_source = TrackingChart(
        0,
        1,
        None,
        EuclideanSpace(1),
        transitions={1: lambda x: x},
        transition_domains={1: lambda x: x[..., 0] <= 0.11},
        domain=lambda x: x[..., 0] <= 0.11,
        jacobian_scale=3.0,
    )
    switching_atlas = Atlas(
        [switching_source, Chart(1, 1, None, EuclideanSpace(1), domain=always)],
        0,
    )

    class LinearMultiField(torch.nn.Module):
        def forward(self, t, x, chart_id):
            del t, chart_id
            return 0.5 * x

    result = integrate_multichart(
        LinearMultiField(),
        switching_atlas,
        torch.tensor([[0.1]], dtype=torch.float64),
        0,
        0.0,
        0.4,
        0.2,
        compute_divergence=False,
    )
    assert len(result.transition_events) == 1
    torch.testing.assert_close(
        result.transition_events[0].transition_jacobian,
        torch.tensor([[[3.0]]], dtype=torch.float64),
    )
    assert switching_source.transition_calls == 1
    assert switching_source.jacobian_calls == 1


def test_atlas_selection_and_overlap_loss_intersect_metric_domains() -> None:
    always = lambda x: torch.ones_like(x[..., 0], dtype=torch.bool)
    source_metric = EuclideanSpace(1)
    target_metric = AnalyticMetric(
        1,
        lambda x: x.new_ones(*x.shape[:-1], 1, 1),
        domain_fn=lambda x: x[..., 0].abs() < 1.0,
    )
    source = Chart(
        0,
        1,
        None,
        source_metric,
        transitions={1: lambda x: x + 10.0},
        transition_domains={1: always},
        domain=always,
    )
    target = Chart(1, 1, None, target_metric, domain=always)
    atlas = Atlas([source, target], 0)
    x = torch.zeros(3, 1)

    selection = atlas.find_chart(x, source_chart=0, prefer=1)
    assert selection.chart_id == 0
    assert selection.candidates == (0,)

    field = MultiChartVectorField(atlas, hidden_dim=4, n_layers=1)
    loss = overlap_consistency_loss(
        field,
        atlas,
        x,
        chart_alpha=0,
        chart_beta=1,
        t=torch.zeros(3),
    )
    torch.testing.assert_close(loss, torch.zeros_like(loss))
    loss.backward()
    assert all(parameter.grad is not None for parameter in field.parameters())


def test_public_divergence_validates_domain_before_field_execution() -> None:
    field_calls = 0
    metric = AnalyticMetric(
        1,
        lambda x: x.new_ones(*x.shape[:-1], 1, 1),
        domain_fn=lambda x: torch.zeros_like(x[..., 0], dtype=torch.bool),
    )

    def field(x: torch.Tensor) -> torch.Tensor:
        nonlocal field_calls
        field_calls += 1
        return x

    x = torch.zeros(1, 1, requires_grad=True)
    with pytest.raises(ValueError, match="outside"):
        divergence(field, x, metric)
    assert field_calls == 0
