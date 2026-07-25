"""Dynamic-chart domain, transition, and tensor-transformation oracles."""

from __future__ import annotations

import pytest
import torch

from geomflow.torch import (
    Atlas,
    Chart,
    ChartDomainError,
    EuclideanSpace,
    Sphere2DAtlas,
    SphereStereographicMetric,
    Transition,
    integrate_multichart,
    pullback_covector,
    pushforward_vector,
    replay_transition_pullbacks,
    transform_metric,
)


def _finite(x: torch.Tensor) -> torch.Tensor:
    return torch.isfinite(x).all(dim=-1)


def test_chart_domain_is_distinct_from_sample_coverage() -> None:
    samples = torch.tensor([[0.0], [0.1]], dtype=torch.float64)
    chart = Chart(
        0,
        1,
        samples,
        EuclideanSpace(1),
        domain=lambda x: x[..., 0].abs() < 2.0,
    )
    point = torch.tensor([[1.0]], dtype=torch.float64)
    assert chart.contains(point).item()
    assert not chart.heuristically_covered(point).item()


def test_atlas_query_maps_known_source_coordinates_and_reports_ambiguity() -> None:
    shift = lambda x: x + 10.0
    chart0 = Chart(
        0,
        1,
        None,
        EuclideanSpace(1),
        transitions={1: Transition(shift, _finite)},
        domain=lambda x: x[..., 0] < 1.0,
    )
    chart1 = Chart(
        1,
        1,
        None,
        EuclideanSpace(1),
        domain=lambda x: x[..., 0] > 9.0,
    )
    atlas = Atlas([chart0, chart1], 0)
    x = torch.tensor([[0.25]], dtype=torch.float64)
    selected = atlas.find_chart(x, source_chart=0, prefer=1)
    assert selected.chart_id == 1
    assert selected.candidates == (0, 1)
    assert torch.equal(selected.coordinates, x + 10.0)


def test_sphere_transition_rejects_excluded_pole_coordinate() -> None:
    atlas = Sphere2DAtlas()
    with pytest.raises(ChartDomainError, match="outside overlap"):
        atlas[0].transition_to(1, torch.zeros(1, 2, dtype=torch.float64))


def test_stereographic_tangent_covector_metric_and_density_transformations() -> None:
    atlas = Sphere2DAtlas()
    x = torch.tensor([[0.5, -0.75], [1.25, 0.4]], dtype=torch.float64)
    y = atlas[0].transition_to(1, x)
    jacobian = atlas[0].jacobian(1, x)
    inverse_jacobian = atlas[1].jacobian(0, y)
    vector = torch.tensor([[0.3, 1.1], [-0.7, 0.2]], dtype=torch.float64)
    covector_y = torch.tensor([[1.2, -0.4], [0.1, 0.8]], dtype=torch.float64)

    vector_y = pushforward_vector(vector, jacobian)
    covector_x = pullback_covector(covector_y, jacobian)
    assert torch.allclose(pushforward_vector(vector_y, inverse_jacobian), vector)
    assert torch.allclose(
        (covector_x * vector).sum(-1), (covector_y * vector_y).sum(-1)
    )

    metric_x = SphereStereographicMetric(2).metric(x)
    metric_y = SphereStereographicMetric(2).metric(y)
    assert torch.allclose(transform_metric(metric_x, jacobian), metric_y)
    norm_x = torch.einsum("bi,bij,bj->b", vector, metric_x, vector)
    norm_y = torch.einsum("bi,bij,bj->b", vector_y, metric_y, vector_y)
    assert torch.allclose(norm_x, norm_y)

    log_rho = torch.tensor([-1.3, 0.7], dtype=torch.float64)
    log_q_x = log_rho + torch.log(SphereStereographicMetric(2).sqrt_det(x))
    log_abs_det = torch.linalg.slogdet(jacobian).logabsdet
    log_q_y = log_q_x - log_abs_det
    recovered = log_q_y - torch.log(SphereStereographicMetric(2).sqrt_det(y))
    assert torch.allclose(recovered, log_rho)


class _LinearField(torch.nn.Module):
    def __init__(self, coefficient: torch.Tensor) -> None:
        super().__init__()
        self.coefficient = coefficient

    def forward(self, t: torch.Tensor, x: torch.Tensor, chart_id: int) -> torch.Tensor:
        del t, chart_id
        return self.coefficient * x


def _threshold_atlas(with_switch: bool) -> Atlas:
    upper = 0.11 if with_switch else 1.0
    chart0 = Chart(
        0,
        1,
        None,
        EuclideanSpace(1),
        transitions={1: Transition(lambda x: x, lambda x: x[..., 0] <= upper)},
        domain=lambda x: x[..., 0] <= upper,
    )
    chart1 = Chart(1, 1, None, EuclideanSpace(1), domain=_finite)
    return Atlas([chart0, chart1], 0)


def test_nonzero_flow_value_density_and_gradient_ignore_valid_switch_schedule() -> None:
    coefficient = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    x0 = torch.tensor([[0.1]], dtype=torch.float64, requires_grad=True)
    switched = integrate_multichart(
        _LinearField(coefficient), _threshold_atlas(True), x0, 0, 0.0, 0.4, 0.2
    )
    switched_loss = switched.x_final.sum() + switched.divergence_integral.sum()
    switched_grads = torch.autograd.grad(switched_loss, (x0, coefficient))

    coefficient_ref = coefficient.detach().clone().requires_grad_(True)
    x0_ref = x0.detach().clone().requires_grad_(True)
    unswitched = integrate_multichart(
        _LinearField(coefficient_ref),
        _threshold_atlas(False),
        x0_ref,
        0,
        0.0,
        0.4,
        0.2,
    )
    reference_loss = unswitched.x_final.sum() + unswitched.divergence_integral.sum()
    reference_grads = torch.autograd.grad(reference_loss, (x0_ref, coefficient_ref))

    assert len(switched.transition_events) == 1
    terminal_covector = torch.tensor([[0.7]], dtype=torch.float64)
    assert torch.allclose(
        replay_transition_pullbacks(switched.transition_events, terminal_covector),
        terminal_covector,
    )
    assert torch.allclose(switched.x_final, unswitched.x_final, atol=2e-8)
    assert torch.allclose(
        switched.divergence_integral, unswitched.divergence_integral, atol=2e-8
    )
    assert torch.allclose(switched_grads[0], reference_grads[0], atol=2e-7)
    assert torch.allclose(switched_grads[1], reference_grads[1], atol=2e-7)
