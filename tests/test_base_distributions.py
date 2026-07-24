"""Phase 2 base-measure, normalization, domain, and API contract tests."""

from __future__ import annotations

import math

import pytest
import torch

from geomflow.torch import (
    AnalyticMetric,
    AtlasBaseDistribution,
    EuclideanSpace,
    InducedMetric,
    ManifoldCNF,
    PoincareDisk,
    PoincareDiskCoordinateBase,
    Sphere2DAtlas,
    SphereStereographicMetric,
    StandardNormalCoordinateBase,
    Torus2D,
    UniformAngleCoordinateBase,
    cnf_nll,
    cnf_nll_multichart,
)


DTYPE = torch.float64


def _zero_field(model: ManifoldCNF) -> None:
    for parameter in model.vf.parameters():
        parameter.data.zero_()


def test_coordinate_base_converts_arbitrary_leading_shapes_to_volume() -> None:
    """MATH-300--307/MATH-320--323: one explicit measure conversion path."""
    scale = 2.5
    metric = AnalyticMetric(
        2,
        lambda x: scale**2
        * torch.eye(2, device=x.device, dtype=x.dtype).expand(*x.shape[:-1], 2, 2),
        sqrt_det_fn=lambda x: torch.full(
            x.shape[:-1], scale**2, device=x.device, dtype=x.dtype
        ),
    )
    base = StandardNormalCoordinateBase(2)
    x = torch.randn(2, 3, 4, 2, dtype=DTYPE)
    expected_log_q = -0.5 * (
        2 * math.log(2 * math.pi) + x.square().sum(dim=-1)
    )
    torch.testing.assert_close(
        base.log_prob_volume(x, metric), expected_log_q - 2 * math.log(scale)
    )

    samples = base.sample((2, 3, 4), device=x.device, dtype=x.dtype)
    assert samples.shape == x.shape
    assert samples.dtype == x.dtype
    assert samples.device == x.device


def test_euclidean_zero_flow_nll_and_samples_share_standard_normal() -> None:
    """MATH-310--312: identity volume makes coordinate and volume density equal."""
    model = ManifoldCNF(EuclideanSpace(3), hidden_dim=4, n_layers=1).double()
    _zero_field(model)
    samples, chart = model.sample(64)
    expected = -0.5 * (
        3 * math.log(2 * math.pi) + samples.square().sum(dim=-1)
    )
    torch.testing.assert_close(model.log_prob(samples), expected)
    assert chart is None


@pytest.mark.parametrize(
    "metric",
    [
        SphereStereographicMetric(2),
        InducedMetric(2, lambda x: torch.cat((2.0 * x, x[..., :1] * 0.0), dim=-1)),
    ],
    ids=["sphere-stereographic", "induced-metric"],
)
def test_unrestricted_single_chart_presets_convert_coordinate_base(
    metric: AnalyticMetric,
) -> None:
    """MATH-320--324/MATH-359: unrestricted presets use normalized q/sqrt(g)."""
    model = ManifoldCNF(metric, hidden_dim=4, n_layers=1).double()
    _zero_field(model)
    samples, _ = model.sample(64)
    log_q = -0.5 * (
        2 * math.log(2 * math.pi) + samples.square().sum(dim=-1)
    )
    torch.testing.assert_close(
        model.log_prob(samples) + torch.log(metric.sqrt_det(samples)), log_q
    )


def test_poincare_base_is_normalized_and_strictly_inside_disk() -> None:
    """MATH-304--305/MATH-323/MATH-359: normalized disk-supported base."""
    metric = PoincareDisk(2)
    base = PoincareDiskCoordinateBase(2)
    torch.manual_seed(12)
    samples = base.sample((20_000,), device=torch.device("cpu"), dtype=DTYPE)
    assert base.contains(samples).all()

    radius_squared = samples.square().sum(dim=-1)
    u = samples / torch.sqrt(1.0 - radius_squared).unsqueeze(-1)
    log_q_u = -0.5 * (2 * math.log(2 * math.pi) + u.square().sum(dim=-1))
    log_abs_det_du_dx = -2.0 * torch.log1p(-radius_squared)
    recovered_log_q = base.log_prob_volume(samples, metric) + torch.log(
        metric.sqrt_det(samples)
    )
    torch.testing.assert_close(recovered_log_q, log_q_u + log_abs_det_du_dx)

    model = ManifoldCNF(metric, hidden_dim=4, n_layers=1).double()
    _zero_field(model)
    model_samples, _ = model.sample(128)
    assert base.contains(model_samples).all()
    torch.testing.assert_close(
        model.log_prob(model_samples), base.log_prob_volume(model_samples, metric)
    )


def test_torus_base_normalizes_over_canonical_angle_cell() -> None:
    """MATH-305/MATH-323/MATH-359: rho*dV recovers uniform angle measure."""
    metric = Torus2D(R=2.0, r=0.75)
    base = UniformAngleCoordinateBase(2)
    grid = torch.linspace(-math.pi, math.pi, 801, dtype=DTYPE)[:-1]
    theta, phi = torch.meshgrid(grid, grid, indexing="ij")
    x = torch.stack((theta, phi), dim=-1)
    cell_area = (2.0 * math.pi / grid.numel()) ** 2
    mass = (
        torch.exp(base.log_prob_volume(x, metric)) * metric.sqrt_det(x)
    ).sum() * cell_area
    torch.testing.assert_close(mass, torch.tensor(1.0, dtype=DTYPE), atol=2e-14, rtol=0)

    model = ManifoldCNF(metric, hidden_dim=4, n_layers=1).double()
    _zero_field(model)
    samples, _ = model.sample(128)
    assert base.contains(samples).all()
    torch.testing.assert_close(
        model.log_prob(samples), base.log_prob_volume(samples, metric)
    )


def test_atlas_base_is_scalar_across_reference_chart_transition() -> None:
    """MATH-308/MATH-330--334: reference chart is explicit and rho is scalar."""
    atlas = Sphere2DAtlas(n_samples=300, seed=4)
    base = AtlasBaseDistribution(StandardNormalCoordinateBase(2), 0)
    x_reference = torch.tensor([[0.7, 1.2], [-1.1, 0.8]], dtype=DTYPE)
    x_other = atlas[0].transition_to(1, x_reference)
    torch.testing.assert_close(
        base.log_prob_volume(x_reference, atlas, 0),
        base.log_prob_volume(x_other, atlas, 1),
    )

    model = ManifoldCNF(
        atlas, hidden_dim=4, n_layers=1, base_distribution=base
    ).double()
    _zero_field(model)
    samples, chart = model.sample(64)
    torch.testing.assert_close(
        model.log_prob(samples, start_chart=chart),
        base.log_prob_volume(samples, atlas, chart),
    )


def test_loss_apis_use_the_same_measure_explicit_base() -> None:
    """MATH-340--342: low- and high-level APIs use one base-density contract."""
    metric = PoincareDisk(1)
    model = ManifoldCNF(metric, hidden_dim=4, n_layers=1, dt=0.2).double()
    _zero_field(model)
    data = torch.tensor([[-0.4], [0.2]], dtype=DTYPE)
    expected = -model.log_prob(data).mean()
    actual = cnf_nll(
        model.vf,
        metric,
        data,
        dt=model.dt,
        base_distribution=model.base_distribution,
    )
    torch.testing.assert_close(actual, expected)

    atlas = Sphere2DAtlas(n_samples=300, seed=5)
    atlas_model = ManifoldCNF(atlas, hidden_dim=4, n_layers=1, dt=0.2).double()
    _zero_field(atlas_model)
    atlas_data = torch.tensor([[0.8, 0.9], [1.1, 0.7]], dtype=DTYPE)
    expected_atlas = -atlas_model.log_prob(atlas_data, start_chart=0).mean()
    actual_atlas = cnf_nll_multichart(
        atlas_model.vf,
        atlas,
        atlas_data,
        start_chart=0,
        dt=atlas_model.dt,
        base_distribution=atlas_model.base_distribution,
    )
    torch.testing.assert_close(actual_atlas, expected_atlas)


def test_invalid_or_ambiguous_custom_base_is_rejected() -> None:
    """MATH-322/MATH-334: callbacks without a declared volume measure fail."""
    with pytest.raises(TypeError, match="log_prob_volume"):
        ManifoldCNF(EuclideanSpace(2), base_distribution=lambda x: x.sum(-1))

    wrong_reference = AtlasBaseDistribution(StandardNormalCoordinateBase(2), 1)
    with pytest.raises(ValueError, match="reference chart"):
        ManifoldCNF(Sphere2DAtlas(n_samples=50), base_distribution=wrong_reference)
