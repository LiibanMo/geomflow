"""Mathematical domain, topology, and metric tests for built-in presets."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from geomflow.torch import (
    ManifoldCNF,
    PoincareDisk,
    Sphere2DAtlas,
    SphereStereographicMetric,
    Torus2D,
    integrate_rk4,
)


DTYPE = torch.float64


class _ConstantField(nn.Module):
    def __init__(self, value: torch.Tensor):
        super().__init__()
        self.register_buffer("value", value)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.value.expand_as(x)


def test_poincare_domain_is_open_and_metric_is_not_clamped() -> None:
    metric = PoincareDisk(2)
    near = torch.tensor([[1.0 - 1e-6, 0.0]], dtype=DTYPE)
    boundary = torch.tensor([[1.0, 0.0]], dtype=DTYPE)
    outside = torch.tensor([[1.01, 0.0]], dtype=DTYPE)

    assert metric.contains(near).item()
    assert not metric.contains(boundary).item()
    assert not metric.contains(outside).item()
    expected = 4.0 / (1.0 - near[0, 0].square()).square()
    torch.testing.assert_close(metric.metric(near)[0, 0, 0], expected)
    with pytest.raises(ValueError, match="coordinate domain"):
        metric.metric(boundary)
    with pytest.raises(ValueError, match="coordinate domain"):
        metric.sqrt_det(outside)


def test_poincare_integrator_rejects_a_stage_leaving_the_disk() -> None:
    metric = PoincareDisk(2)
    field = _ConstantField(torch.tensor([1.0, 0.0], dtype=DTYPE))
    with pytest.raises(ValueError, match="coordinate domain"):
        integrate_rk4(
            field,
            metric,
            torch.tensor([[0.9, 0.0]], dtype=DTYPE),
            0.0,
            0.2,
            0.2,
            compute_divergence=False,
        )


@pytest.mark.parametrize("R,r", [(1.0, 1.0), (0.5, 1.0), (2.0, 0.0), (math.inf, 1.0)])
def test_torus_rejects_invalid_radii(R: float, r: float) -> None:
    with pytest.raises(ValueError, match="R > r > 0"):
        Torus2D(R, r)


def test_torus_metric_field_and_coordinates_respect_identifications() -> None:
    metric = Torus2D(2.0, 0.75)
    x = torch.tensor([[0.4, -1.2]], dtype=DTYPE)
    shifted = x + torch.tensor([[2.0 * math.pi, -2.0 * math.pi]], dtype=DTYPE)
    torch.testing.assert_close(metric.metric(x), metric.metric(shifted))
    torch.testing.assert_close(metric.sqrt_det(x), metric.sqrt_det(shifted))
    torch.testing.assert_close(metric.canonicalize(shifted), x)

    model = ManifoldCNF(metric, hidden_dim=8, n_layers=1).double()
    time = torch.tensor([0.3], dtype=DTYPE)
    torch.testing.assert_close(model.vf(time, x), model.vf(time, shifted))


def test_torus_trajectory_crosses_both_angular_seams() -> None:
    metric = Torus2D()
    field = _ConstantField(torch.tensor([1.0, 1.5], dtype=DTYPE))
    result = integrate_rk4(
        field,
        metric,
        torch.tensor([[3.0, 3.0]], dtype=DTYPE),
        0.0,
        0.4,
        0.1,
        track_trajectory=True,
        compute_divergence=False,
    )
    expected = torch.remainder(
        torch.tensor([[3.4, 3.6]], dtype=DTYPE) + math.pi, 2.0 * math.pi
    ) - math.pi
    torch.testing.assert_close(result.x_final, expected)
    assert all(((state >= -math.pi) & (state < math.pi)).all() for _, state, _ in result.trajectory)


def test_sphere_radius_scales_metric_inverse_and_volume() -> None:
    x = torch.tensor([[0.3, -0.7]], dtype=DTYPE)
    unit = SphereStereographicMetric(2, radius=1.0)
    scaled = SphereStereographicMetric(2, radius=2.5)
    torch.testing.assert_close(scaled.metric(x), 2.5**2 * unit.metric(x))
    torch.testing.assert_close(scaled.inverse(x), unit.inverse(x) / 2.5**2)
    torch.testing.assert_close(scaled.sqrt_det(x), 2.5**2 * unit.sqrt_det(x))
    tangent = torch.tensor([[1.2, -0.4]], dtype=DTYPE)
    cotangent = torch.tensor([[0.7, 1.1]], dtype=DTYPE)
    unit_inner = torch.einsum("...i,...ij,...j->...", tangent, unit.metric(x), tangent)
    scaled_inner = torch.einsum(
        "...i,...ij,...j->...", tangent, scaled.metric(x), tangent
    )
    torch.testing.assert_close(scaled_inner, 2.5**2 * unit_inner)
    torch.testing.assert_close(
        torch.einsum("...ij,...j->...i", scaled.inverse(x), cotangent),
        torch.einsum("...ij,...j->...i", unit.inverse(x), cotangent) / 2.5**2,
    )
    with pytest.raises(ValueError, match="radius"):
        SphereStereographicMetric(2, radius=0.0)


def test_sphere_transition_is_exact_only_on_the_overlap() -> None:
    atlas = Sphere2DAtlas(n_samples=10)
    x = torch.tensor([[0.5, -1.25]], dtype=DTYPE, requires_grad=True)
    mapped = atlas[0].transition_to(1, x)
    torch.testing.assert_close(atlas[1].transition_to(0, mapped), x)

    radius_squared = x.square().sum(dim=-1, keepdim=True)
    eye = torch.eye(2, dtype=DTYPE).unsqueeze(0)
    expected_jacobian = eye / radius_squared.unsqueeze(-1) - (
        2.0 * x.unsqueeze(-1) * x.unsqueeze(-2) / radius_squared.square().unsqueeze(-1)
    )
    torch.testing.assert_close(atlas[0].jacobian(1, x), expected_jacobian)
    with pytest.raises(ValueError, match="outside overlap"):
        atlas[0].transition_to(1, torch.zeros(1, 2, dtype=DTYPE))
