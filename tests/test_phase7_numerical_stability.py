"""Phase 7 stability, precision, and mixed-precision policy gates."""

import pytest
import torch

from conftest import supported_device_dtype_cases
from geomflow.torch import (
    AnalyticMetric,
    PoincareDisk,
    Sphere2DAtlas,
    SphereStereographicMetric,
    StandardNormalCoordinateBase,
    christoffel,
    gradient,
    integrate_rk4,
)


CASES = supported_device_dtype_cases()


def _spd_metric(x: torch.Tensor) -> torch.Tensor:
    one = torch.ones_like(x[..., 0])
    return torch.stack(
        (
            torch.stack((2.0 + x[..., 0].square(), 0.2 * one), dim=-1),
            torch.stack((0.2 * one, 1.0 + x[..., 1].square()), dim=-1),
        ),
        dim=-2,
    )


@pytest.mark.parametrize("device_dtype", CASES)
def test_metric_solve_logdet_and_higher_gradients(device_dtype) -> None:
    device, dtype = device_dtype
    x = torch.tensor([[0.2, -0.3], [-0.4, 0.1]], device=device, dtype=dtype)
    x.requires_grad_(True)
    rhs = torch.tensor([[0.7, -0.2], [0.1, 0.9]], device=device, dtype=dtype)
    metric = AnalyticMetric(2, _spd_metric)

    solved = metric.solve(x, rhs)
    reference = torch.linalg.solve(_spd_metric(x), rhs.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(solved, reference)
    sign, reference_logdet = torch.linalg.slogdet(_spd_metric(x))
    assert torch.equal(sign, torch.ones_like(sign))
    torch.testing.assert_close(metric.log_det(x), reference_logdet)

    first = torch.autograd.grad((solved.square().sum() + metric.log_det(x).sum()), x, create_graph=True)[0]
    second = torch.autograd.grad(first.sum(), x)[0]
    assert torch.isfinite(first).all()
    assert torch.isfinite(second).all()


@pytest.mark.parametrize("device_dtype", CASES)
def test_operators_use_stable_metric_solves(device_dtype) -> None:
    device, dtype = device_dtype
    x = torch.tensor([[0.2, -0.3]], device=device, dtype=dtype, requires_grad=True)
    metric = AnalyticMetric(2, _spd_metric)

    actual_gradient = gradient(lambda point: point.square().sum(-1), x, metric)
    expected_gradient = torch.linalg.solve(_spd_metric(x), (2.0 * x).unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(actual_gradient, expected_gradient)
    gamma = christoffel(metric, x)
    assert gamma.shape == (1, 2, 2, 2)
    assert torch.isfinite(gamma).all()


def test_debug_metric_reports_singular_point_context() -> None:
    def singular_metric(x: torch.Tensor) -> torch.Tensor:
        result = torch.eye(2, dtype=x.dtype, device=x.device).expand(*x.shape[:-1], 2, 2).clone()
        result[..., 1, 1] = 0.0
        return result

    metric = AnalyticMetric(2, singular_metric, debug_validation=True)
    with pytest.raises(ValueError, match=r"positive-definite.*batch index 0.*point="):
        metric.metric(torch.tensor([[0.25, -0.5]]))


@pytest.mark.parametrize("dtype", [torch.int64, torch.complex64, torch.float16, torch.bfloat16])
def test_unsupported_dtypes_are_rejected_at_public_boundaries(dtype) -> None:
    metric = PoincareDisk(2)
    point = torch.zeros((2, 2), dtype=dtype)
    with pytest.raises(TypeError, match="float32 or torch.float64"):
        metric.metric(point)
    with pytest.raises(TypeError, match="unsupported dtype"):
        StandardNormalCoordinateBase(2).sample((2,), device=torch.device("cpu"), dtype=dtype)
    with pytest.raises(TypeError, match="float32 or torch.float64"):
        integrate_rk4(lambda _t, x: torch.zeros_like(x), metric, point, 0.0, 1.0, 0.1)


def test_autocast_is_explicitly_unsupported() -> None:
    x = torch.zeros((2, 2), dtype=torch.float32)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        with pytest.raises(RuntimeError, match="automatic mixed precision is unsupported"):
            integrate_rk4(
                lambda _t, point: torch.zeros_like(point),
                PoincareDisk(2),
                x,
                0.0,
                0.1,
                0.1,
            )


@pytest.mark.parametrize("device_dtype", CASES)
def test_boundary_policy_keeps_accepted_metrics_and_transitions_finite(device_dtype) -> None:
    device, dtype = device_dtype
    poincare = PoincareDisk(2)
    near_boundary = torch.tensor([[1.0 - 8.0 * torch.finfo(dtype).eps, 0.0]], device=device, dtype=dtype)
    assert poincare.contains(near_boundary).all()
    assert torch.isfinite(poincare.metric(near_boundary)).all()
    assert torch.isfinite(poincare.log_det(near_boundary)).all()

    sphere = SphereStereographicMetric(2)
    safe_scale = torch.finfo(dtype).max ** 0.25 / 4.0
    safe = torch.tensor([[safe_scale, 0.0]], device=device, dtype=dtype)
    unsafe = torch.tensor([[safe_scale * 8.0, 0.0]], device=device, dtype=dtype)
    assert sphere.contains(safe).all()
    assert torch.isfinite(sphere.metric(safe)).all()
    assert not sphere.contains(unsafe).any()

    atlas = Sphere2DAtlas()
    tiny = torch.tensor([[torch.finfo(dtype).tiny ** 0.75, 0.0]], device=device, dtype=dtype)
    assert not atlas[0].can_transition_to(1, tiny).any()


@pytest.mark.parametrize("device_dtype", CASES)
def test_long_float32_and_float64_trajectory_is_finite(device_dtype) -> None:
    device, dtype = device_dtype
    x0 = torch.tensor([[0.2, -0.15], [-0.1, 0.25]], device=device, dtype=dtype)

    def contracting_field(_time: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return -0.05 * x

    result = integrate_rk4(
        contracting_field,
        PoincareDisk(2),
        x0,
        0.0,
        10.0,
        0.02,
    )
    assert torch.isfinite(result.x_final).all()
    assert torch.isfinite(result.divergence_integral).all()
