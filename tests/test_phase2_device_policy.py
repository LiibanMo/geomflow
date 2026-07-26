"""Device, dtype, callback, and persistent geometry policy tests."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from conftest import requires_cuda
from geomflow import preprocess
from geomflow.torch import AnalyticMetric, Atlas, Chart, EuclideanSpace, ManifoldCNF


def test_preprocess_preserves_tensor_device_and_supports_explicit_device() -> None:
    source = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
    preserved = preprocess(source, dtype=torch.float32)
    explicit = preprocess(np.array([[1.0, 2.0]]), dtype=torch.float64, device="cpu")

    assert preserved.device == source.device
    assert preserved.dtype == torch.float32
    assert explicit.device.type == "cpu"
    assert explicit.dtype == torch.float64


def test_list_preprocessing_does_not_round_through_float32() -> None:
    value = 1.000000000000001
    result = preprocess([[value]], dtype=torch.float64)
    assert result.item() == value


def test_metric_callback_rejects_wrong_leading_shape() -> None:
    metric = AnalyticMetric(2, lambda x: torch.eye(2, dtype=x.dtype).expand(1, 2, 2))
    with pytest.raises(ValueError, match="metric_fn: expected shape"):
        metric.metric(torch.zeros(3, 2))


def test_metric_callback_rejects_wrong_dtype() -> None:
    metric = AnalyticMetric(
        2,
        lambda x: torch.eye(2, dtype=torch.float64).expand(*x.shape[:-1], 2, 2),
    )
    with pytest.raises(ValueError, match="expected device=cpu, dtype=torch.float32"):
        metric.metric(torch.zeros(3, 2, dtype=torch.float32))


def test_debug_metric_validation_checks_finiteness_and_symmetry() -> None:
    metric = AnalyticMetric(
        2,
        lambda x: x.new_tensor([[1.0, 1.0], [0.0, 1.0]]).expand(
            *x.shape[:-1], 2, 2
        ),
        debug_validation=True,
    )
    with pytest.raises(ValueError, match="must be symmetric"):
        metric.metric(torch.zeros(1, 2))


def test_model_rejects_input_dtype_mismatch_early() -> None:
    model = ManifoldCNF(EuclideanSpace(2), hidden_dim=4, n_layers=1)
    with pytest.raises(
        ValueError,
        match=r"ManifoldCNF.log_prob: expected input device=cpu, dtype=torch.float32",
    ):
        model.log_prob(torch.zeros(2, 2, dtype=torch.float64))


def test_sample_rejects_requested_device_mismatch() -> None:
    model = ManifoldCNF(EuclideanSpace(2), hidden_dim=4, n_layers=1)
    with pytest.raises(ValueError, match="ManifoldCNF.sample: expected device=cpu"):
        model.sample(2, device="meta")


def test_atlas_samples_follow_dtype_and_state_dict() -> None:
    chart = Chart(7, 2, torch.randn(8, 2), EuclideanSpace(2))
    model = ManifoldCNF(Atlas([chart], reference_chart_id=7), hidden_dim=4, n_layers=1)
    model = model.double()

    assert chart.samples is not None
    assert chart.samples.dtype == torch.float64
    assert "_atlas_samples_7" in model.state_dict()

    replacement = model.state_dict()
    replacement["_atlas_samples_7"] = torch.full_like(
        replacement["_atlas_samples_7"], 2.0
    )
    model.load_state_dict(replacement)
    torch.testing.assert_close(chart.samples, torch.full_like(chart.samples, 2.0))


@pytest.mark.optional
@requires_cuda
def test_atlas_samples_round_trip_between_cpu_and_cuda() -> None:
    chart = Chart(7, 2, torch.randn(8, 2), EuclideanSpace(2))
    model = ManifoldCNF(Atlas([chart], reference_chart_id=7), hidden_dim=4, n_layers=1)

    model = model.to(device="cuda", dtype=torch.float64)
    assert chart.samples is not None
    assert chart.samples.device.type == "cuda"
    assert chart.samples.dtype == torch.float64

    model = model.cpu()
    assert chart.samples.device.type == "cpu"
    assert chart.samples.dtype == torch.float64


@pytest.mark.parametrize("factory", [EuclideanSpace])
def test_builtin_metric_ignores_default_dtype(factory) -> None:
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        x = torch.randn(3, 2, dtype=torch.float32)
        metric = factory(2)
        assert metric.metric(x).dtype == x.dtype
        assert metric.inverse(x).dtype == x.dtype
        assert metric.sqrt_det(x).dtype == x.dtype
    finally:
        torch.set_default_dtype(previous)
