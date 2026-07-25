"""Phase 3 device-native atlas coverage and selection tests."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.spatial import KDTree

from geomflow.torch import AnalyticMetric, Atlas, Chart, Transition
from conftest import requires_cuda


def _metric(dim: int) -> AnalyticMetric:
    def metric_fn(x: torch.Tensor) -> torch.Tensor:
        return torch.eye(dim, device=x.device, dtype=x.dtype).expand(
            *x.shape[:-1], dim, dim
        )

    return AnalyticMetric(dim, metric_fn)


def _sample_chart(
    samples: torch.Tensor, *, chunk_size: int | None = None, chart_id: int = 0
) -> Chart:
    return Chart(
        chart_id,
        samples.shape[1],
        samples,
        _metric(samples.shape[1]),
        distance_chunk_size=chunk_size,
    )


@pytest.mark.parametrize("chunk_size", [1, 7, 10_000])
def test_torch_membership_matches_scipy_reference(chunk_size: int) -> None:
    generator = torch.Generator().manual_seed(400)
    samples = torch.randn(53, 3, generator=generator, dtype=torch.float64)
    chart = _sample_chart(samples, chunk_size=chunk_size)
    queries = torch.cat(
        (
            torch.randn(97, 3, generator=generator, dtype=torch.float64),
            samples[:8],
        )
    )

    distances, _ = KDTree(samples.numpy()).query(queries.numpy(), k=chart.k)
    expected = np.asarray(distances).reshape(len(queries), -1).max(axis=1) <= chart.radius

    actual = chart.is_inside(queries)
    assert actual.device == queries.device
    assert actual.dtype == torch.bool
    np.testing.assert_array_equal(actual.numpy(), expected)


def test_membership_supports_noncontiguous_multidimensional_batches() -> None:
    samples = torch.randn(31, 2, dtype=torch.float64)
    chart = _sample_chart(samples, chunk_size=5)
    queries = torch.randn(4, 3, 2, dtype=torch.float64).transpose(0, 1)
    assert not queries.is_contiguous()

    mask = chart.is_inside(queries)

    assert mask.shape == queries.shape[:-1]
    assert torch.equal(mask, chart.is_inside(queries.contiguous()))


def test_equal_distance_ties_have_deterministic_membership() -> None:
    samples = torch.tensor([[-1.0], [1.0]], dtype=torch.float64)
    chart = Chart(0, 1, samples, _metric(1), k=1, distance_chunk_size=1)
    query = torch.zeros(1, 1, dtype=torch.float64)

    assert torch.equal(chart.is_inside(query), chart.is_inside(query))


def test_atlas_selection_is_stable_and_transition_keeps_gradients() -> None:
    always = lambda x: torch.ones(x.shape[:-1], device=x.device, dtype=torch.bool)
    shift = lambda x: x + 1.0
    source = Chart(
        2,
        1,
        None,
        _metric(1),
        transitions={1: Transition(shift, always), 0: Transition(shift, always)},
        domain=always,
    )
    atlas = Atlas(
        [source, Chart(1, 1, None, _metric(1), domain=always),
         Chart(0, 1, None, _metric(1), domain=always)],
        reference_chart_id=0,
    )
    x = torch.tensor([[2.0]], dtype=torch.float64, requires_grad=True)

    selection = atlas.find_chart(x, source_chart=2, prefer=0)
    selection.coordinates.sum().backward()

    assert selection.chart_id == 0
    assert selection.candidates == (0, 1, 2)
    torch.testing.assert_close(x.grad, torch.ones_like(x))


def test_atlas_rejects_empty_batch() -> None:
    atlas = Atlas([_sample_chart(torch.randn(8, 2))], reference_chart_id=0)
    with pytest.raises(ValueError, match="non-empty batch"):
        atlas.best_chart(torch.empty(0, 2), current=0)


@pytest.mark.optional
@requires_cuda
def test_cuda_membership_has_no_host_transfer() -> None:
    samples = torch.randn(256, 2, device="cuda")
    chart = _sample_chart(samples, chunk_size=37)
    queries = torch.randn(64, 2, device="cuda")

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        mask = chart.is_inside(queries)
    names = {event.key for event in prof.key_averages()}

    assert mask.device.type == "cuda"
    assert not names.intersection({"aten::cpu", "aten::to", "aten::_to_copy"})
