#!/usr/bin/env python3
"""Measure distinct multi-chart control, rejection, and transition workloads."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import statistics
import time

import torch

from geomflow.torch import (
    Atlas,
    Chart,
    ManifoldVectorField,
    MultiChartVectorField,
    Sphere2DAtlas,
    SphereStereographicMetric,
    Transition,
    EuclideanSpace,
    integrate_multichart,
    integrate_rk4,
)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def samples(operation, device: torch.device, repetitions: int = 8) -> list[float]:
    operation()
    values = []
    for _ in range(repetitions):
        synchronize(device)
        start = time.perf_counter_ns()
        result = operation()
        synchronize(device)
        values.append((time.perf_counter_ns() - start) / 1e6)
        del result
    return values


def no_switch_control(device: torch.device) -> list[dict[str, object]]:
    torch.manual_seed(0)
    single = ManifoldVectorField(2, hidden_dim=32, n_layers=2).to(device)
    atlas = Sphere2DAtlas()
    multi = MultiChartVectorField(atlas, hidden_dim=32, n_layers=2).to(device)
    multi.head(0).load_state_dict(single.state_dict())
    x = (0.1 * torch.randn(256, 2, device=device)).requires_grad_(True)
    kwargs = {"t0": 0.0, "t1": 1.0, "dt": 1.0 / 16.0}

    single_samples = samples(
        lambda: integrate_rk4(single, SphereStereographicMetric(2), x, **kwargs),
        device,
    )
    atlas_samples = samples(
        lambda: integrate_multichart(multi, atlas, x, start_chart=0, **kwargs),
        device,
    )
    structural = integrate_multichart(
        multi, atlas, x, start_chart=0, record_statistics=True, **kwargs
    )
    assert structural.statistics is not None
    overhead = statistics.median(atlas_samples) / statistics.median(single_samples) - 1.0
    return [
        {
            "workload": "single-chart-sphere-control",
            "samples_ms": single_samples,
        },
        {
            "workload": "atlas-no-switch",
            "samples_ms": atlas_samples,
            "overhead_fraction": overhead,
            "statistics": asdict(structural.statistics),
        },
    ]


class ConstantMultiField(torch.nn.Module):
    def __init__(self, velocity: float) -> None:
        super().__init__()
        self.velocity = velocity

    def forward(self, time, state, chart_id):
        del time, chart_id
        return torch.full_like(state, self.velocity)


def switching_atlas(limit: float) -> Atlas:
    def domain(x: torch.Tensor) -> torch.Tensor:
        return torch.isfinite(x).all(dim=-1) & (x[..., 0] <= limit)

    def overlap(x: torch.Tensor) -> torch.Tensor:
        return domain(x) & (x[..., 0] >= 0.75 * limit)

    def transition(x: torch.Tensor) -> torch.Tensor:
        return -x

    def jacobian(x: torch.Tensor) -> torch.Tensor:
        return -torch.ones(x.shape + (1,), device=x.device, dtype=x.dtype)

    metric = EuclideanSpace(1)
    chart0 = Chart(
        0,
        1,
        None,
        metric,
        transitions={1: Transition(transition, overlap, jacobian)},
        domain=domain,
    )
    chart1 = Chart(
        1,
        1,
        None,
        metric,
        transitions={0: Transition(transition, overlap, jacobian)},
        domain=domain,
    )
    return Atlas([chart0, chart1], 0)


def transition_case(
    name: str,
    device: torch.device,
    *,
    limit: float,
    velocity: float,
    initial: float,
    t1: float,
    dt: float,
) -> dict[str, object]:
    atlas = switching_atlas(limit)
    x = torch.full((256, 1), initial, device=device)
    result = integrate_multichart(
        ConstantMultiField(velocity).to(device),
        atlas,
        x,
        0,
        0.0,
        t1,
        dt,
        max_subdivisions=16,
        record_statistics=True,
    )
    assert result.statistics is not None
    return {
        "workload": name,
        "transition_count": len(result.transition_events),
        "statistics": asdict(result.statistics),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 10 multi-chart performance gate requires CUDA")
    device = torch.device("cuda")
    result = {"schema_version": 1, "status": "running", "records": [], "failures": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")

    def checkpoint() -> None:
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        temporary.replace(args.output)

    checkpoint()
    result["records"].extend(no_switch_control(device))
    checkpoint()
    result["records"].append(
        transition_case(
            "deterministic-frequent-switch",
            device,
            limit=0.11,
            velocity=2.0,
            initial=0.1,
            t1=1.0,
            dt=0.1,
        )
    )
    checkpoint()
    result["records"].append(
        transition_case(
            "late-stage-rejection",
            device,
            limit=0.19,
            velocity=1.0,
            initial=0.0,
            t1=0.2,
            dt=0.2,
        )
    )
    checkpoint()
    result["records"].append(
        transition_case(
            "bounded-bisection",
            device,
            limit=0.11,
            velocity=0.5,
            initial=0.1,
            t1=0.4,
            dt=0.4,
        )
    )
    no_switch = next(
        record for record in result["records"] if record["workload"] == "atlas-no-switch"
    )
    if no_switch["overhead_fraction"] > 0.20:
        result["failures"].append(
            f"no-switch atlas overhead {no_switch['overhead_fraction']:.3f} > 0.200"
        )
    if no_switch["statistics"]["scalar_decision_count"] > 17:
        result["failures"].append("no-switch atlas exceeded one decision per accepted step")
    result["status"] = "failed" if result["failures"] else "passed"
    checkpoint()
    return 0 if not result["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
