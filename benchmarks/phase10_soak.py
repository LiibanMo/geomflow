"""Bounded CUDA training soak for single-chart and multi-chart CNFs."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from pathlib import Path

import torch

from geomflow.torch import EuclideanSpace, ManifoldCNF, Sphere2DAtlas


def run_case(name: str, model: ManifoldCNF, data: torch.Tensor, iterations: int) -> dict:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    allocated = []
    reserved = []
    losses = []
    torch.cuda.reset_peak_memory_stats(data.device)

    for _ in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        loss = model.training_loss(data)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize(data.device)
        losses.append(float(loss.detach()))
        allocated.append(torch.cuda.memory_allocated(data.device))
        reserved.append(torch.cuda.memory_reserved(data.device))

    tail = allocated[iterations // 2 :]
    growth = max(tail) - min(tail)
    tolerance = 16 * 1024 * 1024
    assert all(torch.isfinite(torch.tensor(losses)))
    assert growth <= tolerance, f"{name} allocated-memory growth {growth} > {tolerance}"
    return {
        "name": name,
        "status": "passed",
        "iterations": iterations,
        "final_loss": losses[-1],
        "tail_allocated_growth_bytes": growth,
        "growth_tolerance_bytes": tolerance,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(data.device),
        "peak_reserved_bytes": max(reserved),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=40)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 10 soak requires CUDA")

    torch.manual_seed(104)
    device = torch.device("cuda", 0)
    single = ManifoldCNF(EuclideanSpace(2), hidden_dim=8, n_layers=1, dt=0.25).to(device)
    multichart = ManifoldCNF(
        Sphere2DAtlas(n_samples=128, seed=104), hidden_dim=8, n_layers=1, dt=0.25
    ).to(device)
    single_data = torch.randn(32, 2, device=device) * 0.2
    chart_data = torch.randn(32, 2, device=device) * 0.1 + 0.5
    cases = [
        run_case("single_chart_direct", single, single_data, args.iterations),
        run_case("multi_chart_direct", multichart, chart_data, args.iterations),
    ]
    properties = torch.cuda.get_device_properties(device)
    result = {
        "phase": 10,
        "status": "passed",
        "cases": cases,
        "environment": {
            "gpu": properties.name,
            "gpu_memory_bytes": properties.total_memory,
            "driver": subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                text=True,
            ).splitlines()[0],
            "cuda": torch.version.cuda,
            "pytorch": torch.__version__,
            "python": platform.python_version(),
            "os": platform.platform(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
