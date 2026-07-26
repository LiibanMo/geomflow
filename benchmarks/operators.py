"""Standalone differential-operator microbenchmarks for Phase 4."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from geomflow.torch import (
    AnalyticMetric,
    batched_jacobian,
    christoffel,
    covariant_derivative_tensor,
    divergence,
    gradient,
)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--dtype", choices=("float32", "float64"), required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--dimension", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    dtype = getattr(torch, args.dtype)
    dim = args.dimension
    x = torch.randn(args.batch_size, dim, device=device, dtype=dtype)
    x.requires_grad_(True)

    def metric_fn(point: torch.Tensor) -> torch.Tensor:
        diagonal = 1.0 + point.square()
        return torch.diag_embed(diagonal)

    def derivative_fn(point: torch.Tensor) -> torch.Tensor:
        eye = torch.eye(dim, device=point.device, dtype=point.dtype)
        return 2.0 * torch.einsum("...i,ij,ik->...ijk", point, eye, eye)

    metric = AnalyticMetric(dim, metric_fn, derivative_fn=derivative_fn)
    matrix = torch.randn(dim, dim, device=device, dtype=dtype)
    vf = lambda point: torch.tanh(point @ matrix)
    operations = {
        "batched_jacobian": lambda: batched_jacobian(vf, x),
        "metric_derivative": lambda: metric.derivative(x),
        "christoffel": lambda: christoffel(metric, x),
        "divergence_forward": lambda: divergence(vf, x, metric),
        "gradient": lambda: gradient(lambda point: point.sin().sum(-1), x, metric),
        "covariant_derivative": lambda: covariant_derivative_tensor(vf, x, metric),
    }

    records = []
    for name, operation in operations.items():
        for _ in range(args.warmup):
            operation()
        synchronize(device)
        samples = []
        for _ in range(args.repetitions):
            synchronize(device)
            start = time.perf_counter_ns()
            result = operation()
            if name == "divergence_forward":
                torch.autograd.grad(result.sum(), x, retain_graph=True)
            synchronize(device)
            samples.append((time.perf_counter_ns() - start) / 1e6)
        records.append(
            {
                "operator": name,
                "median_ms": statistics.median(samples),
                "timing_samples_ms": samples,
            }
        )

    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "dtype": str(dtype),
        "batch_size": args.batch_size,
        "dimension": dim,
    }
    if device.type == "cuda":
        metadata["gpu"] = torch.cuda.get_device_name(device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"run": metadata, "operators": records}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
