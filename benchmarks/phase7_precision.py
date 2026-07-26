"""CUDA stability and precision characterization for the Phase 7 gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import platform
from pathlib import Path
import subprocess
import time

import torch

from geomflow.torch import AnalyticMetric, PoincareDisk, integrate_rk4


class StableField(torch.nn.Module):
    def __init__(self, device: torch.device, dtype: torch.dtype) -> None:
        super().__init__()
        self.matrix = torch.nn.Parameter(
            torch.tensor([[-0.05, 0.02], [-0.01, -0.04]], device=device, dtype=dtype)
        )

    def forward(self, time: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        del time
        return state @ self.matrix.T


def metadata() -> dict[str, object]:
    def command(*args: str) -> str:
        return subprocess.run(args, capture_output=True, text=True, check=False).stdout.strip()

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": command("git", "rev-parse", "HEAD") or "unavailable",
        "os": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "driver": command(
            "nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"
        )
        or "unavailable",
        "gpu": torch.cuda.get_device_name(),
    }


def trajectory(dtype: torch.dtype) -> dict[str, object]:
    device = torch.device("cuda")
    field = StableField(device, dtype)
    x = torch.tensor([[0.2, -0.15], [-0.1, 0.25]], device=device, dtype=dtype)
    x.requires_grad_(True)
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = integrate_rk4(field, PoincareDisk(2), x, 0.0, 4.0, 1.0 / 64.0)
    loss = result.x_final.square().sum() + result.divergence_integral.sum()
    loss.backward()
    torch.cuda.synchronize()
    gradients = [x.grad, *(parameter.grad for parameter in field.parameters())]
    return {
        "dtype": str(dtype).removeprefix("torch."),
        "seconds": time.perf_counter() - start,
        "state": result.x_final.detach().cpu().tolist(),
        "divergence": result.divergence_integral.detach().cpu().tolist(),
        "loss": loss.item(),
        "gradients": torch.cat([value.detach().reshape(-1).cpu() for value in gradients]).tolist(),
        "finite": all(value is not None and torch.isfinite(value).all().item() for value in gradients),
    }


def low_precision_probe(dtype: torch.dtype) -> dict[str, object]:
    device = torch.device("cuda")
    x = torch.tensor([[0.2, -0.1]], device=device, dtype=dtype)
    matrix = torch.tensor([[[2.0, 0.2], [0.2, 1.0]]], device=device, dtype=dtype)
    results: dict[str, object] = {"dtype": str(dtype).removeprefix("torch.")}
    for name, operation in {
        "cholesky": lambda: torch.linalg.cholesky(matrix),
        "slogdet": lambda: torch.linalg.slogdet(matrix),
        "higher_order_autograd": lambda: torch.autograd.functional.hessian(
            lambda value: value.square().sum(), x.requires_grad_(True)
        ),
    }.items():
        try:
            output = operation()
            tensors = output if isinstance(output, tuple) else (output,)
            results[name] = {
                "supported": True,
                "finite": all(torch.isfinite(value).all().item() for value in tensors),
            }
        except (RuntimeError, TypeError) as error:
            results[name] = {"supported": False, "error": str(error)}
    metric = AnalyticMetric(2, lambda value: matrix.expand(*value.shape[:-1], 2, 2))
    try:
        metric.metric(x)
    except TypeError as error:
        results["production_policy"] = {"accepted": False, "error": str(error)}
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 7 benchmark requires CUDA")

    records = [trajectory(dtype) for dtype in (torch.float32, torch.float64)]
    if not all(record["finite"] for record in records):
        raise AssertionError("supported precision produced non-finite gradients")
    float32, float64 = records
    torch.testing.assert_close(
        torch.tensor(float32["state"]), torch.tensor(float64["state"]), rtol=8e-4, atol=8e-5
    )
    torch.testing.assert_close(
        torch.tensor(float32["gradients"]),
        torch.tensor(float64["gradients"]),
        rtol=3e-3,
        atol=3e-4,
    )
    probes = [low_precision_probe(dtype) for dtype in (torch.float16, torch.bfloat16)]
    if any(probe["production_policy"]["accepted"] for probe in probes):
        raise AssertionError("unsupported low precision passed the production boundary")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"environment": metadata(), "trajectories": records, "low_precision": probes}, indent=2)
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
