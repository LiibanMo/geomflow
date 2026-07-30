"""CUDA correctness, memory, and runtime gate for the intrinsic adjoint."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import platform
from pathlib import Path
import subprocess
import sys
import time

import torch

from geomflow.torch import (
    EuclideanSpace,
    cnf_nll,
    intrinsic_adjoint_nll,
)


class LinearField(torch.nn.Module):
    def __init__(self, *, device: torch.device, dtype: torch.dtype) -> None:
        super().__init__()
        self.matrix = torch.nn.Parameter(
            torch.tensor([[0.1, -0.2], [0.15, 0.05]], device=device, dtype=dtype)
        )

    def forward(self, time: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        del time
        return state @ self.matrix.T


def metadata() -> dict[str, object]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    driver = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": revision or "unavailable",
        "os": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "driver": driver or "unavailable",
        "gpu": torch.cuda.get_device_name(),
    }


def run(mode: str, steps: int, dtype: torch.dtype) -> dict[str, object]:
    torch.manual_seed(7)
    device = torch.device("cuda")
    field = LinearField(device=device, dtype=dtype)
    data = torch.randn(2048, 2, device=device, dtype=dtype, requires_grad=True)
    loss_fn = cnf_nll if mode == "direct" else intrinsic_adjoint_nll
    dt = 1.0 / steps

    loss_fn(field, EuclideanSpace(2), data, dt=dt).backward()
    torch.cuda.synchronize()
    field.zero_grad(set_to_none=False)
    data.grad.zero_()
    torch.cuda.empty_cache()
    fixed = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    loss = loss_fn(field, EuclideanSpace(2), data, dt=dt)
    loss.backward()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    gradients = (data.grad, *(parameter.grad for parameter in field.parameters()))
    peak = torch.cuda.max_memory_allocated()
    return {
        "mode": mode,
        "steps": steps,
        "dtype": str(dtype).removeprefix("torch."),
        "seconds": elapsed,
        "fixed_allocated_bytes": fixed,
        "peak_allocated_bytes": peak,
        "adjusted_peak_bytes": peak - fixed,
        "loss": loss.item(),
        "gradients_finite": all(
            gradient is not None and torch.isfinite(gradient).all().item()
            for gradient in gradients
        ),
        "gradient_vector": torch.cat(
            [gradient.detach().reshape(-1).cpu() for gradient in gradients]
        ).tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--child-mode", choices=("direct", "intrinsic_adjoint"))
    parser.add_argument("--child-steps", type=int)
    parser.add_argument("--child-dtype", choices=("float32", "float64"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 6 benchmark requires CUDA")
    if args.child_mode is not None:
        dtype = getattr(torch, args.child_dtype)
        print(json.dumps(run(args.child_mode, args.child_steps, dtype)))
        return 0

    result = {
        "schema_version": 2,
        "status": "running",
        "environment": metadata(),
        "records": [],
        "memory_growth": {},
        "failures": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")

    def checkpoint() -> None:
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        temporary.replace(args.output)

    checkpoint()
    records = result["records"]
    for dtype in ("float32", "float64"):
        for steps in (16, 128):
            for mode in ("direct", "intrinsic_adjoint"):
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--output",
                        str(args.output),
                        "--child-mode",
                        mode,
                        "--child-steps",
                        str(steps),
                        "--child-dtype",
                        dtype,
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                records.append(json.loads(completed.stdout))
                checkpoint()
    failures = result["failures"]
    for dtype in ("float32", "float64"):
        selected = [record for record in records if record["dtype"] == dtype]
        by_key = {(record["mode"], record["steps"]): record for record in selected}
        low = by_key[("intrinsic_adjoint", 16)]["adjusted_peak_bytes"]
        high = by_key[("intrinsic_adjoint", 128)]["adjusted_peak_bytes"]
        growth = high / low if low > 0 else math.inf
        result["memory_growth"][dtype] = growth
        if low < 1_000_000:
            failures.append(
                f"{dtype} adjoint adjusted allocation {low} is fixed-cost dominated"
            )
        if growth > 1.25:
            failures.append(f"{dtype} adjoint peak-memory growth {growth:.3f} > 1.25")
        if not all(record["gradients_finite"] for record in selected):
            failures.append(f"{dtype} produced missing or non-finite gradients")
        tolerance = 2e-4 if dtype == "float32" else 2e-10
        for steps in (16, 128):
            direct = torch.tensor(by_key[("direct", steps)]["gradient_vector"])
            adjoint = torch.tensor(
                by_key[("intrinsic_adjoint", steps)]["gradient_vector"]
            )
            try:
                torch.testing.assert_close(
                    adjoint, direct, rtol=tolerance, atol=tolerance
                )
            except AssertionError as error:
                failures.append(f"{dtype} step {steps} gradient parity: {error}")

    result["status"] = "failed" if failures else "passed"
    checkpoint()
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
