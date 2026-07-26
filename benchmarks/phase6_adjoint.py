"""CUDA correctness, memory, and runtime gate for the intrinsic adjoint."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import platform
from pathlib import Path
import subprocess
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
    data = torch.randn(4, 2, device=device, dtype=dtype, requires_grad=True)
    loss_fn = cnf_nll if mode == "direct" else intrinsic_adjoint_nll
    dt = 1.0 / steps

    loss_fn(field, EuclideanSpace(2), data, dt=dt).backward()
    torch.cuda.synchronize()
    field.zero_grad(set_to_none=True)
    data.grad = None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    loss = loss_fn(field, EuclideanSpace(2), data, dt=dt)
    loss.backward()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    gradients = (data.grad, *(parameter.grad for parameter in field.parameters()))
    return {
        "mode": mode,
        "steps": steps,
        "dtype": str(dtype).removeprefix("torch."),
        "seconds": elapsed,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
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
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 6 benchmark requires CUDA")

    records = [
        run(mode, steps, dtype)
        for dtype in (torch.float32, torch.float64)
        for steps in (16, 128)
        for mode in ("direct", "intrinsic_adjoint")
    ]
    for dtype in ("float32", "float64"):
        selected = [record for record in records if record["dtype"] == dtype]
        by_key = {(record["mode"], record["steps"]): record for record in selected}
        growth = (
            by_key[("intrinsic_adjoint", 128)]["peak_allocated_bytes"]
            / by_key[("intrinsic_adjoint", 16)]["peak_allocated_bytes"]
        )
        if growth > 1.25:
            raise AssertionError(f"{dtype} adjoint peak-memory growth {growth:.3f} > 1.25")
        if not all(record["gradients_finite"] for record in selected):
            raise AssertionError(f"{dtype} produced missing or non-finite gradients")
        tolerance = 2e-4 if dtype == "float32" else 2e-10
        for steps in (16, 128):
            direct = torch.tensor(by_key[("direct", steps)]["gradient_vector"])
            adjoint = torch.tensor(
                by_key[("intrinsic_adjoint", steps)]["gradient_vector"]
            )
            torch.testing.assert_close(
                adjoint, direct, rtol=tolerance, atol=tolerance
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"environment": metadata(), "records": records}, indent=2) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
