#!/usr/bin/env python3
"""Persistent worker for version-isolated Phase 10 timing samples."""

from __future__ import annotations

import argparse
from dataclasses import replace
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

import torch

package_root = os.environ.get("GEOMFLOW_BENCHMARK_PACKAGE_ROOT")
if package_root:
    sys.path.insert(0, package_root)
    sys.meta_path = [
        finder
        for finder in sys.meta_path
        if not (
            "editable" in type(finder).__module__.lower()
            and "geomflow" in type(finder).__module__.lower()
        )
    ]

import geomflow
from geomflow.torch import integrate_multichart, integrate_rk4

from scenarios import BenchmarkCase, make_geometry, make_input, make_model


def tensor_hash(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def model_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor_hash(tensor).encode())
    return digest.hexdigest()


def package_fingerprint(package_directory: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    files = [
        path
        for path in package_directory.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
    ]
    for path in sorted(
        files, key=lambda item: item.relative_to(package_directory).as_posix()
    ):
        relative = path.relative_to(package_directory).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        contents = path.read_bytes()
        digest.update(len(contents).to_bytes(8, "big"))
        digest.update(contents)
    return {
        "package_root": str(package_directory),
        "package_sha256": digest.hexdigest(),
        "package_file_count": len(files),
    }


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class Worker:
    def __init__(self, device: torch.device, revision: str) -> None:
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA worker requested without an available CUDA device")
        self.device = device
        self.revision = revision
        self.case: BenchmarkCase | None = None
        self.geometry = None
        self.model: torch.nn.Module | None = None
        self.model_key: tuple[Any, ...] | None = None
        self.x: torch.Tensor | None = None

    def describe(self) -> dict[str, Any]:
        package_file = Path(geomflow.__file__).resolve()
        if package_root and not package_file.is_relative_to(
            Path(package_root).resolve()
        ):
            raise RuntimeError(
                f"imported geomflow from {geomflow.__file__}, expected {package_root}"
            )
        result: dict[str, Any] = {
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "geomflow_path": str(package_file),
            "declared_revision": self.revision,
            "device": str(self.device),
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "cpu": platform.processor() or platform.machine(),
            "cpu_count": os.cpu_count(),
            "process_affinity": (
                sorted(os.sched_getaffinity(0))
                if hasattr(os, "sched_getaffinity")
                else None
            ),
            "thread_environment": {
                name: os.environ.get(name)
                for name in (
                    "MKL_NUM_THREADS",
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                )
            },
            "dependency_versions": {
                name: importlib.metadata.version(name)
                for name in ("numpy", "scipy", "torch")
            },
            "ram_bytes": (
                os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
                if hasattr(os, "sysconf")
                else None
            ),
            "torch_backends": {
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            },
            **package_fingerprint(package_file.parent),
        }
        if self.device.type == "cuda":
            properties = torch.cuda.get_device_properties(self.device)
            result["gpu"] = {
                "name": properties.name,
                "uuid": str(properties.uuid),
                "total_memory": properties.total_memory,
                "capability": list(torch.cuda.get_device_capability(self.device)),
            }
            result["nvidia_driver"] = (
                subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=driver_version",
                        "--format=csv,noheader",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                .stdout.splitlines()[0]
                .strip()
            )
            result["cudnn"] = torch.backends.cudnn.version()
        return result

    def prepare(self, raw_case: dict[str, Any]) -> dict[str, Any]:
        dtype = getattr(torch, raw_case["dtype"])
        self.case = BenchmarkCase(
            scenario=raw_case["scenario"],
            batch_size=int(raw_case["batch_size"]),
            dim=int(raw_case["dim"]),
            hidden_width=int(raw_case["hidden_width"]),
            hidden_depth=int(raw_case["hidden_depth"]),
            steps=int(raw_case["steps"]),
            dtype=dtype,
            device=self.device,
            divergence_mode="exact",
            workload=raw_case["workload"],
            seed=int(raw_case["seed"]),
        )
        torch.manual_seed(self.case.seed)
        fresh_geometry = make_geometry(self.case)
        source_case = replace(self.case, device=torch.device("cpu"))
        fresh_model = make_model(source_case, fresh_geometry).to(self.device)
        model_key = (
            self.case.scenario,
            self.case.dim,
            self.case.hidden_width,
            self.case.hidden_depth,
            self.case.dtype,
            self.case.seed,
        )
        if self.model is not None and self.model_key == model_key:
            self.model.load_state_dict(fresh_model.state_dict())
        else:
            self.geometry = fresh_geometry
            self.model = fresh_model
            self.model_key = model_key
        self.x = make_input(self.case)
        return {
            "case_id": self.case.case_id,
            "model_hash": model_hash(self.model),
            "input_hash": tensor_hash(self.x),
        }

    def _operation(self) -> tuple[object, torch.Tensor | None]:
        assert self.case is not None and self.model is not None and self.x is not None
        kwargs = {
            "t0": 0.0,
            "t1": 1.0,
            "dt": 1.0 / self.case.steps,
            "compute_divergence": True,
        }
        if self.case.scenario == "sphere-atlas":
            result = integrate_multichart(
                self.model, self.geometry, self.x, start_chart=0, **kwargs
            )
        else:
            result = integrate_rk4(self.model, self.geometry, self.x, **kwargs)
        loss = None
        if self.case.workload == "backward":
            loss = result.x_final.square().mean() + result.divergence_integral.mean()
        return result, loss

    def sample(self) -> dict[str, Any]:
        assert self.model is not None
        self.model.zero_grad(set_to_none=True)
        gc.collect()
        synchronize(self.device)
        total_start = time.perf_counter_ns()
        result, loss = self._operation()
        synchronize(self.device)
        forward_end = time.perf_counter_ns()
        if loss is not None:
            loss.backward()
            synchronize(self.device)
        total_end = time.perf_counter_ns()
        sample = {
            "forward_ms": (forward_end - total_start) / 1e6,
            "backward_ms": (total_end - forward_end) / 1e6,
            "wall_ms": (total_end - total_start) / 1e6,
            "backend": getattr(
                result, "_execution_backend", "component-gradient-eager"
            ),
            "fallback_reason": getattr(result, "_fallback_reason", None),
        }
        del result, loss
        return sample

    def warmup(self, count: int) -> dict[str, int]:
        for _ in range(count):
            self.sample()
        return {"iterations": count}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    worker = Worker(torch.device(args.device), args.revision)

    for line in sys.stdin:
        request = json.loads(line)
        try:
            action = request["action"]
            if action == "describe":
                result = worker.describe()
            elif action == "prepare":
                result = worker.prepare(request["case"])
            elif action == "warmup":
                result = worker.warmup(int(request["count"]))
            elif action == "sample":
                result = worker.sample()
            elif action == "close":
                print(json.dumps({"ok": True, "result": {}}), flush=True)
                return 0
            else:
                raise ValueError(f"unknown action: {action}")
            response = {"ok": True, "result": result}
        except Exception as error:
            response = {"ok": False, "error": f"{type(error).__name__}: {error}"}
        print(json.dumps(response), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
