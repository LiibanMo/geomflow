"""Phase 8 compilation characterization and profiling benchmark.

This benchmark records observations; it does not encode release thresholds or
claim that a configuration is faster. Run it on the designated CUDA Vast host.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import platform
from pathlib import Path
import statistics
import subprocess
import time
from typing import Callable

import torch

from geomflow.torch import (
    EuclideanSpace,
    ManifoldVectorField,
    MultiChartVectorField,
    Sphere2DAtlas,
    clear_compilation_cache,
    compilation_cache_info,
    divergence,
    integrate_multichart,
    integrate_rk4,
    intrinsic_adjoint_nll,
)


TensorWork = Callable[[], torch.Tensor]


def command(*args: str) -> str:
    try:
        return subprocess.run(
            args, capture_output=True, text=True, check=False
        ).stdout.strip()
    except OSError:
        return ""


def metadata(args: argparse.Namespace) -> dict[str, object]:
    cuda = torch.cuda.is_available()
    cpu_count = os.cpu_count()
    ram = None
    try:
        ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        pass
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": command("git", "rev-parse", "HEAD") or "unavailable",
        "git_dirty": bool(command("git", "status", "--porcelain")),
        "os": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "logical_cpu_count": cpu_count,
        "ram_bytes": ram,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_compile_available": hasattr(torch, "compile"),
        "cuda_available": cuda,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if cuda else None,
        "gpu": torch.cuda.get_device_name() if cuda else None,
        "gpu_capability": list(torch.cuda.get_device_capability()) if cuda else None,
        "driver": command(
            "nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"
        )
        or "unavailable",
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "compile_backend": "mixed exact eager and TorchInductor",
        "compile_dynamic_shapes": True,
        "nvtx": "disabled: profiler events and JSON are the benchmark outputs",
    }


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_samples(
    work: TensorWork, device: torch.device, warmup: int, repetitions: int
) -> list[float]:
    for _ in range(warmup):
        work()
    synchronize(device)
    samples = []
    for _ in range(repetitions):
        start = time.perf_counter_ns()
        work()
        synchronize(device)
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return samples


def summarize(samples: list[float]) -> dict[str, object]:
    return {
        "milliseconds": samples,
        "median_milliseconds": statistics.median(samples),
        "minimum_milliseconds": min(samples),
    }


def make_field(device: torch.device, dtype: torch.dtype, width: int) -> ManifoldVectorField:
    torch.manual_seed(8128)
    return ManifoldVectorField(2, width, 1).to(device=device, dtype=dtype)


def direct_work(
    field: ManifoldVectorField,
    data: torch.Tensor,
    steps: int,
    compiled: bool,
) -> TensorWork:
    metric = EuclideanSpace(2)

    def work() -> torch.Tensor:
        field.zero_grad(set_to_none=True)
        data.grad = None
        result = integrate_rk4(
            field,
            metric,
            data,
            0.0,
            1.0,
            1.0 / steps,
            compile=compiled,
        )
        loss = result.x_final.square().mean() + result.divergence_integral.mean()
        loss.backward()
        return loss

    return work


def adjoint_work(
    field: ManifoldVectorField, data: torch.Tensor, steps: int
) -> TensorWork:
    metric = EuclideanSpace(2)

    def work() -> torch.Tensor:
        field.zero_grad(set_to_none=True)
        data.grad = None
        loss = intrinsic_adjoint_nll(field, metric, data, dt=1.0 / steps)
        loss.backward()
        return loss

    return work


def cold_warm_latency(
    device: torch.device, dtype: torch.dtype, batch: int, width: int, steps: int
) -> dict[str, object]:
    clear_compilation_cache()
    field = make_field(device, dtype, width)
    data = torch.randn(batch, 2, device=device, dtype=dtype, requires_grad=True)
    work = direct_work(field, data, steps, True)
    synchronize(device)
    start = time.perf_counter_ns()
    work()
    synchronize(device)
    cold = (time.perf_counter_ns() - start) / 1e6
    start = time.perf_counter_ns()
    work()
    synchronize(device)
    warm = (time.perf_counter_ns() - start) / 1e6
    return {
        "device": device.type,
        "cold_milliseconds": cold,
        "warm_milliseconds": warm,
        "cache_info": compilation_cache_info()._asdict(),
    }


def parity(
    device: torch.device, dtype: torch.dtype, batch: int, width: int, steps: int
) -> dict[str, object]:
    eager_field = make_field(device, dtype, width)
    compiled_field = make_field(device, dtype, width)
    source = torch.randn(batch, 2, device=device, dtype=dtype)
    eager_x = source.clone().requires_grad_(True)
    compiled_x = source.clone().requires_grad_(True)
    eager = integrate_rk4(
        eager_field, EuclideanSpace(2), eager_x, 0.0, 1.0, 1.0 / steps
    )
    clear_compilation_cache()
    compiled = integrate_rk4(
        compiled_field,
        EuclideanSpace(2),
        compiled_x,
        0.0,
        1.0,
        1.0 / steps,
        compile=True,
    )
    eager_loss = eager.x_final.square().mean() + eager.divergence_integral.mean()
    compiled_loss = compiled.x_final.square().mean() + compiled.divergence_integral.mean()
    eager_loss.backward()
    compiled_loss.backward()
    synchronize(device)
    pairs = [
        ("state", eager.x_final, compiled.x_final),
        ("divergence", eager.divergence_integral, compiled.divergence_integral),
        ("input_gradient", eager_x.grad, compiled_x.grad),
    ]
    pairs.extend(
        (f"parameter_gradient:{name}", eager_parameter.grad, compiled_parameter.grad)
        for (name, eager_parameter), compiled_parameter in zip(
            eager_field.named_parameters(), compiled_field.parameters()
        )
    )
    errors = {}
    for name, expected, actual in pairs:
        assert expected is not None and actual is not None
        difference = (actual - expected).abs()
        errors[name] = {
            "max_absolute": difference.max().item(),
            "max_relative": (
                difference / expected.abs().clamp_min(torch.finfo(dtype).eps)
            ).max().item(),
            "finite": bool(torch.isfinite(actual).all().item()),
        }
    return {"device": device.type, "dtype": str(dtype).split(".")[-1], "errors": errors}


def dynamic_batches(
    device: torch.device,
    dtype: torch.dtype,
    batches: list[int],
    width: int,
    steps: int,
) -> dict[str, object]:
    clear_compilation_cache()
    field = make_field(device, dtype, width)
    metric = EuclideanSpace(2)
    shapes = []
    for batch in batches:
        result = integrate_rk4(
            field,
            metric,
            torch.randn(batch, 2, device=device, dtype=dtype),
            0.0,
            1.0,
            1.0 / steps,
            compute_divergence=False,
            compile=True,
        )
        shapes.append(list(result.x_final.shape))
    synchronize(device)
    return {
        "batches": batches,
        "result_shapes": shapes,
        "cache_info": compilation_cache_info()._asdict(),
    }


def graph_break_report(
    name: str, function: Callable[..., object], *inputs: object
) -> dict[str, object]:
    try:
        report = torch._dynamo.explain(function)(*inputs)
        reasons = []
        for item in getattr(report, "break_reasons", []):
            reasons.append(str(getattr(item, "reason", item)))
        return {
            "name": name,
            "status": "reported",
            "graph_count": getattr(report, "graph_count", None),
            "graph_break_count": getattr(report, "graph_break_count", None),
            "break_reasons": reasons,
        }
    except Exception as error:
        return {"name": name, "status": "error", "error": f"{type(error).__name__}: {error}"}


def graph_break_reports(
    device: torch.device, dtype: torch.dtype, batch: int, width: int, steps: int
) -> list[dict[str, object]]:
    field = make_field(device, dtype, width)
    metric = EuclideanSpace(2)
    x = torch.randn(batch, 2, device=device, dtype=dtype)
    atlas = Sphere2DAtlas(n_samples=32, seed=8)
    multichart_field = MultiChartVectorField(atlas, hidden_dim=width, n_layers=1).to(
        device=device, dtype=dtype
    )
    chart_x = torch.zeros(batch, 2, device=device, dtype=dtype)
    return [
        graph_break_report(
            "vector_field",
            lambda value: field(value.new_full(value.shape[:-1], 0.5), value),
            x,
        ),
        graph_break_report(
            "operators",
            lambda value: divergence(
                lambda point: field(point.new_full(point.shape[:-1], 0.5), point),
                value,
                metric,
            ),
            x,
        ),
        graph_break_report("single_chart", lambda value: integrate_rk4(field, metric, value, 0.0, 1.0, 1.0 / steps).x_final, x),
        graph_break_report("multichart", lambda value: integrate_multichart(multichart_field, atlas, value, 0, 0.0, 1.0, 1.0 / steps, compute_divergence=False).x_final, chart_x),
    ]


def profile_work(work: TensorWork, device: torch.device, steps: int) -> dict[str, object]:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
        torch.cuda.reset_peak_memory_stats(device)
    with torch.profiler.profile(
        activities=activities, profile_memory=True, record_shapes=False
    ) as profile:
        work()
        synchronize(device)
    events = profile.events()
    cuda_events = [event for event in events if str(event.device_type).lower().endswith("cuda")]
    kernel_names = sorted({event.name for event in cuda_events})
    return {
        "steps": steps,
        "cuda_kernel_event_count": len(cuda_events),
        "cuda_kernel_events_per_step": len(cuda_events) / steps,
        "unique_cuda_kernel_count": len(kernel_names),
        "unique_cuda_kernel_names": kernel_names,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else None,
        "profiler_self_cuda_memory_usage_bytes": sum(
            getattr(event, "self_cuda_memory_usage", 0) for event in events
        ),
    }


def non_default_stream(
    dtype: torch.dtype, batch: int, width: int, steps: int
) -> dict[str, object]:
    device = torch.device("cuda")
    field = make_field(device, dtype, width)
    data = torch.randn(batch, 2, device=device, dtype=dtype)
    stream = torch.cuda.Stream(device=device)
    clear_compilation_cache()
    with torch.cuda.stream(stream):
        result = integrate_rk4(
            field,
            EuclideanSpace(2),
            data,
            0.0,
            1.0,
            1.0 / steps,
            compute_divergence=False,
            compile=True,
        )
        event = torch.cuda.Event()
        event.record(stream)
    torch.cuda.current_stream(device).wait_event(event)
    return {
        "completed": bool(event.query()),
        "finite": bool(torch.isfinite(result.x_final).all().item()),
        "result_device": str(result.x_final.device),
    }


def runtime_matrix(
    device: torch.device,
    dtype: torch.dtype,
    batch: int,
    width: int,
    steps: int,
    warmup: int,
    repetitions: int,
) -> list[dict[str, object]]:
    rows = []
    for mode, compiled in (("direct_eager", False), ("direct_compiled", True)):
        clear_compilation_cache()
        field = make_field(device, dtype, width)
        data = torch.randn(batch, 2, device=device, dtype=dtype, requires_grad=True)
        samples = timed_samples(
            direct_work(field, data, steps, compiled), device, warmup, repetitions
        )
        rows.append({"mode": mode, "device": device.type, **summarize(samples)})
    field = make_field(device, dtype, width)
    data = torch.randn(batch, 2, device=device, dtype=dtype, requires_grad=True)
    samples = timed_samples(adjoint_work(field, data, steps), device, warmup, repetitions)
    rows.append(
        {
            "mode": "intrinsic_adjoint_eager",
            "device": device.type,
            "compilation": "unsupported: intrinsic adjoint is eager-only",
            **summarize(samples),
        }
    )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/phase8_cuda.json"))
    parser.add_argument("--quick", action="store_true", help="Use the smallest affordable smoke workload")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dynamic-batches", type=int, nargs="+", default=[8, 32, 96])
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--allow-cpu-only", action="store_true", help="Run CPU coverage when CUDA is unavailable")
    args = parser.parse_args()
    if args.quick:
        args.batch_size, args.dynamic_batches, args.steps = 8, [2, 5], 2
        args.width, args.warmup, args.repetitions = 4, 1, 2
    if min(args.batch_size, args.steps, args.width, args.repetitions) < 1 or args.warmup < 0:
        parser.error("batch size, steps, width, and repetitions must be positive; warmup must be nonnegative")
    if not torch.cuda.is_available() and not args.allow_cpu_only:
        raise RuntimeError("Phase 8 benchmark requires CUDA; use --allow-cpu-only for a smoke run")

    dtype = getattr(torch, args.dtype)
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.insert(0, torch.device("cuda"))
    cuda_or_cpu = devices[0]
    result: dict[str, object] = {
        "schema_version": 1,
        "environment": metadata(args),
        "compile_scope": {
            "default": "eager; compilation is optional via integrate_rk4(..., compile=True)",
            "supported": "single-chart direct-autograd fixed-step RK4 without trajectory capture or stage callbacks",
            "unsupported": ["multichart integration", "intrinsic adjoint", "user callbacks", "trajectory capture"],
            "fallback": "warning followed by eager execution on the input device",
            "cache": "bounded 8-entry LRU keyed by field, metric, device, dtype, schedule, divergence, and grad mode; batch is dynamic",
        },
        "graph_break_reports": graph_break_reports(
            cuda_or_cpu, dtype, min(args.batch_size, 8), args.width, min(args.steps, 4)
        ),
        "latency": cold_warm_latency(
            cuda_or_cpu, dtype, args.batch_size, args.width, args.steps
        ),
        "dynamic_batches": dynamic_batches(
            cuda_or_cpu, dtype, args.dynamic_batches, args.width, args.steps
        ),
        "parity": [
            parity(device, dtype, args.batch_size, args.width, args.steps)
            for device in devices
        ],
        "runtime": [
            row
            for device in devices
            for row in runtime_matrix(
                device,
                dtype,
                args.batch_size,
                args.width,
                args.steps,
                args.warmup,
                args.repetitions,
            )
        ],
        "cpu_regression": {
            "meaning": "within-run compiled/eager characterization; compare preserved JSON with the frozen baseline for a release regression decision",
        },
        "unsupported_modes": [
            {"mode": "multichart_compiled", "status": "eager-only"},
            {"mode": "intrinsic_adjoint_compiled", "status": "eager-only"},
        ],
    }
    cpu_rows = [row for row in result["runtime"] if row["device"] == "cpu"]
    result["cpu_regression"]["rows"] = cpu_rows
    if torch.cuda.is_available():
        clear_compilation_cache()
        profile_field = make_field(torch.device("cuda"), dtype, args.width)
        profile_data = torch.randn(
            args.batch_size, 2, device="cuda", dtype=dtype, requires_grad=True
        )
        profile_direct = direct_work(profile_field, profile_data, args.steps, True)
        profile_direct()
        synchronize(torch.device("cuda"))
        result["profiler"] = profile_work(profile_direct, torch.device("cuda"), args.steps)
        result["non_default_stream"] = non_default_stream(
            dtype, args.batch_size, args.width, args.steps
        )
    else:
        result["profiler"] = {"status": "not_run", "reason": "CUDA unavailable"}
        result["non_default_stream"] = {"status": "not_run", "reason": "CUDA unavailable"}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
