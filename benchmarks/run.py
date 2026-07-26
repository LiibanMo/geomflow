#!/usr/bin/env python3
"""Run reproducible geomflow PyTorch performance baselines."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path
from statistics import median
from typing import Any, Callable

import torch

from geomflow.torch import integrate_multichart, integrate_rk4

from scenarios import BenchmarkCase, SCENARIOS, make_geometry, make_input, make_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", nargs="+", choices=SCENARIOS, default=["euclidean"])
    parser.add_argument("--batch-size", nargs="+", type=int, default=[8])
    parser.add_argument("--dim", nargs="+", type=int, default=[2])
    parser.add_argument("--hidden-width", nargs="+", type=int, default=[32])
    parser.add_argument("--hidden-depth", nargs="+", type=int, default=[2])
    parser.add_argument("--steps", nargs="+", type=int, default=[8])
    parser.add_argument("--dtype", nargs="+", choices=("float32", "float64"), default=["float32"])
    parser.add_argument("--device", nargs="+", choices=("cpu", "cuda"), default=["cpu"])
    parser.add_argument("--divergence", choices=("exact", "none"), default="exact")
    parser.add_argument("--workload", nargs="+", choices=("forward", "backward", "train"), default=["forward"])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reference", action="store_true", help="compare output with CPU float64")
    parser.add_argument("--profile", action="store_true", help="export one Chrome trace per case")
    parser.add_argument("--trace-dir", type=Path, default=Path("benchmarks/traces"))
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/latest.json"))
    parser.add_argument("--fail-on-error", action="store_true")
    return parser.parse_args()


def git_value(*args: str) -> str:
    try:
        return subprocess.check_output(
            ("git", *args), text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def environment_metadata() -> dict[str, Any]:
    gpu = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            gpu.append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_bytes": properties.total_memory,
                    "capability": list(torch.cuda.get_device_capability(index)),
                }
            )
    revision = git_value("rev-parse", "HEAD")
    dirty = bool(git_value("status", "--porcelain"))
    driver = "unavailable"
    if torch.cuda.is_available():
        try:
            driver = subprocess.check_output(
                ("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"),
                text=True,
                stderr=subprocess.DEVNULL,
            ).splitlines()[0]
        except (OSError, subprocess.CalledProcessError, IndexError):
            driver = "unknown"
    return {
        "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
        "git_revision": revision,
        "git_dirty": dirty,
        "command": sys.argv,
        "hostname": socket.gethostname(),
        "os": platform.platform(),
        "machine": platform.machine(),
        "cpu": platform.processor() or "unknown",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "nvidia_driver": driver,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": gpu,
    }


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_call(fn: Callable[[], Any], device: torch.device) -> tuple[Any, float]:
    if device.type == "cuda":
        synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        value = fn()
        end.record()
        end.synchronize()
        return value, float(start.elapsed_time(end))
    start_ns = time.perf_counter_ns()
    value = fn()
    return value, (time.perf_counter_ns() - start_ns) / 1e6


def integrate(case: BenchmarkCase, model, geometry, x: torch.Tensor):
    kwargs = {
        "t0": 0.0,
        "t1": 1.0,
        "dt": 1.0 / case.steps,
        "compute_divergence": case.divergence_mode == "exact",
    }
    if case.scenario == "sphere-atlas":
        return integrate_multichart(model, geometry, x, start_chart=0, **kwargs)
    return integrate_rk4(model, geometry, x, **kwargs)


def scalar_objective(result) -> torch.Tensor:
    return result.x_final.square().mean() + result.divergence_integral.mean()


def one_iteration(case: BenchmarkCase, model, geometry, x, optimizer=None):
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    result, forward_ms = time_call(lambda: integrate(case, model, geometry, x), case.device)
    backward_ms = 0.0
    optimizer_ms = 0.0
    if case.workload in {"backward", "train"}:
        loss = scalar_objective(result)
        _, backward_ms = time_call(loss.backward, case.device)
    if case.workload == "train":
        _, optimizer_ms = time_call(optimizer.step, case.device)
    return result, {
        "forward_ms": forward_ms,
        "backward_ms": backward_ms,
        "optimizer_step_ms": optimizer_ms,
        "wall_ms": forward_ms + backward_ms + optimizer_ms,
    }


def reference_error(case: BenchmarkCase, model, x, actual) -> dict[str, float]:
    reference_case = BenchmarkCase(
        **{
            **case.__dict__,
            "device": torch.device("cpu"),
            "dtype": torch.float64,
            "workload": "forward",
        }
    )
    geometry = make_geometry(reference_case)
    reference_model = make_model(reference_case, geometry)
    reference_model.load_state_dict(
        {name: value.detach().cpu().double() for name, value in model.state_dict().items()}
    )
    reference_x = x.detach().cpu().double()
    reference = integrate(reference_case, reference_model, geometry, reference_x)
    state_error = (actual.x_final.detach().cpu().double() - reference.x_final).abs().max()
    div_error = (
        actual.divergence_integral.detach().cpu().double()
        - reference.divergence_integral
    ).abs().max()
    return {
        "state_max_abs_error": state_error.item(),
        "divergence_max_abs_error": div_error.item(),
        "max_abs_error": torch.maximum(state_error, div_error).item(),
    }


def profile_case(case, model, geometry, x, output: Path) -> dict[str, Any]:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if case.device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    output.parent.mkdir(parents=True, exist_ok=True)
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as profiler:
        one_iteration(case, model, geometry, x)
    profiler.export_chrome_trace(str(output))
    events = profiler.key_averages()
    return {
        "trace": str(output),
        "aten_to_count": sum(event.count for event in events if event.key == "aten::to"),
        "aten_to_copy_count": sum(
            event.count for event in events if event.key == "aten::_to_copy"
        ),
    }


def run_case(case: BenchmarkCase, args: argparse.Namespace) -> dict[str, Any]:
    if case.device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    torch.manual_seed(case.seed)
    geometry = make_geometry(case)
    model = make_model(case, geometry)
    x = make_input(case)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) if case.workload == "train" else None
    for _ in range(args.warmup):
        one_iteration(case, model, geometry, x, optimizer)

    samples = []
    peaks = []
    result = None
    for _ in range(args.repetitions):
        if case.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(case.device)
        result, timing = one_iteration(case, model, geometry, x, optimizer)
        samples.append(timing)
        if case.device.type == "cuda":
            peaks.append(
                {
                    "allocated_bytes": torch.cuda.max_memory_allocated(case.device),
                    "reserved_bytes": torch.cuda.max_memory_reserved(case.device),
                }
            )

    wall_ms = median(sample["wall_ms"] for sample in samples)
    record: dict[str, Any] = {
        "case_id": case.case_id,
        "status": "passed",
        "parameters": {
            **case.__dict__,
            "dtype": str(case.dtype),
            "device": str(case.device),
        },
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "timing_samples": samples,
        "median_wall_ms": wall_ms,
        "samples_per_second": case.batch_size / (wall_ms / 1e3),
        "accepted_steps_per_second": case.steps / (wall_ms / 1e3),
        "memory_samples": peaks,
    }
    if args.reference:
        record["error"] = reference_error(case, model, x, result)
    if args.profile:
        record["profiler"] = profile_case(
            case, model, geometry, x, args.trace_dir / f"{case.case_id}.json"
        )
    return record


def cases(args: argparse.Namespace):
    for scenario in args.scenario:
        for batch_size in args.batch_size:
            for dim in args.dim:
                for width in args.hidden_width:
                    for depth in args.hidden_depth:
                        for steps in args.steps:
                            for dtype_name in args.dtype:
                                for device_name in args.device:
                                    for workload in args.workload:
                                        yield BenchmarkCase(
                                            scenario,
                                            batch_size,
                                            dim,
                                            width,
                                            depth,
                                            steps,
                                            getattr(torch, dtype_name),
                                            torch.device(device_name),
                                            args.divergence,
                                            workload,
                                            args.seed,
                                        )


def print_table(records: list[dict[str, Any]]) -> None:
    print(f"{'case':70} {'wall ms':>10} {'samples/s':>12} {'status':>8}")
    for record in records:
        print(
            f"{record['case_id'][:70]:70} "
            f"{record.get('median_wall_ms', math.nan):10.3f} "
            f"{record.get('samples_per_second', math.nan):12.1f} "
            f"{record['status']:>8}"
        )


def main() -> int:
    args = parse_args()
    if args.warmup < 0 or args.repetitions < 1:
        raise SystemExit("--warmup must be nonnegative and --repetitions positive")
    records = []
    failed = False
    for case in cases(args):
        try:
            records.append(run_case(case, args))
        except Exception as error:
            failed = True
            records.append(
                {
                    "case_id": case.case_id,
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    payload = {"schema_version": 1, "run": environment_metadata(), "cases": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print_table(records)
    print(f"results: {args.output}")
    return 1 if failed and args.fail_on_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
