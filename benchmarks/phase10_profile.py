#!/usr/bin/env python3
"""Profile scoped exact-divergence CUDA solver hot paths."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import torch

from geomflow.torch import (
    Atlas,
    Chart,
    EuclideanSpace,
    Transition,
    integrate_multichart,
    integrate_rk4,
)

from scenarios import BenchmarkCase, make_geometry, make_input, make_model

MAX_SYNCHRONIZATION_DURATION_FRACTION = 0.05
SYNCHRONIZATION_LIMITS = {
    "euclidean": 84,
    "sphere-atlas": 68,
    "sphere-atlas-forced-switch": 68,
}


def transfer_bytes(event: dict[str, object]) -> int | None:
    args = event.get("args")
    if not isinstance(args, dict) or "bytes" not in args:
        return None
    try:
        return int(args["bytes"])
    except (TypeError, ValueError):
        return None


def event_duration_us(event: dict[str, object]) -> float | None:
    duration = event.get("dur")
    if isinstance(duration, bool) or not isinstance(duration, (int, float)):
        return None
    duration = float(duration)
    return duration if duration >= 0.0 and math.isfinite(duration) else None


def event_in_scope(event: dict[str, object], start: float, end: float) -> bool:
    try:
        timestamp = float(event["ts"])
    except (KeyError, TypeError, ValueError):
        return False
    duration = event_duration_us(event) or 0.0
    return (
        math.isfinite(timestamp) and timestamp <= end and timestamp + duration >= start
    )


def is_cuda_synchronization(name: str) -> bool:
    normalized = name.lower()
    return normalized.startswith("cu") and "synchroniz" in normalized


def profiler_summary(
    profiler,
    trace_path: Path,
    *,
    scope_name: str,
    end_to_end_us: float,
) -> dict[str, object]:
    profiler.export_chrome_trace(str(trace_path))
    averages = profiler.key_averages()
    counts: dict[str, int] = {}
    for event in averages:
        counts[event.key] = counts.get(event.key, 0) + event.count
    trace = json.loads(trace_path.read_text())
    if isinstance(trace, dict):
        events = trace.get("traceEvents", [])
    elif isinstance(trace, list):
        events = trace
    else:
        raise ValueError("profiler trace must be a JSON object or array")
    if not isinstance(events, list):
        raise ValueError("profiler traceEvents must be an array")

    scopes = [
        event
        for event in events
        if isinstance(event, dict) and event.get("name") == scope_name
    ]
    if not scopes:
        raise ValueError(f"profiler trace is missing scope {scope_name!r}")
    try:
        scope_ranges = [(float(event["ts"]), float(event["dur"])) for event in scopes]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"profiler scope {scope_name!r} has invalid timing") from error
    if any(
        not math.isfinite(start) or not math.isfinite(duration) or duration < 0.0
        for start, duration in scope_ranges
    ):
        raise ValueError(f"profiler scope {scope_name!r} has invalid timing")
    scope_start = min(start for start, _duration in scope_ranges)
    scope_end = max(start + duration for start, duration in scope_ranges)

    copies = []
    launch_count = 0
    synchronizations = []
    scoped_to_count = 0
    scoped_to_copy_count = 0
    allocation_event_count = 0
    graph_break_count = 0
    autograd_engine_call_count = 0
    operator_counts: dict[str, int] = {}
    kernel_ranges = []
    for event in events:
        if not isinstance(event, dict):
            continue
        if not event_in_scope(event, scope_start, scope_end):
            continue
        name = str(event.get("name", ""))
        category = str(event.get("cat", ""))
        if name.startswith("aten::"):
            operator_counts[name] = operator_counts.get(name, 0) + 1
        if "autograd::engine::evaluate_function" in name:
            autograd_engine_call_count += 1
        if category == "kernel":
            duration = event_duration_us(event)
            if duration is not None:
                kernel_ranges.append((float(event["ts"]), duration))
        if name == "cudaLaunchKernel":
            launch_count += 1
        if name == "aten::to":
            scoped_to_count += 1
        if name == "aten::_to_copy":
            scoped_to_copy_count += 1
        if name == "[memory]":
            allocation_event_count += 1
        if "graph break" in name.lower() or "graph_break" in name.lower():
            graph_break_count += 1
        if is_cuda_synchronization(name):
            synchronizations.append(
                {"name": name, "duration_us": event_duration_us(event)}
            )
        if "Memcpy DtoH" in name or "Memcpy HtoD" in name:
            copies.append({"name": name, "bytes": transfer_bytes(event)})
    unknown_copies = [copy for copy in copies if copy["bytes"] is None]
    non_scalar_copies = [
        copy for copy in copies if copy["bytes"] is not None and copy["bytes"] > 8
    ]
    scalar_copies = [
        copy for copy in copies if copy["bytes"] is not None and copy["bytes"] <= 8
    ]
    unknown_synchronization_durations = [
        event for event in synchronizations if event["duration_us"] is None
    ]
    synchronization_duration_us = sum(
        event["duration_us"]
        for event in synchronizations
        if event["duration_us"] is not None
    )
    if not math.isfinite(end_to_end_us) or end_to_end_us <= 0.0:
        raise ValueError("end-to-end profile duration must be positive")
    kernel_ranges.sort()
    kernel_gaps_us = [
        max(0.0, start - (previous_start + previous_duration))
        for (previous_start, previous_duration), (start, _duration) in zip(
            kernel_ranges, kernel_ranges[1:]
        )
    ]
    return {
        "linear_count": counts.get("aten::linear", 0),
        "cuda_launch_count": launch_count,
        "synchronization_count": len(synchronizations),
        "synchronizations": synchronizations,
        "synchronization_duration_us": synchronization_duration_us,
        "synchronization_duration_fraction": synchronization_duration_us
        / end_to_end_us,
        "unknown_synchronization_durations": unknown_synchronization_durations,
        "end_to_end_us": end_to_end_us,
        "copies": copies,
        "scalar_copy_count": len(scalar_copies),
        "scalar_copy_bytes": sum(copy["bytes"] for copy in scalar_copies),
        "non_scalar_copies": non_scalar_copies,
        "materializing_full_transfer_bytes": sum(
            copy["bytes"] for copy in non_scalar_copies
        ),
        "unknown_copies": unknown_copies,
        "aten_to_count": scoped_to_count,
        "aten_to_noop_count": scoped_to_count - scoped_to_copy_count,
        "aten_to_copy_count": scoped_to_copy_count,
        "allocation_event_count": allocation_event_count,
        "graph_break_count": graph_break_count,
        "autograd_engine_call_count": autograd_engine_call_count,
        "operator_counts": operator_counts,
        "cuda_kernel_count": len(kernel_ranges),
        "cuda_kernel_gap_count": len(kernel_gaps_us),
        "cuda_kernel_gap_total_us": sum(kernel_gaps_us),
        "cuda_kernel_gap_max_us": max(kernel_gaps_us, default=0.0),
    }


def profiler_failures(
    summary: dict[str, object],
    *,
    expected_linear_count: int | None,
    synchronization_limit: int,
) -> list[str]:
    failures = []
    if (
        expected_linear_count is not None
        and summary["linear_count"] != expected_linear_count
    ):
        failures.append(
            f"expected {expected_linear_count} linear calls, "
            f"observed {summary['linear_count']}"
        )
    if summary["aten_to_copy_count"]:
        failures.append(
            f"observed {summary['aten_to_copy_count']} materializing _to_copy calls"
        )
    if summary["non_scalar_copies"]:
        failures.append(
            f"observed non-scalar host transfers: {summary['non_scalar_copies']}"
        )
    if summary["unknown_copies"]:
        failures.append(
            f"host transfers had unknown byte counts: {summary['unknown_copies']}"
        )
    if summary["graph_break_count"]:
        failures.append(f"observed {summary['graph_break_count']} graph breaks")
    if summary["unknown_synchronization_durations"]:
        failures.append(
            "CUDA synchronizations had unknown durations: "
            f"{summary['unknown_synchronization_durations']}"
        )
    if summary["synchronization_count"] > synchronization_limit:
        failures.append(
            f"CUDA synchronization count {summary['synchronization_count']} > "
            f"{synchronization_limit}"
        )
    if (
        summary["synchronization_duration_fraction"]
        > MAX_SYNCHRONIZATION_DURATION_FRACTION
    ):
        failures.append(
            "CUDA synchronization duration fraction "
            f"{summary['synchronization_duration_fraction']:.3f} > "
            f"{MAX_SYNCHRONIZATION_DURATION_FRACTION:.3f}"
        )
    return failures


def timed_cuda_operation(operation):
    torch.cuda.synchronize()
    start = time.perf_counter_ns()
    result = operation()
    torch.cuda.synchronize()
    return result, (time.perf_counter_ns() - start) / 1e3


def saved_tensor_summary(operation) -> dict[str, int]:
    count = 0
    total_bytes = 0

    def pack(tensor: torch.Tensor):
        nonlocal count, total_bytes
        count += 1
        total_bytes += tensor.numel() * tensor.element_size()
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        result = operation()
    del result
    return {"saved_tensor_count": count, "saved_tensor_bytes": total_bytes}


def run_case(
    scenario: str,
    workload: str,
    batch_size: int,
    trace_path: Path,
    expected_backend: str | None = None,
    force_eager: bool = False,
) -> dict[str, object]:
    case = BenchmarkCase(
        scenario=scenario,
        batch_size=batch_size,
        dim=2,
        hidden_width=32,
        hidden_depth=2,
        steps=16,
        dtype=torch.float32,
        device=torch.device("cuda"),
        divergence_mode="exact",
        workload=workload,
        seed=0,
    )
    torch.manual_seed(case.seed)
    geometry = make_geometry(case)
    model = make_model(case, geometry)
    x = make_input(case)
    kwargs = {
        "t0": 0.0,
        "t1": 1.0,
        "dt": 1.0 / case.steps,
        "compile": False if force_eager else None,
    }

    def operation():
        if scenario == "sphere-atlas":
            result = integrate_multichart(model, geometry, x, start_chart=0, **kwargs)
        else:
            result = integrate_rk4(model, geometry, x, **kwargs)
        if workload == "backward":
            loss = result.x_final.square().mean() + result.divergence_integral.mean()
            loss.backward()
        return result

    model.zero_grad(set_to_none=True)
    operation()
    torch.cuda.synchronize()
    model.zero_grad(set_to_none=True)
    timed_result, end_to_end_us = timed_cuda_operation(operation)
    assert torch.isfinite(timed_result.x_final).all()
    assert torch.isfinite(timed_result.divergence_integral).all()
    del timed_result
    model.zero_grad(set_to_none=True)
    saved_tensors = saved_tensor_summary(operation)
    torch.cuda.synchronize()
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    with torch.profiler.profile(
        activities=(
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as profiler:
        model.zero_grad(set_to_none=True)
        with torch.profiler.record_function("geomflow_solver"):
            result = operation()
            torch.cuda.synchronize()
    peak_allocated_bytes = torch.cuda.max_memory_allocated()
    assert torch.isfinite(result.x_final).all()
    assert torch.isfinite(result.divergence_integral).all()
    summary = profiler_summary(
        profiler,
        trace_path,
        scope_name="geomflow_solver",
        end_to_end_us=end_to_end_us,
    )
    failures = profiler_failures(
        summary,
        expected_linear_count=None,
        synchronization_limit=SYNCHRONIZATION_LIMITS[scenario],
    )
    backend = getattr(result, "_execution_backend", "component-gradient-eager")
    fallback_reason = getattr(result, "_fallback_reason", None)
    if expected_backend is not None and backend != expected_backend:
        failures.append(f"expected backend {expected_backend}, observed {backend}")
    if fallback_reason is not None:
        failures.append(f"solver used fallback: {fallback_reason}")
    return {
        "scenario": scenario,
        "workload": workload,
        "batch_size": batch_size,
        "trace": str(trace_path),
        "rk_step_count": case.steps,
        "rk_stage_count": case.steps * 4,
        "functional_transform_attempt_count": 0,
        "functional_transform_fallback_count": 0,
        "exact_divergence_strategy": (
            "compiled-tensor-value-and-trace"
            if backend == "inductor"
            else (
                "tensor-value-and-trace"
                if backend == "tensor-eager"
                else "component-gradient-with-connected-value"
            )
        ),
        "backend": backend,
        "fallback_reason": fallback_reason,
        "execution_variant": "tensor-eager-oracle" if force_eager else "production",
        "peak_allocated_bytes": peak_allocated_bytes,
        **saved_tensors,
        **summary,
        "failures": failures,
    }


def run_switch_case(trace_path: Path) -> dict[str, object]:
    limit = 0.11
    finite = lambda x: torch.isfinite(x).all(dim=-1)
    overlap = lambda x: x[..., 0] <= limit
    chart0 = Chart(
        0,
        1,
        None,
        EuclideanSpace(1),
        transitions={1: Transition(lambda x: x, overlap)},
        domain=overlap,
    )
    chart1 = Chart(1, 1, None, EuclideanSpace(1), domain=finite)
    atlas = Atlas([chart0, chart1], 0)

    class SwitchField(torch.nn.Module):
        def forward(self, time, state, chart_id):
            del time, chart_id
            return 0.5 * state

    field = SwitchField().cuda()
    x = torch.full((256, 1), 0.1, device="cuda")

    def operation():
        return integrate_multichart(
            field, atlas, x, 0, 0.0, 0.4, 0.2, max_subdivisions=12
        )

    warm = operation()
    if len(warm.transition_events) != 1:
        raise AssertionError("forced-switch profile did not record one transition")
    torch.cuda.synchronize()
    timed_result, end_to_end_us = timed_cuda_operation(operation)
    if len(timed_result.transition_events) != 1:
        raise AssertionError("timed forced-switch run did not record one transition")
    del timed_result
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.profiler.profile(
        activities=(
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as profiler:
        with torch.profiler.record_function("geomflow_solver_switch"):
            result = operation()
            torch.cuda.synchronize()
    summary = profiler_summary(
        profiler,
        trace_path,
        scope_name="geomflow_solver_switch",
        end_to_end_us=end_to_end_us,
    )
    failures = profiler_failures(
        summary,
        expected_linear_count=None,
        synchronization_limit=SYNCHRONIZATION_LIMITS["sphere-atlas-forced-switch"],
    )
    if len(result.transition_events) != 1:
        failures.append("forced-switch run did not preserve one transition")
    return {
        "scenario": "sphere-atlas-forced-switch",
        "trace": str(trace_path),
        "transition_count": len(result.transition_events),
        **summary,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--paired", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 10 profiling requires CUDA")

    result = {"schema_version": 2, "status": "running", "records": [], "failures": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")

    def checkpoint() -> None:
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        temporary.replace(args.output)

    checkpoint()
    if args.paired is None:
        profile_cases = [
            {
                "scenario": scenario,
                "workload": workload,
                "batch_size": 256,
                "expected_backend": None,
            }
            for scenario in ("euclidean", "sphere-atlas")
            for workload in ("forward", "backward")
        ]
    else:
        paired = json.loads(args.paired.read_text())
        profile_cases = []
        for gate in paired.get("speed_gates", {}).values():
            backends = gate.get("backends", [])
            if len(backends) != 1 or not isinstance(backends[0], str):
                raise ValueError("each selected speed gate must identify one backend")
            profile_cases.append({**gate["case"], "expected_backend": backends[0]})
        if len(profile_cases) != 4:
            failure = (
                "paired evidence must contain four selected speed gates; "
                f"found {len(profile_cases)}"
            )
            result["status"] = "failed"
            result["failures"] = [failure]
            checkpoint()
            raise ValueError(failure)

    records = result["records"]
    for case in profile_cases:
        scenario = case["scenario"]
        workload = case["workload"]
        batch_size = int(case["batch_size"])
        records.append(
            run_case(
                scenario,
                workload,
                batch_size,
                args.trace_dir
                / f"ci-vast-profile-{scenario}-{workload}-b{batch_size}.json",
                case["expected_backend"],
            )
        )
        checkpoint()
        records.append(
            run_case(
                scenario,
                workload,
                batch_size,
                args.trace_dir
                / f"ci-vast-profile-{scenario}-{workload}-b{batch_size}-eager.json",
                "tensor-eager",
                force_eager=True,
            )
        )
        checkpoint()
    records.append(
        run_switch_case(args.trace_dir / "ci-vast-profile-forced-switch.json")
    )
    checkpoint()
    failures = [failure for record in records for failure in record["failures"]]
    result["status"] = "failed" if failures else "passed"
    result["failures"] = failures
    checkpoint()
    if failures:
        raise AssertionError("; ".join(failures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
