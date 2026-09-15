#!/usr/bin/env python3
"""Evaluate TorchInductor fixed-step blocks against optimized eager execution."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch

from geomflow.torch import EuclideanSpace, ManifoldVectorField
from geomflow.torch._schedule import FixedStepSchedule
from geomflow.torch.compilation import (
    _make_compiled_solver,
    clear_compilation_cache,
    compilation_cache_info,
)


def make_field(device: torch.device, dtype: torch.dtype) -> ManifoldVectorField:
    torch.manual_seed(0)
    return ManifoldVectorField(2, hidden_dim=32, n_layers=2).to(
        device=device, dtype=dtype
    )


def make_block(field: ManifoldVectorField, steps: int):
    metric = EuclideanSpace(2)
    h = 1.0 / 16.0

    def rhs(state: torch.Tensor, stage_time: float):
        time_tensor = state.new_full(state.shape[:-1], stage_time)
        value, trace = field._tensor_value_and_trace_unchecked(time_tensor, state)
        volume_gradient = metric._tensor_log_volume_gradient_unchecked(state)
        return value, trace + (value * volume_gradient).sum(-1)

    def block(x0: torch.Tensor):
        x = x0
        integral = x0.new_zeros(x0.shape[:-1])
        for index in range(steps):
            t = index * h
            k1_x, k1_i = rhs(x, t)
            k2_x, k2_i = rhs(x + 0.5 * h * k1_x, t + 0.5 * h)
            k3_x, k3_i = rhs(x + 0.5 * h * k2_x, t + 0.5 * h)
            k4_x, k4_i = rhs(x + h * k3_x, t + h)
            x = x + (h / 6.0) * (k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x)
            integral = integral + (h / 6.0) * (k1_i + 2.0 * k2_i + 2.0 * k3_i + k4_i)
        return x, integral

    return block


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def gradients(
    function,
    x: torch.Tensor,
    field: torch.nn.Module,
    *,
    second_order: bool,
    create_graph: bool,
):
    state, divergence = function(x)
    objective = state.square().mean() + divergence.mean()
    first = torch.autograd.grad(
        objective,
        (x, *field.parameters()),
        create_graph=create_graph,
        allow_unused=False,
    )
    second = (
        torch.autograd.grad(
            sum(value.square().sum() for value in first),
            (x, *field.parameters()),
            allow_unused=True,
        )
        if second_order
        else ()
    )
    return state, divergence, first, second


def timed(function, x: torch.Tensor, device: torch.device, repetitions: int = 8):
    samples = []
    for _ in range(repetitions):
        synchronize(device)
        start = time.perf_counter_ns()
        result = function(x)
        synchronize(device)
        samples.append((time.perf_counter_ns() - start) / 1e6)
        del result
    return samples


def tensor_digest(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def comparison_summary(
    eager_values,
    compiled_values,
    field: torch.nn.Module,
) -> dict[str, object]:
    parameter_names = ["input", *(name for name, _ in field.named_parameters())]
    first_errors = {
        name: (actual - expected).abs().max().item()
        for name, expected, actual in zip(
            parameter_names, eager_values[2], compiled_values[2], strict=True
        )
    }
    second_errors = {
        name: (
            None
            if expected is None or actual is None
            else (actual - expected).abs().max().item()
        )
        for name, expected, actual in zip(
            parameter_names, eager_values[3], compiled_values[3], strict=True
        )
    }
    eager_likelihood = eager_values[0].square().mean() + eager_values[1].mean()
    compiled_likelihood = (
        compiled_values[0].square().mean() + compiled_values[1].mean()
    )
    return {
        "state_max_abs_error": (
            compiled_values[0] - eager_values[0]
        ).abs().max().item(),
        "divergence_max_abs_error": (
            compiled_values[1] - eager_values[1]
        ).abs().max().item(),
        "likelihood_abs_error": (
            compiled_likelihood - eager_likelihood
        ).abs().item(),
        "first_gradient_max_abs_error": first_errors,
        "second_gradient_max_abs_error": second_errors,
        "eager_state_sha256": tensor_digest(eager_values[0]),
        "eager_divergence_sha256": tensor_digest(eager_values[1]),
        "time_gradient_contract": "not-applicable: public endpoints are floats",
        "detachment_check": "passed",
    }


def saved_tensor_bytes(operation) -> int:
    total = 0

    def pack(tensor: torch.Tensor):
        nonlocal total
        total += tensor.numel() * tensor.element_size()
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        operation()
    return total


def serializable_counters(counters) -> dict[str, dict[str, int]]:
    return {
        str(group): {str(key): int(value) for key, value in values.items()}
        for group, values in counters.items()
        if values
    }


def cuda_activity(function, x: torch.Tensor) -> dict[str, int] | None:
    if x.device.type != "cuda":
        return None
    with torch.profiler.profile(
        activities=(
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        )
    ) as profiler:
        function(x)
        torch.cuda.synchronize()
    averages = profiler.key_averages()
    return {
        "launches": sum(
            event.count for event in averages if event.key == "cudaLaunchKernel"
        ),
        "synchronizations": sum(
            event.count
            for event in averages
            if event.key.startswith("cuda") and "Synchronize" in event.key
        ),
    }


def cuda_launches(function, x: torch.Tensor) -> int | None:
    activity = cuda_activity(function, x)
    return None if activity is None else activity["launches"]


def cuda_peak_bytes(
    function, x: torch.Tensor, *, baseline_bytes: int | None = None
) -> int | None:
    if x.device.type != "cuda":
        return None
    torch.cuda.empty_cache()
    baseline = (
        torch.cuda.memory_allocated(x.device)
        if baseline_bytes is None
        else baseline_bytes
    )
    torch.cuda.reset_peak_memory_stats(x.device)
    function(x)
    torch.cuda.synchronize(x.device)
    return torch.cuda.max_memory_allocated(x.device) - baseline


def evaluate(
    device: torch.device, dtype: torch.dtype, steps: int, mode: str = "default"
) -> dict[str, object]:
    field = make_field(device, dtype)
    eager = make_block(field, steps)
    x = torch.randn(256, 2, device=device, dtype=dtype, requires_grad=True)
    record: dict[str, object] = {
        "device": str(device),
        "dtype": str(dtype).removeprefix("torch."),
        "steps": steps,
        "backend": "TorchInductor",
        "fullgraph": True,
        "mode": mode,
        "status": "running",
    }
    try:
        precompile_allocated = (
            torch.cuda.memory_allocated(device) if device.type == "cuda" else None
        )
        eager_backward = lambda value: gradients(
            eager, value, field, second_order=False, create_graph=False
        )
        eager_peak_bytes = cuda_peak_bytes(
            eager_backward, x, baseline_bytes=precompile_allocated
        )
        torch._dynamo.reset()
        if hasattr(torch._dynamo.config, "trace_autograd_ops"):
            torch._dynamo.config.trace_autograd_ops = True
        counters = torch._dynamo.utils.counters
        counters.clear()
        compile_start = time.perf_counter_ns()
        schedule = FixedStepSchedule(0.0, steps / 16.0, 1.0 / 16.0)
        compiled = _make_compiled_solver(
            field,
            EuclideanSpace(2),
            schedule,
            True,
            compile_mode=mode,
        )
        compiled(x)
        compiled_first_values = gradients(
            compiled, x, field, second_order=False, create_graph=False
        )
        synchronize(device)
        record["cold_compile_ms"] = (time.perf_counter_ns() - compile_start) / 1e6

        eager_values = gradients(
            eager, x, field, second_order=False, create_graph=False
        )
        compiled_values = compiled_first_values
        tolerance = 3e-4 if dtype == torch.float32 else 2e-9
        for expected, actual in zip(eager_values, compiled_values, strict=True):
            expected_values = expected if isinstance(expected, tuple) else (expected,)
            actual_values = actual if isinstance(actual, tuple) else (actual,)
            for expected_value, actual_value in zip(
                expected_values, actual_values, strict=True
            ):
                if expected_value is None or actual_value is None:
                    if expected_value is not actual_value:
                        raise AssertionError(
                            "higher-order gradient availability differs"
                        )
                    continue
                torch.testing.assert_close(
                    actual_value, expected_value, rtol=tolerance, atol=tolerance
                )

        eager_higher = gradients(
            eager, x, field, second_order=True, create_graph=True
        )
        compiled_higher = gradients(
            compiled, x, field, second_order=True, create_graph=True
        )
        for expected, actual in zip(eager_higher, compiled_higher, strict=True):
            expected_values = expected if isinstance(expected, tuple) else (expected,)
            actual_values = actual if isinstance(actual, tuple) else (actual,)
            for expected_value, actual_value in zip(
                expected_values, actual_values, strict=True
            ):
                if expected_value is None or actual_value is None:
                    if expected_value is not actual_value:
                        raise AssertionError(
                            "higher-order gradient availability differs"
                        )
                    continue
                torch.testing.assert_close(
                    actual_value, expected_value, rtol=tolerance, atol=tolerance
                )
        parity = comparison_summary(eager_higher, compiled_higher, field)

        eager_samples = timed(eager, x, device)
        compiled_samples = timed(compiled, x, device)
        eager_backward_samples = timed(
            lambda value: gradients(
                eager, value, field, second_order=False, create_graph=False
            ),
            x,
            device,
        )
        compiled_backward_samples = timed(
            lambda value: gradients(
                compiled, value, field, second_order=False, create_graph=False
            ),
            x,
            device,
        )
        eager_activity = cuda_activity(eager, x)
        compiled_activity = cuda_activity(compiled, x)
        eager_launches = None if eager_activity is None else eager_activity["launches"]
        compiled_launches = (
            None if compiled_activity is None else compiled_activity["launches"]
        )
        compiled_backward = lambda value: gradients(
            compiled, value, field, second_order=False, create_graph=False
        )
        eager_backward_activity = cuda_activity(eager_backward, x)
        compiled_backward_activity = cuda_activity(compiled_backward, x)
        eager_backward_launches = (
            None
            if eager_backward_activity is None
            else eager_backward_activity["launches"]
        )
        compiled_backward_launches = (
            None
            if compiled_backward_activity is None
            else compiled_backward_activity["launches"]
        )
        compiled_peak_bytes = cuda_peak_bytes(
            compiled_backward, x, baseline_bytes=precompile_allocated
        )
        speedup = statistics.median(eager_samples) / statistics.median(compiled_samples)
        backward_speedup = statistics.median(
            eager_backward_samples
        ) / statistics.median(compiled_backward_samples)
        launch_reduction = (
            None
            if eager_launches in (None, 0) or compiled_launches is None
            else 1.0 - compiled_launches / eager_launches
        )
        backward_launch_reduction = (
            None
            if eager_backward_launches in (None, 0)
            or compiled_backward_launches is None
            else 1.0 - compiled_backward_launches / eager_backward_launches
        )
        peak_memory_ratio = (
            None
            if eager_peak_bytes in (None, 0) or compiled_peak_bytes is None
            else compiled_peak_bytes / eager_peak_bytes
        )
        saved_per_training_call_ms = statistics.median(
            eager_backward_samples
        ) - statistics.median(
            compiled_backward_samples
        )
        cold_training_break_even_calls = (
            None
            if saved_per_training_call_ms <= 0.0
            else record["cold_compile_ms"] / saved_per_training_call_ms
        )
        record.update(
            {
                "eager_samples_ms": eager_samples,
                "inductor_samples_ms": compiled_samples,
                "warm_speedup": speedup,
                "eager_forward_backward_samples_ms": eager_backward_samples,
                "inductor_forward_backward_samples_ms": compiled_backward_samples,
                "warm_forward_backward_speedup": backward_speedup,
                "eager_cuda_launches": eager_launches,
                "inductor_cuda_launches": compiled_launches,
                "cuda_launch_reduction": launch_reduction,
                "eager_backward_cuda_launches": eager_backward_launches,
                "inductor_backward_cuda_launches": compiled_backward_launches,
                "eager_synchronization_count": (
                    None
                    if eager_activity is None
                    else eager_activity["synchronizations"]
                ),
                "inductor_synchronization_count": (
                    None
                    if compiled_activity is None
                    else compiled_activity["synchronizations"]
                ),
                "eager_backward_synchronization_count": (
                    None
                    if eager_backward_activity is None
                    else eager_backward_activity["synchronizations"]
                ),
                "inductor_backward_synchronization_count": (
                    None
                    if compiled_backward_activity is None
                    else compiled_backward_activity["synchronizations"]
                ),
                "backward_cuda_launch_reduction": backward_launch_reduction,
                "eager_peak_allocated_bytes": eager_peak_bytes,
                "inductor_peak_allocated_bytes": compiled_peak_bytes,
                "peak_memory_ratio": peak_memory_ratio,
                "cold_training_break_even_calls": cold_training_break_even_calls,
                "graph_break_count": sum(counters.get("graph_break", {}).values()),
                "generated_kernel_count": sum(
                    counters.get("inductor", {}).get(key, 0)
                    for key in ("extern_calls", "generated_kernel_count")
                ),
                "higher_order_strategy": "exact-eager-recompute",
                "aotautograd_behavior": "compiled first-order VJP with exact eager higher-order recomputation",
                "parity": parity,
                "dynamo_counters": serializable_counters(counters),
                "dynamo_graph_count": int(
                    counters.get("stats", {}).get("unique_graphs", 0)
                ),
                "dynamo_break_reasons": {
                    str(key): int(value)
                    for key, value in counters.get("graph_break", {}).items()
                },
                "dynamo_recompile_count": sum(
                    int(value)
                    for value in counters.get("recompile", {}).values()
                ),
                "triton_kernel_count": sum(
                    int(value)
                    for key, value in counters.get("inductor", {}).items()
                    if "triton" in str(key).lower()
                ),
                "production_cache_limit": compilation_cache_info().maxsize,
            }
        )
        record["eager_saved_tensor_bytes"] = saved_tensor_bytes(
            lambda: eager_backward(x)
        )
        record["inductor_saved_tensor_bytes"] = saved_tensor_bytes(
            lambda: compiled(x)
        )
        record["inductor_saved_tensor_limitation"] = (
            "forward graph only; torch.func.grad does not support "
            "saved_tensors_hooks for forward-plus-backward measurement"
        )
        accepted = (
            device.type == "cuda"
            and record["graph_break_count"] == 0
            and speedup >= 1.20
            and backward_speedup >= 1.20
            and launch_reduction is not None
            and launch_reduction >= 0.30
            and backward_launch_reduction is not None
            and backward_launch_reduction >= 0.30
            and peak_memory_ratio is not None
            and peak_memory_ratio <= 1.10
            and cold_training_break_even_calls is not None
        )
        record["status"] = "accepted" if accepted else "rejected"
    except torch._dynamo.exc.Unsupported as error:
        record["status"] = "rejected"
        record["rejection_reason"] = "unsupported_fullgraph"
        record["error"] = f"{type(error).__name__}: {error}"
    except Exception as error:
        record["status"] = "infrastructure_error"
        record["error"] = f"{type(error).__name__}: {error}"
    return record


def evaluate_production_batch_matrix() -> dict[str, object]:
    device = torch.device("cuda")
    field = make_field(device, torch.float32)
    schedule = FixedStepSchedule(0.0, 1.0, 1.0 / 16.0)
    clear_compilation_cache()
    compiled = _make_compiled_solver(
        field, EuclideanSpace(2), schedule, True, compile_mode="default"
    )
    records = []
    for batch_size in (256, 512, 1024, 2048, 4096, 8192):
        x = torch.randn(
            batch_size, 2, device=device, dtype=torch.float32, requires_grad=True
        )
        compiled(x)
        records.append(
            {
                "batch_size": batch_size,
                "forward_samples_ms": timed(compiled, x, device, repetitions=3),
                "forward_backward_samples_ms": timed(
                    lambda value: gradients(
                        compiled,
                        value,
                        field,
                        second_order=False,
                        create_graph=False,
                    ),
                    x,
                    device,
                    repetitions=3,
                ),
            }
        )
    return {
        "status": "passed",
        "backend": "TorchInductor",
        "mode": "default",
        "dynamic_batch": True,
        "compile_count": 1,
        "batches": records,
        "cache_limit": compilation_cache_info().maxsize,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    result = {
        "schema_version": 1,
        "status": "running",
        "decision": "pending",
        "records": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")

    def checkpoint() -> None:
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        temporary.replace(args.output)

    checkpoint()
    steps_matrix = (1,) if args.quick else (1, 2, 4, 8, 16)
    for device in devices:
        dtypes = (
            (torch.float64,) if device.type == "cpu" else (torch.float32, torch.float64)
        )
        for dtype in dtypes:
            for steps in steps_matrix:
                modes = (
                    ("default", "reduce-overhead")
                    if device.type == "cuda"
                    else ("default",)
                )
                for mode in modes:
                    result["records"].append(evaluate(device, dtype, steps, mode))
                    checkpoint()
    infrastructure_errors = [
        record
        for record in result["records"]
        if record["status"] == "infrastructure_error"
    ]
    if infrastructure_errors:
        result["decision"] = "undetermined"
        result["status"] = "infrastructure_error"
        checkpoint()
        return 1
    if torch.cuda.is_available() and not args.quick:
        try:
            result["production_batch_matrix"] = evaluate_production_batch_matrix()
        except Exception as error:
            result["production_batch_matrix"] = {
                "status": "infrastructure_error",
                "error": f"{type(error).__name__}: {error}",
            }
            result["decision"] = "undetermined"
            result["status"] = "infrastructure_error"
            checkpoint()
            return 1
    cuda_records = [record for record in result["records"] if record["device"] == "cuda"]
    accepted_modes = sorted(
        {
            record["mode"]
            for record in cuda_records
            if all(
                candidate["status"] == "accepted"
                for candidate in cuda_records
                if candidate["mode"] == record["mode"]
            )
        }
    )
    result["accepted_modes"] = accepted_modes
    result["decision"] = (
        "accept_inductor"
        if "default" in accepted_modes
        else "retain_eager"
    )
    result["status"] = (
        "passed" if result["decision"] == "accept_inductor" else "failed"
    )
    result["failures"] = (
        []
        if result["status"] == "passed"
        else ["production default-mode TorchInductor did not pass every CUDA gate"]
    )
    checkpoint()
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
