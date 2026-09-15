#!/usr/bin/env python3
"""Compare exact built-in divergence strategies in complete CUDA solves."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import torch

from geomflow.torch import ManifoldVectorField


STRATEGIES = (
    "component-vjp",
    "batched-vjp",
    "vmap-jacrev",
    "chunked-jacrev",
    "jacfwd",
    "jvp-trace",
    "explicit-tangent",
)


def make_field() -> ManifoldVectorField:
    torch.manual_seed(0)
    return ManifoldVectorField(2, hidden_dim=32, n_layers=2).cuda()


def value_and_trace(
    field: ManifoldVectorField,
    time_tensor: torch.Tensor,
    state: torch.Tensor,
    strategy: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if strategy == "explicit-tangent":
        return field._tensor_value_and_trace_unchecked(time_tensor, state)

    value = field._forward_unchecked(time_tensor, state)
    if strategy == "component-vjp":
        rows = [
            torch.autograd.grad(
                value[..., index].sum(),
                state,
                create_graph=True,
                retain_graph=True,
            )[0][..., index]
            for index in range(field.dim)
        ]
        return value, torch.stack(rows, dim=-1).sum(-1)

    if strategy == "batched-vjp":
        basis = value.new_zeros(field.dim, *value.shape)
        for index in range(field.dim):
            basis[index, ..., index] = 1.0
        jacobian_rows = torch.autograd.grad(
            value,
            state,
            grad_outputs=basis,
            is_grads_batched=True,
            create_graph=True,
        )[0]
        trace = sum(
            jacobian_rows[index, ..., index] for index in range(field.dim)
        )
        return value, trace

    def point_value(point: torch.Tensor, point_time: torch.Tensor) -> torch.Tensor:
        return field._forward_unchecked(point_time, point)

    if strategy in {"vmap-jacrev", "chunked-jacrev"}:
        jacobian_fn = torch.func.vmap(torch.func.jacrev(point_value, argnums=0))
        if strategy == "chunked-jacrev":
            chunk_size = min(1024, state.shape[0])
            jacobian = torch.cat(
                [
                    jacobian_fn(
                        state[start : start + chunk_size],
                        time_tensor[start : start + chunk_size],
                    )
                    for start in range(0, state.shape[0], chunk_size)
                ],
                dim=0,
            )
        else:
            jacobian = jacobian_fn(state, time_tensor)
        return value, jacobian.diagonal(dim1=-2, dim2=-1).sum(-1)

    if strategy == "jacfwd":
        jacobian = torch.func.vmap(torch.func.jacfwd(point_value, argnums=0))(
            state, time_tensor
        )
        return value, jacobian.diagonal(dim1=-2, dim2=-1).sum(-1)

    if strategy == "jvp-trace":
        trace = state.new_zeros(state.shape[:-1])
        for index in range(field.dim):
            tangent = torch.zeros_like(state)
            tangent[..., index] = 1.0
            _, directional = torch.func.jvp(
                lambda current: field._forward_unchecked(time_tensor, current),
                (state,),
                (tangent,),
            )
            trace = trace + directional[..., index]
        return value, trace

    raise ValueError(f"unknown derivative strategy {strategy!r}")


def solve(
    field: ManifoldVectorField, x0: torch.Tensor, strategy: str
) -> tuple[torch.Tensor, torch.Tensor]:
    x = x0
    integral = x0.new_zeros(x0.shape[:-1])
    step_size = 1.0 / 16.0

    def rhs(state: torch.Tensor, stage_time: float):
        time_tensor = state.new_full(state.shape[:-1], stage_time)
        return value_and_trace(field, time_tensor, state, strategy)

    for index in range(16):
        start = index * step_size
        half = step_size / 2.0
        k1_x, k1_i = rhs(x, start)
        k2_x, k2_i = rhs(x + half * k1_x, start + half)
        k3_x, k3_i = rhs(x + half * k2_x, start + half)
        k4_x, k4_i = rhs(x + step_size * k3_x, start + step_size)
        x = x + (step_size / 6.0) * (k1_x + 2 * k2_x + 2 * k3_x + k4_x)
        integral = integral + (step_size / 6.0) * (
            k1_i + 2 * k2_i + 2 * k3_i + k4_i
        )
    return x, integral


def objective(result: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    state, integral = result
    return state.square().mean() + integral.mean()


def timed(operation, repetitions: int = 3) -> list[float]:
    samples = []
    for _ in range(repetitions):
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        result = operation()
        torch.cuda.synchronize()
        samples.append((time.perf_counter_ns() - start) / 1e6)
        del result
    return samples


def operation_counts(operation) -> dict[str, int]:
    with torch.profiler.profile(
        activities=(
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        )
    ) as profiler:
        operation()
        torch.cuda.synchronize()
    averages = profiler.key_averages()
    return {
        "cuda_launches": sum(
            event.count for event in averages if event.key == "cudaLaunchKernel"
        ),
        "autograd_engine_calls": sum(
            event.count
            for event in averages
            if event.key.startswith("autograd::engine::evaluate_function")
        ),
    }


def saved_tensor_bytes(operation) -> int:
    saved = 0

    def pack(tensor: torch.Tensor):
        nonlocal saved
        saved += tensor.numel() * tensor.element_size()
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        operation()
    return saved


def measure(
    field: ManifoldVectorField,
    batch_size: int,
    strategy: str,
) -> dict[str, object]:
    generator = torch.Generator(device="cpu").manual_seed(0)
    base = torch.randn(batch_size, 2, generator=generator).cuda()

    def forward():
        return solve(field, base.detach().requires_grad_(True), strategy)

    def forward_backward():
        field.zero_grad(set_to_none=True)
        x = base.detach().requires_grad_(True)
        result = solve(field, x, strategy)
        objective(result).backward()
        return result

    forward_backward()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    backward_samples = timed(forward_backward)
    peak_bytes = torch.cuda.max_memory_allocated()
    saved_bytes = None
    saved_tensor_limitation = None
    try:
        saved_bytes = saved_tensor_bytes(forward_backward)
    except RuntimeError as error:
        if "don't yet support saved tensor hooks" not in str(error):
            raise
        saved_tensor_limitation = str(error)
    forward_counts = operation_counts(forward)
    forward_backward_counts = operation_counts(forward_backward)
    return {
        "batch_size": batch_size,
        "forward_samples_ms": timed(forward),
        "forward_backward_samples_ms": backward_samples,
        "forward_cuda_launches": forward_counts["cuda_launches"],
        "forward_backward_cuda_launches": forward_backward_counts["cuda_launches"],
        "forward_autograd_engine_calls": forward_counts["autograd_engine_calls"],
        "forward_backward_autograd_engine_calls": forward_backward_counts[
            "autograd_engine_calls"
        ],
        "saved_tensor_bytes": saved_bytes,
        "saved_tensor_limitation": saved_tensor_limitation,
        "peak_allocated_bytes": peak_bytes,
    }


def correctness(field: ManifoldVectorField, strategy: str) -> dict[str, float]:
    torch.manual_seed(7)
    reference_x = torch.randn(16, 2, device="cuda", requires_grad=True)
    actual_x = reference_x.detach().clone().requires_grad_(True)
    reference = solve(field, reference_x, "component-vjp")
    actual = solve(field, actual_x, strategy)
    reference_gradients = torch.autograd.grad(
        objective(reference), (reference_x, *field.parameters()), retain_graph=True
    )
    actual_gradients = torch.autograd.grad(
        objective(actual), (actual_x, *field.parameters()), retain_graph=True
    )
    return {
        "state_max_abs_error": (actual[0] - reference[0]).abs().max().item(),
        "integral_max_abs_error": (actual[1] - reference[1]).abs().max().item(),
        "gradient_max_abs_error": max(
            (left - right).abs().max().item()
            for left, right in zip(actual_gradients, reference_gradients, strict=True)
        ),
    }


def complete_solve_score(record: dict[str, object]) -> float:
    return statistics.geometric_mean(
        sample
        for measurement in record["measurements"]
        for sample in (
            statistics.median(measurement["forward_samples_ms"]),
            statistics.median(measurement["forward_backward_samples_ms"]),
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("derivative candidate comparison requires CUDA")

    batches = (256,) if args.quick else (256, 512, 1024, 2048, 4096, 8192)
    result = {
        "schema_version": 1,
        "status": "running",
        "selected_strategy": None,
        "records": [],
        "failures": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")

    def checkpoint() -> None:
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        temporary.replace(args.output)

    checkpoint()
    field = make_field()
    for strategy in STRATEGIES:
        record: dict[str, object] = {"strategy": strategy, "status": "running"}
        result["records"].append(record)
        checkpoint()
        try:
            record["correctness"] = correctness(field, strategy)
            record["measurements"] = []
            for batch_size in batches:
                record["measurements"].append(
                    measure(field, batch_size, strategy)
                )
                checkpoint()
            tolerance = 5e-4
            record["status"] = (
                "passed"
                if max(record["correctness"].values()) <= tolerance
                else "failed"
            )
        except Exception as error:
            record["status"] = "unsupported"
            record["error"] = f"{type(error).__name__}: {error}"
            torch.cuda.empty_cache()
        checkpoint()

    passing = [record for record in result["records"] if record["status"] == "passed"]
    if not passing:
        result["status"] = "failed"
        result["failures"] = ["no exact derivative candidate completed"]
    else:
        selected = min(
            passing,
            key=complete_solve_score,
        )
        result["selected_strategy"] = selected["strategy"]
        if selected["strategy"] != "explicit-tangent":
            result["failures"].append(
                "complete-solve measurements did not select explicit-tangent"
            )
        result["status"] = "failed" if result["failures"] else "passed"
    checkpoint()
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
