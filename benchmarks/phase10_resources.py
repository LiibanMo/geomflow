#!/usr/bin/env python3
"""Measure direct-autograd CUDA batch-memory scaling in fresh processes."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import subprocess
import sys

import torch

from geomflow.torch import integrate_multichart, integrate_rk4

from scenarios import BenchmarkCase, make_geometry, make_input, make_model


def zero_gradients_preserving_storage(model: torch.nn.Module) -> None:
    model.zero_grad(set_to_none=False)


def child_measurement(scenario: str, batch_size: int) -> dict[str, object]:
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
        workload="backward",
        seed=0,
    )
    torch.manual_seed(case.seed)
    geometry = make_geometry(case)
    model = make_model(case, geometry)
    kwargs = {"t0": 0.0, "t1": 1.0, "dt": 1.0 / case.steps}

    warm_case = BenchmarkCase(**{**case.__dict__, "batch_size": 1})
    warm_x = make_input(warm_case)
    if scenario == "sphere-atlas":
        warm_result = integrate_multichart(
            model, geometry, warm_x, start_chart=0, **kwargs
        )
    else:
        warm_result = integrate_rk4(model, geometry, warm_x, **kwargs)
    (warm_result.x_final.sum() + warm_result.divergence_integral.sum()).backward()
    torch.cuda.synchronize()
    del warm_x, warm_result
    gc.collect()
    zero_gradients_preserving_storage(model)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    fixed = torch.cuda.memory_allocated()

    x = make_input(case)
    torch.cuda.reset_peak_memory_stats()
    if scenario == "sphere-atlas":
        result = integrate_multichart(model, geometry, x, start_chart=0, **kwargs)
    else:
        result = integrate_rk4(model, geometry, x, **kwargs)
    loss = result.x_final.square().mean() + result.divergence_integral.mean()
    loss.backward()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    measured_parameters = (
        model.parameters_for_chart(0)
        if scenario == "sphere-atlas"
        else model.parameters()
    )
    return {
        "scenario": scenario,
        "batch_size": batch_size,
        "fixed_allocated_bytes": fixed,
        "fixed_includes_parameter_gradients": True,
        "peak_allocated_bytes": peak,
        "adjusted_peak_bytes": peak - fixed,
        "gradients_finite": all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all().item()
            for parameter in measured_parameters
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--child-scenario", choices=("euclidean", "sphere-atlas"))
    parser.add_argument("--child-batch", type=int)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 10 resource gate requires CUDA")
    if args.child_scenario is not None:
        print(json.dumps(child_measurement(args.child_scenario, args.child_batch)))
        return 0
    if args.output is None:
        parser.error("--output is required in parent mode")

    result = {"schema_version": 2, "status": "running", "records": [], "failures": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")

    def checkpoint() -> None:
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        temporary.replace(args.output)

    checkpoint()
    records = result["records"]
    for scenario in ("euclidean", "sphere-atlas"):
        for batch in (256, 512):
            completed = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--child-scenario",
                    scenario,
                    "--child-batch",
                    str(batch),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            records.append(json.loads(completed.stdout))
            checkpoint()

    failures = []
    ratios = {}
    for scenario in ("euclidean", "sphere-atlas"):
        selected = {row["batch_size"]: row for row in records if row["scenario"] == scenario}
        adjusted = [selected[batch]["adjusted_peak_bytes"] for batch in (256, 512)]
        if any(value <= 0 for value in adjusted):
            failures.append(f"{scenario} adjusted peak allocation is not positive")
            ratios[scenario] = None
            continue
        ratio = adjusted[1] / adjusted[0]
        ratios[scenario] = ratio
        if ratio > 2.2:
            failures.append(f"{scenario} adjusted 2B/B memory ratio {ratio:.3f} > 2.2")
        if not all(row["gradients_finite"] for row in selected.values()):
            failures.append(f"{scenario} produced missing or non-finite gradients")

    result["status"] = "failed" if failures else "passed"
    result["adjusted_memory_ratios"] = ratios
    result["failures"] = failures
    checkpoint()
    if failures:
        raise AssertionError("; ".join(failures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
