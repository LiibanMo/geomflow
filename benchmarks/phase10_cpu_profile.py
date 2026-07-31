#!/usr/bin/env python3
"""Capture function-level CPU call counts for one frozen exact-divergence solve."""

from __future__ import annotations

import argparse
import cProfile
import json
import os
from pathlib import Path
import pstats
import sys

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


CATEGORIES = {
    "compatibility": ("compatib",),
    "field": ("forward", "_solver_forward"),
    "divergence": ("divergence", "_coordinate_derivative"),
    "metric_domain": ("metric", "contains", "validate_points"),
    "chart_routing": ("chart", "transition", "rk4_trial"),
    "schedule": ("schedule", "__iter__"),
    "allocation": ("zeros", "empty", "new_full", "clone"),
    "autograd": ("autograd", "grad"),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    case = BenchmarkCase(
        scenario="sphere-atlas",
        batch_size=256,
        dim=2,
        hidden_width=32,
        hidden_depth=2,
        steps=16,
        dtype=torch.float32,
        device=torch.device("cpu"),
        divergence_mode="exact",
        workload="forward",
        seed=0,
    )
    torch.manual_seed(case.seed)
    geometry = make_geometry(case)
    model = make_model(case, geometry)
    x = make_input(case)

    def operation() -> None:
        kwargs = {"t0": 0.0, "t1": 1.0, "dt": 1.0 / case.steps}
        if case.scenario == "sphere-atlas":
            integrate_multichart(model, geometry, x, start_chart=0, **kwargs)
        else:
            integrate_rk4(model, geometry, x, **kwargs)

    operation()
    profiler = cProfile.Profile()
    profiler.enable()
    operation()
    profiler.disable()
    stats = pstats.Stats(profiler)
    functions = []
    categories = {category: 0 for category in CATEGORIES}
    for (filename, line, name), values in stats.stats.items():
        primitive_calls, total_calls, own_seconds, cumulative_seconds, _callers = values
        normalized = f"{filename}:{name}".lower()
        matched = [
            category
            for category, needles in CATEGORIES.items()
            if any(needle in normalized for needle in needles)
        ]
        if not matched:
            continue
        for category in matched:
            categories[category] += total_calls
        functions.append(
            {
                "file": filename,
                "line": line,
                "function": name,
                "primitive_calls": primitive_calls,
                "total_calls": total_calls,
                "own_seconds": own_seconds,
                "cumulative_seconds": cumulative_seconds,
                "categories": matched,
            }
        )
    result = {
        "schema_version": 1,
        "status": "passed",
        "label": args.label,
        "geomflow_path": str(Path(geomflow.__file__).resolve()),
        "case": case.case_id,
        "total_calls": stats.total_calls,
        "primitive_calls": stats.prim_calls,
        "category_call_counts": categories,
        "functions": sorted(
            functions, key=lambda item: item["cumulative_seconds"], reverse=True
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
