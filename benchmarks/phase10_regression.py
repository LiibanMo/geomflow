"""Evaluate paired Phase 10 baseline and release-candidate benchmarks."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-case-regression", type=float, default=1.10)
    parser.add_argument("--max-geomean-regression", type=float, default=1.05)
    return parser.parse_args()


def load_cases(path: Path) -> tuple[dict, dict[str, dict]]:
    document = json.loads(path.read_text())
    cases = {case["case_id"]: case for case in document["cases"]}
    failed = [case_id for case_id, case in cases.items() if case["status"] != "passed"]
    if failed:
        raise AssertionError(f"failed benchmark cases in {path}: {failed}")
    return document["run"], cases


def cpu_key(case: dict) -> tuple:
    parameters = case["parameters"]
    return tuple((key, value) for key, value in parameters.items() if key != "device")


def main() -> int:
    args = parse_args()
    baseline_run, baseline_cases = load_cases(args.baseline)
    candidate_run, candidate_cases = load_cases(args.candidate)
    baseline_cpu = {
        cpu_key(case): case
        for case in baseline_cases.values()
        if case["parameters"]["device"] == "cpu"
    }
    candidate_cpu = {
        cpu_key(case): case
        for case in candidate_cases.values()
        if case["parameters"]["device"] == "cpu"
    }
    candidate_cuda = {
        cpu_key(case): case
        for case in candidate_cases.values()
        if case["parameters"]["device"] == "cuda"
    }
    if baseline_cpu.keys() != candidate_cpu.keys():
        raise AssertionError("baseline and candidate CPU scenario matrices differ")

    cpu_rows = []
    ratios = []
    failures = []
    for key in sorted(baseline_cpu, key=str):
        baseline = baseline_cpu[key]
        candidate = candidate_cpu[key]
        ratio = candidate["median_wall_ms"] / baseline["median_wall_ms"]
        ratios.append(ratio)
        row = {
            "case": dict(key),
            "baseline_ms": baseline["median_wall_ms"],
            "candidate_ms": candidate["median_wall_ms"],
            "wall_time_ratio": ratio,
        }
        cpu_rows.append(row)
        if ratio > args.max_case_regression:
            failures.append(f"CPU case regression {ratio:.3f} exceeds {args.max_case_regression:.3f}: {dict(key)}")

    geomean_ratio = math.exp(sum(math.log(ratio) for ratio in ratios) / len(ratios))
    if geomean_ratio > args.max_geomean_regression:
        failures.append(
            f"CPU geometric-mean regression {geomean_ratio:.3f} exceeds "
            f"{args.max_geomean_regression:.3f}"
        )

    speed_rows = []
    eligible_scenarios = set()
    for key, cpu_case in sorted(candidate_cpu.items(), key=lambda item: str(item[0])):
        cuda_case = candidate_cuda.get(key)
        parameters = dict(key)
        if cuda_case is None or parameters["batch_size"] < 256 or cpu_case["median_wall_ms"] < 100:
            continue
        required = 1.5 if parameters["scenario"] == "sphere-atlas" else 2.0
        speedup = cpu_case["median_wall_ms"] / cuda_case["median_wall_ms"]
        speed_rows.append({"case": parameters, "speedup": speedup, "required": required})
        eligible_scenarios.add(parameters["scenario"])
        if speedup < required:
            failures.append(f"CUDA speedup {speedup:.3f} is below {required:.3f}: {parameters}")

    required_scenarios = {"euclidean", "sphere-atlas"}
    missing = required_scenarios - eligible_scenarios
    if missing:
        failures.append(f"no eligible CPU >=100 ms CUDA speed case for: {sorted(missing)}")

    result = {
        "schema_version": 1,
        "status": "failed" if failures else "passed",
        "baseline_run": baseline_run,
        "candidate_run": candidate_run,
        "thresholds": {
            "max_case_regression": args.max_case_regression,
            "max_geomean_regression": args.max_geomean_regression,
            "single_chart_min_speedup": 2.0,
            "multi_chart_min_speedup": 1.5,
        },
        "cpu_geomean_wall_time_ratio": geomean_ratio,
        "cpu_cases": cpu_rows,
        "cuda_speedups": speed_rows,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if failures:
        raise AssertionError("; ".join(failures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
