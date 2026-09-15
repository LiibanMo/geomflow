#!/usr/bin/env python3
"""Run drift-balanced Phase 10 baseline, CPU, and CUDA comparisons."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-python", type=Path, required=True)
    parser.add_argument("--candidate-python", type=Path, required=True)
    parser.add_argument("--baseline-package-root", type=Path)
    parser.add_argument("--candidate-package-root", type=Path)
    parser.add_argument("--baseline-revision", required=True)
    parser.add_argument("--candidate-revision", required=True)
    parser.add_argument("--baseline-wheel", type=Path)
    parser.add_argument("--candidate-wheel", type=Path)
    parser.add_argument("--offer", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--worker", type=Path, default=Path(__file__).with_name("phase10_worker.py")
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).with_name("phase10_manifest.json"),
    )
    parser.add_argument("--quartets", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--skip-cuda", action="store_true")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


class WorkerClient:
    def __init__(
        self,
        python: Path,
        worker: Path,
        device: str,
        revision: str,
        package_root: Path | None,
    ) -> None:
        environment = os.environ.copy()
        if package_root is not None:
            environment["PYTHONPATH"] = str(package_root)
            environment["GEOMFLOW_BENCHMARK_PACKAGE_ROOT"] = str(package_root)
        self.process = subprocess.Popen(
            [str(python), str(worker), "--device", device, "--revision", revision],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=environment,
        )
        self.label = f"{revision[:8]}-{device}"

    def request(self, action: str, **payload: Any) -> Any:
        if self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError(f"worker {self.label} has no protocol streams")
        self.process.stdin.write(json.dumps({"action": action, **payload}) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(
                f"worker {self.label} exited unexpectedly with {self.process.poll()}"
            )
        response = json.loads(line)
        if not response["ok"]:
            raise RuntimeError(f"worker {self.label}: {response['error']}")
        return response["result"]

    def close(self) -> None:
        if self.process.poll() is None:
            try:
                self.request("close")
            finally:
                return_code = self.process.wait(timeout=10)
        else:
            return_code = self.process.wait(timeout=10)
        if return_code != 0:
            raise RuntimeError(f"worker {self.label} exited with status {return_code}")


def validate_worker_environments(environments: dict[str, dict[str, Any]]) -> None:
    baseline = environments["baseline_cpu"]
    candidate = environments["candidate_cpu"]
    if baseline["python"] != candidate["python"]:
        raise AssertionError("baseline and candidate Python versions differ")
    if baseline["torch"] != candidate["torch"]:
        raise AssertionError("baseline and candidate PyTorch versions differ")
    for key in (
        "dependency_versions",
        "process_affinity",
        "thread_environment",
        "torch_num_threads",
        "torch_num_interop_threads",
    ):
        if baseline[key] != candidate[key]:
            raise AssertionError(f"baseline and candidate {key} differ")

    baseline_root = Path(baseline["package_root"]).resolve()
    candidate_root = Path(candidate["package_root"]).resolve()
    if baseline_root == candidate_root:
        raise AssertionError("baseline and candidate imported the same package root")
    if baseline["package_sha256"] == candidate["package_sha256"]:
        raise AssertionError("baseline and candidate package contents are identical")

    candidate_cuda = environments.get("candidate_cuda")
    if candidate_cuda is not None:
        if candidate_cuda["package_sha256"] != candidate["package_sha256"]:
            raise AssertionError("candidate CPU and CUDA package contents differ")
        if Path(candidate_cuda["package_root"]).resolve() != candidate_root:
            raise AssertionError("candidate CPU and CUDA package roots differ")
        if candidate_cuda["python"] != candidate["python"]:
            raise AssertionError("candidate CPU and CUDA Python versions differ")
        if candidate_cuda["torch"] != candidate["torch"]:
            raise AssertionError("candidate CPU and CUDA PyTorch versions differ")
        for key in ("dependency_versions", "process_affinity", "thread_environment"):
            if candidate_cuda[key] != candidate[key]:
                raise AssertionError(f"candidate CPU and CUDA {key} differ")


def incomplete_reasons(*, quick: bool, skip_cuda: bool) -> list[str]:
    reasons = []
    if quick:
        reasons.append("quick mode does not produce release evidence")
    if skip_cuda:
        reasons.append("CUDA speed gates were skipped")
    return reasons


def overall_status(payload: dict[str, Any]) -> str:
    if payload["failures"]:
        return "failed"
    if (
        payload["incomplete"]
        or payload["inconclusive"]
        or not payload["manifest"]["release_matrix_complete"]
    ):
        return "inconclusive"
    return "passed"


def case_payload(scenario: str, batch: int, workload: str) -> dict[str, Any]:
    return {
        "scenario": scenario,
        "batch_size": batch,
        "dim": 2,
        "hidden_width": 32,
        "hidden_depth": 2,
        "steps": 16,
        "dtype": "float32",
        "workload": workload,
        "seed": 0,
    }


def case_name(case: dict[str, Any]) -> str:
    return f"{case['scenario']}-b{case['batch_size']}-{case['workload']}"


def verify_prepared(*prepared: dict[str, Any]) -> None:
    model_hashes = {item["model_hash"] for item in prepared}
    input_hashes = {item["input_hash"] for item in prepared}
    if len(model_hashes) != 1 or len(input_hashes) != 1:
        raise AssertionError(
            "paired workers did not receive an identical workload: "
            f"model_hashes={model_hashes}, input_hashes={input_hashes}"
        )


def prepare(workers: list[WorkerClient], case: dict[str, Any], warmup: int) -> None:
    prepared = [worker.request("prepare", case=case) for worker in workers]
    verify_prepared(*prepared)
    for worker in workers:
        worker.request("warmup", count=warmup)


def balanced_blocks(
    first: WorkerClient,
    second: WorkerClient,
    quartets: int,
    orders: tuple[str, str],
    labels: tuple[str, str] = ("A", "B"),
) -> list[dict[str, Any]]:
    blocks = []
    for index in range(quartets):
        order = orders[index % 2]
        block: dict[str, Any] = {"order": order, "first": [], "second": []}
        for label in order:
            if label == labels[0]:
                key, worker = "first", first
            elif label == labels[1]:
                key, worker = "second", second
            else:
                raise ValueError(f"unknown block label {label!r}")
            block[key].append(worker.request("sample"))
        blocks.append(block)
    return blocks


def block_log_ratios(blocks: list[dict[str, Any]]) -> list[float]:
    ratios = []
    for block in blocks:
        first = statistics.fmean(math.log(item["wall_ms"]) for item in block["first"])
        second = statistics.fmean(math.log(item["wall_ms"]) for item in block["second"])
        ratios.append(second - first)
    return ratios


def deterministic_block_bootstrap_bounds(
    values: list[float], alpha: float, *, seed: int = 0, resamples: int = 20_000
) -> dict[str, Any]:
    if not values:
        raise ValueError("confidence bounds require at least one sample")
    if not math.isfinite(alpha) or not 0.0 < alpha <= 0.5:
        raise ValueError("alpha must be finite and in (0, 0.5]")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("confidence-bound samples must be finite")

    if len(values) < 2:
        raise ValueError("block bootstrap requires at least two blocks")
    if resamples < 1_000:
        raise ValueError("block bootstrap requires at least 1000 resamples")
    random_source = random.Random(seed)
    estimates = sorted(
        statistics.median(random_source.choices(values, k=len(values)))
        for _ in range(resamples)
    )
    lower_index = max(0, math.floor(alpha * resamples) - 1)
    upper_index = min(resamples - 1, math.ceil((1.0 - alpha) * resamples) - 1)
    return {
        "lower": estimates[lower_index],
        "upper": estimates[upper_index],
        "requested_alpha": alpha,
        "resamples": resamples,
        "seed": seed,
        "bound_method": "deterministic percentile block bootstrap of medians",
    }


# Kept as the public helper name used by benchmark tests and eligibility checks.
order_statistic_bounds = deterministic_block_bootstrap_bounds


def ratio_summary(
    blocks: list[dict[str, Any]],
    *,
    alpha: float,
) -> dict[str, Any]:
    logs = block_log_ratios(blocks)
    bounds = order_statistic_bounds(logs, alpha)
    estimate_log = statistics.median(logs)
    return {
        "ratio": math.exp(estimate_log),
        "lower": math.exp(bounds["lower"]),
        "upper": math.exp(bounds["upper"]),
        "bound_method": bounds["bound_method"],
        "bound_requested_alpha": bounds["requested_alpha"],
        "bound_resamples": bounds["resamples"],
        "bound_seed": bounds["seed"],
        "block_log_ratios": logs,
        "blocks": blocks,
    }


def aggregate_ratio_summary(
    summaries: list[dict[str, Any]],
) -> dict[str, float]:
    block_counts = {len(summary["block_log_ratios"]) for summary in summaries}
    if len(block_counts) != 1:
        raise ValueError("aggregate summaries have unequal block counts")
    aggregate_logs = [
        statistics.fmean(summary["block_log_ratios"][index] for summary in summaries)
        for index in range(block_counts.pop())
    ]
    bounds = deterministic_block_bootstrap_bounds(aggregate_logs, 0.05)
    estimate = statistics.median(aggregate_logs)
    return {
        "ratio": math.exp(estimate),
        "lower": math.exp(bounds["lower"]),
        "upper": math.exp(bounds["upper"]),
        "block_log_ratios": aggregate_logs,
        "bound_method": bounds["bound_method"],
    }


def atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def worker_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def optional_file_digest(path: Path | None) -> str | None:
    return None if path is None else worker_digest(path)


def validate_measurements(payload: dict[str, Any]) -> None:
    expected_cpu = {
        case_name(case_payload(scenario, batch, workload))
        for scenario in payload["manifest"]["scenarios"]
        for workload in payload["manifest"]["workloads"]
        for batch in payload["manifest"]["regression_batches"]
    }
    if set(payload["cpu_regression"]) != expected_cpu:
        raise AssertionError("CPU results contain duplicate or missing workloads")
    for name, summary in payload["cpu_regression"].items():
        blocks = summary.get("blocks", [])
        if len(blocks) != payload["manifest"]["quartets"]:
            raise AssertionError(f"{name} has an invalid quartet count")
        samples = [
            sample
            for block in blocks
            for side in ("first", "second")
            for sample in block[side]
        ]
        if any(
            not math.isfinite(float(sample.get("wall_ms", 0.0)))
            or float(sample.get("wall_ms", 0.0)) <= 0.0
            for sample in samples
        ):
            raise AssertionError(f"{name} has non-finite or non-positive timings")
    if payload["manifest"]["release_matrix_complete"]:
        expected_speed = {
            f"{scenario}-{workload}"
            for scenario in payload["manifest"]["scenarios"]
            for workload in payload["manifest"]["workloads"]
        }
        if not set(payload["speed_gates"]) <= expected_speed:
            raise AssertionError("CUDA results contain unexpected speed gates")


def main() -> int:
    args = parse_args()
    frozen_manifest = json.loads(args.manifest.read_text())
    if frozen_manifest["baseline_revision"] != args.baseline_revision:
        raise SystemExit("baseline revision differs from the frozen manifest")
    if not args.quick and (
        args.quartets != frozen_manifest["quartets"]
        or args.warmup != frozen_manifest["warmup"]
    ):
        raise SystemExit("release sampling differs from the frozen manifest")
    if args.quartets < 1 or args.warmup < 0:
        raise SystemExit("quartets must be positive and warmup nonnegative")
    if not args.quick and args.quartets % 2:
        raise SystemExit("non-quick runs require an even number of quartets")
    if args.quick:
        args.quartets, args.warmup = 2, 1
        regression_batches = [8]
        crossover_batches = [1, 8]
    else:
        regression_batches = frozen_manifest["regression_batches"]
        crossover_batches = frozen_manifest["crossover_batches"]

    payload: dict[str, Any] = {
        "schema_version": 2,
        "status": "running",
        "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
        "manifest": {
            **frozen_manifest,
            "regression_batches": regression_batches,
            "crossover_batches": crossover_batches,
            "quartets": args.quartets,
            "warmup": args.warmup,
            "mode": (
                "quick" if args.quick else "cpu-only" if args.skip_cuda else "release"
            ),
            "release_matrix_complete": not args.quick and not args.skip_cuda,
            "baseline_revision": args.baseline_revision,
            "candidate_revision": args.candidate_revision,
            "baseline_wheel_sha256": optional_file_digest(args.baseline_wheel),
            "candidate_wheel_sha256": optional_file_digest(args.candidate_wheel),
            "worker_sha256": worker_digest(args.worker),
            "scenarios_sha256": worker_digest(args.worker.with_name("scenarios.py")),
            "frozen_manifest_path": str(args.manifest.resolve()),
            "frozen_manifest_sha256": worker_digest(args.manifest),
        },
        "cpu_regression": {},
        "crossover": {},
        "speed_gates": {},
        "failures": [],
        "inconclusive": [],
        "incomplete": incomplete_reasons(quick=args.quick, skip_cuda=args.skip_cuda),
    }
    atomic_write(args.output, payload)
    if args.offer is not None:
        payload["vast_offer"] = json.loads(args.offer.read_text())
        atomic_write(args.output, payload)

    workers: list[WorkerClient] = []
    try:
        baseline = WorkerClient(
            args.baseline_python,
            args.worker,
            "cpu",
            args.baseline_revision,
            args.baseline_package_root,
        )
        workers.append(baseline)
        candidate_cpu = WorkerClient(
            args.candidate_python,
            args.worker,
            "cpu",
            args.candidate_revision,
            args.candidate_package_root,
        )
        workers.append(candidate_cpu)
        candidate_cuda = None
        if not args.skip_cuda:
            candidate_cuda = WorkerClient(
                args.candidate_python,
                args.worker,
                "cuda",
                args.candidate_revision,
                args.candidate_package_root,
            )
            workers.append(candidate_cuda)

        environments = {
            "baseline_cpu": baseline.request("describe"),
            "candidate_cpu": candidate_cpu.request("describe"),
        }
        if candidate_cuda is not None:
            environments["candidate_cuda"] = candidate_cuda.request("describe")
        payload["environments"] = environments
        validate_worker_environments(environments)

        cpu_case_count = 4 * len(regression_batches)
        cpu_alpha = 0.5 if args.quick else 0.05 / max(cpu_case_count, 1)
        cpu_summaries = []
        for scenario in ("euclidean", "sphere-atlas"):
            for workload in ("forward", "backward"):
                for batch in regression_batches:
                    case = case_payload(scenario, batch, workload)
                    prepare([baseline, candidate_cpu], case, args.warmup)
                    blocks = balanced_blocks(
                        baseline, candidate_cpu, args.quartets, ("ABBA", "BAAB")
                    )
                    summary = ratio_summary(
                        blocks,
                        alpha=cpu_alpha,
                    )
                    summary["case"] = case
                    summary["decision"] = (
                        "passed"
                        if summary["upper"] <= 1.10
                        else "failed" if summary["lower"] > 1.10 else "inconclusive"
                    )
                    payload["cpu_regression"][case_name(case)] = summary
                    cpu_summaries.append(summary)
                    if summary["decision"] == "failed":
                        payload["failures"].append(
                            f"CPU regression lower bound {summary['lower']:.3f} "
                            f"> 1.100: {case_name(case)}"
                        )
                    elif summary["decision"] == "inconclusive":
                        payload["inconclusive"].append(
                            f"CPU regression interval crosses 1.100: {case_name(case)}"
                        )
                    atomic_write(args.output, payload)

        aggregate = aggregate_ratio_summary(cpu_summaries)
        geomean_ratio = aggregate["ratio"]
        geomean_upper = aggregate["upper"]
        geomean_lower = aggregate["lower"]
        geomean_decision = (
            "passed"
            if geomean_upper <= 1.05
            else "failed" if geomean_lower > 1.05 else "inconclusive"
        )
        payload["cpu_geomean"] = {
            "ratio": geomean_ratio,
            "lower": geomean_lower,
            "upper": geomean_upper,
            "decision": geomean_decision,
        }
        if geomean_decision == "failed":
            payload["failures"].append(
                f"CPU geometric-mean lower bound {geomean_lower:.3f} > 1.050"
            )
        elif geomean_decision == "inconclusive":
            payload["inconclusive"].append("CPU geometric-mean interval crosses 1.050")

        if candidate_cuda is not None:
            speed_alpha = 0.05 / 4.0
            for scenario in ("euclidean", "sphere-atlas"):
                for workload in ("forward", "backward"):
                    family = f"{scenario}-{workload}"
                    pilot = []
                    for batch in crossover_batches:
                        case = case_payload(scenario, batch, workload)
                        prepare([candidate_cpu, candidate_cuda], case, 1)
                        cpu_sample = candidate_cpu.request("sample")
                        cuda_sample = candidate_cuda.request("sample")
                        pilot.append(
                            {
                                "batch_size": batch,
                                "cpu_ms": cpu_sample["wall_ms"],
                                "cuda_ms": cuda_sample["wall_ms"],
                                "speedup": cpu_sample["wall_ms"]
                                / cuda_sample["wall_ms"],
                            }
                        )
                    payload["crossover"][family] = pilot

                    if args.quick:
                        continue
                    selected = None
                    eligibility = []
                    for row in pilot:
                        if row["batch_size"] < 256:
                            continue
                        case = case_payload(scenario, row["batch_size"], workload)
                        prepare([candidate_cpu], case, args.warmup)
                        cpu_samples = [
                            candidate_cpu.request("sample")["wall_ms"]
                            for _ in range(2 * args.quartets)
                        ]
                        duration_bounds = order_statistic_bounds(
                            cpu_samples, speed_alpha
                        )
                        cpu_lower = duration_bounds["lower"]
                        eligibility.append(
                            {
                                "batch_size": row["batch_size"],
                                "samples_ms": cpu_samples,
                                "lower_ms": cpu_lower,
                                "bound_method": duration_bounds["bound_method"],
                                "bound_requested_alpha": duration_bounds[
                                    "requested_alpha"
                                ],
                                "bound_resamples": duration_bounds["resamples"],
                                "bound_seed": duration_bounds["seed"],
                            }
                        )
                        if cpu_lower < 100.0:
                            continue
                        prepare([candidate_cpu, candidate_cuda], case, args.warmup)
                        blocks = balanced_blocks(
                            candidate_cpu,
                            candidate_cuda,
                            args.quartets,
                            ("CGGC", "GCCG"),
                            ("C", "G"),
                        )
                        summary = ratio_summary(
                            blocks,
                            alpha=speed_alpha,
                        )
                        speedup = 1.0 / summary["ratio"]
                        speedup_lower = 1.0 / summary["upper"]
                        speedup_upper = 1.0 / summary["lower"]
                        required = 1.5 if scenario == "sphere-atlas" else 2.0
                        decision = (
                            "passed"
                            if speedup_lower >= required
                            else (
                                "failed" if speedup_upper < required else "inconclusive"
                            )
                        )
                        selected = {
                            "case": case,
                            "cpu_duration_lower_ms": cpu_lower,
                            "speedup": speedup,
                            "lower": speedup_lower,
                            "upper": speedup_upper,
                            "required": required,
                            "decision": decision,
                            "blocks": blocks,
                        }
                        selected_samples = [
                            sample for block in blocks for sample in block["second"]
                        ]
                        selected_backends = sorted(
                            {sample.get("backend") for sample in selected_samples}
                        )
                        fallback_reasons = sorted(
                            {
                                sample["fallback_reason"]
                                for sample in selected_samples
                                if sample.get("fallback_reason") is not None
                            }
                        )
                        selected["backends"] = selected_backends
                        selected["fallback_reasons"] = fallback_reasons
                        break
                    payload["crossover"][family + "-eligibility"] = eligibility
                    if selected is None:
                        payload["failures"].append(
                            f"no objectively eligible CUDA speed batch: {family}"
                        )
                    else:
                        payload["speed_gates"][family] = selected
                        if selected["decision"] == "failed":
                            payload["failures"].append(
                                f"CUDA speedup upper bound {selected['upper']:.3f} < "
                                f"{selected['required']:.3f}: {family}"
                            )
                        elif selected["decision"] == "inconclusive":
                            payload["inconclusive"].append(
                                f"CUDA speedup interval crosses {selected['required']:.3f}: {family}"
                            )
                        if len(selected["backends"]) != 1:
                            payload["failures"].append(
                                f"CUDA gate used multiple backends {selected['backends']}: {family}"
                            )
                        if selected["fallback_reasons"]:
                            payload["failures"].append(
                                f"CUDA gate used fallback {selected['fallback_reasons']}: {family}"
                            )
                    atomic_write(args.output, payload)

        selected_backends = {
            backend
            for gate in payload["speed_gates"].values()
            for backend in gate.get("backends", [])
        }
        payload["selected_cuda_backend"] = (
            next(iter(selected_backends)) if len(selected_backends) == 1 else None
        )
        validate_measurements(payload)
        payload["status"] = overall_status(payload)
    except Exception as error:
        payload["status"] = "infrastructure_error"
        payload["failures"].append(f"{type(error).__name__}: {error}")
    finally:
        for worker in reversed(workers):
            try:
                worker.close()
            except Exception as error:
                payload["failures"].append(
                    f"worker cleanup: {type(error).__name__}: {error}"
                )
                payload["status"] = "infrastructure_error"
        atomic_write(args.output, payload)

    print(
        json.dumps(
            {
                "status": payload["status"],
                "failures": payload["failures"],
                "inconclusive": payload["inconclusive"],
                "incomplete": payload["incomplete"],
            },
            indent=2,
        )
    )
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
