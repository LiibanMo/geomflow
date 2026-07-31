#!/usr/bin/env python3
"""Reconstruct and validate Phase 10 paired release evidence from raw samples."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from phase10_paired import (
    aggregate_ratio_summary,
    case_name,
    case_payload,
    deterministic_block_bootstrap_bounds,
    ratio_summary,
    validate_worker_environments,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _close(actual: Any, expected: Any, name: str) -> None:
    if isinstance(expected, float):
        _require(
            isinstance(actual, (int, float))
            and math.isclose(float(actual), expected, rel_tol=1e-12, abs_tol=1e-12),
            f"{name} differs: expected {expected!r}, got {actual!r}",
        )
    else:
        _require(
            actual == expected, f"{name} differs: expected {expected!r}, got {actual!r}"
        )


def _verify_samples(samples: list[dict[str, Any]], name: str) -> None:
    _require(len(samples) == 2, f"{name} must contain exactly two samples")
    for index, sample in enumerate(samples):
        for timing in ("forward_ms", "backward_ms", "wall_ms"):
            value = sample.get(timing)
            _require(
                isinstance(value, (int, float))
                and math.isfinite(float(value))
                and float(value) >= 0.0
                and (timing != "wall_ms" or float(value) > 0.0),
                f"{name}[{index}].{timing} must be finite and nonnegative",
            )


def _verify_blocks(
    blocks: list[dict[str, Any]], quartets: int, orders: tuple[str, str], name: str
) -> None:
    _require(len(blocks) == quartets, f"{name} has an invalid quartet count")
    for index, block in enumerate(blocks):
        _require(
            block.get("order") == orders[index % 2], f"{name} has wrong block order"
        )
        _verify_samples(block.get("first", []), f"{name}.blocks[{index}].first")
        _verify_samples(block.get("second", []), f"{name}.blocks[{index}].second")


def _verify_ratio_summary(
    actual: dict[str, Any],
    *,
    alpha: float,
    quartets: int,
    orders: tuple[str, str],
    name: str,
) -> dict[str, Any]:
    blocks = actual.get("blocks", [])
    _verify_blocks(blocks, quartets, orders, name)
    expected = ratio_summary(blocks, alpha=alpha)
    for key in (
        "ratio",
        "lower",
        "upper",
        "bound_method",
        "bound_requested_alpha",
        "bound_resamples",
        "bound_seed",
        "block_log_ratios",
    ):
        _close(actual.get(key), expected[key], f"{name}.{key}")
    return expected


def _verify_manifest(payload: dict[str, Any], frozen: dict[str, Any]) -> None:
    manifest = payload.get("manifest", {})
    expected = {
        "scenarios": frozen["scenarios"],
        "workloads": frozen["workloads"],
        "regression_batches": frozen["regression_batches"],
        "crossover_batches": frozen["crossover_batches"],
        "dimension": frozen["dimension"],
        "hidden_width": frozen["hidden_width"],
        "hidden_depth": frozen["hidden_depth"],
        "steps": frozen["steps"],
        "dtype": frozen["dtype"],
        "quartets": frozen["quartets"],
        "warmup": frozen["warmup"],
        "baseline_revision": frozen["baseline_revision"],
        "mode": "release",
        "release_matrix_complete": True,
    }
    for key, value in expected.items():
        _require(
            manifest.get(key) == value, f"manifest.{key} differs from frozen protocol"
        )
    for key in (
        "baseline_wheel_sha256",
        "candidate_wheel_sha256",
        "worker_sha256",
        "scenarios_sha256",
        "frozen_manifest_sha256",
    ):
        digest = manifest.get(key)
        _require(
            isinstance(digest, str)
            and len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest),
            f"manifest.{key} is not a SHA-256 digest",
        )


def verify_release_payload(payload: dict[str, Any], frozen: dict[str, Any]) -> None:
    """Raise ``AssertionError`` unless every decision reproduces from raw evidence."""
    _require(payload.get("schema_version") == 2, "unsupported paired schema")
    _verify_manifest(payload, frozen)
    validate_worker_environments(payload.get("environments", {}))

    manifest = payload["manifest"]
    quartets = manifest["quartets"]
    expected_cpu = {
        case_name(case_payload(scenario, batch, workload))
        for scenario in manifest["scenarios"]
        for workload in manifest["workloads"]
        for batch in manifest["regression_batches"]
    }
    _require(
        set(payload.get("cpu_regression", {})) == expected_cpu, "CPU case set differs"
    )

    cpu_case_count = len(expected_cpu)
    cpu_alpha = 0.05 / cpu_case_count
    cpu_summaries = []
    expected_failures = []
    expected_inconclusive = []
    for name, summary in payload["cpu_regression"].items():
        expected = _verify_ratio_summary(
            summary,
            alpha=cpu_alpha,
            quartets=quartets,
            orders=("ABBA", "BAAB"),
            name=name,
        )
        expected_decision = (
            "passed"
            if expected["upper"] <= 1.10
            else "failed" if expected["lower"] > 1.10 else "inconclusive"
        )
        _require(
            summary.get("decision") == expected_decision, f"{name} decision differs"
        )
        cpu_summaries.append(expected)
        if expected_decision == "failed":
            expected_failures.append(
                f"CPU regression lower bound {expected['lower']:.3f} > 1.100: {name}"
            )
        elif expected_decision == "inconclusive":
            expected_inconclusive.append(
                f"CPU regression interval crosses 1.100: {name}"
            )

    aggregate = aggregate_ratio_summary(cpu_summaries)
    expected_geomean_decision = (
        "passed"
        if aggregate["upper"] <= 1.05
        else "failed" if aggregate["lower"] > 1.05 else "inconclusive"
    )
    for key in ("ratio", "lower", "upper"):
        _close(
            payload.get("cpu_geomean", {}).get(key),
            aggregate[key],
            f"cpu_geomean.{key}",
        )
    _require(
        payload.get("cpu_geomean", {}).get("decision") == expected_geomean_decision,
        "CPU geometric-mean decision differs",
    )
    if expected_geomean_decision == "failed":
        expected_failures.append(
            f"CPU geometric-mean lower bound {aggregate['lower']:.3f} > 1.050"
        )
    elif expected_geomean_decision == "inconclusive":
        expected_inconclusive.append("CPU geometric-mean interval crosses 1.050")

    expected_family_order = [
        f"{scenario}-{workload}"
        for scenario in manifest["scenarios"]
        for workload in manifest["workloads"]
    ]
    expected_families = set(expected_family_order)
    _require(
        set(payload.get("speed_gates", {})) == expected_families,
        "speed-gate set differs",
    )
    for family in expected_family_order:
        pilot = payload.get("crossover", {}).get(family)
        _require(isinstance(pilot, list), f"{family} crossover pilot is missing")
        _require(
            [row.get("batch_size") for row in pilot] == manifest["crossover_batches"],
            f"{family} crossover batches differ from the frozen matrix",
        )
        for row in pilot:
            for key in ("cpu_ms", "cuda_ms", "speedup"):
                value = row.get(key)
                _require(
                    isinstance(value, (int, float))
                    and math.isfinite(float(value))
                    and float(value) > 0.0,
                    f"{family} crossover {key} must be finite and positive",
                )
            _close(row["speedup"], row["cpu_ms"] / row["cuda_ms"], f"{family}.speedup")

        eligibility = payload.get("crossover", {}).get(f"{family}-eligibility")
        _require(
            isinstance(eligibility, list) and eligibility,
            f"{family} eligibility is missing",
        )
        selected = payload["speed_gates"][family]
        _require(
            selected.get("case", {}).get("batch_size")
            == eligibility[-1].get("batch_size"),
            f"{family} did not select the first eligible batch",
        )
        for index, row in enumerate(eligibility):
            samples = row.get("samples_ms", [])
            _require(
                len(samples) == 2 * quartets,
                f"{family} eligibility sample count differs",
            )
            _require(
                all(
                    isinstance(value, (int, float)) and float(value) > 0.0
                    for value in samples
                ),
                f"{family} eligibility samples must be positive",
            )
            _require(
                all(math.isfinite(float(value)) for value in samples),
                f"{family} eligibility samples must be finite",
            )
            bounds = deterministic_block_bootstrap_bounds(samples, 0.05 / 4.0)
            _close(
                row.get("lower_ms"), bounds["lower"], f"{family}.eligibility[{index}]"
            )
            if index + 1 < len(eligibility):
                _require(
                    bounds["lower"] < 100.0,
                    f"{family} skipped an earlier eligible batch",
                )
            else:
                _require(
                    bounds["lower"] >= 100.0, f"{family} selected an ineligible batch"
                )

        blocks = selected.get("blocks", [])
        _verify_blocks(blocks, quartets, ("CGGC", "GCCG"), family)
        summary = ratio_summary(blocks, alpha=0.05 / 4.0)
        speedup = 1.0 / summary["ratio"]
        lower = 1.0 / summary["upper"]
        upper = 1.0 / summary["lower"]
        required = 1.5 if family.startswith("sphere-atlas-") else 2.0
        for key, value in (("speedup", speedup), ("lower", lower), ("upper", upper)):
            _close(selected.get(key), value, f"{family}.{key}")
        _close(selected.get("required"), required, f"{family}.required")
        expected_decision = (
            "passed"
            if lower >= required
            else "failed" if upper < required else "inconclusive"
        )
        _require(
            selected.get("decision") == expected_decision, f"{family} decision differs"
        )
        for block in selected["blocks"]:
            for sample in (*block["first"], *block["second"]):
                backend = sample.get("backend")
                _require(
                    isinstance(backend, str) and backend, f"{family} backend is missing"
                )
                _require(
                    sample.get("fallback_reason") is None,
                    f"{family} used fallback: {sample}",
                )
        observed_backends = sorted(
            {
                sample["backend"]
                for block in selected["blocks"]
                for sample in block["second"]
            }
        )
        _require(
            selected.get("backends") == observed_backends,
            f"{family} backend summary differs",
        )
        _require(
            selected.get("fallback_reasons") == [], f"{family} fallback summary differs"
        )
        if expected_decision == "failed":
            expected_failures.append(
                f"CUDA speedup upper bound {upper:.3f} < {required:.3f}: {family}"
            )
        elif expected_decision == "inconclusive":
            expected_inconclusive.append(
                f"CUDA speedup interval crosses {required:.3f}: {family}"
            )

    _require(
        payload.get("failures") == expected_failures, "failure list does not reproduce"
    )
    _require(
        payload.get("inconclusive") == expected_inconclusive,
        "inconclusive list does not reproduce",
    )
    expected_status = (
        "failed"
        if expected_failures
        else "inconclusive" if expected_inconclusive else "passed"
    )
    selected_backends = {
        backend
        for gate in payload["speed_gates"].values()
        for backend in gate["backends"]
    }
    _require(len(selected_backends) == 1, "release gates used different CUDA backends")
    _require(
        payload.get("selected_cuda_backend") == next(iter(selected_backends)),
        "selected CUDA backend differs from raw samples",
    )
    _require(
        payload.get("status") == expected_status, "top-level status does not reproduce"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).with_name("phase10_manifest.json"),
    )
    parser.add_argument("--allow-failed", action="store_true")
    args = parser.parse_args()
    payload = json.loads(args.evidence.read_text())
    frozen = json.loads(args.manifest.read_text())
    try:
        verify_release_payload(payload, frozen)
    except (AssertionError, KeyError, TypeError, ValueError) as error:
        print(f"invalid paired evidence: {error}")
        return 1
    print(
        json.dumps(
            {
                key: payload.get(key, [])
                for key in ("status", "failures", "inconclusive")
            },
            indent=2,
        )
    )
    if payload.get("status") != "passed" and not args.allow_failed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
