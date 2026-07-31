"""Deterministic tests for the Phase 10 benchmark gates."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


def _benchmark_module(name: str):
    benchmark_dir = Path(__file__).parents[1] / "benchmarks"
    path = benchmark_dir / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"test_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(benchmark_dir))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def _module():
    return _benchmark_module("phase10_paired")


def _profile_module():
    return _benchmark_module("phase10_profile")


def _compiler_module():
    return _benchmark_module("phase10_compiler")


def _block(first: tuple[float, float], second: tuple[float, float]):
    return {
        "first": [{"wall_ms": value} for value in first],
        "second": [{"wall_ms": value} for value in second],
    }


def test_ratio_summary_recovers_multiplicative_change() -> None:
    paired = _module()
    blocks = [
        _block(
            (10.0 + index, 10.3 + index),
            (10.5 + 1.05 * index, 10.815 + 1.05 * index),
        )
        for index in range(5)
    ]
    summary = paired.ratio_summary(blocks, alpha=0.05)
    assert 1.04 < summary["ratio"] < 1.06
    assert summary["lower"] <= summary["ratio"] <= summary["upper"]

    aggregate = paired.aggregate_ratio_summary([summary, summary])
    assert 1.04 < aggregate["ratio"] < 1.06
    assert aggregate["lower"] <= aggregate["ratio"] <= aggregate["upper"]


def test_balanced_quartets_cancel_linear_order_drift() -> None:
    paired = _module()
    blocks = [
        _block((100.0, 103.0), (101.0, 102.0)),
        _block((102.0, 101.0), (100.0, 103.0)),
    ]
    logs = paired.block_log_ratios(blocks)
    assert max(abs(value) for value in logs) < 5e-4


def test_deterministic_block_bootstrap_uses_requested_alpha() -> None:
    paired = _module()
    values = [float(value) for value in range(10)]
    first = paired.order_statistic_bounds(values, 0.0125)
    second = paired.order_statistic_bounds(values, 0.0125)
    assert first == second
    assert first["requested_alpha"] == 0.0125
    assert first["resamples"] == 20_000
    assert first["lower"] <= 4.5 <= first["upper"]


def test_ratio_summary_rejects_single_block() -> None:
    paired = _module()
    blocks = [_block((1.0, 1.0), (1.0, 1.0))]
    with pytest.raises(ValueError, match="at least two"):
        paired.ratio_summary(blocks, alpha=0.00625)


def test_incomplete_modes_never_produce_passed_status() -> None:
    paired = _module()
    for quick, skip_cuda in ((True, False), (False, True), (True, True)):
        payload = {
            "failures": [],
            "inconclusive": [],
            "incomplete": paired.incomplete_reasons(
                quick=quick, skip_cuda=skip_cuda
            ),
            "manifest": {"release_matrix_complete": False},
        }
        assert payload["incomplete"]
        assert paired.overall_status(payload) == "inconclusive"
        payload["incomplete"] = []
        assert paired.overall_status(payload) == "inconclusive"

    assert paired.overall_status(
        {
            "failures": [],
            "inconclusive": [],
            "incomplete": [],
            "manifest": {"release_matrix_complete": True},
        }
    ) == "passed"


def test_worker_identity_uses_measured_package_content() -> None:
    paired = _module()
    baseline = {
        "python": "3.12.0",
        "torch": "2.7.1",
        "package_root": "/packages/baseline/geomflow",
        "package_sha256": "baseline-bytes",
        "dependency_versions": {"torch": "2.7.1"},
        "process_affinity": [0],
        "thread_environment": {},
        "torch_num_threads": 1,
        "torch_num_interop_threads": 1,
    }
    candidate = {
        "python": "3.12.0",
        "torch": "2.7.1",
        "package_root": "/packages/candidate/geomflow",
        "package_sha256": "candidate-bytes",
        "dependency_versions": {"torch": "2.7.1"},
        "process_affinity": [0],
        "thread_environment": {},
        "torch_num_threads": 1,
        "torch_num_interop_threads": 1,
    }
    paired.validate_worker_environments(
        {"baseline_cpu": baseline, "candidate_cpu": candidate}
    )

    with pytest.raises(AssertionError, match="contents are identical"):
        paired.validate_worker_environments(
            {
                "baseline_cpu": baseline,
                "candidate_cpu": {**candidate, "package_sha256": "baseline-bytes"},
            }
        )
    with pytest.raises(AssertionError, match="same package root"):
        paired.validate_worker_environments(
            {
                "baseline_cpu": baseline,
                "candidate_cpu": {
                    **candidate,
                    "package_root": baseline["package_root"],
                },
            }
        )


def test_worker_package_fingerprint_hashes_package_bytes(tmp_path: Path) -> None:
    worker = _benchmark_module("phase10_worker")
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "module.py").write_text("value = 1\n")
    (second / "module.py").write_text("value = 1\n")
    assert worker.package_fingerprint(first)[
        "package_sha256"
    ] == worker.package_fingerprint(second)["package_sha256"]
    (second / "module.py").write_text("value = 2\n")
    assert worker.package_fingerprint(first)[
        "package_sha256"
    ] != worker.package_fingerprint(second)["package_sha256"]


def test_balanced_blocks_support_named_cpu_cuda_order() -> None:
    paired = _module()
    execution = []

    class FakeWorker:
        def __init__(self, label: str) -> None:
            self.label = label

        def request(self, action: str):
            assert action == "sample"
            execution.append(self.label)
            return {"wall_ms": 1.0}

    blocks = paired.balanced_blocks(
        FakeWorker("C"), FakeWorker("G"), 2, ("CGGC", "GCCG"), ("C", "G")
    )
    assert execution == list("CGGCGCCG")
    assert [block["order"] for block in blocks] == ["CGGC", "GCCG"]
    assert all(len(block["first"]) == len(block["second"]) == 2 for block in blocks)


def test_profiler_transfer_bytes_fails_closed() -> None:
    profile = _profile_module()
    assert profile.transfer_bytes({"args": {"bytes": 4}}) == 4
    assert profile.transfer_bytes({"args": {"bytes": "4096"}}) == 4096
    assert profile.transfer_bytes({"args": {}}) is None
    assert profile.transfer_bytes({"args": {"bytes": "unknown"}}) is None


def test_profiler_gates_all_sync_variants_durations_and_transfers(
    tmp_path: Path,
) -> None:
    profile = _profile_module()
    events = [
        {"name": "solver", "ts": 100.0, "dur": 900.0},
        {"name": "cudaStreamSynchronize", "ts": 200.0, "dur": 10.0},
        {"name": "cudaDeviceSynchronize", "ts": 300.0, "dur": 20.0},
        {"name": "cudaEventSynchronize", "ts": 400.0, "dur": 30.0},
        {"name": "cuCtxSynchronize", "ts": 500.0, "dur": 5.0},
        {"name": "cudaUnknownSynchronize", "ts": 550.0},
        {"name": "cudaDeviceSynchronize", "ts": 1200.0, "dur": 500.0},
        {
            "name": "Memcpy DtoH (Device -> Pageable)",
            "ts": 600.0,
            "dur": 1.0,
            "args": {"bytes": 4096},
        },
        {"name": "aten::to", "ts": 700.0, "dur": 1.0},
        {"name": "aten::_to_copy", "ts": 701.0, "dur": 1.0},
    ]

    class FakeProfiler:
        def export_chrome_trace(self, path: str) -> None:
            Path(path).write_text(json.dumps({"traceEvents": events}))

        def key_averages(self):
            return [
                SimpleNamespace(key="aten::linear", count=192),
                SimpleNamespace(key="aten::to", count=1),
                SimpleNamespace(key="aten::_to_copy", count=2),
            ]

    summary = profile.profiler_summary(
        FakeProfiler(),
        tmp_path / "trace.json",
        scope_name="solver",
        end_to_end_us=1000.0,
    )
    assert summary["synchronization_count"] == 5
    assert summary["synchronization_duration_us"] == pytest.approx(65.0)
    assert summary["synchronization_duration_fraction"] == pytest.approx(0.065)

    failures = profile.profiler_failures(
        summary, expected_linear_count=192, synchronization_limit=4
    )
    assert summary["aten_to_noop_count"] == 0
    assert any("_to_copy" in failure for failure in failures)
    assert any("non-scalar host transfers" in failure for failure in failures)
    assert any("unknown durations" in failure for failure in failures)
    assert any("synchronization count" in failure for failure in failures)
    assert any("duration fraction" in failure for failure in failures)


def test_resource_baseline_preserves_allocated_gradient_storage() -> None:
    resources = _benchmark_module("phase10_resources")
    model = torch.nn.Linear(2, 1)
    model(torch.ones(3, 2)).sum().backward()
    pointers = [parameter.grad.data_ptr() for parameter in model.parameters()]

    resources.zero_gradients_preserving_storage(model)

    for parameter, pointer in zip(model.parameters(), pointers, strict=True):
        assert parameter.grad is not None
        assert parameter.grad.data_ptr() == pointer
        assert torch.count_nonzero(parameter.grad) == 0


def test_compiler_harness_errors_are_not_candidate_rejections(monkeypatch) -> None:
    compiler = _compiler_module()

    def fail_compile(*_args, **_kwargs):
        raise RuntimeError("broken harness")

    monkeypatch.setattr(compiler.torch, "compile", fail_compile)

    record = compiler.evaluate(torch.device("cpu"), torch.float64, 1)

    assert record["status"] == "infrastructure_error"
    assert record["error"] == "RuntimeError: broken harness"


def test_compiler_unsupported_fullgraph_is_a_candidate_rejection(monkeypatch) -> None:
    compiler = _compiler_module()

    def reject_graph(*_args, **_kwargs):
        raise compiler.torch._dynamo.exc.Unsupported("unsupported graph")

    monkeypatch.setattr(compiler.torch, "compile", reject_graph)

    record = compiler.evaluate(torch.device("cpu"), torch.float64, 1)

    assert record["status"] == "rejected"
    assert record["rejection_reason"] == "unsupported_fullgraph"


def test_compiler_harness_errors_fail_the_evidence_run(monkeypatch, tmp_path) -> None:
    compiler = _compiler_module()
    output = tmp_path / "compiler.json"
    monkeypatch.setattr(sys, "argv", ["phase10_compiler.py", "--output", str(output), "--quick"])
    monkeypatch.setattr(compiler.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        compiler,
        "evaluate",
        lambda *_args: {"status": "infrastructure_error", "error": "broken"},
    )

    assert compiler.main() == 1
    result = json.loads(output.read_text())
    assert result["status"] == "infrastructure_error"
    assert result["decision"] == "undetermined"


def test_run_defaults_fail_closed_with_five_repetitions(monkeypatch) -> None:
    run = _benchmark_module("run")
    monkeypatch.setattr(sys, "argv", ["run.py"])
    args = run.parse_args()
    assert args.repetitions == 5
    assert not args.allow_errors


def test_backward_objective_executes_inside_timed_region(monkeypatch) -> None:
    run = _benchmark_module("run")
    case = SimpleNamespace(device=torch.device("cpu"), workload="backward")
    result = SimpleNamespace()
    timed = False
    objective_observed = []

    class FakeModel:
        def zero_grad(self, *, set_to_none: bool) -> None:
            assert set_to_none

    def fake_time_call(function, _device):
        nonlocal timed
        timed = True
        try:
            return function(), 1.0
        finally:
            timed = False

    def fake_objective(_result):
        objective_observed.append(timed)
        return torch.tensor(1.0, requires_grad=True)

    monkeypatch.setattr(run, "integrate", lambda *_args: result)
    monkeypatch.setattr(run, "time_call", fake_time_call)
    monkeypatch.setattr(run, "scalar_objective", fake_objective)
    _, timing = run.one_iteration(case, FakeModel(), None, None)
    assert objective_observed == [True]
    assert timing["backward_ms"] == 1.0
    assert timing["wall_ms"] >= 0.0


def test_run_case_errors_return_nonzero_unless_explicitly_allowed(
    monkeypatch, tmp_path: Path
) -> None:
    run = _benchmark_module("run")
    args = SimpleNamespace(
        warmup=0,
        repetitions=5,
        output=tmp_path / "failed.json",
        allow_errors=False,
    )
    case = SimpleNamespace(case_id="broken-case")

    monkeypatch.setattr(run, "parse_args", lambda: args)
    monkeypatch.setattr(run, "cases", lambda _args: [case])

    def fail_case(*_args):
        raise RuntimeError("boom")

    monkeypatch.setattr(run, "run_case", fail_case)
    monkeypatch.setattr(run, "environment_metadata", lambda: {})
    monkeypatch.setattr(run, "print_table", lambda _records: None)

    assert run.main() == 1
    assert json.loads(args.output.read_text())["status"] == "failed"
    args.allow_errors = True
    args.output = tmp_path / "allowed.json"
    assert run.main() == 0
    assert json.loads(args.output.read_text())["status"] == "failed"
