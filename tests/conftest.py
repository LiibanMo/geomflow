"""Deterministic test policy and mandatory-skip enforcement."""
import os

import random

import numpy as np
import pytest
import torch


_MANDATORY_SKIPS: list[str] = []


def available_test_devices(*, include_mps: bool = False) -> tuple[torch.device, ...]:
    """Return devices that can execute tensors in this test process."""
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    if include_mps and torch.backends.mps.is_available():
        devices.append(torch.device("mps"))
    return tuple(devices)


def supported_device_dtype_cases(
    *, include_mps: bool = False
) -> tuple[object, ...]:
    """Build explicit production device/dtype test parameters."""
    cases: list[object] = []
    for device in available_test_devices(include_mps=include_mps):
        dtypes = (torch.float32, torch.float64)
        if device.type == "mps":
            dtypes = (torch.float32,)
        for dtype in dtypes:
            marks = pytest.mark.optional if device.type != "cpu" else ()
            cases.append(
                pytest.param(
                    (device, dtype),
                    marks=marks,
                    id=f"{device.type}-{str(dtype).removeprefix('torch.')}",
                )
            )
    return tuple(cases)


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA-capable PyTorch runtime"
)
requires_multiple_cuda_devices = pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="requires at least two CUDA devices"
)
requires_mps = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="requires an available MPS backend"
)


@pytest.fixture(params=supported_device_dtype_cases())
def device_dtype(request: pytest.FixtureRequest) -> tuple[torch.device, torch.dtype]:
    """Provide supported CPU/CUDA production cases and verify placement."""
    device, dtype = request.param
    probe = torch.empty(1, device=device, dtype=dtype)
    assert probe.device.type == device.type
    assert probe.dtype == dtype
    return device, dtype


@pytest.fixture(autouse=True)
def _deterministic_cpu_test() -> None:
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)


def pytest_configure() -> None:
    _MANDATORY_SKIPS.clear()
    if os.getenv("GEOMFLOW_REQUIRE_CUDA") == "1" and not torch.cuda.is_available():
        raise pytest.UsageError(
            "GEOMFLOW_REQUIRE_CUDA=1 but CUDA is unavailable; mandatory GPU validation cannot run"
        )


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    cuda_required = os.getenv("GEOMFLOW_REQUIRE_CUDA") == "1"
    skip_reason = str(report.longrepr).lower() if report.skipped else ""
    unavailable_cuda = not cuda_required and "cuda" in skip_reason
    optional = "optional" in report.keywords and not (
        cuda_required and "gpu" in report.keywords
    )
    if report.skipped and not optional and not unavailable_cuda:
        _MANDATORY_SKIPS.append(report.nodeid)


def pytest_sessionfinish(session: pytest.Session) -> None:
    if _MANDATORY_SKIPS:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
