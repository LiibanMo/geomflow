"""Deterministic test policy and mandatory-skip enforcement."""

import random

import numpy as np
import pytest
import torch


_MANDATORY_SKIPS: list[str] = []


@pytest.fixture(autouse=True)
def _deterministic_cpu_test() -> None:
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)


def pytest_configure() -> None:
    _MANDATORY_SKIPS.clear()


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    if report.skipped and "optional" not in report.keywords:
        _MANDATORY_SKIPS.append(report.nodeid)


def pytest_sessionfinish(session: pytest.Session) -> None:
    if _MANDATORY_SKIPS:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
