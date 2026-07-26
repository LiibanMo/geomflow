"""Deterministic host-side schedules for fixed-step integrators."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterator


@dataclass(frozen=True)
class ScalarStep:
    """One exactly bounded scalar-time step."""

    index: int
    start: float
    end: float

    @property
    def size(self) -> float:
        return self.end - self.start


class FixedStepSchedule:
    """Lazy deterministic schedule with an exact final endpoint."""

    def __init__(self, t0: float, t1: float, dt: float) -> None:
        self.t0 = _finite_scalar("t0", t0)
        self.t1 = _finite_scalar("t1", t1)
        try:
            self.dt = float(dt)
        except (TypeError, ValueError) as error:
            raise TypeError("dt must be a real scalar") from error
        if not math.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("dt must be a finite positive step magnitude")

        duration = abs(self.t1 - self.t0)
        self.direction = 1.0 if self.t1 >= self.t0 else -1.0
        full_steps = math.floor(duration / self.dt)
        remainder = duration - full_steps * self.dt
        tolerance = max(math.ulp(duration), math.ulp(self.dt)) * 4.0
        if remainder <= tolerance:
            remainder = 0.0
        elif self.dt - remainder <= tolerance:
            full_steps += 1
            remainder = 0.0
        self.full_steps = full_steps
        self.remainder = remainder
        self.step_count = full_steps + int(remainder > 0.0)

    def __len__(self) -> int:
        return self.step_count

    def __iter__(self) -> Iterator[ScalarStep]:
        for index in range(self.step_count):
            start = self.t0 + self.direction * self.dt * index
            end = (
                self.t1
                if index == self.step_count - 1
                else self.t0 + self.direction * self.dt * (index + 1)
            )
            yield ScalarStep(index, start, end)


def _finite_scalar(name: str, value: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a real scalar") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def checkpoint_due(index: int, total: int, interval: int) -> bool:
    """Return whether an accepted step is a requested or final checkpoint."""
    return index == total or index % interval == 0


def validate_checkpoint_interval(interval: int) -> int:
    if isinstance(interval, bool) or not isinstance(interval, int) or interval <= 0:
        raise ValueError("checkpoint_interval must be a positive integer")
    return interval
