"""Deterministic benchmark scenario construction."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from geomflow.torch import (
    EuclideanSpace,
    InducedMetric,
    ManifoldVectorField,
    MultiChartVectorField,
    PoincareDisk,
    Sphere2DAtlas,
    SphereStereographicMetric,
    Torus2D,
)


SCENARIOS = (
    "euclidean",
    "sphere",
    "torus",
    "poincare",
    "induced",
    "sphere-atlas",
)


@dataclass(frozen=True)
class BenchmarkCase:
    scenario: str
    batch_size: int
    dim: int
    hidden_width: int
    hidden_depth: int
    steps: int
    dtype: torch.dtype
    device: torch.device
    divergence_mode: str
    workload: str
    seed: int

    @property
    def case_id(self) -> str:
        dtype = str(self.dtype).removeprefix("torch.")
        return (
            f"{self.scenario}-{self.device.type}-{dtype}-b{self.batch_size}"
            f"-d{self.dim}-w{self.hidden_width}x{self.hidden_depth}"
            f"-s{self.steps}-{self.workload}-{self.divergence_mode}"
        )


def make_geometry(case: BenchmarkCase):
    if case.scenario == "euclidean":
        return EuclideanSpace(case.dim)
    if case.scenario == "sphere":
        return SphereStereographicMetric(case.dim)
    if case.scenario == "torus":
        if case.dim != 2:
            raise ValueError("torus requires --dim 2")
        return Torus2D()
    if case.scenario == "poincare":
        return PoincareDisk(case.dim)
    if case.scenario == "induced":
        def immersion(x: torch.Tensor) -> torch.Tensor:
            return torch.cat((x, x.square().sum(dim=-1, keepdim=True)), dim=-1)

        return InducedMetric(case.dim, immersion)
    if case.scenario == "sphere-atlas":
        if case.dim != 2:
            raise ValueError("sphere-atlas requires --dim 2")
        return Sphere2DAtlas()
    raise ValueError(f"unknown scenario: {case.scenario}")


def make_model(case: BenchmarkCase, geometry) -> torch.nn.Module:
    if case.scenario == "sphere-atlas":
        model = MultiChartVectorField(
            geometry,
            hidden_dim=case.hidden_width,
            n_layers=case.hidden_depth,
        )
    else:
        model = ManifoldVectorField(
            case.dim,
            hidden_dim=case.hidden_width,
            n_layers=case.hidden_depth,
            periodic=case.scenario == "torus",
        )
    return model.to(device=case.device, dtype=case.dtype)


def make_input(case: BenchmarkCase) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(case.seed)
    x = torch.randn(case.batch_size, case.dim, generator=generator, dtype=torch.float64)
    if case.scenario in {"sphere", "sphere-atlas", "induced"}:
        x = 0.25 * x
    elif case.scenario == "torus":
        x = torch.pi * torch.tanh(x)
    elif case.scenario == "poincare":
        norm = x.norm(dim=-1, keepdim=True).clamp_min(1.0)
        x = 0.7 * x / norm
    return x.to(device=case.device, dtype=case.dtype)
