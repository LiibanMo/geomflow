"""Base distributions with density explicitly relative to Riemannian volume."""

from __future__ import annotations

import math
from typing import Protocol, runtime_checkable

import torch

from ._utils import validate_supported_dtype, validate_supported_floating_tensor
from .analytic_metric import AnalyticMetric
from .atlas import Atlas


@runtime_checkable
class BaseDistribution(Protocol):
    """Normalized base law whose public density is relative to ``dV_g``."""

    dim: int

    def sample(
        self,
        sample_shape: torch.Size | tuple[int, ...],
        *,
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor: ...

    def log_prob_volume(
        self, x: torch.Tensor, metric: AnalyticMetric
    ) -> torch.Tensor: ...

    def contains(self, x: torch.Tensor) -> torch.Tensor: ...


class CoordinateBaseDistribution:
    """Coordinate law converted to a density relative to ``dV_g`` exactly."""

    def __init__(self, dim: int):
        if dim < 1:
            raise ValueError("base distribution dimension must be positive")
        self.dim = dim

    def log_prob_coordinate(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def log_prob_volume(
        self, x: torch.Tensor, metric: AnalyticMetric
    ) -> torch.Tensor:
        self._validate_points(x, metric)
        if not self.contains(x).all():
            raise ValueError("point lies outside the base distribution support")
        return self.log_prob_coordinate(x) - 0.5 * metric.log_det(x)

    def _validate_points(self, x: torch.Tensor, metric: AnalyticMetric) -> None:
        if x.ndim < 1 or x.shape[-1] != self.dim:
            raise ValueError(f"expected points with shape (..., {self.dim})")
        if metric.dim != self.dim:
            raise ValueError("base distribution and metric dimensions differ")
        validate_supported_floating_tensor(x, "base distribution log_prob")


class StandardNormalCoordinateBase(CoordinateBaseDistribution):
    """Standard normal in chart-coordinate measure, not an intrinsic Gaussian."""

    def sample(
        self,
        sample_shape: torch.Size | tuple[int, ...],
        *,
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        validate_supported_dtype(dtype, "base samples")
        return torch.randn(
            *sample_shape, self.dim, device=device, dtype=dtype, generator=generator
        )

    def log_prob_coordinate(self, x: torch.Tensor) -> torch.Tensor:
        return -0.5 * (
            self.dim * math.log(2.0 * math.pi) + x.square().sum(dim=-1)
        )

    def contains(self, x: torch.Tensor) -> torch.Tensor:
        return torch.isfinite(x).all(dim=-1)


class UniformAngleCoordinateBase(CoordinateBaseDistribution):
    """Uniform coordinate law on the canonical ``[-pi, pi)^d`` angle cell."""

    def sample(
        self,
        sample_shape: torch.Size | tuple[int, ...],
        *,
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        validate_supported_dtype(dtype, "base samples")
        shape = (*sample_shape, self.dim)
        return (2.0 * math.pi) * torch.rand(
            shape, device=device, dtype=dtype, generator=generator
        ) - math.pi

    def log_prob_coordinate(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full(
            x.shape[:-1],
            -self.dim * math.log(2.0 * math.pi),
            device=x.device,
            dtype=x.dtype,
        )

    def contains(self, x: torch.Tensor) -> torch.Tensor:
        return ((x >= -math.pi) & (x < math.pi) & torch.isfinite(x)).all(dim=-1)


class PoincareDiskCoordinateBase(CoordinateBaseDistribution):
    """Pushforward of a standard normal onto the open unit ball."""

    def sample(
        self,
        sample_shape: torch.Size | tuple[int, ...],
        *,
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        validate_supported_dtype(dtype, "base samples")
        u = torch.randn(
            *sample_shape, self.dim, device=device, dtype=dtype, generator=generator
        )
        return u / torch.sqrt(1.0 + u.square().sum(dim=-1, keepdim=True))

    def log_prob_coordinate(self, x: torch.Tensor) -> torch.Tensor:
        radius_squared = x.square().sum(dim=-1)
        one_minus_radius_squared = 1.0 - radius_squared
        normal_radius_squared = radius_squared / one_minus_radius_squared
        return (
            -0.5 * (self.dim * math.log(2.0 * math.pi) + normal_radius_squared)
            - 0.5 * (self.dim + 2) * torch.log(one_minus_radius_squared)
        )

    def contains(self, x: torch.Tensor) -> torch.Tensor:
        return torch.isfinite(x).all(dim=-1) & (x.square().sum(dim=-1) < 1.0)


class AtlasBaseDistribution:
    """A coordinate base law associated with one explicit atlas reference chart."""

    def __init__(
        self, coordinate_base: CoordinateBaseDistribution, reference_chart_id: int
    ):
        self.coordinate_base = coordinate_base
        self.reference_chart_id = reference_chart_id
        self.dim = coordinate_base.dim

    def sample(
        self,
        sample_shape: torch.Size | tuple[int, ...],
        *,
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        return self.coordinate_base.sample(
            sample_shape, device=device, dtype=dtype, generator=generator
        )

    def contains(self, x: torch.Tensor) -> torch.Tensor:
        return self.coordinate_base.contains(x)

    def log_prob_volume(
        self, x: torch.Tensor, atlas: Atlas, chart_id: int
    ) -> torch.Tensor:
        if atlas.reference_chart_id != self.reference_chart_id:
            raise ValueError("base reference chart does not match atlas reference chart")
        if chart_id != self.reference_chart_id:
            try:
                x = atlas[chart_id].transition_to(self.reference_chart_id, x)
            except KeyError as error:
                raise ValueError("point cannot be mapped into the base reference chart") from error
        return self.coordinate_base.log_prob_volume(
            x, atlas[self.reference_chart_id].analytic_metric
        )


def validate_base_distribution(base: BaseDistribution, dim: int) -> None:
    """Reject ambiguous callbacks that do not declare volume-density semantics."""
    if not isinstance(base, BaseDistribution):
        raise TypeError(
            "base_distribution must implement sample, contains, and log_prob_volume"
        )
    if base.dim != dim:
        raise ValueError("base distribution and manifold dimensions differ")
