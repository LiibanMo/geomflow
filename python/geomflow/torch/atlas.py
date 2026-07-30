"""Coordinate charts, overlap-validated transitions, and atlas queries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch

from ._utils import batched_jacobian, validate_supported_floating_tensor
from .analytic_metric import AnalyticMetric

DomainPredicate = Callable[[torch.Tensor], torch.Tensor]
TransitionMap = Callable[[torch.Tensor], torch.Tensor]
TransitionJacobian = Callable[[torch.Tensor], torch.Tensor]


class ChartDomainError(ValueError):
    """Raised when coordinates are outside a chart or transition domain."""


@dataclass(frozen=True)
class Transition:
    """A coordinate transition and its declared source-chart overlap domain."""

    map: TransitionMap
    source_domain: DomainPredicate
    jacobian: TransitionJacobian | None = None


@dataclass(frozen=True)
class ChartSelection:
    """A deterministic atlas query result in the selected coordinates."""

    chart_id: int
    coordinates: torch.Tensor
    candidates: tuple[int, ...]


def _all_finite(x: torch.Tensor) -> torch.Tensor:
    return torch.isfinite(x).all(dim=-1)


class Chart:
    """A coordinate chart with a mathematical domain and optional coverage heuristic.

    ``domain`` is an exact or conservative mathematical predicate. If it is
    omitted, the sample-based k-nearest-neighbour predicate is used as an
    explicitly heuristic conservative domain for learned/user atlases.
    """

    def __init__(
        self,
        chart_id: int,
        dim: int,
        samples: Optional[torch.Tensor],
        analytic_metric: AnalyticMetric,
        transitions: Optional[dict[int, Transition | TransitionMap]] = None,
        k: Optional[int] = None,
        domain: Optional[DomainPredicate] = None,
        transition_domains: Optional[dict[int, DomainPredicate]] = None,
        distance_chunk_size: Optional[int] = None,
    ):
        if dim <= 0:
            raise ValueError("chart dimension must be positive")
        if analytic_metric.dim != dim:
            raise ValueError("chart and metric dimensions must agree")
        if samples is not None:
            if samples.dim() != 2 or samples.shape[1] != dim or samples.shape[0] == 0:
                raise ValueError("samples must have shape (N, dim) with N > 0")
            validate_supported_floating_tensor(samples, "chart samples")
            if not torch.isfinite(samples).all():
                raise ValueError("samples must be finite floating-point coordinates")
        if k is not None and k <= 0:
            raise ValueError("k must be positive")
        if distance_chunk_size is not None and distance_chunk_size <= 0:
            raise ValueError("distance_chunk_size must be positive")

        self.chart_id = chart_id
        self.dim = dim
        self.samples = None if samples is None else samples.detach()
        self.analytic_metric = analytic_metric
        self.k = k if k is not None else min(2 * dim, 15)
        self.distance_chunk_size = distance_chunk_size
        self._domain = domain
        self.transitions: dict[int, Transition] = {}
        transition_domains = transition_domains or {}
        for target, transition in (transitions or {}).items():
            if isinstance(transition, Transition):
                self.transitions[target] = transition
            else:
                predicate = transition_domains.get(target, self.contains)
                self.transitions[target] = Transition(transition, predicate)

        self.radius = None
        if self.samples is not None:
            self.radius = self._coverage_radius(self.samples)
        if self._domain is None and self.samples is None:
            self._domain = _all_finite

    def _coverage_radius(self, samples: torch.Tensor) -> float:
        """Compute the fixed coverage radius during atlas construction."""
        from scipy.spatial import KDTree

        sample_np = samples.detach().cpu().numpy()
        tree = KDTree(sample_np)
        k_eff = min(self.k, len(sample_np))
        distances, _ = tree.query(sample_np, k=k_eff)
        distances = np.asarray(distances).reshape(len(sample_np), -1)
        return max(
            float(np.percentile(distances.max(axis=1), 95)) * 2.5,
            float(np.finfo(sample_np.dtype).eps),
        )

    def set_samples(self, samples: torch.Tensor) -> None:
        """Replace persistent samples and recompute static coverage metadata."""
        if samples.dim() != 2 or samples.shape[1] != self.dim or samples.shape[0] == 0:
            raise ValueError("samples must have shape (N, dim) with N > 0")
        validate_supported_floating_tensor(samples, "chart samples")
        if not torch.isfinite(samples).all():
            raise ValueError("samples must be finite floating-point coordinates")
        self.samples = samples.detach()
        self.radius = self._coverage_radius(self.samples)

    @staticmethod
    def _validate_mask(name: str, mask: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if mask.shape != x.shape[:-1]:
            raise ValueError(f"{name} must return shape x.shape[:-1]")
        if mask.device != x.device or mask.dtype != torch.bool:
            raise ValueError(
                f"{name}: expected device={x.device}, dtype=torch.bool; got "
                f"device={mask.device}, dtype={mask.dtype}"
            )
        return mask

    def heuristically_covered(self, x: torch.Tensor) -> torch.Tensor:
        """Return the k-NN sample-coverage heuristic, never chart validity."""
        if self.samples is None or self.radius is None:
            raise RuntimeError("this chart has no sample-based coverage heuristic")
        if x.shape[-1] != self.dim:
            raise ValueError("coordinate dimension does not match chart")
        validate_supported_floating_tensor(x, "chart coverage")
        if self.samples.device != x.device or self.samples.dtype != x.dtype:
            raise ValueError(
                "chart coverage: expected samples on "
                f"device={x.device}, dtype={x.dtype}; got "
                f"device={self.samples.device}, dtype={self.samples.dtype}"
            )
        original_shape = x.shape[:-1]
        x_flat = x.detach().reshape(-1, self.dim)
        k_eff = min(self.k, len(self.samples))
        chunk_size = self.distance_chunk_size or min(len(self.samples), 4096)
        nearest = x_flat.new_full((len(x_flat), k_eff), torch.inf)
        for start in range(0, len(self.samples), chunk_size):
            distances = torch.cdist(
                x_flat,
                self.samples[start : start + chunk_size],
                compute_mode="donot_use_mm_for_euclid_dist",
            )
            candidates = torch.cat((nearest, distances), dim=-1)
            nearest = candidates.topk(k_eff, dim=-1, largest=False, sorted=True).values
        return (nearest[..., -1] <= self.radius).reshape(original_shape)

    def contains(self, x: torch.Tensor) -> torch.Tensor:
        """Return the exact or declared conservative chart-domain mask."""
        if x.shape[-1] != self.dim:
            raise ValueError("coordinate dimension does not match chart")
        validate_supported_floating_tensor(x, "chart membership")
        predicate = self._domain
        mask = self.heuristically_covered(x) if predicate is None else predicate(x)
        return self._validate_mask("chart domain predicate", mask, x) & _all_finite(x)

    def is_inside(self, x: torch.Tensor) -> torch.Tensor:
        """Compatibility alias for :meth:`contains`."""
        return self.contains(x)

    def can_transition_to(self, target_id: int, x: torch.Tensor) -> torch.Tensor:
        """Return whether the declared source overlap contains each point."""
        if target_id not in self.transitions:
            return torch.zeros(x.shape[:-1], device=x.device, dtype=torch.bool)
        mask = self.transitions[target_id].source_domain(x)
        return (
            self.contains(x)
            & self.analytic_metric.contains(x)
            & self._validate_mask("transition domain", mask, x)
        )

    def transition_to(self, target_id: int, x: torch.Tensor) -> torch.Tensor:
        """Map coordinates after requiring every point to lie in the overlap."""
        if target_id not in self.transitions:
            raise KeyError(f"no transition from chart {self.chart_id} to {target_id}")
        if not self.can_transition_to(target_id, x).all():
            raise ChartDomainError(
                f"coordinates are outside overlap {self.chart_id}->{target_id}"
            )
        return self._transition_unchecked(target_id, x)

    def _transition_unchecked(self, target_id: int, x: torch.Tensor) -> torch.Tensor:
        """Apply a transition whose source-overlap mask was already accepted."""
        mapped = self.transitions[target_id].map(x)
        if not isinstance(mapped, torch.Tensor):
            raise TypeError("transition map must return a torch.Tensor")
        if mapped.shape != x.shape:
            raise ValueError(
                f"transition map: expected shape {tuple(x.shape)}; "
                f"got {tuple(mapped.shape)}"
            )
        if mapped.device != x.device or mapped.dtype != x.dtype:
            raise ValueError(
                f"transition map: expected device={x.device}, dtype={x.dtype}; "
                f"got device={mapped.device}, dtype={mapped.dtype}"
            )
        return mapped

    def jacobian(self, target_id: int, x: torch.Tensor) -> torch.Tensor:
        """Return ``D psi_target,self`` on the declared overlap."""
        if not self.can_transition_to(target_id, x).all():
            raise ChartDomainError(
                f"coordinates are outside overlap {self.chart_id}->{target_id}"
            )
        return self._jacobian_unchecked(target_id, x)

    def _jacobian_unchecked(self, target_id: int, x: torch.Tensor) -> torch.Tensor:
        """Differentiate a transition whose source overlap was already accepted."""
        transition = self.transitions[target_id]
        if transition.jacobian is not None:
            value = transition.jacobian(x)
            expected = x.shape + (self.dim,)
            if value.shape != expected:
                raise ValueError(
                    f"transition jacobian: expected shape {tuple(expected)}; "
                    f"got {tuple(value.shape)}"
                )
            if value.device != x.device or value.dtype != x.dtype:
                raise ValueError(
                    "transition jacobian: expected "
                    f"device={x.device}, dtype={x.dtype}; got "
                    f"device={value.device}, dtype={value.dtype}"
                )
            return value
        return batched_jacobian(transition.map, x)


class Atlas:
    """A finite atlas using one chart identifier for an entire input batch."""

    def __init__(self, charts: list[Chart], reference_chart_id: int):
        if not charts:
            raise ValueError("an atlas must contain at least one chart")
        if len({chart.chart_id for chart in charts}) != len(charts):
            raise ValueError("chart identifiers must be unique")
        if len({chart.dim for chart in charts}) != 1:
            raise ValueError("all atlas charts must have the same dimension")
        self.charts = {chart.chart_id: chart for chart in charts}
        if reference_chart_id not in self.charts:
            raise ValueError("reference chart is not in the atlas")
        for chart in charts:
            unknown = set(chart.transitions) - set(self.charts)
            if unknown:
                raise ValueError(f"chart {chart.chart_id} has unknown targets {unknown}")
        self.reference_chart_id = reference_chart_id

    def __getitem__(self, chart_id: int) -> Chart:
        return self.charts[chart_id]

    def find_chart(
        self,
        x: torch.Tensor,
        source_chart: int,
        prefer: Optional[int] = None,
    ) -> ChartSelection:
        """Select a chart after mapping from the known source coordinates.

        The source chart is retained when valid unless a valid ``prefer`` is
        supplied. Remaining ties are resolved by ascending chart identifier.
        """
        if source_chart not in self.charts:
            raise ValueError(f"unknown source chart {source_chart}")
        if x.numel() == 0:
            raise ValueError("atlas queries require a non-empty batch")
        candidates: dict[int, torch.Tensor] = {}
        source = self[source_chart]
        if (source.contains(x) & source.analytic_metric.contains(x)).all():
            candidates[source_chart] = x
        for target in sorted(source.transitions):
            if not source.can_transition_to(target, x).all():
                continue
            mapped = (
                source._transition_unchecked(target, x)
                if type(source) is Chart
                else source.transition_to(target, x)
            )
            target_chart = self[target]
            if (
                target_chart.contains(mapped)
                & target_chart.analytic_metric.contains(mapped)
            ).all():
                candidates[target] = mapped
        if not candidates:
            raise ChartDomainError("no chart covers the complete batch")
        ordered = tuple(sorted(candidates))
        if prefer is not None and prefer in candidates:
            selected = prefer
        elif source_chart in candidates:
            selected = source_chart
        else:
            selected = ordered[0]
        return ChartSelection(selected, candidates[selected], ordered)

    def best_chart(self, x: torch.Tensor, current: int) -> tuple[int, torch.Tensor]:
        """Compatibility query using deterministic, domain-valid selection."""
        selection = self.find_chart(x, source_chart=current)
        return selection.chart_id, selection.coordinates
