"""Multi-chart atlas for Riemannian manifolds.

Each chart covers a patch of the manifold.  Charts are linked by
user-supplied transition maps (:math:`\\psi_{\\beta\\alpha}
= \\varphi_\\beta \\circ \\varphi_\\alpha^{-1}`).

Chart validity is determined by a :math:`k`-NN ball around sample points.
The metric in each chart is user-provided analytic closed-form.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch

from ._utils import batched_jacobian
from .analytic_metric import AnalyticMetric


class Chart:
    """A coordinate chart patch with an analytic metric.

    Parameters
    ----------
    chart_id : int
        Unique identifier.
    dim : int
        Manifold dimension (same for all charts in one atlas).
    samples : torch.Tensor
        Shape ``(N, dim)`` — sample point coordinates in this chart.
    analytic_metric : AnalyticMetric
        User-supplied analytic metric function for this chart.
    transitions : dict[int, callable], optional
        Map from target chart id to a callable ``ψ(x_self) → x_target``.
        Must be torch-differentiable if gradients flow across chart boundaries.
    k : int, optional
        Number of nearest neighbours for validity (default ``min(2·dim, 15)``).
    """

    def __init__(
        self,
        chart_id: int,
        dim: int,
        samples: torch.Tensor,
        analytic_metric: AnalyticMetric,
        transitions: Optional[dict[int, Callable[[torch.Tensor], torch.Tensor]]] = None,
        k: Optional[int] = None,
    ):
        self.chart_id = chart_id
        self.dim = dim
        self.samples = samples.detach()
        self.analytic_metric = analytic_metric
        self.transitions = transitions or {}
        self.k = k or min(2 * dim, 15)

        sample_np = self.samples.cpu().numpy()
        k_eff = min(self.k, len(sample_np))
        from scipy.spatial import KDTree

        self._tree = KDTree(sample_np)
        distances, _ = self._tree.query(sample_np, k=k_eff)
        self.radius = float(np.percentile(distances.max(axis=1), 95)) * 2.5  # generous margin

    def is_inside(self, x: torch.Tensor) -> torch.Tensor:
        """Return a boolean mask for points inside the chart coverage."""
        x_np = x.detach().cpu().numpy()
        original_shape = x_np.shape[:-1]
        x_flat = x_np.reshape(-1, self.dim)
        k_eff = min(self.k, len(self.samples))
        distances, _ = self._tree.query(x_flat, k=k_eff)
        max_dist = distances.max(axis=1)
        inside_flat = max_dist <= self.radius
        mask_np = inside_flat.reshape(original_shape)
        return torch.tensor(mask_np, device=x.device)

    def transition_to(self, target_id: int, x: torch.Tensor) -> torch.Tensor:
        """Map coordinates from this chart to *target_id*."""
        fn = self.transitions[target_id]
        return fn(x)

    def jacobian(self, target_id: int, x: torch.Tensor) -> torch.Tensor:
        """Jacobian :math:`J_{\\beta\\alpha}` of the transition map."""
        fn = lambda x_: self.transitions[target_id](x_)
        return batched_jacobian(fn, x)  # (..., dim, dim)


class Atlas:
    """Collection of charts covering a Riemannian manifold.

    One chart is designated as the *reference* where the base density lives.

    Parameters
    ----------
    charts : list of Chart
    reference_chart_id : int
    """

    def __init__(
        self,
        charts: list[Chart],
        reference_chart_id: int,
    ):
        self.charts: dict[int, Chart] = {c.chart_id: c for c in charts}
        self.reference_chart_id = reference_chart_id

    def __getitem__(self, chart_id: int) -> Chart:
        return self.charts[chart_id]

    def find_chart(self, x: torch.Tensor, prefer: Optional[int] = None) -> int:
        """Return a chart id that fully covers *x*."""
        if prefer is not None and self.charts[prefer].is_inside(x).all():
            return prefer
        for cid, chart in self.charts.items():
            if chart.is_inside(x).all():
                return cid
        raise ValueError("Point is not covered by any chart in the atlas.")

    def best_chart(self, x: torch.Tensor, current: Optional[int] = None) -> tuple[int, torch.Tensor]:
        """Pick the chart with the largest validity margin and map *x* into it."""
        if current is not None and self.charts[current].is_inside(x).all():
            return current, x

        best_id: Optional[int] = None
        best_margin = -1.0
        best_x_mapped: Optional[torch.Tensor] = None

        for cid, chart in self.charts.items():
            if current is not None and cid != current:
                if cid not in self.charts[current].transitions:
                    continue
                try:
                    x_cid = self.charts[current].transition_to(cid, x)
                except Exception:
                    continue
            else:
                x_cid = x

            if not chart.is_inside(x_cid).all():
                continue

            x_np = x_cid.detach().cpu().numpy().reshape(-1, chart.dim)
            k_eff = min(chart.k, len(chart.samples))
            dist, _ = chart._tree.query(x_np, k=k_eff)
            margin = float((chart.radius - dist.max(axis=1).mean()) / chart.radius)
            if margin > best_margin:
                best_margin = margin
                best_id = cid
                best_x_mapped = x_cid

        if best_id is None or best_x_mapped is None:
            raise ValueError("No chart covers the point.")

        return best_id, best_x_mapped
