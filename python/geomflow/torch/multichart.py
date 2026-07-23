"""Multi-chart vector field — one head per chart, with overlap consistency."""

from __future__ import annotations

import torch
import torch.nn as nn

from .atlas import Atlas
from .vector_field import ManifoldVectorField


class MultiChartVectorField(nn.Module):
    """Vector-field module parametrised per chart.

    Each chart gets its own ``ManifoldVectorField`` head.  Only the
    vector-field parameters are trained; the manifold metric is fixed.

    Parameters
    ----------
    atlas : Atlas
        The atlas (used to determine dimension).
    hidden_dim : int
        Width of each per-chart MLP.
    n_layers : int
        Number of hidden layers per chart.
    """

    def __init__(self, atlas: Atlas, hidden_dim: int = 64, n_layers: int = 2):
        super().__init__()
        dim = atlas.charts[next(iter(atlas.charts))].dim
        self.dim = dim
        self._heads = nn.ModuleDict()
        for cid in atlas.charts:
            self._heads[str(cid)] = ManifoldVectorField(dim, hidden_dim, n_layers)

    def forward(
        self, t: torch.Tensor, x: torch.Tensor, chart_id: int
    ) -> torch.Tensor:
        """Evaluate vector field in a given chart.

        Parameters
        ----------
        t : Tensor
            Time, broadcastable to ``(...,)``.
        x : Tensor
            Coordinates in chart *chart_id*, shape ``(..., dim)``.
        chart_id : int

        Returns
        -------
        f : Tensor
            Vector components in the same chart, shape ``(..., dim)``.
        """
        return self._heads[str(chart_id)](t, x)

    def head(self, chart_id: int) -> ManifoldVectorField:
        """Access the per-chart head."""
        return self._heads[str(chart_id)]

    def parameters_for_chart(self, chart_id: int):
        """Yield parameters belonging to a specific chart's head."""
        return self._heads[str(chart_id)].parameters()


def overlap_consistency_loss(
    vf: MultiChartVectorField,
    atlas: Atlas,
    x_alpha: torch.Tensor,
    chart_alpha: int,
    chart_beta: int,
    t: torch.Tensor,
) -> torch.Tensor:
    r"""Penalise disagreement between vector-field heads in an overlap region.

    .. math::

        \\mathcal{L}_{\\text{vf}} =
        \\big\\| f_\\beta(y) - J_{\\beta\\alpha}\\, f_\\alpha(x) \\big\\|_2^2

    Parameters
    ----------
    vf : MultiChartVectorField
    atlas : Atlas
    x_alpha : Tensor
        Coordinates in chart α.
    chart_alpha, chart_beta : int
    t : Tensor
        Time.
    """
    from .transforms import pushforward_vector

    f_a = vf(t, x_alpha, chart_alpha)
    x_beta = atlas[chart_alpha].transition_to(chart_beta, x_alpha)
    J = atlas[chart_alpha].jacobian(chart_beta, x_alpha)
    f_a_pushed = pushforward_vector(f_a, J)
    f_b = vf(t, x_beta, chart_beta)
    return ((f_b - f_a_pushed) ** 2).sum(dim=-1).mean()
