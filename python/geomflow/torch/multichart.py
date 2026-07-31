"""Chartwise vector fields with an intrinsic overlap compatibility penalty."""

from __future__ import annotations

import torch
import torch.nn as nn

from .atlas import Atlas
from .vector_field import ManifoldVectorField, _has_global_execution_hooks


class MultiChartVectorField(nn.Module):
    """Vector-field module parametrised per chart.

    Each chart gets an independent head. These heads represent only an
    approximately global field during penalty-based training; exact globality
    requires transition compatibility.

    Parameters
    ----------
    atlas : Atlas
        The atlas (used to determine dimension).
    hidden_dim : int
        Width of each per-chart MLP.
    n_layers : int
        Number of hidden layers per chart.
    """

    def __init__(
        self,
        atlas: Atlas,
        hidden_dim: int = 64,
        n_layers: int = 2,
        activation: type[nn.Module] = nn.SiLU,
    ):
        super().__init__()
        dim = atlas.charts[next(iter(atlas.charts))].dim
        self.dim = dim
        self._heads = nn.ModuleDict()
        for cid in atlas.charts:
            self._heads[str(cid)] = ManifoldVectorField(
                dim, hidden_dim, n_layers, activation=activation
            )

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

    def _forward_unchecked(
        self, t: torch.Tensor, x: torch.Tensor, chart_id: int
    ) -> torch.Tensor:
        """Evaluate one head after solver-level compatibility validation."""
        return self._heads[str(chart_id)]._forward_unchecked(t, x)

    def _solver_forward(
        self, t: torch.Tensor, x: torch.Tensor, chart_id: int
    ) -> torch.Tensor:
        """Preserve custom dispatch and hooks while accelerating plain heads."""
        head = self._heads[str(chart_id)]
        has_hooks = bool(
            self._forward_hooks
            or self._forward_pre_hooks
            or self._backward_hooks
            or self._backward_pre_hooks
            or head._forward_hooks
            or head._forward_pre_hooks
            or head._backward_hooks
            or head._backward_pre_hooks
        )
        if (
            type(self) is MultiChartVectorField
            and type(head) is ManifoldVectorField
            and not has_hooks
            and not _has_global_execution_hooks()
            and getattr(self, "_compiled_call_impl", None) is None
            and getattr(head, "_compiled_call_impl", None) is None
        ):
            return head._forward_unchecked(t, x)
        return self(t, x, chart_id)

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
    coordinate_chart: int | None = None,
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

    def zero_loss() -> torch.Tensor:
        return sum((parameter.sum() * 0.0) for parameter in vf.parameters())

    coordinate_chart = chart_alpha if coordinate_chart is None else coordinate_chart
    if coordinate_chart != chart_alpha:
        source = atlas[coordinate_chart]
        if chart_alpha not in source.transitions:
            raise ValueError(
                f"no direct transition {coordinate_chart}->{chart_alpha} for overlap data"
            )
        source_mask = source.can_transition_to(chart_alpha, x_alpha)
        x_alpha = x_alpha[source_mask]
        if t.dim() > 0:
            t = t[source_mask]
        if x_alpha.shape[0] == 0:
            return zero_loss()
        x_alpha = source.transition_to(chart_alpha, x_alpha)
        target = atlas[chart_alpha]
        target_valid = target.contains(x_alpha) & target.analytic_metric.contains(
            x_alpha
        )
        x_alpha = x_alpha[target_valid]
        if t.dim() > 0:
            t = t[target_valid]
        if x_alpha.shape[0] == 0:
            return zero_loss()

    chart = atlas[chart_alpha]
    valid = chart.can_transition_to(chart_beta, x_alpha)
    x_alpha = x_alpha[valid]
    if t.dim() > 0:
        t = t[valid]
    if x_alpha.shape[0] == 0:
        return zero_loss()
    x_beta = chart.transition_to(chart_beta, x_alpha)
    target = atlas[chart_beta]
    target_valid = target.contains(x_beta) & target.analytic_metric.contains(x_beta)
    x_alpha = x_alpha[target_valid]
    x_beta = x_beta[target_valid]
    if t.dim() > 0:
        t = t[target_valid]
    if x_alpha.shape[0] == 0:
        return zero_loss()

    f_a = vf(t, x_alpha, chart_alpha)
    J = chart.jacobian(chart_beta, x_alpha)
    f_a_pushed = pushforward_vector(f_a, J)
    f_b = vf(t, x_beta, chart_beta)
    residual = f_b - f_a_pushed
    metric = atlas[chart_beta].analytic_metric.metric(x_beta)
    return torch.einsum("...i,...ij,...j->...", residual, metric, residual).mean()
