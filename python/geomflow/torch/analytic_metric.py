"""Analytic metric wrapper — metrics are user-supplied closed-form functions.

This matches Mohamud's paper assumption: the Riemannian manifold
(M, g) is known beforehand.  Only the vector field is learned.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch


class AnalyticMetric:
    """Wrap a user-provided analytic Riemannian metric.

    Parameters
    ----------
    metric_fn : callable
        Maps ``(..., dim)`` → ``(..., dim, dim)`` returning the metric
        tensor ``G(x)`` in a single coordinate chart.  Must be a
        torch-differentiable expression.
    inverse_fn : callable, optional
        If known, maps ``x`` → ``G(x)^{-1}``.  Otherwise computed with
        ``torch.linalg.inv``.
    sqrt_det_fn : callable, optional
        If known, maps ``x`` → ``sqrt(det G(x))``.  Otherwise computed.
    derivative_fn : callable, optional
        Maps ``x`` (requires_grad=True) → ``∂g_ij/∂x_k`` of shape
        ``(..., dim, dim, dim)``.  If omitted, evaluated via autograd.

    All callables take ``x : torch.Tensor`` with shape ``(..., dim)``.
    """

    def __init__(
        self,
        dim: int,
        metric_fn: Callable[[torch.Tensor], torch.Tensor],
        inverse_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        sqrt_det_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        derivative_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        domain_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        canonicalize_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        if dim < 1:
            raise ValueError("metric dimension must be positive")
        self.dim = dim
        self._metric_fn = metric_fn
        self._inverse_fn = inverse_fn
        self._sqrt_det_fn = sqrt_det_fn
        self._derivative_fn = derivative_fn
        self._domain_fn = domain_fn
        self._canonicalize_fn = canonicalize_fn

    def contains(self, x: torch.Tensor) -> torch.Tensor:
        """Return whether coordinates lie in this preset's declared domain."""
        if x.ndim < 1 or x.shape[-1] != self.dim:
            raise ValueError(f"expected points with shape (..., {self.dim})")
        finite = torch.isfinite(x).all(dim=-1)
        if self._domain_fn is None:
            return finite
        mask = self._domain_fn(x)
        if mask.shape != x.shape[:-1]:
            raise ValueError("metric domain predicate must return shape x.shape[:-1]")
        return finite & mask.to(device=x.device, dtype=torch.bool)

    def validate_points(self, x: torch.Tensor) -> None:
        """Raise an actionable error when any coordinate is outside the domain."""
        if not self.contains(x).all():
            raise ValueError("point lies outside the metric coordinate domain")

    def canonicalize(self, x: torch.Tensor) -> torch.Tensor:
        """Return the canonical representative of identified coordinates."""
        self.validate_points(x)
        return x if self._canonicalize_fn is None else self._canonicalize_fn(x)

    def metric(self, x: torch.Tensor) -> torch.Tensor:
        """Return metric tensor ``G(x)`` of shape ``(..., dim, dim)``."""
        self.validate_points(x)
        G = self._metric_fn(x)
        if G.shape[-2:] != (self.dim, self.dim):
            raise ValueError(
                f"metric_fn returned shape {G.shape} but expected (..., {self.dim}, {self.dim})"
            )
        return G

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """Return inverse metric ``G(x)^{-1}``."""
        self.validate_points(x)
        if self._inverse_fn is not None:
            return self._inverse_fn(x)
        return torch.linalg.inv(self.metric(x))

    def sqrt_det(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``sqrt(det G(x))``."""
        self.validate_points(x)
        if self._sqrt_det_fn is not None:
            return self._sqrt_det_fn(x)
        return torch.sqrt(torch.linalg.det(self.metric(x)))

    def log_det(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``log(det G(x))``."""
        return torch.log(torch.linalg.det(self.metric(x)))

    def derivative(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``∂g_ij/∂x_k`` of shape ``(..., dim, dim, dim)``.

        Robust to metrics that do not depend on ``x`` (e.g. constant
        Euclidean metric): returns zeros in that case instead of raising.
        If ``x`` requires gradients, the result remains connected to that
        exact tensor so higher coordinate derivatives are available. For a
        non-differentiable input, a differentiable clone is used only to
        evaluate the numerical derivative value.
        """
        if self._derivative_fn is not None:
            return self._derivative_fn(x)

        x_grad = x if x.requires_grad else x.clone().requires_grad_(True)
        G = self.metric(x_grad)
        dG: list[torch.Tensor] = []
        for i in range(self.dim):
            dG_row: list[torch.Tensor] = []
            for j in range(self.dim):
                component = G[..., i, j]
                if component.requires_grad:
                    (g,) = torch.autograd.grad(
                        component.sum(),
                        x_grad,
                        create_graph=True,
                        retain_graph=True,
                        allow_unused=True,
                    )
                else:
                    g = None
                if g is None:
                    g = torch.zeros_like(x_grad)
                dG_row.append(g)
            dG.append(torch.stack(dG_row, dim=-2))
        return torch.stack(dG, dim=-3)
