"""Analytic metric wrapper — metrics are user-supplied closed-form functions.

This matches Mohamud's paper assumption: the Riemannian manifold
(M, g) is known beforehand.  Only the vector field is learned.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch

from ._utils import batched_jacobian, validate_supported_floating_tensor


class AnalyticMetric:
    """Wrap a user-provided analytic Riemannian metric.

    Parameters
    ----------
    metric_fn : callable
        Maps ``(..., dim)`` → ``(..., dim, dim)`` returning the metric
        tensor ``G(x)`` in a single coordinate chart.  Must be a
        torch-differentiable expression.
    inverse_fn : callable, optional
        If known, maps ``x`` → ``G(x)^{-1}``.  Otherwise computed by solving
        against the identity. Downstream inverse actions should use
        :meth:`solve` instead of materializing this matrix.
    sqrt_det_fn : callable, optional
        If known, maps ``x`` → ``sqrt(det G(x))``.  Otherwise computed.
    derivative_fn : callable, optional
        Maps ``x`` (requires_grad=True) → ``∂g_ij/∂x_k`` of shape
        ``(..., dim, dim, dim)``.  If omitted, evaluated via autograd.
    log_volume_gradient_fn : callable, optional
        Maps ``x`` to ``partial_i log sqrt(det g)`` with shape ``(..., dim)``.
        Exact divergence uses this closed form when supplied.

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
        debug_validation: bool = False,
        log_volume_gradient_fn: Optional[
            Callable[[torch.Tensor], torch.Tensor]
        ] = None,
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
        self.debug_validation = debug_validation
        self._log_volume_gradient_fn = log_volume_gradient_fn

    def _validate_callback_output(
        self,
        name: str,
        value: torch.Tensor,
        x: torch.Tensor,
        expected_shape: torch.Size,
        *,
        symmetric: bool = False,
    ) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name}: expected a torch.Tensor result")
        if value.shape != expected_shape:
            raise ValueError(
                f"{name}: expected shape {tuple(expected_shape)}; "
                f"got {tuple(value.shape)}"
            )
        if value.device != x.device or value.dtype != x.dtype:
            raise ValueError(
                f"{name}: expected device={x.device}, dtype={x.dtype}; "
                f"got device={value.device}, dtype={value.dtype}"
            )
        validate_supported_floating_tensor(value, name)
        if self.debug_validation:
            if not torch.isfinite(value).all():
                raise ValueError(f"{name}: result must be finite")
            if symmetric and not torch.allclose(value, value.transpose(-1, -2)):
                raise ValueError(f"{name}: result must be symmetric")
        return value

    def contains(self, x: torch.Tensor) -> torch.Tensor:
        """Return whether coordinates lie in this preset's declared domain."""
        if x.ndim < 1 or x.shape[-1] != self.dim:
            raise ValueError(f"expected points with shape (..., {self.dim})")
        validate_supported_floating_tensor(x, "AnalyticMetric.contains")
        finite = torch.isfinite(x).all(dim=-1)
        if self._domain_fn is None:
            return finite
        mask = self._domain_fn(x)
        if mask.shape != x.shape[:-1]:
            raise ValueError("metric domain predicate must return shape x.shape[:-1]")
        if mask.device != x.device or mask.dtype != torch.bool:
            raise ValueError(
                "metric domain predicate: expected "
                f"device={x.device}, dtype=torch.bool; got "
                f"device={mask.device}, dtype={mask.dtype}"
            )
        return finite & mask

    def validate_points(self, x: torch.Tensor) -> None:
        """Raise an actionable error when any coordinate is outside the domain."""
        if not self.contains(x).all():
            raise ValueError("point lies outside the metric coordinate domain")

    def canonicalize(self, x: torch.Tensor) -> torch.Tensor:
        """Return the canonical representative of identified coordinates."""
        self.validate_points(x)
        return self._canonicalize_unchecked(x)

    def _canonicalize_unchecked(self, x: torch.Tensor) -> torch.Tensor:
        """Canonicalize coordinates already validated by a solver stage."""
        if self._canonicalize_fn is None:
            return x
        return self._validate_callback_output(
            "canonicalize_fn", self._canonicalize_fn(x), x, x.shape
        )

    def metric(self, x: torch.Tensor) -> torch.Tensor:
        """Return metric tensor ``G(x)`` of shape ``(..., dim, dim)``."""
        self.validate_points(x)
        return self._metric_unchecked(x)

    def _metric_unchecked(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the metric at coordinates validated by the caller."""
        G = self._metric_fn(x)
        G = self._validate_callback_output(
            "metric_fn",
            G,
            x,
            x.shape[:-1] + (self.dim, self.dim),
            symmetric=True,
        )
        if self.debug_validation:
            _, info = torch.linalg.cholesky_ex(G)
            failed = info != 0
            if failed.any():
                flat_index = failed.reshape(-1).nonzero()[0, 0].item()
                point = x.reshape(-1, self.dim)[flat_index].detach().cpu().tolist()
                raise ValueError(
                    "metric_fn: metric must be symmetric positive-definite; "
                    f"failed at batch index {flat_index}, point={point}"
                )
        return G

    def solve(self, x: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        """Return the solution of ``G(x) y = rhs`` using SPD factorization."""
        G = self.metric(x)
        vector_rhs = rhs.ndim == x.ndim
        expected = x.shape if vector_rhs else x.shape[:-1] + (self.dim, rhs.shape[-1])
        if rhs.shape != expected:
            raise ValueError(
                f"metric solve: expected rhs shape {tuple(expected)}; "
                f"got {tuple(rhs.shape)}"
            )
        if rhs.device != x.device or rhs.dtype != x.dtype:
            raise ValueError(
                f"metric solve: expected device={x.device}, dtype={x.dtype}; "
                f"got device={rhs.device}, dtype={rhs.dtype}"
            )
        factor = torch.linalg.cholesky(G)
        solution = torch.cholesky_solve(
            rhs.unsqueeze(-1) if vector_rhs else rhs, factor
        )
        return solution.squeeze(-1) if vector_rhs else solution

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """Return inverse metric ``G(x)^{-1}``."""
        self.validate_points(x)
        if self._inverse_fn is not None:
            return self._validate_callback_output(
                "inverse_fn",
                self._inverse_fn(x),
                x,
                x.shape[:-1] + (self.dim, self.dim),
                symmetric=True,
            )
        identity = torch.eye(self.dim, device=x.device, dtype=x.dtype)
        identity = identity.expand(x.shape[:-1] + (self.dim, self.dim))
        return self.solve(x, identity)

    def sqrt_det(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``sqrt(det G(x))``."""
        self.validate_points(x)
        return self._sqrt_det_unchecked(x)

    def _sqrt_det_unchecked(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the volume factor at coordinates validated by the caller."""
        if self._sqrt_det_fn is not None:
            return self._validate_callback_output(
                "sqrt_det_fn", self._sqrt_det_fn(x), x, x.shape[:-1]
            )
        return torch.exp(0.5 * self._log_det_unchecked(x))

    @property
    def has_log_volume_gradient(self) -> bool:
        """Whether an exact analytic ``d log sqrt(|g|)`` callback is available."""
        return self._log_volume_gradient_fn is not None

    def log_volume_gradient(self, x: torch.Tensor) -> torch.Tensor:
        """Return coordinate derivatives of ``log sqrt(|g|)``."""
        self.validate_points(x)
        return self._log_volume_gradient_unchecked(x)

    def _log_volume_gradient_unchecked(self, x: torch.Tensor) -> torch.Tensor:
        if self._log_volume_gradient_fn is None:
            raise RuntimeError("no analytic log-volume gradient is available")
        return self._validate_callback_output(
            "log_volume_gradient_fn",
            self._log_volume_gradient_fn(x),
            x,
            x.shape,
        )

    def log_det(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``log(det G(x))``."""
        self.validate_points(x)
        return self._log_det_unchecked(x)

    def _log_det_unchecked(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the log determinant after caller-side point validation."""
        if self._sqrt_det_fn is not None:
            sqrt_det = self._validate_callback_output(
                "sqrt_det_fn", self._sqrt_det_fn(x), x, x.shape[:-1]
            )
            if self.debug_validation and (sqrt_det <= 0).any():
                raise ValueError("sqrt_det_fn: result must be positive")
            return 2.0 * torch.log(sqrt_det)
        factor = torch.linalg.cholesky(self._metric_unchecked(x))
        return 2.0 * torch.log(factor.diagonal(dim1=-2, dim2=-1)).sum(-1)

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
            return self._validate_callback_output(
                "derivative_fn",
                self._derivative_fn(x),
                x,
                x.shape[:-1] + (self.dim, self.dim, self.dim),
            )

        self.validate_points(x)

        def flattened_metric(point: torch.Tensor) -> torch.Tensor:
            return self._metric_fn(point).reshape(
                point.shape[:-1] + (self.dim * self.dim,)
            )

        derivative = batched_jacobian(flattened_metric, x)
        return derivative.reshape(x.shape[:-1] + (self.dim, self.dim, self.dim))
