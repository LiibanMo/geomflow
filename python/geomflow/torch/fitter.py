"""High-level ManifoldCNF fitter for Continuous Normalizing Flows on Riemannian manifolds.

Provides an intuitive, scikit-learn/PyTorch-style API for fitting density
models on manifolds using Mohamud's intrinsic Riemannian CNF formulation.
"""

from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn as nn

from .analytic_metric import AnalyticMetric
from ._utils import (
    module_device_dtype,
    validate_generator_device,
    validate_points,
    validate_tensor_module_compatibility,
)
from .atlas import Atlas
from .base_distribution import (
    AtlasBaseDistribution,
    BaseDistribution,
    StandardNormalCoordinateBase,
    validate_base_distribution,
)
from .adjoint import cnf_log_prob, intrinsic_adjoint_nll
from .integrator import integrate_rk4
from .multichart import MultiChartVectorField, overlap_consistency_loss
from .multichart_integrator import (
    cnf_log_prob_multichart,
    cnf_nll_multichart,
    integrate_multichart,
)
from .vector_field import (
    ManifoldVectorField,
    coordinate_jacobian_regularizer,
    weight_decay_loss,
)


class ManifoldCNF(nn.Module):
    """General Manifold-Constraint CNF Fitter using intrinsic Riemannian geometry.

    Parameters
    ----------
    manifold : AnalyticMetric or Atlas
        The manifold geometry specified as either a single-chart :class:`AnalyticMetric`
        or a multi-chart :class:`Atlas`.
    hidden_dim : int
        Hidden dimension of neural vector field MLPs. Default: 64.
    n_layers : int
        Number of hidden layers in neural vector field MLPs. Default: 2.
    dt : float
        ODE integration step size. Default: 0.05.
    """

    def __init__(
        self,
        manifold: Union[AnalyticMetric, Atlas],
        hidden_dim: int = 64,
        n_layers: int = 2,
        dt: float = 0.05,
        base_distribution: BaseDistribution | AtlasBaseDistribution | None = None,
        activation: type[nn.Module] = nn.SiLU,
    ):
        super().__init__()
        self.manifold = manifold
        self.is_multichart = isinstance(manifold, Atlas)
        self.dt = dt

        if self.is_multichart:
            self.atlas: Atlas = manifold
            self.vf = MultiChartVectorField(
                self.atlas, hidden_dim, n_layers, activation=activation
            )
            self.dim = self.atlas.charts[next(iter(self.atlas.charts))].dim
            self.base_distribution = base_distribution or AtlasBaseDistribution(
                StandardNormalCoordinateBase(self.dim), self.atlas.reference_chart_id
            )
            if not isinstance(self.base_distribution, AtlasBaseDistribution):
                raise TypeError("an atlas requires AtlasBaseDistribution")
            if self.base_distribution.reference_chart_id != self.atlas.reference_chart_id:
                raise ValueError("base reference chart does not match atlas reference chart")
            for chart_id, chart in self.atlas.charts.items():
                if chart.samples is not None:
                    self.register_buffer(f"_atlas_samples_{chart_id}", chart.samples)
        else:
            self.metric: AnalyticMetric = manifold
            self.dim = self.metric.dim
            self.vf = ManifoldVectorField(
                self.dim,
                hidden_dim,
                n_layers,
                activation=activation,
                periodic=getattr(self.metric, "coordinate_topology", None)
                == "angles modulo 2*pi",
            )
            self.base_distribution = base_distribution or getattr(
                self.metric,
                "default_base_distribution",
                StandardNormalCoordinateBase(self.dim),
            )
            validate_base_distribution(self.base_distribution, self.dim)
        self.fit_diagnostics: list[dict[str, float]] = []

    def _sync_atlas_samples(self) -> None:
        if not self.is_multichart:
            return
        for chart_id, chart in self.atlas.charts.items():
            name = f"_atlas_samples_{chart_id}"
            if name in self._buffers:
                chart.set_samples(self._buffers[name])

    def _apply(self, fn, recurse: bool = True):
        result = super()._apply(fn, recurse)
        self._sync_atlas_samples()
        return result

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        self._sync_atlas_samples()
        return result

    def log_prob(
        self,
        x_data: torch.Tensor,
        start_chart: int = 0,
    ) -> torch.Tensor:
        """Evaluate exact log-likelihood log p(x) under the manifold CNF.

        Parameters
        ----------
        x_data : Tensor
            Coordinates of shape ``(batch, dim)``.
        start_chart : int
            Starting chart id (only used if multi-chart atlas).

        Returns
        -------
        log_p : Tensor
            Log-likelihood values of shape ``(batch,)``.
        """
        validate_tensor_module_compatibility(x_data, self, "ManifoldCNF.log_prob")
        validate_points(x_data, self.dim, "ManifoldCNF.log_prob")
        if self.is_multichart:
            if start_chart not in self.atlas.charts:
                raise ValueError(
                    f"ManifoldCNF.log_prob: unknown start chart {start_chart}"
                )
            return cnf_log_prob_multichart(
                self.vf,
                self.atlas,
                x_data,
                start_chart,
                dt=self.dt,
                base_distribution=self.base_distribution,
            )
        return cnf_log_prob(
            self.vf,
            self.metric,
            x_data,
            self.dt,
            base_distribution=self.base_distribution,
        )

    def forward(self, x_data: torch.Tensor, start_chart: int = 0) -> torch.Tensor:
        """Return log likelihoods through the standard ``nn.Module`` boundary.

        DistributedDataParallel users should call the wrapped module rather
        than invoking :meth:`fit` on ``ddp.module``.
        """
        return self.log_prob(x_data, start_chart=start_chart)

    def training_loss(
        self, x_data: torch.Tensor, start_chart: int = 0
    ) -> torch.Tensor:
        """Return the unregularized mean NLL for external optimizer loops."""
        return -self(x_data, start_chart=start_chart).mean()

    def sample(
        self,
        n_samples: int,
        start_chart: Optional[int] = None,
        device: Optional[torch.device | str] = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, Optional[int]]:
        """Sample from the learned manifold CNF by integrating t = 0 -> 1.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate.
        start_chart : int, optional
            Starting chart id for base distribution (defaults to atlas reference_chart_id).
        device : torch.device, optional

        Returns
        -------
        x_samples : Tensor
            Generated samples on the manifold, shape ``(n_samples, dim)``.
        final_chart : int or None
            Final chart ID if multi-chart, else None.
        """
        model_device, dtype = module_device_dtype(self, "ManifoldCNF.sample")
        if device is not None and torch.device(device) != model_device:
            raise ValueError(
                f"ManifoldCNF.sample: expected device={model_device}, "
                f"dtype={dtype}; got requested device={torch.device(device)}, "
                f"dtype={dtype}"
            )
        device = model_device
        if n_samples < 1:
            raise ValueError("ManifoldCNF.sample: n_samples must be positive")
        validate_generator_device(generator, device, "ManifoldCNF.sample")
        x0 = self.base_distribution.sample(
            (n_samples,), device=device, dtype=dtype, generator=generator
        )
        if not self.base_distribution.contains(x0).all():
            raise RuntimeError("base sampler produced a point outside its declared support")

        if self.is_multichart:
            cid = start_chart if start_chart is not None else self.atlas.reference_chart_id
            if cid not in self.atlas.charts:
                raise ValueError(f"ManifoldCNF.sample: unknown start chart {cid}")
            if cid != self.atlas.reference_chart_id:
                try:
                    x0 = self.atlas[self.atlas.reference_chart_id].transition_to(cid, x0)
                except KeyError as error:
                    raise ValueError(
                        "base samples cannot be mapped from the reference chart "
                        f"to requested chart {cid}"
                    ) from error
            res = integrate_multichart(
                self.vf,
                self.atlas,
                x0,
                start_chart=cid,
                t0=0.0,
                t1=1.0,
                dt=self.dt,
                compute_divergence=False,
            )
            return res.x_final, res.chart_final
        else:
            res = integrate_rk4(
                self.vf,
                self.metric,
                x0,
                t0=0.0,
                t1=1.0,
                dt=self.dt,
                compute_divergence=False,
            )
            return res.x_final, None

    def fit(
        self,
        x_data: torch.Tensor,
        epochs: int = 100,
        batch_size: int = 64,
        lr: float = 1e-3,
        lipschitz_weight: float = 1e-3,
        weight_decay_weight: float = 1e-4,
        overlap_weight: float = 0.01,
        start_chart: int = 0,
        verbose: bool = True,
        gradient_mode: str = "direct",
        log_every: int | None = None,
        max_grad_norm: float | None = 1.0,
        error_if_nonfinite: bool = True,
        generator: torch.Generator | None = None,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> list[float]:
        """Fit the manifold CNF to target data using Adam optimizer.

        Parameters
        ----------
        x_data : Tensor
            Target dataset on the manifold, shape ``(N, dim)``.
        epochs : int
            Number of training epochs.
        batch_size : int
            Batch size.
        lr : float
            Learning rate.
        lipschitz_weight : float
            Coordinate-Jacobian engineering regularization weight. This is not
            an intrinsic Lipschitz bound or a theorem assumption.
        weight_decay_weight : float
            L2 weight decay regularization weight.
        overlap_weight : float
            Overlap consistency loss weight (for multi-chart atlas).
        start_chart : int
            Chart ID for x_data (for multi-chart atlas).
        verbose : bool
            Print training progress loss.
        gradient_mode : {"direct", "intrinsic_adjoint"}
            Single-chart gradient implementation. Direct autograd is the
            default; the intrinsic adjoint reduces trajectory memory through
            deterministic replay. Multi-chart training requires ``direct``.

        Returns
        -------
        loss_history : list of float
        """
        if gradient_mode not in {"direct", "intrinsic_adjoint"}:
            raise ValueError(
                "gradient_mode must be 'direct' or 'intrinsic_adjoint'"
            )
        if self.is_multichart and gradient_mode == "intrinsic_adjoint":
            raise ValueError(
                "intrinsic_adjoint is not supported for multi-chart training"
            )
        validate_tensor_module_compatibility(x_data, self, "ManifoldCNF.fit")
        validate_points(x_data, self.dim, "ManifoldCNF.fit", nonempty=True)
        model_device, _ = module_device_dtype(self, "ManifoldCNF.fit")
        validate_generator_device(generator, model_device, "ManifoldCNF.fit")
        if self.is_multichart and start_chart not in self.atlas.charts:
            raise ValueError(f"ManifoldCNF.fit: unknown start chart {start_chart}")
        if epochs < 1 or batch_size < 1:
            raise ValueError("ManifoldCNF.fit: epochs and batch_size must be positive")
        if log_every is not None and log_every < 1:
            raise ValueError("ManifoldCNF.fit: log_every must be positive or None")
        optimizer = optimizer or torch.optim.Adam(self.vf.parameters(), lr=lr)
        N = x_data.shape[0]
        epoch_losses: list[torch.Tensor] = []
        epoch_overlaps: list[torch.Tensor] = []
        self.fit_diagnostics = []

        for epoch in range(epochs):
            perm = torch.randperm(N, device=x_data.device, generator=generator)
            epoch_loss = x_data.new_zeros(())
            num_batches = 0
            epoch_overlap = x_data.new_zeros(())

            for i in range(0, N, batch_size):
                batch_indices = perm[i : i + batch_size]
                x_batch = x_data[batch_indices]

                optimizer.zero_grad()

                if self.is_multichart:
                    nll = cnf_nll_multichart(
                        self.vf,
                        self.atlas,
                        x_batch,
                        start_chart=start_chart,
                        dt=self.dt,
                        base_distribution=self.base_distribution,
                    )
                    loss = nll
                    overlap_penalty = nll.new_zeros(())
                    if overlap_weight > 0.0 and len(self.atlas.charts) > 1:
                        pair_losses = []
                        seen_pairs: set[frozenset[int]] = set()
                        for cid_a, chart_a in self.atlas.charts.items():
                            for cid_b in chart_a.transitions:
                                pair = frozenset((cid_a, cid_b))
                                if pair in seen_pairs:
                                    continue
                                seen_pairs.add(pair)
                                t_rand = torch.rand(
                                    x_batch.shape[0],
                                    device=x_batch.device,
                                    dtype=x_batch.dtype,
                                    generator=generator,
                                )
                                pair_losses.append(
                                    overlap_consistency_loss(
                                        self.vf,
                                        self.atlas,
                                        x_batch,
                                        chart_alpha=cid_a,
                                        chart_beta=cid_b,
                                        t=t_rand,
                                        coordinate_chart=start_chart,
                                    )
                                )
                        if pair_losses:
                            overlap_penalty = torch.stack(pair_losses).mean()
                            loss = loss + overlap_weight * overlap_penalty
                    if lipschitz_weight > 0.0:
                        coordinate_penalties = [
                            coordinate_jacobian_regularizer(
                                self.vf.head(cid), x_batch, t=0.5
                            )
                            for cid in self.atlas.charts
                        ]
                        loss = loss + lipschitz_weight * torch.stack(
                            coordinate_penalties
                        ).mean()
                    if weight_decay_weight > 0.0:
                        loss = loss + weight_decay_weight * weight_decay_loss(self.vf)
                    epoch_overlap += overlap_penalty.detach()
                else:
                    if gradient_mode == "intrinsic_adjoint":
                        nll = intrinsic_adjoint_nll(
                            self.vf,
                            self.metric,
                            x_batch,
                            dt=self.dt,
                            base_distribution=self.base_distribution,
                        )
                    else:
                        log_p = self.log_prob(x_batch)
                        nll = -log_p.mean()
                    loss = nll
                    if lipschitz_weight > 0.0:
                        loss = loss + lipschitz_weight * coordinate_jacobian_regularizer(
                            self.vf, x_batch, t=0.5
                        )
                    if weight_decay_weight > 0.0:
                        loss = loss + weight_decay_weight * weight_decay_loss(self.vf)

                try:
                    loss.backward()
                    if max_grad_norm is not None:
                        nn.utils.clip_grad_norm_(
                            self.vf.parameters(),
                            max_grad_norm,
                            error_if_nonfinite=error_if_nonfinite,
                        )
                    optimizer.step()
                    for parameter, state in optimizer.state.items():
                        for name, value in state.items():
                            if (
                                torch.is_tensor(value)
                                and value.device != parameter.device
                                and not (name == "step" and value.ndim == 0)
                            ):
                                raise RuntimeError(
                                    "ManifoldCNF.fit: optimizer state device mismatch; "
                                    f"parameter is on {parameter.device}, state is on {value.device}"
                                )
                except torch.cuda.OutOfMemoryError as error:
                    raise torch.cuda.OutOfMemoryError(
                        "ManifoldCNF.fit: CUDA out of memory; no batch-size "
                        "change or CPU fallback was attempted"
                    ) from error

                epoch_loss += loss.detach()
                num_batches += 1

            avg_loss_tensor = epoch_loss / num_batches
            avg_overlap_tensor = epoch_overlap / num_batches
            epoch_losses.append(avg_loss_tensor)
            epoch_overlaps.append(avg_overlap_tensor)

            effective_log_every = log_every or max(1, epochs // 5)
            if verbose and (
                epoch % effective_log_every == 0 or epoch == epochs - 1
            ):
                avg_loss = float(avg_loss_tensor.cpu())
                print(f"[Epoch {epoch:3d}/{epochs:3d}] Loss: {avg_loss:.4f}")

        loss_history = torch.stack(epoch_losses).cpu().tolist()
        overlap_history = torch.stack(epoch_overlaps).cpu().tolist()
        self.fit_diagnostics = [
            {"loss": loss, "overlap_residual": overlap}
            for loss, overlap in zip(loss_history, overlap_history)
        ]
        return loss_history
