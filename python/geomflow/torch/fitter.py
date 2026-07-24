"""High-level ManifoldCNF fitter for Continuous Normalizing Flows on Riemannian manifolds.

Provides an intuitive, scikit-learn/PyTorch-style API for fitting density
models on manifolds using Mohamud's intrinsic Riemannian CNF formulation.
"""

from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn as nn

from .analytic_metric import AnalyticMetric
from .atlas import Atlas
from .base_distribution import (
    AtlasBaseDistribution,
    BaseDistribution,
    StandardNormalCoordinateBase,
    validate_base_distribution,
)
from .integrator import integrate_rk4
from .multichart import MultiChartVectorField, overlap_consistency_loss
from .multichart_integrator import cnf_nll_multichart, integrate_multichart
from .vector_field import ManifoldVectorField, lipschitz_regularizer, weight_decay_loss


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
    ):
        super().__init__()
        self.manifold = manifold
        self.is_multichart = isinstance(manifold, Atlas)
        self.dt = dt

        if self.is_multichart:
            self.atlas: Atlas = manifold
            self.vf = MultiChartVectorField(self.atlas, hidden_dim, n_layers)
            self.dim = self.atlas.charts[next(iter(self.atlas.charts))].dim
            self.base_distribution = base_distribution or AtlasBaseDistribution(
                StandardNormalCoordinateBase(self.dim), self.atlas.reference_chart_id
            )
            if not isinstance(self.base_distribution, AtlasBaseDistribution):
                raise TypeError("an atlas requires AtlasBaseDistribution")
            if self.base_distribution.reference_chart_id != self.atlas.reference_chart_id:
                raise ValueError("base reference chart does not match atlas reference chart")
        else:
            self.metric: AnalyticMetric = manifold
            self.dim = self.metric.dim
            self.vf = ManifoldVectorField(self.dim, hidden_dim, n_layers)
            self.base_distribution = base_distribution or getattr(
                self.metric,
                "default_base_distribution",
                StandardNormalCoordinateBase(self.dim),
            )
            validate_base_distribution(self.base_distribution, self.dim)

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
        if self.is_multichart:
            res = integrate_multichart(
                self.vf,
                self.atlas,
                x_data,
                start_chart=start_chart,
                t0=1.0,
                t1=0.0,
                dt=self.dt,
                compute_divergence=True,
            )
            return self.base_distribution.log_prob_volume(
                res.x_final, self.atlas, res.chart_final
            ) + res.divergence_integral
        else:
            res = integrate_rk4(
                self.vf,
                self.metric,
                x_data,
                t0=1.0,
                t1=0.0,
                dt=self.dt,
            )
            return self.base_distribution.log_prob_volume(
                res.x_final, self.metric
            ) + res.divergence_integral

    def sample(
        self,
        n_samples: int,
        start_chart: Optional[int] = None,
        device: Optional[torch.device] = None,
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
        if device is None:
            device = next(self.parameters()).device

        dtype = next(self.parameters()).dtype
        x0 = self.base_distribution.sample(
            (n_samples,), device=device, dtype=dtype
        )
        if not self.base_distribution.contains(x0).all():
            raise RuntimeError("base sampler produced a point outside its declared support")

        if self.is_multichart:
            cid = start_chart if start_chart is not None else self.atlas.reference_chart_id
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
            Lipschitz regularization weight.
        weight_decay_weight : float
            L2 weight decay regularization weight.
        overlap_weight : float
            Overlap consistency loss weight (for multi-chart atlas).
        start_chart : int
            Chart ID for x_data (for multi-chart atlas).
        verbose : bool
            Print training progress loss.

        Returns
        -------
        loss_history : list of float
        """
        optimizer = torch.optim.Adam(self.vf.parameters(), lr=lr)
        N = x_data.shape[0]
        loss_history: list[float] = []

        for epoch in range(epochs):
            perm = torch.randperm(N, device=x_data.device)
            epoch_loss = 0.0
            num_batches = 0

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
                    if overlap_weight > 0.0 and len(self.atlas.charts) > 1:
                        # Compute overlap loss across connected chart pairs
                        for cid_a, chart_a in self.atlas.charts.items():
                            for cid_b in chart_a.transitions:
                                t_rand = torch.rand(x_batch.shape[0], device=x_batch.device)
                                overlap = overlap_consistency_loss(
                                    self.vf,
                                    self.atlas,
                                    x_batch,
                                    chart_alpha=cid_a,
                                    chart_beta=cid_b,
                                    t=t_rand,
                                )
                                loss = loss + overlap_weight * overlap
                else:
                    log_p = self.log_prob(x_batch)
                    nll = -log_p.mean()
                    loss = nll
                    if lipschitz_weight > 0.0:
                        loss = loss + lipschitz_weight * lipschitz_regularizer(
                            self.vf, x_batch, t=0.5
                        )
                    if weight_decay_weight > 0.0:
                        loss = loss + weight_decay_weight * weight_decay_loss(self.vf)

                loss.backward()
                nn.utils.clip_grad_norm_(self.vf.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / max(num_batches, 1)
            loss_history.append(avg_loss)

            if verbose and (epoch % max(1, epochs // 5) == 0 or epoch == epochs - 1):
                print(f"[Epoch {epoch:3d}/{epochs:3d}] Loss: {avg_loss:.4f}")

        return loss_history
