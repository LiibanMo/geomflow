"""Multi-chart validation on S^2 using stereographic coordinates."""

from __future__ import annotations

import torch

from geomflow.torch import (
    AnalyticMetric,
    Atlas,
    Chart,
    MultiChartVectorField,
    EuclideanSpace,
    batched_jacobian,
    cnf_nll_multichart,
    divergence,
    integrate_multichart,
    overlap_consistency_loss,
    pushforward_vector,
    transform_metric,
)


class _CompatibleLinearField(torch.nn.Module):
    def forward(
        self, t: torch.Tensor, x: torch.Tensor, chart_id: int
    ) -> torch.Tensor:
        del t, chart_id
        return 0.5 * x


class _ThresholdChart:
    def __init__(self, upper_bound: float | None) -> None:
        self.analytic_metric = EuclideanSpace(1)
        self.upper_bound = upper_bound

    def is_inside(self, x: torch.Tensor) -> torch.Tensor:
        if self.upper_bound is None:
            return torch.ones(x.shape[:-1], dtype=torch.bool, device=x.device)
        return x[..., 0] <= self.upper_bound


class _IdentityTransitionAtlas:
    reference_chart_id = 0

    def __init__(self) -> None:
        self.charts = {0: _ThresholdChart(0.11), 1: _ThresholdChart(None)}

    def __getitem__(self, chart_id: int) -> _ThresholdChart:
        return self.charts[chart_id]

    def best_chart(
        self, x: torch.Tensor, current: int
    ) -> tuple[int, torch.Tensor]:
        assert current == 0
        return 1, x


def _stereographic_metric():
    def metric_fn(x: torch.Tensor) -> torch.Tensor:
        r2 = (x * x).sum(dim=-1, keepdim=True)
        lam = 4.0 / ((1.0 + r2) ** 2)
        eye = torch.eye(2, device=x.device, dtype=x.dtype)
        return lam.unsqueeze(-1) * eye

    def sqrt_det_fn(x: torch.Tensor) -> torch.Tensor:
        r2 = (x * x).sum(dim=-1)
        return 4.0 / ((1.0 + r2) ** 2)

    return AnalyticMetric(2, metric_fn, sqrt_det_fn=sqrt_det_fn)


# Chart A: stereographic from North pole; Chart B: from South pole.
# A -> B : inversion (u, v) -> (u, v) / (u^2 + v^2)
def _transition_A_to_B(x: torch.Tensor) -> torch.Tensor:
    r2 = (x * x).sum(dim=-1, keepdim=True)
    return x / r2.clamp_min(1e-8)


def _transition_B_to_A(x: torch.Tensor) -> torch.Tensor:
    return _transition_A_to_B(x)


def _jacobian_A_to_B(x: torch.Tensor) -> torch.Tensor:
    x = x.detach().requires_grad_(True)
    return batched_jacobian(_transition_A_to_B, x)


def _make_atlas(hidden_dim: int = 32, n_samples: int = 300, seed: int = 0):
    torch.manual_seed(seed)
    metric_a = _stereographic_metric()
    metric_b = _stereographic_metric()
    atlas = Atlas(
        [
            Chart(
                0, 2, torch.randn(n_samples, 2) * 1.5,
                metric_a, transitions={1: _transition_A_to_B},
            ),
            Chart(
                1, 2, torch.randn(n_samples, 2) * 1.5,
                metric_b, transitions={0: _transition_B_to_A},
            ),
        ],
        reference_chart_id=0,
    )
    vf = MultiChartVectorField(atlas, hidden_dim=hidden_dim)
    return atlas, vf


def test_metric_transform_invariant():
    metric_a = _stereographic_metric()
    metric_b = _stereographic_metric()

    x_a = torch.randn(16, 2) * 0.8 + 1.5  # away from origin
    x_a.requires_grad_(True)
    J = _jacobian_A_to_B(x_a)

    G_a = metric_a.metric(x_a)
    G_b_pred = transform_metric(G_a, J)
    x_b = _transition_A_to_B(x_a).detach().requires_grad_(True)
    G_b_true = metric_b.metric(x_b)

    rel_err = (G_b_pred - G_b_true).norm() / G_b_true.norm().clamp_min(1e-8)
    print("  metric transform rel err:", rel_err.item())
    assert rel_err.item() < 1e-4

def _jacobian_A_to_B_analytic(x: torch.Tensor) -> torch.Tensor:
    r2 = (x * x).sum(dim=-1, keepdim=True).unsqueeze(-1)
    eye = torch.eye(2, device=x.device, dtype=x.dtype)
    outer = torch.einsum("...i,...j->...ij", x, x)
    return (eye / r2) - (2.0 * outer / (r2**2))


def test_divergence_invariant():
    torch.manual_seed(1)
    metric_a = _stereographic_metric()
    metric_b = _stereographic_metric()

    def vf_a(x_: torch.Tensor) -> torch.Tensor:
        return torch.stack([x_[..., 1], -x_[..., 0]], dim=-1) + 0.1 * x_

    def vf_b(x_b_: torch.Tensor) -> torch.Tensor:
        x_a_ = _transition_B_to_A(x_b_)
        J_ba = _jacobian_A_to_B_analytic(x_a_)
        f_a_val = vf_a(x_a_)
        return pushforward_vector(f_a_val, J_ba)

    x_a = torch.randn(16, 2) * 0.8 + 1.5
    x_a.requires_grad_(True)

    div_a = divergence(vf_a, x_a, metric_a)

    x_b = _transition_A_to_B(x_a).detach().requires_grad_(True)
    div_b = divergence(vf_b, x_b, metric_b)

    max_diff = (div_a - div_b).abs().max().item()
    print("  divergence max diff:", max_diff)
    assert max_diff < 1e-3


def test_multichart_integrator_density_independent_of_start_chart():
    atlas, vf = _make_atlas(hidden_dim=32, seed=2)
    # Zero out vector field so both chart heads represent the same zero vector field.
    for p in vf.parameters():
        torch.nn.init.zeros_(p)

    x0_a = torch.randn(8, 2) * 0.4 + 1.2  # a point well inside chart 0's ball
    x0_b = _transition_A_to_B(x0_a)  # the same manifold points, in chart 1

    result_from_a = integrate_multichart(
        vf, atlas, x0_a, start_chart=0, t0=1.0, t1=0.0, dt=0.05
    )
    result_from_b = integrate_multichart(
        vf, atlas, x0_b, start_chart=1, t0=1.0, t1=0.0, dt=0.05
    )

    # Map both results into the reference chart (chart 0) for comparison.
    x_final_a = result_from_a.x_final
    if result_from_a.chart_final != atlas.reference_chart_id:
        x_final_a = atlas[result_from_a.chart_final].transition_to(
            atlas.reference_chart_id, x_final_a
        )

    x_final_b = result_from_b.x_final
    if result_from_b.chart_final != atlas.reference_chart_id:
        x_final_b = atlas[result_from_b.chart_final].transition_to(
            atlas.reference_chart_id, x_final_b
        )

    x_diff = (x_final_a - x_final_b).abs().max().item()
    log_det_diff = (
        result_from_a.divergence_integral - result_from_b.divergence_integral
    ).abs().max().item()
    print("  final-x max diff:", x_diff, " log_det max diff:", log_det_diff)
    assert x_diff < 1e-2
    assert log_det_diff < 1e-2


def test_nonzero_divergence_is_scalar_across_identity_transition():
    """MATH-422/MATH-424/MATH-426: density has no chart Jacobian jump."""
    atlas = _IdentityTransitionAtlas()
    field = _CompatibleLinearField()
    x0 = torch.tensor([[0.1]], dtype=torch.float64)

    switched = integrate_multichart(
        field, atlas, x0, start_chart=0, t0=0.0, t1=0.4, dt=0.2,
        track_trajectory=True,
    )
    target_only = integrate_multichart(
        field, atlas, x0, start_chart=1, t0=0.0, t1=0.4, dt=0.2,
    )

    torch.testing.assert_close(switched.x_final, target_only.x_final)
    torch.testing.assert_close(
        switched.divergence_integral, target_only.divergence_integral
    )
    torch.testing.assert_close(
        switched.divergence_integral, torch.tensor([0.2], dtype=torch.float64)
    )
    assert switched.chart_final == 1
    assert len(switched.transition_events) == 1
    assert switched.transition_events[0].time == 0.2
    assert switched.transition_events[0].source_chart == 0
    assert switched.transition_events[0].target_chart == 1


def test_multichart_sphere_training():
    torch.manual_seed(3)
    atlas, vf = _make_atlas(hidden_dim=32, seed=3)

    # Target: a tight Gaussian cluster in chart-0 coordinates near (1.2, 1.2),
    # matching the chart's sample distribution.
    target = torch.randn(64, 2) * 0.2 + torch.tensor([1.2, 1.2])

    opt = torch.optim.Adam(vf.parameters(), lr=0.01)
    initial_nll = cnf_nll_multichart(vf, atlas, target[:16], start_chart=0, dt=0.1).item()

    for step in range(80):
        opt.zero_grad()
        nll = cnf_nll_multichart(vf, atlas, target, start_chart=0, dt=0.1)
        overlap = overlap_consistency_loss(
            vf, atlas, target, chart_alpha=0, chart_beta=1,
            t=torch.zeros(target.shape[0]),
        )
        loss = nll + 0.01 * overlap
        loss.backward()
        torch.nn.utils.clip_grad_norm_(vf.parameters(), 1.0)
        opt.step()
        if step % 20 == 0:
            print(f"  [train] step {step:3d} NLL={nll.item():.3f}")

    final_nll = cnf_nll_multichart(vf, atlas, target[:16], start_chart=0, dt=0.1).item()
    print(f"  initial NLL: {initial_nll:.3f}  final NLL: {final_nll:.3f}")
    assert final_nll < initial_nll - 0.1


if __name__ == "__main__":
    print("=== Multi-chart S^2 smoke tests ===\n")
    test_metric_transform_invariant()
    test_divergence_invariant()
    test_multichart_integrator_density_independent_of_start_chart()
    test_multichart_sphere_training()
    print("\n✅ Multi-chart sphere tests passed!")
