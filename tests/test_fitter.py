"""Smoke tests for geomflow.torch ManifoldCNF fitter and built-in manifold presets."""

from __future__ import annotations

import torch

from geomflow.torch import (
    EuclideanSpace,
    HyperbolicSpace,
    InducedMetric,
    ManifoldCNF,
    PoincareDisk,
    Sphere2DAtlas,
    SphereStereographicMetric,
    Torus2D,
)


def test_single_chart_fitter_euclidean():
    torch.manual_seed(42)
    metric = EuclideanSpace(2)
    cnf = ManifoldCNF(metric, hidden_dim=16, dt=0.1)

    # Generated data cluster near (0.5, 0.5)
    data = torch.randn(32, 2) * 0.3 + 0.5
    initial_log_p = cnf.log_prob(data[:8]).mean().item()

    losses = cnf.fit(data, epochs=30, batch_size=16, lr=0.01, verbose=False)
    final_log_p = cnf.log_prob(data[:8]).mean().item()

    print(f"  Euclidean CNF initial log_p: {initial_log_p:.3f}, final log_p: {final_log_p:.3f}")
    assert final_log_p > initial_log_p + 0.1
    assert len(losses) == 30

    samples, _ = cnf.sample(10)
    assert samples.shape == (10, 2)


def test_torus_preset():
    metric = Torus2D(R=2.0, r=1.0)
    x = torch.tensor([[0.0, 0.0], [0.5, 0.5]], requires_grad=True)
    G = metric.metric(x)
    assert G.shape == (2, 2, 2)
    torch.testing.assert_close(G[..., 0, 0], (2.0 + torch.cos(x[..., 1])) ** 2)
    torch.testing.assert_close(G[..., 1, 1], torch.ones(2))
    sqrtg = metric.sqrt_det(x)
    torch.testing.assert_close(sqrtg, 2.0 + torch.cos(x[..., 1]))

    cnf = ManifoldCNF(metric, hidden_dim=16, dt=0.1)
    log_p = cnf.log_prob(x)
    assert log_p.shape == (2,)


def test_poincare_disk_preset():
    metric = PoincareDisk(2)
    x = torch.tensor([[0.1, 0.2], [-0.3, 0.4]])
    G = metric.metric(x)
    assert G.shape == (2, 2, 2)
    conformal_factor = 4.0 / (1.0 - x.square().sum(dim=-1)).square()
    torch.testing.assert_close(G[..., 0, 0], conformal_factor)
    torch.testing.assert_close(G[..., 1, 1], conformal_factor)

    cnf = ManifoldCNF(metric, hidden_dim=16, dt=0.1)
    for parameter in cnf.vf.parameters():
        torch.nn.init.zeros_(parameter)
    samples, _ = cnf.sample(5)
    assert samples.shape == (5, 2)
    assert metric.contains(samples).all()


def test_induced_metric():
    # Surface z = x^2 + y^2 immersion into R^3
    def immersion(x: torch.Tensor) -> torch.Tensor:
        z = (x * x).sum(dim=-1, keepdim=True)
        return torch.cat([x, z], dim=-1)

    metric = InducedMetric(2, immersion)
    x = torch.randn(4, 2, requires_grad=True)
    G = metric.metric(x)
    assert G.shape == (4, 2, 2)


def test_multichart_fitter_sphere():
    torch.manual_seed(42)
    atlas = Sphere2DAtlas(n_samples=200, seed=42)
    cnf = ManifoldCNF(atlas, hidden_dim=16, dt=0.1)

    data = torch.randn(32, 2) * 0.2 + torch.tensor([1.0, 1.0])
    initial_log_p = cnf.log_prob(data[:8], start_chart=0).mean().item()

    losses = cnf.fit(data, epochs=20, batch_size=16, lr=0.01, verbose=False)
    final_log_p = cnf.log_prob(data[:8], start_chart=0).mean().item()

    print(f"  Sphere Atlas initial log_p: {initial_log_p:.3f}, final log_p: {final_log_p:.3f}")
    assert final_log_p > initial_log_p

    samples, final_chart = cnf.sample(8, start_chart=0)
    assert samples.shape == (8, 2)


def test_fit_records_global_field_compatibility_diagnostics():
    model = ManifoldCNF(Sphere2DAtlas(), hidden_dim=4, n_layers=1, dt=0.25)
    data = torch.tensor([[0.5, 0.1], [0.8, -0.2]])
    history = model.fit(
        data,
        epochs=1,
        batch_size=2,
        lipschitz_weight=0.0,
        weight_decay_weight=0.0,
        overlap_weight=0.01,
        verbose=False,
    )
    assert len(history) == len(model.fit_diagnostics) == 1
    assert model.fit_diagnostics[0]["overlap_residual"] >= 0.0


if __name__ == "__main__":
    print("=== Testing ManifoldCNF Fitter & Presets ===\n")
    test_single_chart_fitter_euclidean()
    test_torus_preset()
    test_poincare_disk_preset()
    test_induced_metric()
    test_multichart_fitter_sphere()
    print("\n✅ All ManifoldCNF fitter tests passed!")
