"""Smoke tests for geomflow.torch with analytic user-supplied metrics."""

from __future__ import annotations

import torch
import torch.nn as nn

from geomflow.torch import (
    AnalyticMetric,
    ManifoldVectorField,
    christoffel,
    cnf_nll,
    divergence,
    integrate_rk4,
    lipschitz_regularizer,
    weight_decay_loss,
)


def _euclidean_metric(dim: int):
    def metric_fn(x: torch.Tensor) -> torch.Tensor:
        eye = torch.eye(dim, device=x.device, dtype=x.dtype)
        return eye.expand(*x.shape[:-1], -1, -1)

    def inverse_fn(x: torch.Tensor) -> torch.Tensor:
        eye = torch.eye(dim, device=x.device, dtype=x.dtype)
        return eye.expand(*x.shape[:-1], -1, -1)

    def sqrt_det_fn(x: torch.Tensor) -> torch.Tensor:
        return torch.ones(x.shape[:-1], device=x.device, dtype=x.dtype)

    return AnalyticMetric(dim, metric_fn, inverse_fn, sqrt_det_fn)


def test_metric_and_christoffel():
    dim = 2
    metric = _euclidean_metric(dim)
    x = torch.randn(8, dim, requires_grad=True)
    G = metric.metric(x)
    assert G.shape == (8, dim, dim)
    assert torch.allclose(G, torch.eye(dim), atol=1e-6)
    Gamma = christoffel(metric, x)
    assert Gamma.abs().max().item() < 1e-5
    print("  metric & Christoffel OK")


def test_divergence_euclidean():
    dim = 2
    metric = _euclidean_metric(dim)

    def vf(x: torch.Tensor) -> torch.Tensor:
        return x

    x = torch.randn(8, dim, requires_grad=True)
    div_val = divergence(vf, x, metric)
    assert torch.allclose(div_val, torch.full_like(div_val, dim), atol=0.05)
    print("  divergence Euclidean OK")


def test_integrator_zero_field():
    dim = 2
    metric = _euclidean_metric(dim)
    vf = ManifoldVectorField(dim=dim, hidden_dim=32)
    for p in vf.parameters():
        nn.init.zeros_(p)

    x0 = torch.randn(4, dim)
    result = integrate_rk4(vf, metric, x0, t0=0.0, t1=1.0, dt=0.05)
    assert torch.allclose(result.x_final, x0, atol=1e-6)
    print("  integrator zero-field OK")


def test_metric_derivative_zero_without_requires_grad():
    """derivative() must not raise even if x has no grad tracking, and must
    be exactly zero for a constant (Euclidean) metric."""
    dim = 3
    metric = _euclidean_metric(dim)
    x = torch.randn(5, dim)  # no requires_grad
    dG = metric.derivative(x)
    assert dG.shape == (5, dim, dim, dim)
    assert torch.all(dG == 0.0)
    print("  metric derivative (no grad) OK")


def test_integrator_batch_shapes():
    """Time-tensor broadcasting must work for single points, 1-D batches,
    and 2-D (multi-axis) batches."""
    dim = 2
    metric = _euclidean_metric(dim)
    vf = ManifoldVectorField(dim=dim, hidden_dim=16)

    for shape in [(dim,), (4, dim), (3, 5, dim)]:
        x0 = torch.randn(*shape)
        result = integrate_rk4(vf, metric, x0, t0=0.0, t1=0.2, dt=0.05)
        assert result.x_final.shape == shape
        assert result.log_det.shape == shape[:-1]
    print("  integrator batch-shape broadcasting OK")


def test_cnf_training():
    torch.manual_seed(42)
    dim = 2
    metric = _euclidean_metric(dim)
    vf = ManifoldVectorField(dim=dim, hidden_dim=32)

    # Target data: Gaussian cluster around (0.5, 0.5)
    data = torch.randn(64, dim) * 0.3 + 0.5

    opt = torch.optim.Adam(vf.parameters(), lr=0.005, weight_decay=1e-4)
    initial_nll = cnf_nll(vf, metric, data[:16], dt=0.1).item()
    print(f"  initial NLL: {initial_nll:.3f}")

    for step in range(80):
        opt.zero_grad()
        nll = cnf_nll(vf, metric, data, dt=0.1)
        lip = lipschitz_regularizer(vf, data, t=0.5)
        loss = nll + 0.01 * lip
        loss.backward()
        opt.step()
        if step % 25 == 0:
            print(f"  [train] step {step:3d} NLL={nll.item():.3f}")

    final_nll = cnf_nll(vf, metric, data[:16], dt=0.1).item()
    print(f"  final NLL: {final_nll:.3f}")
    assert final_nll < initial_nll - 0.1
    print("  CNF training OK")


def test_cnf_training_with_regularizers():
    """cnf_nll's built-in lipschitz_weight / weight_decay_weight path runs
    and still decreases NLL."""
    torch.manual_seed(7)
    dim = 2
    metric = _euclidean_metric(dim)
    vf = ManifoldVectorField(dim=dim, hidden_dim=32)
    data = torch.randn(64, dim) * 0.3 + 0.5

    opt = torch.optim.Adam(vf.parameters(), lr=0.005)
    initial = cnf_nll(
        vf, metric, data[:16], dt=0.1,
        lipschitz_weight=1e-3, weight_decay_weight=1e-4,
    ).item()

    for _ in range(60):
        opt.zero_grad()
        loss = cnf_nll(
            vf, metric, data, dt=0.1,
            lipschitz_weight=1e-3, weight_decay_weight=1e-4,
        )
        loss.backward()
        opt.step()

    final = cnf_nll(
        vf, metric, data[:16], dt=0.1,
        lipschitz_weight=1e-3, weight_decay_weight=1e-4,
    ).item()
    assert final < initial
    assert weight_decay_loss(vf).item() > 0.0
    print("  CNF training with regularizers OK")


if __name__ == "__main__":
    print("=== geomflow.torch analytic metric smoke tests ===\n")
    test_metric_and_christoffel()
    test_metric_derivative_zero_without_requires_grad()
    test_divergence_euclidean()
    test_integrator_zero_field()
    test_integrator_batch_shapes()
    test_cnf_training()
    test_cnf_training_with_regularizers()
    print("\n✅ All smoke tests passed!")
