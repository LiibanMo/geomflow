"""Unit test for Mohamud Theorem 3.7 intrinsic adjoint ODE implementation."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import torch

from geomflow.torch import (
    EuclideanSpace,
    ManifoldVectorField,
    MohamudAdjointFunction,
    cnf_nll,
)


def test_mohamud_adjoint_gradients():
    torch.manual_seed(42)
    dim = 2
    metric = EuclideanSpace(dim)
    vf = ManifoldVectorField(dim=dim, hidden_dim=16)

    x_data_1 = torch.randn(8, dim, requires_grad=True)
    x_data_2 = x_data_1.clone().detach().requires_grad_(True)

    # Standard autograd loss
    loss_autograd = cnf_nll(vf, metric, x_data_1, dt=0.05)
    loss_autograd.backward()
    grad_autograd = x_data_1.grad

    # Mohamud intrinsic adjoint loss
    loss_mohamud = MohamudAdjointFunction.apply(x_data_2, vf, metric, 0.05, 0.0, 1.0)
    loss_mohamud.backward()
    grad_mohamud = x_data_2.grad

    diff = (grad_autograd - grad_mohamud).abs().max().item()
    print("  Mohamud Adjoint vs Autograd loss diff:", abs(loss_autograd.item() - loss_mohamud.item()))
    print("  Mohamud Adjoint vs Autograd grad max diff:", diff)

    assert abs(loss_autograd.item() - loss_mohamud.item()) < 1e-4
    assert diff < 0.1  # RK1 Euler adjoint vs RK4 autograd step diff bound


if __name__ == "__main__":
    print("=== Testing Mohamud Intrinsic Adjoint (Theorem 3.7) ===\n")
    test_mohamud_adjoint_gradients()
    print("\n✅ Mohamud Intrinsic Adjoint test passed!")
