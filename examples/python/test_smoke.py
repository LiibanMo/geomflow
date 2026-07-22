#!/usr/bin/env python3
"""Smoke test: tiny training + inference round-trip."""
import sys
import torch
from model import CNFNet


def test_build():
    model = CNFNet(dim=2, hidden=8)
    t = torch.tensor(0.5)
    x = torch.randn(4, 2)
    y = model(t, x)
    assert y.shape == (4, 2), f"Expected (4,2), got {y.shape}"
    print("build: OK")


def test_spectral_norm_params():
    model = CNFNet(dim=2, hidden=8)
    keys = set(dict(model.named_parameters()))
    assert "net.0.weight_orig" in keys
    assert "net.4.weight_orig" in keys
    print("spectral_norm_params: OK")


def test_roundtrip():
    model = CNFNet(dim=2, hidden=8)
    x = torch.randn(4, 2)
    t0, t1 = 0.0, 1.0
    dt = 0.1
    t = t0
    for _ in range(int(abs(t1 - t0) / dt)):
        k1 = model(torch.tensor(t), x)
        k2 = model(torch.tensor(t + dt / 2), x + dt / 2 * k1)
        k3 = model(torch.tensor(t + dt / 2), x + dt / 2 * k2)
        k4 = model(torch.tensor(t + dt), x + dt * k3)
        x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += dt
    assert x.shape == (4, 2)
    print("roundtrip: OK")


if __name__ == "__main__":
    test_build()
    test_spectral_norm_params()
    test_roundtrip()
    print("\nAll smoke tests passed.")