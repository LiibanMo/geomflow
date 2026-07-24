"""Safety tests for the experimental intrinsic adjoint implementation."""

from __future__ import annotations

import pytest
import torch

from geomflow.torch import (
    EuclideanSpace,
    ManifoldVectorField,
    IntrinsicAdjointFunction,
)


def test_experimental_adjoint_rejects_parameter_training():
    """MATH-630--632: unsupported adjoint parameter gradients fail explicitly."""
    torch.manual_seed(42)
    dim = 2
    metric = EuclideanSpace(dim)
    vf = ManifoldVectorField(dim=dim, hidden_dim=16)

    x_data = torch.randn(8, dim, requires_grad=True)
    with pytest.raises(RuntimeError, match="does not return parameter gradients"):
        IntrinsicAdjointFunction.apply(x_data, vf, metric, 0.05, 0.0, 1.0)


if __name__ == "__main__":
    print("=== Testing Mohamud Intrinsic Adjoint (Theorem 3.7) ===\n")
    test_experimental_adjoint_rejects_parameter_training()
    print("\n✅ Mohamud Intrinsic Adjoint test passed!")
