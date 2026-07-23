"""Batched helper moved to a utils file to avoid circular imports."""

from __future__ import annotations

import torch


def batched_jacobian(
    fn: callable, x: torch.Tensor
) -> torch.Tensor:
    """Compute Jacobian of ``fn`` for a batch, returning ``(..., dim_out, dim_in)``.

    Uses a loop over output components and autograd per batch element.
    Slower than ``torch.autograd.functional.jacobian`` but handles arbitrary
    batch shapes.
    """
    *batch_shape, dim_in = x.shape
    y = fn(x)
    dim_out = y.shape[-1]

    jac = torch.zeros(*batch_shape, dim_out, dim_in, device=x.device, dtype=x.dtype)

    for i in range(dim_out):
        x_grad = x.detach().clone().requires_grad_(True)
        y = fn(x_grad)
        (g,) = torch.autograd.grad(
            y[..., i].sum(), x_grad, create_graph=True, retain_graph=True
        )
        jac[..., i, :] = g
    return jac
