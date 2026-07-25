"""Batched helper moved to a utils file to avoid circular imports."""

from __future__ import annotations

import torch


def batched_jacobian(
    fn: callable, x: torch.Tensor
) -> torch.Tensor:
    """Compute Jacobian of ``fn`` for a batch, returning ``(..., dim_out, dim_in)``.

    ``fn`` must act pointwise over every leading batch dimension. Summing each
    output component then gives the per-sample Jacobian without materializing
    the zero cross-sample blocks of a full coupled-batch Jacobian.
    """
    if x.ndim < 1:
        raise ValueError("x must have a final coordinate dimension")

    *batch_shape, dim_in = x.shape
    x_grad = x if x.requires_grad else x.clone().requires_grad_(True)
    y = fn(x_grad)
    if y.ndim < 1 or y.shape[:-1] != x.shape[:-1]:
        raise ValueError(
            "fn must preserve all leading batch dimensions and return "
            "shape (..., dim_out)"
        )
    dim_out = y.shape[-1]

    rows: list[torch.Tensor] = []
    for i in range(dim_out):
        component = y[..., i]
        if component.requires_grad:
            (row,) = torch.autograd.grad(
                component.sum(),
                x_grad,
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )
        else:
            row = None
        if row is None:
            row = torch.zeros(*batch_shape, dim_in, device=x.device, dtype=x.dtype)
        rows.append(row)
    return torch.stack(rows, dim=-2)
