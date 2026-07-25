"""Batched helper moved to a utils file to avoid circular imports."""

from __future__ import annotations

import torch
import torch.nn as nn


_SUPPORTED_FLOAT_DTYPES = (torch.float32, torch.float64)


def validate_supported_floating_tensor(
    tensor: torch.Tensor, operation: str
) -> None:
    """Validate the production floating-point policy without moving data."""
    if tensor.dtype not in _SUPPORTED_FLOAT_DTYPES:
        supported = ", ".join(str(dtype) for dtype in _SUPPORTED_FLOAT_DTYPES)
        raise TypeError(
            f"{operation}: expected a tensor with dtype {supported}; "
            f"got device={tensor.device}, dtype={tensor.dtype}"
        )


def module_device_dtype(
    module: nn.Module, operation: str
) -> tuple[torch.device, torch.dtype]:
    """Return the unique device and floating dtype of module state."""
    tensors = [*module.parameters(), *module.buffers()]
    if not tensors:
        raise ValueError(f"{operation}: module has no parameters or buffers")
    devices = {tensor.device for tensor in tensors}
    dtypes = {tensor.dtype for tensor in tensors if tensor.is_floating_point()}
    if len(devices) != 1 or len(dtypes) != 1:
        raise ValueError(
            f"{operation}: module must use one device and floating dtype; "
            f"got devices={sorted(map(str, devices))}, "
            f"dtypes={sorted(map(str, dtypes))}"
        )
    device = next(iter(devices))
    dtype = next(iter(dtypes))
    if dtype not in _SUPPORTED_FLOAT_DTYPES:
        raise TypeError(
            f"{operation}: module on {device} has unsupported dtype {dtype}; "
            "expected torch.float32 or torch.float64"
        )
    return device, dtype


def validate_tensor_module_compatibility(
    tensor: torch.Tensor, module: nn.Module | callable, operation: str
) -> None:
    """Reject input/module placement mismatches without implicit transfers."""
    validate_supported_floating_tensor(tensor, operation)
    if not isinstance(module, nn.Module):
        return
    if not any(True for _ in module.parameters()) and not any(
        True for _ in module.buffers()
    ):
        return
    expected_device, expected_dtype = module_device_dtype(module, operation)
    if tensor.device != expected_device or tensor.dtype != expected_dtype:
        raise ValueError(
            f"{operation}: expected input device={expected_device}, "
            f"dtype={expected_dtype}; got device={tensor.device}, "
            f"dtype={tensor.dtype}"
        )


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
