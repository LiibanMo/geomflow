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
    """Return independent-sample Jacobians with shape ``(..., out, in)``.

    ``fn`` is evaluated on one point at a time and must map ``(in,)`` to
    ``(out,)``. Arbitrary leading batch dimensions are flattened for the
    transform and restored in the result.
    """
    if x.ndim < 1:
        raise ValueError("x must have a final coordinate dimension")

    *batch_shape, dim_in = x.shape
    x_grad = x if x.requires_grad else x.clone().requires_grad_(True)

    def point_fn(point: torch.Tensor) -> torch.Tensor:
        result = fn(point)
        if result.ndim != 1:
            raise ValueError(
                "fn must preserve leading batch dimensions and map each point "
                "to a one-dimensional output"
            )
        return result

    def autograd_fallback() -> torch.Tensor:
        y = fn(x_grad)
        if y.ndim < 1 or y.shape[:-1] != x.shape[:-1]:
            raise ValueError(
                "fn must preserve all leading batch dimensions and return "
                "shape (..., dim_out)"
            )
        rows: list[torch.Tensor] = []
        for component in y.unbind(-1):
            row = None
            if component.requires_grad:
                (row,) = torch.autograd.grad(
                    component.sum(),
                    x_grad,
                    create_graph=True,
                    retain_graph=True,
                    allow_unused=True,
                )
            rows.append(torch.zeros_like(x_grad) if row is None else row)
        return torch.stack(rows, dim=-2)

    if not batch_shape:
        return torch.func.jacrev(point_fn)(x_grad)

    flat_x = x_grad.reshape(-1, dim_in)
    if flat_x.shape[0] == 0:
        prototype = fn(x_grad)
        if prototype.ndim < 1 or prototype.shape[:-1] != x.shape[:-1]:
            raise ValueError(
                "fn must preserve all leading batch dimensions and return "
                "shape (..., dim_out)"
            )
        return x.new_empty(*batch_shape, prototype.shape[-1], dim_in)

    if x.device.type == "cpu":
        return autograd_fallback()

    try:
        jacobian = torch.vmap(torch.func.jacrev(point_fn))(flat_x)
    except RuntimeError as error:
        message = str(error)
        if "vmap:" not in message and "functorch" not in message:
            raise
        return autograd_fallback()
    return jacobian.reshape(*batch_shape, jacobian.shape[-2], dim_in)
