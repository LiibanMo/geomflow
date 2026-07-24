"""Small analytic oracles independent of geomflow production solvers."""

from __future__ import annotations

import math

import torch


def scalar_linear_state(
    x: torch.Tensor, a: float | torch.Tensor, ta: float, tb: float
) -> torch.Tensor:
    """Exact solution of x_dot = a*x."""
    scale = torch.as_tensor(a, dtype=x.dtype, device=x.device)
    return x * torch.exp(scale * (tb - ta))


def linear_divergence_integral(trace: float, ta: float, tb: float) -> float:
    """Exact signed divergence integral for x_dot = A*x."""
    return trace * (tb - ta)


def linear_log_density_change(trace: float, ta: float, tb: float) -> float:
    """Exact Riemannian log-density change for x_dot = A*x."""
    return -linear_divergence_integral(trace, ta, tb)


def time_linear_state(x: torch.Tensor, ta: float, tb: float) -> torch.Tensor:
    """Exact solution of x_dot = t*x."""
    return x * math.exp(0.5 * (tb * tb - ta * ta))


def quadratic_flow_quantities(
    x: torch.Tensor, theta: torch.Tensor, duration: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """State, divergence integral, and their theta derivatives for x_dot=theta*x^2."""
    denominator = 1.0 - theta * x * duration
    state = x / denominator
    divergence_integral = -2.0 * torch.log(denominator)
    state_derivative = x.square() * duration / denominator.square()
    divergence_derivative = 2.0 * x * duration / denominator
    return state, divergence_integral, state_derivative, divergence_derivative


def observed_order(errors: list[float]) -> tuple[float, ...]:
    """Estimate convergence orders for successive step halvings."""
    if len(errors) < 2 or any(error <= 0.0 for error in errors):
        raise ValueError("errors must contain at least two positive values")
    return tuple(math.log2(coarse / fine) for coarse, fine in zip(errors, errors[1:]))


def central_difference(fn, value: float, step: float = 1e-6) -> float:
    """Independent scalar central difference."""
    if step <= 0.0:
        raise ValueError("step must be positive")
    return (fn(value + step) - fn(value - step)) / (2.0 * step)
