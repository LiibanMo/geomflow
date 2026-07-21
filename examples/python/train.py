#!/usr/bin/env python3
"""Train a Continuous Normalizing Flow on a 2D Riemannian manifold.

Uses PyTorch with a purely intrinsic forward/backward integrator.
The learned vector field transports a standard normal base distribution
to a target distribution specified by the manifold parameter.

Usage:
    python train.py --manifold euclidean [--epochs 200] [--lr 0.01] [--output model.pt]
    python train.py --manifold sphere
    python train.py --manifold torus
"""

import argparse
import math

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

from manifolds.sphere import generate_target_data as gen_sphere
from manifolds.torus import generate_target_data as gen_torus


def generate_target_euclidean(n: int = 5000, seed: int = 123) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    centers = np.array([[3, 3], [3, -3], [-3, 3], [-3, -3]])
    pick = rng.integers(0, 4, n)
    data = centers[pick] + rng.normal(0, 0.3, (n, 2))
    return torch.tensor(data, dtype=torch.float32)


class LambdaScheduler:
    """Warmup scheduler for regularization coefficients.

    Epochs 0..warmup_start:  zero  (let NLL guide initial alignment).
    warmup_start..warmup_end:  linear ramp to final values.
    warmup_end+:  constant.
    """

    def __init__(
        self,
        warmup_start: int = 20,
        warmup_end: int = 60,
        lambda_kinetic_max: float = 1e-3,
        lambda_jac_max: float = 1e-4,
    ):
        self.warmup_start = warmup_start
        self.warmup_end = warmup_end
        self.lambda_k_max = lambda_kinetic_max
        self.lambda_j_max = lambda_jac_max

    def get(self, epoch: int) -> tuple[float, float]:
        if epoch < self.warmup_start:
            return 0.0, 0.0
        if epoch < self.warmup_end:
            frac = (epoch - self.warmup_start) / (self.warmup_end - self.warmup_start)
            return frac * self.lambda_k_max, frac * self.lambda_j_max
        return self.lambda_k_max, self.lambda_j_max


class CNFNet(nn.Module):
    """Simple MLP vector field f(t, x; theta)."""

    def __init__(self, dim: int = 2, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            spectral_norm(nn.Linear(dim + 1, hidden)),
            nn.Tanh(),
            spectral_norm(nn.Linear(hidden, hidden)),
            nn.Tanh(),
            spectral_norm(nn.Linear(hidden, dim)),
        )

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if t.dim() == 0:
            t = t.expand(x.shape[0], 1)
        elif t.dim() == 1:
            t = t.unsqueeze(-1)
        tx = torch.cat([t, x], dim=-1)
        return self.net(tx)


def rk4_step(fn, t, x, dt):
    k1 = fn(t, x)
    k2 = fn(t + dt / 2, x + dt / 2 * k1)
    k3 = fn(t + dt / 2, x + dt / 2 * k2)
    k4 = fn(t + dt, x + dt * k3)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def integrate(fn, x0, t0, t1, dt, track: bool = False):
    """Forward integration with trajectory tracking."""
    x = x0
    trajectory = [x] if track else []
    forward = t1 > t0
    sign = 1.0 if forward else -1.0
    h = sign * abs(dt)
    t = t0

    while (forward and t < t1) or (not forward and t > t1):
        step = h
        if forward and t + step > t1:
            step = t1 - t
        elif not forward and t + step < t1:
            step = t1 - t

        x = rk4_step(fn, t, x, step)
        t += step
        if track:
            trajectory.append(x.clone())

    return x, trajectory


@torch.enable_grad()
def compute_loss(
    model: CNFNet, z: torch.Tensor,
    t0: float, t1: float, dt: float, manifold: str,
    lambda_kinetic: float, lambda_jac: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n_steps = int(abs(t1 - t0) / dt)
    device = z.device

    x = z
    log_det = torch.zeros(z.shape[0], device=device)
    kinetic_loss = torch.zeros(z.shape[0], device=device)
    jac_reg_loss = torch.zeros(z.shape[0], device=device)

    for step in range(n_steps):
        t_val = t0 + step * dt
        h = dt if step < n_steps - 1 else t1 - (t0 + step * dt)
        ts = torch.tensor(t_val, device=device)
        ts_half = torch.tensor(t_val + h / 2, device=device)
        ts_next = torch.tensor(t_val + h, device=device)

        # RK4 integration (unchanged)
        k1 = model(ts, x)
        k2 = model(ts_half, x + h / 2 * k1)
        k3 = model(ts_half, x + h / 2 * k2)
        k4 = model(ts_next, x + h * k3)
        dx = h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        x_new = x + dx

        # Hutchinson estimator: divergence (trace) + Frobenius norm squared
        f_val = model(ts, x)
        v = torch.randn_like(x)
        Jv = torch.autograd.grad(
            f_val, x, v, create_graph=True, retain_graph=True
        )[0]

        div = (v * Jv).sum(dim=-1)
        frob_sq = (Jv * Jv).sum(dim=-1)

        log_det = log_det - h * div
        kinetic_loss = kinetic_loss + h * (f_val * f_val).sum(dim=-1)
        jac_reg_loss = jac_reg_loss + h * frob_sq

        x = x_new

    log_base = -0.5 * (z * z).sum(dim=-1) - math.log(2 * math.pi)
    log_model = log_base - log_det

    if manifold == "euclidean":
        sigma_sq = 0.09
        mus = torch.tensor(
            [[3, 3], [3, -3], [-3, 3], [-3, -3]], device=device, dtype=torch.float32
        )
        diffs = x.unsqueeze(1) - mus.unsqueeze(0)
        log_probs = -0.5 * (diffs**2).sum(dim=-1) / sigma_sq - math.log(2 * math.pi * sigma_sq)
        log_target = torch.logsumexp(log_probs, dim=-1) - math.log(4)
    elif manifold == "sphere":
        sigma_sq = 0.09
        mus = torch.tensor(
            [[1.57, 0.5], [1.57, 3.64]], device=device, dtype=torch.float32
        )
        diffs = x.unsqueeze(1) - mus.unsqueeze(0)
        log_probs = -0.5 * (diffs**2).sum(dim=-1) / sigma_sq - math.log(2 * math.pi * sigma_sq)
        log_target = torch.logsumexp(log_probs, dim=-1) - math.log(2)
    elif manifold == "torus":
        sigma_sq = 0.09
        cos_y = torch.cos(x[:, 1:2])
        sin_y = torch.sin(x[:, 1:2])
        xy = torch.cat([cos_y, sin_y], dim=-1)
        mus = torch.tensor([[1, 0], [-1, 0]], device=device, dtype=torch.float32)
        diffs = xy.unsqueeze(1) - mus.unsqueeze(0)
        log_probs = -0.5 * (diffs**2).sum(dim=-1) / sigma_sq - math.log(2 * math.pi * sigma_sq)
        log_target = torch.logsumexp(log_probs, dim=-1) - math.log(2)
    else:
        raise ValueError(f"Unknown manifold: {manifold}")

    nll_loss = (-log_model + log_target).mean()
    reg_kinetic = lambda_kinetic * kinetic_loss.mean()
    reg_jac = lambda_jac * jac_reg_loss.mean()
    total_loss = nll_loss + reg_kinetic + reg_jac

    return total_loss, nll_loss.detach(), reg_kinetic.detach(), reg_jac.detach()


def main():
    parser = argparse.ArgumentParser(description="Train a CNF on a 2D manifold")
    parser.add_argument(
        "--manifold",
        required=True,
        choices=["euclidean", "sphere", "torus"],
        help="Manifold type",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--output", default="cnf_model.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="Weight decay (L2) for Adam optimizer")
    parser.add_argument("--lambda-kinetic", type=float, default=1e-3,
                        help="Max kinetic-energy regularization coefficient")
    parser.add_argument("--lambda-jac", type=float, default=1e-4,
                        help="Max Jacobian Frobenius norm coefficient")
    args = parser.parse_args()

    device = torch.device(args.device)

    model = CNFNet(dim=2, hidden=32).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = LambdaScheduler(
        warmup_start=20, warmup_end=60,
        lambda_kinetic_max=args.lambda_kinetic,
        lambda_jac_max=args.lambda_jac,
    )

    t0, t1 = 0.0, 1.0

    print(f"Training CNF on {args.manifold} manifold")
    print(f"  epochs={args.epochs}  batch={args.batch_size}  lr={args.lr}"
          f"  dt={args.dt}  wd={args.weight_decay}")
    print(f"  λ_kinetic_max={args.lambda_kinetic}  λ_jac_max={args.lambda_jac}")
    print()

    best_loss = float("inf")

    for epoch in range(args.epochs):
        z = torch.randn(args.batch_size, 2, device=device)
        z.requires_grad_(True)

        lam_k, lam_j = scheduler.get(epoch)
        total_loss, nll_loss, reg_kinetic, reg_jac = compute_loss(
            model, z, t0, t1, args.dt, args.manifold, lam_k, lam_j
        )

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            torch.save(model.state_dict(), args.output)

        if (epoch + 1) % 20 == 0:
            print(
                f"Epoch {epoch+1:>3d}/{args.epochs}"
                f"  total={total_loss.item():.3f}"
                f"  nll={nll_loss.item():.3f}"
                f"  kinetic={reg_kinetic.item():.3f}"
                f"  jac_reg={reg_jac.item():.3f}"
            )

    print(f"\nBest loss: {best_loss:.4f}")
    print(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()