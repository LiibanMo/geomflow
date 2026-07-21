"""Torus manifold: data generation and metric utilities."""

import numpy as np


def generate_target_data(
    n: int = 5000, R: float = 2.0, r: float = 1.0, seed: int = 123
) -> np.ndarray:
    """Generate target point cloud on the torus (2D angular coordinates).

    Produces two clusters concentrated at phi=0 and phi=pi.
    """
    rng = np.random.default_rng(seed)

    theta = rng.uniform(0, 2 * np.pi, n) + rng.normal(0, 0.1, n)
    pick = rng.integers(0, 2, n)
    phi = np.where(pick == 0, rng.normal(0, 0.3, n), rng.normal(np.pi, 0.3, n))
    phi += rng.normal(0, 0.1, n)

    data = np.column_stack([theta, phi])
    return data


def sample_base(n: int, seed: int = 42) -> np.ndarray:
    """Sample from the base (standard normal) distribution in chart coordinates."""
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, (n, 2))


def to_cartesian(points: np.ndarray, R: float = 2.0, r: float = 1.0) -> np.ndarray:
    """Convert torus angular coordinates (major, minor) to Cartesian (x, y, z).

    points: (N, 2) array with columns [major_angle, minor_angle]
    R: major radius (default 2.0)
    r: minor radius (default 1.0)
    Returns: (N, 3) array.
    """
    major = points[:, 0]
    minor = points[:, 1]
    a = R + r * np.cos(minor)
    x = a * np.cos(major)
    y = a * np.sin(major)
    z = r * np.sin(minor)
    return np.column_stack([x, y, z])


def wireframe_points(R: float = 2.0, r: float = 1.0, n_u: int = 40, n_v: int = 20):
    """Generate wireframe mesh points for a torus.

    Returns x, y, z as 2D arrays suitable for plot_wireframe.
    """
    u = np.linspace(0, 2 * np.pi, n_u)
    v = np.linspace(0, 2 * np.pi, n_v)
    u, v = np.meshgrid(u, v)
    a = R + r * np.cos(v)
    x = a * np.cos(u)
    y = a * np.sin(u)
    z = r * np.sin(v)
    return x, y, z