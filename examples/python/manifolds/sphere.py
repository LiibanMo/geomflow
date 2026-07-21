"""Sphere manifold: data generation and metric utilities."""

import numpy as np


def generate_target_data(n: int = 5000, seed: int = 123) -> np.ndarray:
    """Generate target point cloud on the sphere (2D angular coordinates).

    Produces two clusters near the equator (theta ~ pi/2) separated in phi.
    """
    rng = np.random.default_rng(seed)

    theta = rng.normal(np.pi / 2, 0.3, n) + rng.normal(0, 0.05, n)
    phi1 = rng.uniform(0, 2 * np.pi, n // 2)
    phi2 = rng.uniform(0, 2 * np.pi, n - n // 2)
    phi = np.concatenate([phi1, phi2])

    rng.shuffle(phi)

    data = np.column_stack([theta, phi])
    return data


def sample_base(n: int, seed: int = 42) -> np.ndarray:
    """Sample from the base (standard normal) distribution in chart coordinates."""
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, (n, 2))


def to_cartesian(points: np.ndarray, R: float = 1.0) -> np.ndarray:
    """Convert sphere angular coordinates (theta, phi) to Cartesian (x, y, z).

    points: (N, 2) array with columns [theta, phi]
    R: sphere radius (default 1.0)
    Returns: (N, 3) array.
    """
    theta = points[:, 0]
    phi = points[:, 1]
    st = np.sin(theta)
    x = R * st * np.cos(phi)
    y = R * st * np.sin(phi)
    z = R * np.cos(theta)
    return np.column_stack([x, y, z])


def wireframe_points(R: float = 1.0, n_u: int = 30, n_v: int = 30):
    """Generate wireframe mesh points for a sphere.

    Returns x, y, z as 2D arrays suitable for plot_wireframe.
    """
    u = np.linspace(0, 2 * np.pi, n_u)
    v = np.linspace(0, np.pi, n_v)
    u, v = np.meshgrid(u, v)
    x = R * np.sin(v) * np.cos(u)
    y = R * np.sin(v) * np.sin(u)
    z = R * np.cos(v)
    return x, y, z