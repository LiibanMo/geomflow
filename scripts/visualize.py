#!/usr/bin/env python3
"""Visualize CNF inference output as an animated 3D GIF.

Reads CSV frames produced by the C++ infer_cnf executable (x,y,z columns)
and renders them as a 3D scatter plot GIF with an optional wireframe
reference surface.

Usage:
    python visualize.py --input-dir inference_frames --output animation.gif \\
        --surface sphere
"""

import argparse
import glob
import math
import os
import sys

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
except ImportError:
    print("matplotlib is required. Install it with: pip install matplotlib")
    sys.exit(1)

try:
    import imageio.v3 as iio
except ImportError:
    print("imageio is required. Install it with: pip install imageio")
    sys.exit(1)


def load_frame(path: str) -> np.ndarray:
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def sphere_wireframe(R: float = 1.0, n_u: int = 30, n_v: int = 20):
    u = np.linspace(0, 2 * np.pi, n_u)
    v = np.linspace(0, np.pi, n_v)
    u, v = np.meshgrid(u, v)
    x = R * np.sin(v) * np.cos(u)
    y = R * np.sin(v) * np.sin(u)
    z = R * np.cos(v)
    return x, y, z


def torus_wireframe(R: float = 2.0, r: float = 1.0, n_u: int = 40, n_v: int = 20):
    u = np.linspace(0, 2 * np.pi, n_u)
    v = np.linspace(0, 2 * np.pi, n_v)
    u, v = np.meshgrid(u, v)
    a = R + r * np.cos(v)
    x = a * np.cos(u)
    y = a * np.sin(u)
    z = r * np.sin(v)
    return x, y, z


def render_frame(
    points: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    zlim: tuple[float, float],
    surface: str | None = None,
    elev: float = 20,
    azim: float = 30,
    figsize: int = 6,
    point_size: float = 2.0,
) -> np.ndarray:
    fig = plt.figure(figsize=(figsize, figsize))
    ax = fig.add_subplot(111, projection="3d")

    if surface == "sphere":
        xw, yw, zw = sphere_wireframe()
        ax.plot_wireframe(xw, yw, zw, color="gray", alpha=0.15, linewidth=0.3, rstride=2, cstride=2)
    elif surface == "torus":
        xw, yw, zw = torus_wireframe()
        ax.plot_wireframe(xw, yw, zw, color="gray", alpha=0.15, linewidth=0.3, rstride=2, cstride=2)

    if points.shape[0] > 0:
        z_colors = points[:, 2] if points.shape[1] >= 3 else np.zeros_like(points[:, 0])
        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2] if points.shape[1] >= 3 else np.zeros_like(points[:, 0]),
            s=point_size,
            c=z_colors,
            cmap="plasma",
            alpha=0.7,
            edgecolors="none",
        )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)

    fig.canvas.draw()
    img = np.array(fig.canvas.renderer.buffer_rgba())
    plt.close(fig)
    return img[:, :, :3]


def main():
    parser = argparse.ArgumentParser(description="Visualize CNF inference as 3D animated GIF")
    parser.add_argument("--input-dir", default="inference_frames", help="Directory containing frame_*.csv")
    parser.add_argument("--output", default="animation.gif", help="Output GIF path")
    parser.add_argument(
        "--surface", choices=["none", "sphere", "torus"], default="none",
        help="Reference surface wireframe to draw in background"
    )
    parser.add_argument("--xlim", nargs=2, type=float, default=[-3, 3], help="X-axis limits")
    parser.add_argument("--ylim", nargs=2, type=float, default=[-3, 3], help="Y-axis limits")
    parser.add_argument("--zlim", nargs=2, type=float, default=[-3, 3], help="Z-axis limits")
    parser.add_argument("--elev", type=float, default=20, help="Camera elevation (degrees)")
    parser.add_argument("--azim", type=float, default=30, help="Camera azimuth (degrees)")
    parser.add_argument("--rotate", type=float, default=0, help="Total azimuth rotation over animation (degrees, 0=off)")
    parser.add_argument("--duration", type=float, default=0.1, help="Frame duration in seconds")
    parser.add_argument("--point-size", type=float, default=2.0, help="Scatter point size")
    parser.add_argument("--downsample", type=int, default=0, help="Max points per frame (0 = all)")
    args = parser.parse_args()

    frames = sorted(glob.glob(os.path.join(args.input_dir, "frame_*.csv")))
    if not frames:
        print(f"No frame_*.csv files found in {args.input_dir}")
        sys.exit(1)

    print(f"Found {len(frames)} frames")

    surfaces = {"none": None, "sphere": "sphere", "torus": "torus"}
    surface = surfaces.get(args.surface)

    xlim = tuple(args.xlim)
    ylim = tuple(args.ylim)
    zlim = tuple(args.zlim)

    images = []
    for i, frame_path in enumerate(frames):
        points = load_frame(frame_path)
        if points.shape[0] == 0:
            continue

        if args.downsample > 0 and points.shape[0] > args.downsample:
            idx = np.random.choice(points.shape[0], args.downsample, replace=False)
            points = points[idx]

        azim = args.azim
        if args.rotate > 0 and len(frames) > 1:
            azim = args.azim + args.rotate * i / (len(frames) - 1)

        img = render_frame(
            points, xlim, ylim, zlim, surface=surface, elev=args.elev, azim=azim,
            point_size=args.point_size
        )
        images.append(img)
        if (i + 1) % 10 == 0:
            print(f"  Rendered frame {i + 1}/{len(frames)}")

    iio.imwrite(args.output, images, duration=args.duration, loop=0)
    print(f"Saved animation to {args.output}")


if __name__ == "__main__":
    main()