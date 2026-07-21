#!/usr/bin/env python3
"""Visualise C++ CNF inference output with PyVista.

Reads per-frame CSV files produced by ``infer_cnf`` and generates a
paper-grade 3D animation with:

  - semi-transparent manifold surface
  - trajectory streamlines (fading tails) — uses ``trajectories.csv`` when
    available (``--trajectories N`` in infer_cnf), otherwise reconstructs from
    frame CSVs
  - dynamic camera rotation

Usage:
    # basic: animate frame CSVs
    python pv_cpp_vis.py --frames-dir inference_frames --manifold sphere

    # with fine trajectory data from infer_cnf --trajectories 30
    python pv_cpp_vis.py --frames-dir inference_frames --manifold sphere
"""

import argparse
import os
import sys

import numpy as np

try:
    import imageio.v3 as iio
except ImportError:
    print("imageio is required.  Install it with:  pip install imageio")
    sys.exit(1)

try:
    import pyvista as pv
except ImportError:
    print("pyvista is required.  Install it with:  pip install pyvista")
    sys.exit(1)


# ---------------------------------------------------------------------------
# data loaders
# ---------------------------------------------------------------------------

def _load_csv(path: str) -> np.ndarray:
    """Load a CSV with header and N rows of floats."""
    return np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)


def read_frames(frame_dir: str) -> list[tuple[str, np.ndarray]]:
    """Return sorted list of (name, points) for each frame_*.csv."""
    entries = sorted(
        f for f in os.listdir(frame_dir)
        if f.startswith("frame_") and f.endswith(".csv")
    )
    if not entries:
        raise FileNotFoundError(f"No frame_*.csv files found in {frame_dir}")

    frames = []
    for name in entries:
        pts = _load_csv(os.path.join(frame_dir, name))
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(
                f"Expected N×3 array in {name}, got {pts.shape}"
            )
        frames.append((name, pts))
    return frames


def read_trajectories_csv(path: str) -> list[np.ndarray]:
    """Parse ``trajectories.csv`` written by ``infer_cnf --trajectories N``.

    Format:  header ``t,x0,y0,z0,x1,y1,z1,...``
             each row has 1 + 3*N columns.

    Returns a list of (n_steps, 3) arrays, one per trajectory point.
    """
    data = _load_csv(path)
    if data.ndim != 2 or data.shape[1] < 4:
        raise ValueError(f"trajectories.csv: unexpected shape {data.shape}")

    n_cols = data.shape[1]
    # first column = time, remaining 3*N columns
    if (n_cols - 1) % 3 != 0:
        raise ValueError(f"trajectories.csv: {n_cols} columns, expected 1 + 3N")
    n_traj = (n_cols - 1) // 3
    n_steps = data.shape[0]

    lines = []
    for k in range(n_traj):
        col_start = 1 + 3 * k
        line = data[:, col_start : col_start + 3]  # (n_steps, 3)
        lines.append(line)
    return lines


# ---------------------------------------------------------------------------
# surfaces
# ---------------------------------------------------------------------------

def pv_sphere_mesh(radius: float = 1.0) -> pv.PolyData:
    return pv.Sphere(radius=radius, theta_resolution=80, phi_resolution=60)


def pv_torus_mesh(major: float = 2.0, minor: float = 1.0) -> pv.PolyData:
    return pv.ParametricTorus(ringradius=major, crosssectionradius=minor)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualise C++ CNF inference output with PyVista"
    )
    parser.add_argument(
        "--frames-dir", required=True,
        help="Directory containing frame_*.csv (and optionally trajectories.csv)",
    )
    parser.add_argument(
        "--manifold", default="auto",
        choices=["auto", "euclidean", "sphere", "torus"],
    )
    parser.add_argument("--output", default="cpp_animation.gif")
    parser.add_argument("--elev", type=float, default=20)
    parser.add_argument("--azim", type=float, default=30)
    parser.add_argument("--rotate", type=float, default=90,
                        help="Total azimuth rotation (degrees, 0=off)")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--point-size", type=float, default=8.0)
    parser.add_argument("--figsize", type=int, default=6)
    parser.add_argument("--n-trails", type=int, default=30,
                        help="Number of trajectory streamlines (0=off)")
    parser.add_argument("--trail-width", type=float, default=2.0)
    parser.add_argument("--manifold-opacity", type=float, default=0.20)
    parser.add_argument("--window-size", type=int, default=800)
    parser.add_argument("--z-colour", action="store_true",
                        help="Colour points by Z height")
    args = parser.parse_args()

    # ---- read frames ------------------------------------------------------
    frames = read_frames(args.frames_dir)
    n_frames = len(frames)
    n_points = frames[0][1].shape[0]
    print(f"Read {n_frames} frames × {n_points} points from {args.frames_dir}")

    for name, pts in frames:
        if pts.shape[0] != n_points:
            raise ValueError(
                f"Frame {name} has {pts.shape[0]} points, expected {n_points}"
            )

    # ---- auto-detect manifold ---------------------------------------------
    manifold = args.manifold
    if manifold == "auto":
        last = frames[-1][1]
        radii = np.linalg.norm(last, axis=1)
        if np.allclose(radii, 1.0, atol=0.15):
            manifold = "sphere"
            print("Auto-detected manifold: sphere")
        else:
            manifold = "euclidean"
            print("Auto-detected manifold: euclidean (no surface)")

    # ---- trajectory data --------------------------------------------------
    traj_csv = os.path.join(args.frames_dir, "trajectories.csv")
    trail_lines: list[np.ndarray] = []

    if os.path.isfile(traj_csv):
        all_trajs = read_trajectories_csv(traj_csv)
        n_trail = min(args.n_trails, len(all_trajs)) if args.n_trails > 0 else 0
        trail_lines = all_trajs[:n_trail]
        print(f"Loaded {n_trail} trajectories from trajectories.csv "
              f"({trail_lines[0].shape[0]} steps)" if trail_lines else "")
    elif args.n_trails > 0:
        # fallback: reconstruct from frame CSVs
        n_trail = min(args.n_trails, n_points)
        for i in range(n_trail):
            line = np.empty((n_frames, 3), dtype=np.float64)
            for f, (_, pts) in enumerate(frames):
                line[f] = pts[i]
            trail_lines.append(line)
        print(f"Reconstructed {n_trail} trajectories from frame data "
              f"({n_frames} steps each)")

    # ---- PyVista renderer -------------------------------------------------
    ws = args.window_size
    plotter = pv.Plotter(window_size=(ws, ws), off_screen=True)
    plotter.set_background("white")
    plotter.remove_bounds_axes()

    # manifold surface
    if manifold == "sphere":
        surf = pv_sphere_mesh(0.99)
    elif manifold == "torus":
        surf = pv_torus_mesh(2.0, 0.99)
    else:
        surf = None

    if surf is not None:
        plotter.add_mesh(
            surf, color="#90a4ae", opacity=args.manifold_opacity,
            smooth_shading=True, specular=0.1, diffuse=0.8
        )

    # trajectory lines
    for line in trail_lines:
        n_pts = line.shape[0]
        t_arr = np.linspace(0, 1, n_pts)
        pline = pv.lines_from_points(line)
        pline["_time"] = t_arr
        plotter.add_mesh(
            pline, scalars="_time", cmap="viridis",
            line_width=args.trail_width, opacity=0.55, show_scalar_bar=False,
        )

    # initial point cloud
    init_pts = frames[0][1]
    pc = pv.PolyData(init_pts)
    scalar_name = "_z" if args.z_colour else None
    cmap = "plasma"
    if args.z_colour:
        pc["_z"] = init_pts[:, 2]
        pc_actor = plotter.add_mesh(
            pc, scalars=scalar_name, cmap=cmap,
            point_size=args.point_size, render_points_as_spheres=True,
            opacity=0.9, show_scalar_bar=False,
        )
    else:
        pc_actor = plotter.add_mesh(
            pc, color="#2979ff",
            point_size=args.point_size, render_points_as_spheres=True,
            opacity=0.9,
        )

    # camera
    cam_distance = {"sphere": 2.5, "torus": 5.0, "euclidean": 6.0}.get(manifold, 4.0)
    elev_rad = np.radians(args.elev)
    azim_rad = np.radians(args.azim)
    plotter.camera.position = (
        cam_distance * np.cos(elev_rad) * np.cos(azim_rad),
        cam_distance * np.cos(elev_rad) * np.sin(azim_rad),
        cam_distance * np.sin(elev_rad),
    )
    plotter.camera.focal_point = (0, 0, 0)
    plotter.camera.view_up = (0, 0, 1)
    plotter.show(auto_close=False)

    # ---- render frames ----------------------------------------------------
    images = []
    azim_step = args.rotate / max(n_frames - 1, 1) if args.rotate != 0 else 0

    for i, (_, pts) in enumerate(frames):
        vtk_pts = pc.GetPoints()
        for j in range(pts.shape[0]):
            vtk_pts.SetPoint(j, pts[j, 0], pts[j, 1], pts[j, 2])
        vtk_pts.Modified()

        if args.z_colour:
            z_arr = pts[:, 2]
            scalars = pc.point_data[scalar_name]
            for j in range(len(z_arr)):
                scalars[j] = z_arr[j]

        if azim_step != 0:
            plotter.camera.azimuth += azim_step

        plotter.render()
        img = plotter.screenshot(return_img=True)
        images.append(img)

        if (i + 1) % 10 == 0 or i == n_frames - 1:
            print(f"  Rendered frame {i + 1}/{n_frames}")

    plotter.close()

    # ---- write output -----------------------------------------------------
    duration = 1.0 / args.fps
    iio.imwrite(args.output, images, duration=duration, loop=0)
    print(f"Saved animation to {args.output}  ({len(images)} frames, {args.fps} fps)")


if __name__ == "__main__":
    main()