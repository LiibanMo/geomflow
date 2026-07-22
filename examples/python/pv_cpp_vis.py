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

try:
    from manifolds import sphere as sphere_manifold
    from manifolds import torus as torus_manifold
except ImportError:
    sphere_manifold = None  # type: ignore[assignment]
    torus_manifold = None   # type: ignore[assignment]


# ---------------------------------------------------------------------------
# target generation helpers (inlined for C++ pipeline)
# ---------------------------------------------------------------------------

def _generate_target_data(manifold: str, n: int) -> np.ndarray | None:
    rng = np.random.default_rng(123)
    if manifold == "euclidean":
        centers = np.array([[3, 3], [3, -3], [-3, 3], [-3, -3]])
        pick = rng.integers(0, 4, n)
        return centers[pick] + rng.normal(0, 0.3, (n, 2))
    elif manifold == "sphere":
        theta = rng.normal(np.pi / 2, 0.3, n) + rng.normal(0, 0.05, n)
        phi1 = rng.uniform(0, 2 * np.pi, n // 2)
        phi2 = rng.uniform(0, 2 * np.pi, n - n // 2)
        phi = np.concatenate([phi1, phi2])
        rng.shuffle(phi)
        return np.column_stack([theta, phi])
    elif manifold == "torus":
        theta = rng.uniform(0, 2 * np.pi, n) + rng.normal(0, 0.1, n)
        pick = rng.integers(0, 2, n)
        phi = np.where(pick == 0, rng.normal(0, 0.3, n), rng.normal(np.pi, 0.3, n))
        phi += rng.normal(0, 0.1, n)
        return np.column_stack([theta, phi])
    return None


def _chart_to_cartesian(points: np.ndarray, manifold: str) -> np.ndarray:
    if manifold == "sphere":
        theta, phi = points[:, 0], points[:, 1]
        st = np.sin(theta)
        x = st * np.cos(phi)
        y = st * np.sin(phi)
        z = np.cos(theta)
        return np.column_stack([x, y, z])
    elif manifold == "torus":
        major, minor = points[:, 0], points[:, 1]
        a = 2.0 + np.cos(minor)
        x = a * np.cos(major)
        y = a * np.sin(major)
        z = np.sin(minor)
        return np.column_stack([x, y, z])
    else:
        z = np.zeros((points.shape[0], 1))
        return np.column_stack([points, z])


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
    parser.add_argument("--n-trails", type=int, default=0,
                        help="Number of trajectory streamlines (0=off)")
    parser.add_argument("--trail-width", type=float, default=2.0)
    parser.add_argument("--manifold-opacity", type=float, default=0.20)
    parser.add_argument("--window-size", type=int, default=800)
    parser.add_argument("--color-mode", default="transition",
                        choices=["transition", "z", "fixed"],
                        help="Point colouring: red→blue progress, Z-height, or fixed blue")
    parser.add_argument("--test-data", default="",
                        help="Path to CSV of target data (chart coords); if omitted, samples from target distribution")
    parser.add_argument("--hide-target", action="store_true", default=False,
                        help="Hide the green target-distribution reference points")
    parser.add_argument("--hold", type=float, default=2.0,
                        help="Seconds to hold on the final fitted state (0=off)")
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

    # ---- target (original) data -------------------------------------------
    if manifold in ("sphere", "torus", "euclidean"):
        if args.test_data:
            target_chart = np.loadtxt(args.test_data, delimiter=",", skiprows=0, dtype=np.float64)
            if target_chart.ndim == 1:
                target_chart = target_chart.reshape(1, -1)
            if target_chart.shape[1] != 2 and manifold != "euclidean":
                raise ValueError(f"--test-data expected N×2, got {target_chart.shape}")
        else:
            target_chart = _generate_target_data(manifold, n_points)
        if target_chart is not None:
            target_cart = _chart_to_cartesian(target_chart, manifold)
        else:
            target_cart = None
    else:
        target_cart = None

    # ---- colour mode -------------------------------------------------------
    use_transition = args.color_mode == "transition"
    use_z = args.color_mode == "z"

    # initial point cloud
    init_pts = frames[0][1]
    pc = pv.PolyData(init_pts)

    if use_transition:
        rgb = np.zeros((init_pts.shape[0], 3), dtype=np.uint8)
        rgb[:, 0] = 255  # red
        pc["_rgb"] = rgb
        pc_actor = plotter.add_mesh(
            pc, scalars="_rgb", rgb=True,
            point_size=args.point_size, render_points_as_spheres=True,
            opacity=0.9, show_scalar_bar=False,
        )
    elif use_z:
        pc["_z"] = init_pts[:, 2]
        pc_actor = plotter.add_mesh(
            pc, scalars="_z", cmap="plasma",
            point_size=args.point_size, render_points_as_spheres=True,
            opacity=0.9, show_scalar_bar=False,
        )
    else:
        pc_actor = plotter.add_mesh(
            pc, color="#2979ff",
            point_size=args.point_size, render_points_as_spheres=True,
            opacity=0.9,
        )

    # ---- target (original) reference cloud --------------------------------
    if target_cart is not None and not args.hide_target:
        tpc = pv.PolyData(target_cart)
        plotter.add_mesh(
            tpc,
            color="#00e676",
            point_size=args.point_size * 0.9,
            render_points_as_spheres=True,
            opacity=0.85,
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

        if use_transition:
            alpha = i / max(n_frames - 1, 1)
            rgb = pc.point_data["_rgb"]
            r = int(255 * (1 - alpha))
            b = int(255 * alpha)
            for j in range(pts.shape[0]):
                rgb[j, 0] = r
                rgb[j, 1] = 0
                rgb[j, 2] = b
        elif use_z:
            z_arr = pts[:, 2]
            scalars = pc.point_data["_z"]
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

    # ---- hold on final fitted state ----------------------------------------
    n_hold = int(args.hold * args.fps)
    if n_hold > 0 and images:
        last_frame = images[-1]
        for _ in range(n_hold):
            images.append(last_frame)
        print(f"  Appended {n_hold} hold frames ({args.hold}s)")

    # ---- write output -----------------------------------------------------
    duration = 1.0 / args.fps
    iio.imwrite(args.output, images, duration=duration, loop=1)
    print(f"Saved animation to {args.output}  ({len(images)} frames, {args.fps} fps)")


if __name__ == "__main__":
    main()