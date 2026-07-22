#!/usr/bin/env python3
"""Run inference with a trained CNF model and generate a 3D animation via PyVista.

Reads a trained PyTorch model, samples from the base distribution, integrates
through the learned flow, converts to Cartesian coordinates, and produces a
paper-grade animated visualisation with:
  - semi-transparent manifold surface
  - trajectory streamlines (fading tails)
  - density-coloured particles
  - dynamic camera rotation

Usage:
    python infer.py --model sphere_model.pt --manifold sphere --output animation.gif
"""

import argparse
import math
import sys
from typing import Optional

import numpy as np
import torch
from PIL import Image as PILImage

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

from manifolds import sphere as sphere_manifold
from manifolds import torus as torus_manifold


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


from model import CNFNet


# ---------------------------------------------------------------------------
# RK4 integrator
# ---------------------------------------------------------------------------


def rk4_step(model: CNFNet, t: float, x: torch.Tensor, h: float) -> torch.Tensor:
    t_dev = torch.tensor(t, device=x.device)
    k1 = model(t_dev, x)
    k2 = model(t_dev + h / 2, x + h / 2 * k1)
    k3 = model(t_dev + h / 2, x + h / 2 * k2)
    k4 = model(t_dev + h, x + h * k3)
    return x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


# ---------------------------------------------------------------------------
# log-density helper  (finite-difference divergence in chart coords)
# ---------------------------------------------------------------------------


def divergence(
    model: CNFNet, t: float, x: torch.Tensor, eps: float = 1e-4
) -> torch.Tensor:
    t_dev = torch.tensor(t, device=x.device)
    ndim = x.shape[1]
    div = torch.zeros(x.shape[0], device=x.device)
    for d in range(ndim):
        e = torch.zeros_like(x)
        e[:, d] = eps
        fp = model(t_dev, x + e)
        fm = model(t_dev, x - e)
        div += (fp[:, d] - fm[:, d]) / (2 * eps)
    return div


def log_prob_base(z: torch.Tensor) -> torch.Tensor:
    """log-prob of standard normal in chart coordinates."""
    return -0.5 * (z * z).sum(dim=-1) - math.log(2 * math.pi)


# ---------------------------------------------------------------------------
# coordinate conversion
# ---------------------------------------------------------------------------


def convert_points(points: np.ndarray, manifold: str) -> np.ndarray:
    if manifold == "sphere":
        return sphere_manifold.to_cartesian(points)
    elif manifold == "torus":
        return torus_manifold.to_cartesian(points)
    else:
        z = np.zeros((points.shape[0], 1))
        return np.column_stack([points, z])


# ---------------------------------------------------------------------------
# manifold surfaces  (PyVista meshes)
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
        description="Run CNF inference and generate 3D animated visualisation (PyVista)"
    )
    parser.add_argument("--model", required=True, help="Path to trained model .pt file")
    parser.add_argument(
        "--manifold", required=True, choices=["euclidean", "sphere", "torus"]
    )
    parser.add_argument("--output", default="animation.gif", help="Output GIF path")
    parser.add_argument(
        "--title",
        default="",
        help="Text overlay title drawn at top-left of each frame",
    )
    parser.add_argument(
        "--background",
        default="white",
        choices=["white", "dark", "black"],
        help="Plotter background colour",
    )
    parser.add_argument("--n-points", type=int, default=500)
    parser.add_argument("--n-frames", type=int, default=40)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--elev", type=float, default=20)
    parser.add_argument("--azim", type=float, default=30)
    parser.add_argument(
        "--rotate",
        type=float,
        default=90,
        help="Total azimuth rotation (degrees, 0=off)",
    )
    parser.add_argument("--duration", type=float, default=0.1)
    parser.add_argument("--point-size", type=float, default=8.0)
    parser.add_argument("--figsize", type=int, default=6)
    parser.add_argument(
        "--n-trails",
        type=int,
        default=0,
        help="Number of trajectory streamlines to draw (0=off)",
    )
    parser.add_argument(
        "--trail-dt",
        type=float,
        default=0.02,
        help="Time step for trajectory integration (finer = smoother trails)",
    )
    parser.add_argument("--trail-width", type=float, default=2.0)
    parser.add_argument(
        "--no-log-density",
        action="store_true",
        help="Disable log-density colouring (fall back to Z-height)",
    )
    parser.add_argument(
        "--fps", type=int, default=10, help="Frames per second for output GIF"
    )
    parser.add_argument("--manifold-opacity", type=float, default=0.20)
    parser.add_argument(
        "--window-size", type=int, default=800, help="Render resolution (square, px)"
    )
    parser.add_argument(
        "--cam-distance",
        type=float,
        default=0.0,
        help="Camera distance from origin (0 = auto per manifold)",
    )
    parser.add_argument(
        "--test-data",
        default="",
        help="Path to CSV of target data (chart coords); if omitted, samples from target distribution",
    )
    parser.add_argument(
        "--color-mode",
        default="transition",
        choices=["transition", "logp", "z"],
        help="Point colouring: red→blue progress, log-density, or Z-height",
    )
    parser.add_argument(
        "--hide-target", action="store_true", default=False,
        help="Hide the green target-distribution reference points",
    )
    parser.add_argument(
        "--hold", type=float, default=2.0,
        help="Seconds to hold on the final fitted state (0=off)",
    )
    parser.add_argument(
        "--still-pdf", action="store_true", default=False,
        help="Render a single still-frame and export as PDF (disables animation)",
    )
    parser.add_argument(
        "--pdf-size", type=int, default=2400,
        help="Square render resolution in px for --still-pdf (default: %(default)s)",
    )
    parser.add_argument(
        "--pdf-dpi", type=int, default=300,
        help="DPI for PDF export (default: %(default)s)",
    )
    args = parser.parse_args()

    device = torch.device(args.device)

    # ---- load model -------------------------------------------------------
    model = CNFNet(dim=2, hidden=32).to(device)
    model.load_state_dict(
        torch.load(args.model, map_location=device, weights_only=True)
    )
    model.eval()
    print(f"Loaded model from {args.model}")

    manifold = args.manifold
    t0, t1 = 0.0, 1.0
    n_points = args.n_points
    n_frames = args.n_frames

    z = torch.randn(n_points, 2, device=device)  # base samples

    # ---- still PDF branch ---------------------------------------------------
    if args.still_pdf:
        print("Rendering still PDF …")
        title = args.title if args.title else "CNF on S²"
        n_trail_still = args.n_trails if args.n_trails > 0 else 5

        # integrate all points to t=1
        x = z.clone()
        t = float(t0)
        n_steps = int(abs(t1 - t0) / abs(args.dt))
        for _ in range(n_steps):
            step = args.dt
            if t + step > t1:
                step = t1 - t
            if step <= 0:
                break
            x = rk4_step(model, t, x, step)
            t += float(step)
        noise_cart = convert_points(z.detach().cpu().numpy(), manifold)
        fitted_cart = convert_points(x.detach().cpu().numpy(), manifold)

        # target / test data
        if manifold in ("sphere", "torus"):
            gen_fn = {"sphere": sphere_manifold.generate_target_data,
                       "torus": torus_manifold.generate_target_data}
            target_chart_still = gen_fn[manifold](n=n_points)
            target_cart_still = convert_points(target_chart_still, manifold)
        else:
            target_cart_still = None

        # trajectories
        trail_lines_still: list[np.ndarray] = []
        if n_trail_still > 0:
            z_trail = z[:n_trail_still].clone()
            x_trail = z_trail.clone()
            n_ts = int(abs(t1 - t0) / abs(args.trail_dt))
            trail_steps_np: list[np.ndarray] = [
                convert_points(x_trail.detach().cpu().numpy(), manifold)
            ]
            t = float(t0)
            for _ in range(n_ts):
                step = args.trail_dt
                if t + step > t1:
                    step = t1 - t
                if step <= 0:
                    break
                x_trail = rk4_step(model, t, x_trail, step)
                t += float(step)
                trail_steps_np.append(
                    convert_points(x_trail.detach().cpu().numpy(), manifold)
                )
            n_saved = len(trail_steps_np)
            for i in range(n_trail_still):
                line = np.empty((n_saved, 3), dtype=np.float64)
                for s, arr in enumerate(trail_steps_np):
                    line[s] = arr[i]
                trail_lines_still.append(line)
            print(f"  Computed {n_trail_still} trajectories ({n_saved} steps)")

        # PyVista renderer
        ws = args.pdf_size
        bg_colour = "#1a1a2e"
        plotter = pv.Plotter(window_size=(ws, ws), off_screen=True)
        plotter.set_background(bg_colour)
        plotter.remove_bounds_axes()

        # manifold surface
        if manifold == "sphere":
            surf = pv_sphere_mesh(0.99)
            plotter.add_mesh(surf, color="#90a4ae", opacity=args.manifold_opacity,
                             smooth_shading=True, specular=0.1, diffuse=0.8)
        elif manifold == "torus":
            surf = pv_torus_mesh(2.0, 0.99)
            plotter.add_mesh(surf, color="#90a4ae", opacity=args.manifold_opacity,
                             smooth_shading=True, specular=0.1, diffuse=0.8)

        ps = args.point_size * (args.pdf_size / 800.0)

        # noise (red)
        if noise_cart.shape[0] > 0:
            plotter.add_mesh(
                pv.PolyData(noise_cart), color="#d50000", point_size=ps,
                render_points_as_spheres=True, opacity=0.85,
            )

        # fitted (teal blue)
        if fitted_cart.shape[0] > 0:
            plotter.add_mesh(
                pv.PolyData(fitted_cart), color="#00acc1", point_size=ps * 0.9,
                render_points_as_spheres=True, opacity=0.85,
            )

        # target/test (green)
        if target_cart_still is not None and target_cart_still.shape[0] > 0:
            plotter.add_mesh(
                pv.PolyData(target_cart_still), color="#00c853", point_size=ps * 0.8,
                render_points_as_spheres=True, opacity=0.85,
            )

        # trajectories (gold)
        for line in trail_lines_still:
            pline = pv.lines_from_points(line)
            plotter.add_mesh(pline, color="#ffd54f", line_width=2.0, opacity=0.85)

        # camera
        cam_dist = (
            args.cam_distance if args.cam_distance > 0
            else {"sphere": 2.5, "torus": 5.0, "euclidean": 6.0}.get(manifold, 4.0)
        )
        elev_rad = np.radians(args.elev)
        azim_rad = np.radians(args.azim)
        plotter.camera.position = (
            cam_dist * np.cos(elev_rad) * np.cos(azim_rad),
            cam_dist * np.cos(elev_rad) * np.sin(azim_rad),
            cam_dist * np.sin(elev_rad),
        )
        plotter.camera.focal_point = (0, 0, 0)
        plotter.camera.view_up = (0, 0, 1)

        # title
        plotter.add_text(title, position="upper_left",
                         font_size=max(16, ws // 45),
                         color="#e0e0e0", shadow=True)

        plotter.show(auto_close=False)
        plotter.render()
        img = plotter.screenshot(return_img=True)
        plotter.close()

        pil_img = PILImage.fromarray(img)
        pil_img.save(args.output, "PDF", resolution=args.pdf_dpi)
        print(f"Saved still PDF to {args.output}  ({ws}×{ws} px, {args.pdf_dpi} DPI)")
        return

    # ---- per-frame integration  (+ log-density) ---------------------------
    dt_frame = (t1 - t0) / n_frames
    frames_cart: list[np.ndarray] = []
    log_densities: list[Optional[np.ndarray]] = []

    init_np = convert_points(z.detach().cpu().numpy(), manifold)
    frames_cart.append(init_np)

    use_logp = not args.no_log_density
    if use_logp:
        lpb = log_prob_base(z).detach().cpu().numpy()
        log_densities.append(lpb)
    else:
        log_densities.append(None)

    for frame_idx in range(1, n_frames + 1):
        t_target = t0 + frame_idx * dt_frame
        x = z.clone()
        log_det = torch.zeros(n_points, device=device)
        t = float(t0)

        n_steps = int(abs(t_target - t0) / abs(args.dt))
        for _ in range(n_steps):
            t_tensor = torch.tensor(t, device=device)
            step = args.dt
            if (args.dt > 0 and t + step > t_target) or (
                args.dt < 0 and t + step < t_target
            ):
                step = t_target - t

            k1 = model(t_tensor, x)
            k2 = model(t_tensor + step / 2, x + step / 2 * k1)
            k3 = model(t_tensor + step / 2, x + step / 2 * k2)
            k4 = model(t_tensor + step, x + step * k3)
            x = x + step / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

            if use_logp:
                log_det -= step * divergence(model, t, x)

            t += float(step)

        frames_cart.append(convert_points(x.detach().cpu().numpy(), manifold))
        if use_logp:
            lp = (log_prob_base(z) - log_det).detach().cpu().numpy()
            log_densities.append(lp)
        else:
            log_densities.append(None)

    print(f"Integrated {len(frames_cart)} frames")

    # ---- trajectory streamlines  (subset of points) -----------------------
    trail_lines: list[np.ndarray] = []
    n_trail = min(args.n_trails, n_points) if args.n_trails > 0 else 0

    if n_trail > 0:
        z_trail = z[:n_trail].clone()
        x_trail = z_trail.clone()
        n_trail_steps = int(abs(t1 - t0) / abs(args.trail_dt))
        trail_steps_np: list[np.ndarray] = [
            convert_points(x_trail.detach().cpu().numpy(), manifold)
        ]
        t = float(t0)

        for _ in range(n_trail_steps):
            step = args.trail_dt
            if t + step > t1:
                step = t1 - t
            if step <= 0:
                break
            x_trail = rk4_step(model, t, x_trail, step)
            t += step
            trail_steps_np.append(
                convert_points(x_trail.detach().cpu().numpy(), manifold)
            )

        # transpose: list-of-(N,3) -> N × list-of-3  ->  N × (steps,3)
        n_steps_saved = len(trail_steps_np)
        for i in range(n_trail):
            line = np.empty((n_steps_saved, 3), dtype=np.float64)
            for s, arr in enumerate(trail_steps_np):
                line[s] = arr[i]
            trail_lines.append(line)

        print(f"Computed {n_trail} trajectories  ({n_steps_saved} steps each)")

    # ---- target (original) data -------------------------------------------
    if manifold in ("sphere", "torus"):
        if args.test_data:
            target_chart = np.loadtxt(args.test_data, delimiter=",", skiprows=0, dtype=np.float64)
            if target_chart.ndim == 1:
                target_chart = target_chart.reshape(1, -1)
            if target_chart.shape[1] != 2:
                raise ValueError(f"--test-data expected N×2, got {target_chart.shape}")
        else:
            gen_fn = {"sphere": sphere_manifold.generate_target_data,
                       "torus": torus_manifold.generate_target_data}
            target_chart = gen_fn[manifold](n=args.n_points)
        target_cart = convert_points(target_chart, manifold)
    else:
        target_cart = None

    # ---- colour map for points --------------------------------------------
    default_cmap = "plasma"

    # ---- PyVista renderer -------------------------------------------------
    ws = args.window_size
    bg_colour = {"white": "white", "dark": "#1a1a2e", "black": "black"}[args.background]
    plotter = pv.Plotter(window_size=(ws, ws), off_screen=True)
    plotter.set_background(bg_colour)

    # remove axes / grid
    plotter.remove_bounds_axes()

    # ---- manifold surface -------------------------------------------------
    if manifold == "sphere":
        surf = pv_sphere_mesh(0.99)
    elif manifold == "torus":
        surf = pv_torus_mesh(2.0, 0.99)
    else:
        surf = None

    if surf is not None:
        plotter.add_mesh(
            surf,
            color="#90a4ae",
            opacity=args.manifold_opacity,
            smooth_shading=True,
            specular=0.1,
            diffuse=0.8,
        )

    # ---- trajectory lines -------------------------------------------------
    for i, line in enumerate(trail_lines):
        n_pts = line.shape[0]
        t_arr = np.linspace(0, 1, n_pts)
        pline = pv.lines_from_points(line)
        pline["_time"] = t_arr
        plotter.add_mesh(
            pline,
            scalars="_time",
            cmap="viridis",
            line_width=args.trail_width,
            opacity=0.65,
            show_scalar_bar=False,
        )

    # ---- point cloud (initial) --------------------------------------------
    init_pts = frames_cart[0]
    pc = pv.PolyData(init_pts)
    use_transition = args.color_mode == "transition"

    if use_transition:
        rgb = np.zeros((init_pts.shape[0], 3), dtype=np.uint8)
        rgb[:, 0] = 255  # red
        pc["_rgb"] = rgb
        pc_actor = plotter.add_mesh(
            pc,
            scalars="_rgb",
            rgb=True,
            point_size=args.point_size,
            render_points_as_spheres=True,
            opacity=0.9,
            show_scalar_bar=False,
        )
    elif use_logp and log_densities[0] is not None:
        pc["_logp"] = log_densities[0]  # type: ignore[arg-type]
        scalar_name = "_logp"
        pc_actor = plotter.add_mesh(
            pc,
            scalars=scalar_name,
            cmap=default_cmap,
            point_size=args.point_size,
            render_points_as_spheres=True,
            opacity=0.9,
            show_scalar_bar=False,
        )
    else:
        pc["_z"] = init_pts[:, 2]
        scalar_name = "_z"
        pc_actor = plotter.add_mesh(
            pc,
            scalars=scalar_name,
            cmap=default_cmap,
            point_size=args.point_size,
            render_points_as_spheres=True,
            opacity=0.9,
            show_scalar_bar=False,
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

    # ---- camera setup -----------------------------------------------------
    cam_distance = (
        args.cam_distance
        if args.cam_distance > 0
        else {"sphere": 2.5, "torus": 5.0, "euclidean": 6.0}.get(manifold, 4.0)
    )
    plotter.camera.position = (
        cam_distance * np.cos(np.radians(args.elev)) * np.cos(np.radians(args.azim)),
        cam_distance * np.cos(np.radians(args.elev)) * np.sin(np.radians(args.azim)),
        cam_distance * np.sin(np.radians(args.elev)),
    )
    plotter.camera.focal_point = (0, 0, 0)
    plotter.camera.view_up = (0, 0, 1)

    # ---- title overlay ----------------------------------------------------
    if args.title:
        title_colour = "black" if args.background == "white" else "#e0e0e0"
        plotter.add_text(
            args.title,
            position="upper_left",
            font_size=16,
            color=title_colour,
            shadow=True,
        )

    plotter.show(auto_close=False)

    # ---- render frames ----------------------------------------------------
    images = []
    azim_step = args.rotate / max(n_frames, 1)

    for i in range(len(frames_cart)):
        pts = frames_cart[i]

        # update point positions
        vtk_pts = pc.GetPoints()
        for j in range(pts.shape[0]):
            vtk_pts.SetPoint(j, pts[j, 0], pts[j, 1], pts[j, 2])
        vtk_pts.Modified()

        # update colours
        if use_transition:
            alpha = i / max(len(frames_cart) - 1, 1)
            rgb = pc.point_data["_rgb"]
            r = int(255 * (1 - alpha))
            b = int(255 * alpha)
            for j in range(pts.shape[0]):
                rgb[j, 0] = r
                rgb[j, 1] = 0
                rgb[j, 2] = b
        elif use_logp and log_densities[i] is not None:
            arr = log_densities[i]  # type: ignore[assignment]
            scalars = pc.point_data["_logp"]
            for j in range(len(arr)):
                scalars[j] = arr[j]
        elif not use_transition:
            z_arr = pts[:, 2]
            scalars = pc.point_data["_z"]
            for j in range(len(z_arr)):
                scalars[j] = z_arr[j]

        # rotate camera
        if azim_step != 0:
            plotter.camera.azimuth += azim_step

        plotter.render()
        img = plotter.screenshot(return_img=True)
        images.append(img)

        if (i + 1) % 10 == 0 or i == len(frames_cart) - 1:
            print(f"  Rendered frame {i + 1}/{len(frames_cart)}")

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

