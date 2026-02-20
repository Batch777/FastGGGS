#!/usr/bin/env python3
"""Viser-based viewer with server-side CUDA rasterizer rendering for GGGS.

Renders using the actual CUDA render() pipeline, streaming images to the browser.
Guaranteed pixel-perfect match with render.py output.

Usage:
    python webviewer/viser_viewer.py --output_dir output --port 8080
"""

import argparse
import json
import math
import os
import sys
import threading
import time

import numpy as np
import torch
from scipy.spatial.transform import Rotation

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from argparse import ArgumentParser

import viser

from gaussian_renderer import render
from scene.cameras import MiniCam
from scene.gaussian_model import GaussianModel
from utils.graphics_utils import getProjectionMatrix


def find_model_dirs(root: str) -> dict[str, str]:
    """Scan root for directories containing a cfg_args file."""
    models = {}
    for dirpath, _, filenames in os.walk(root):
        if "cfg_args" in filenames:
            rel = os.path.relpath(dirpath, root)
            models[rel] = dirpath
    return dict(sorted(models.items()))


def find_iterations(model_dir: str) -> list[int]:
    """List available point cloud iterations."""
    pc_dir = os.path.join(model_dir, "point_cloud")
    if not os.path.isdir(pc_dir):
        return []
    iters = []
    for name in os.listdir(pc_dir):
        if name.startswith("iteration_"):
            try:
                iters.append(int(name.split("_")[1]))
            except (ValueError, IndexError):
                pass
    return sorted(iters)


def load_cfg(model_dir: str):
    """Parse the saved cfg_args Namespace."""
    from argparse import Namespace
    with open(os.path.join(model_dir, "cfg_args")) as f:
        return eval(f.read())


def load_cameras_json(model_dir: str) -> list[dict]:
    """Load cameras.json for training camera navigation."""
    path = os.path.join(model_dir, "cameras.json")
    if not os.path.isfile(path):
        return []
    with open(path) as f:
        return json.load(f)


def viser_cam_to_minicam(camera, w: int, h: int) -> MiniCam:
    """Convert a viser CameraHandle to a GGGS MiniCam for render().

    viser camera provides:
      - wxyz: quaternion (w,x,y,z) for C2W orientation
      - position: camera center in world coordinates
      - fov: vertical FOV in radians
      - aspect: width / height
    """
    # Quaternion wxyz → C2W rotation matrix
    wxyz = np.asarray(camera.wxyz, dtype=np.float64)
    R_c2w = Rotation.from_quat([wxyz[1], wxyz[2], wxyz[3], wxyz[0]]).as_matrix()
    pos = np.asarray(camera.position, dtype=np.float64)

    # Build W2C matrix: R_w2c = R_c2w^T, t_w2c = -R_w2c @ pos
    R_w2c = R_c2w.T
    t_w2c = -R_w2c @ pos
    W2C = np.eye(4, dtype=np.float32)
    W2C[:3, :3] = R_w2c.astype(np.float32)
    W2C[:3, 3] = t_w2c.astype(np.float32)

    # GGGS stores matrices transposed (column-major convention)
    world_view = torch.from_numpy(W2C).transpose(0, 1).cuda()

    fovy = float(camera.fov)
    fovx = 2.0 * math.atan(camera.aspect * math.tan(fovy * 0.5))

    proj = getProjectionMatrix(0.01, 100.0, fovx, fovy).transpose(0, 1).cuda()
    full_proj = world_view @ proj

    return MiniCam(w, h, fovy, fovx, 0.01, 100.0, world_view, full_proj)


def main():
    parser = argparse.ArgumentParser(description="Viser CUDA Rasterizer Viewer")
    parser.add_argument("--output_dir", "-o", default=None,
                        help="Root directory to scan for trained model(s) on startup")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    # Discover models from initial output_dir (if provided)
    # Key: display label, Value: absolute path to model dir
    models: dict[str, str] = {}
    if args.output_dir:
        models.update(find_model_dirs(args.output_dir))
    if models:
        print(f"Found {len(models)} model(s): {list(models.keys())}")
    else:
        print("No initial models. Use the 'Scan Path' GUI to add server directories.")

    # Shared render state
    state = {
        "gaussians": None,
        "pipeline": None,
        "kernel_size": 0.0,
    }
    render_lock = threading.Lock()

    # Viser server
    server = viser.ViserServer(host="0.0.0.0", port=args.port)

    # --- GUI: Model selection ---
    model_names = list(models.keys()) if models else ["(none)"]
    gui_model = server.gui.add_dropdown(
        "Model", options=model_names, initial_value=model_names[0])

    def _get_iter_opts():
        if gui_model.value == "(none)" or gui_model.value not in models:
            return ["N/A"]
        iters = find_iterations(models[gui_model.value])
        return [str(i) for i in iters] if iters else ["N/A"]

    iter_opts = _get_iter_opts()
    gui_iter = server.gui.add_dropdown(
        "Iteration", options=iter_opts, initial_value=iter_opts[-1])

    @gui_model.on_update
    def _(_):
        opts = _get_iter_opts()
        gui_iter.options = opts
        gui_iter.value = opts[-1]

    gui_load = server.gui.add_button("Load", color="green")

    # --- GUI: Render settings ---
    gui_res = server.gui.add_slider(
        "Resolution Scale", min=0.25, max=1.0, step=0.05, initial_value=0.5)
    gui_sh = server.gui.add_slider(
        "SH Degree", min=0, max=3, step=1, initial_value=3)
    gui_white_bg = server.gui.add_checkbox(
        "White Background", initial_value=False)
    gui_status = server.gui.add_text(
        "Status", initial_value="No model loaded", disabled=True)

    # --- GUI: Server path scanner (at the bottom) ---
    def _refresh_model_dropdown():
        """Sync the Model dropdown with current `models` dict."""
        names = list(models.keys()) if models else ["(none)"]
        gui_model.options = names
        gui_model.value = names[0]

    with server.gui.add_folder("Scan Path"):
        gui_scan_path = server.gui.add_text(
            "Server Path",
            initial_value=os.path.abspath(args.output_dir) if args.output_dir else os.getcwd(),
        )
        gui_scan_btn = server.gui.add_button("Scan")

        @gui_scan_btn.on_click
        def _(_):
            path = gui_scan_path.value.strip()
            if not os.path.isdir(path):
                gui_status.value = f"Not a directory: {path}"
                return
            found = find_model_dirs(path)
            if not found:
                gui_status.value = f"No models in {path}"
                return
            for rel, absdir in found.items():
                label = os.path.join(os.path.basename(path), rel)
                models[label] = absdir
            _refresh_model_dropdown()
            gui_status.value = f"Added {len(found)} model(s) from {path}"
            print(f"Scanned {path}: +{len(found)} model(s), total {len(models)}")

    # Camera folder handle for cleanup on reload
    cam_folder = {"handle": None}

    def _bind_cam_button(button, entry):
        """Attach click handler to navigate client camera to a training view."""
        @button.on_click
        def _(ev: viser.GuiEvent):
            if ev.client is None:
                return
            # cameras.json stores C2W rotation and world position
            R_c2w = np.array(entry["rotation"])
            pos = np.array(entry["position"])
            fov = 2.0 * math.atan(entry["height"] / (2.0 * entry["fy"]))

            # C2W rotation → quaternion (scipy xyzw → viser wxyz)
            q = Rotation.from_matrix(R_c2w).as_quat()  # [x, y, z, w]
            wxyz = (float(q[3]), float(q[0]), float(q[1]), float(q[2]))

            with ev.client.atomic():
                ev.client.camera.wxyz = wxyz
                ev.client.camera.position = pos
                ev.client.camera.fov = fov

    @gui_load.on_click
    def _(event: viser.GuiEvent):
        if gui_model.value == "(none)" or gui_model.value not in models:
            gui_status.value = "No model selected — scan a path first"
            return
        model_dir = models[gui_model.value]
        iter_str = gui_iter.value
        if iter_str == "N/A":
            gui_status.value = "No iterations found"
            return

        iteration = int(iter_str)
        ply_path = os.path.join(
            model_dir, "point_cloud", f"iteration_{iteration}", "point_cloud.ply")
        if not os.path.isfile(ply_path):
            gui_status.value = "PLY not found"
            return

        gui_status.value = "Loading..."

        # Parse saved config
        cfg = load_cfg(model_dir)
        sh_deg = getattr(cfg, "sh_degree", 3)
        sg_deg = getattr(cfg, "sg_degree", 0)
        ks = getattr(cfg, "kernel_size", 0.0)
        white_bg = getattr(cfg, "white_background", False)

        # Pipeline params (defaults: debug=False, convert_SHs_python=False)
        from arguments import PipelineParams
        tmp_parser = ArgumentParser()
        pp = PipelineParams(tmp_parser)
        pipe = pp.extract(tmp_parser.parse_args([]))

        # Load Gaussians (load_ply also loads the precomputed 3D filter)
        gs = GaussianModel(sh_deg, sg_deg)
        gs.load_ply(ply_path)
        gs.active_sh_degree = min(int(gui_sh.value), sh_deg)
        gs.active_sg_degree = sg_deg

        # Update GUI to match config
        gui_white_bg.value = white_bg

        # Update shared state
        state["gaussians"] = gs
        state["pipeline"] = pipe
        state["kernel_size"] = ks

        # Rebuild training camera buttons
        if cam_folder["handle"] is not None:
            cam_folder["handle"].remove()
        cameras = load_cameras_json(model_dir)
        if cameras:
            cam_folder["handle"] = server.gui.add_folder("Training Cameras")
            with cam_folder["handle"]:
                for entry in cameras:
                    name = entry.get("img_name", f"cam_{entry['id']}")
                    btn = server.gui.add_button(name)
                    _bind_cam_button(btn, entry)

        n = gs.get_xyz.shape[0]
        gui_status.value = f"{gui_model.value} | iter {iteration} | {n:,} pts"
        print(f"Loaded {gui_model.value} iter {iteration}: {n:,} Gaussians, ks={ks}")

    # --- Per-client camera rendering ---
    @server.on_client_connect
    def _(client: viser.ClientHandle):
        @client.camera.on_update
        def _(_cam):
            gs = state["gaussians"]
            pipe = state["pipeline"]
            if gs is None or pipe is None:
                return

            # Drop frame if already rendering (debounce)
            if not render_lock.acquire(blocking=False):
                return

            try:
                scale = gui_res.value
                cw = client.camera.image_width
                ch = client.camera.image_height
                if cw is None or ch is None or cw <= 0 or ch <= 0:
                    return
                w = max(64, int(cw * scale))
                h = max(64, int(ch * scale))

                gs.active_sh_degree = min(int(gui_sh.value), gs.max_sh_degree)

                if gui_white_bg.value:
                    bg = torch.ones(3, dtype=torch.float32, device="cuda")
                else:
                    bg = torch.zeros(3, dtype=torch.float32, device="cuda")

                minicam = viser_cam_to_minicam(client.camera, w, h)

                with torch.no_grad():
                    result = render(minicam, gs, pipe, bg,
                                    kernel_size=state["kernel_size"])

                img = result["render"]  # (3, H, W)
                img_np = (
                    img.clamp(0, 1).permute(1, 2, 0).mul(255).byte().cpu().numpy()
                )
                client.scene.set_background_image(
                    img_np, format="jpeg", jpeg_quality=92)
            except Exception as e:
                print(f"Render error: {e}")
            finally:
                render_lock.release()

    print(f"\nViser viewer running at http://localhost:{args.port}")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nShutting down.")


if __name__ == "__main__":
    main()
