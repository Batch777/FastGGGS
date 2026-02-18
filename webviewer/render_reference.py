#!/usr/bin/env python3
"""Render a reference image using the CUDA rasterizer for comparison with the WebGL viewer.

Usage:
    python webviewer/render_reference.py -m output/cuhk_lower [--iteration 30000] [--camera_idx 0]

Outputs:
    webviewer/reference_<idx>.png   — rendered image
    webviewer/reference_camera.json — camera parameters for WebGL viewer comparison
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torchvision

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from arguments import ModelParams, PipelineParams
from gaussian_renderer import GaussianModel, render
from scene.cameras import Camera
from utils.graphics_utils import focal2fov
from utils.system_utils import searchForMaxIteration


def camera_from_json(cam_entry, resolution_scale=1):
    """Reconstruct a Camera object from a cameras.json entry."""
    width = cam_entry['width']
    height = cam_entry['height']
    fx = cam_entry['fx']
    fy = cam_entry['fy']
    position = np.array(cam_entry['position'], dtype=np.float64)
    rotation = np.array(cam_entry['rotation'], dtype=np.float64)  # C2W rotation

    # Apply resolution scaling
    width = round(width / resolution_scale)
    height = round(height / resolution_scale)
    fx = fx / resolution_scale
    fy = fy / resolution_scale

    # Camera constructor takes R (C2W rotation) and T (W2C translation)
    # From cameras.json: rotation = C2W rotation, position = camera center in world
    # W2C = [R_c2w^T | -R_c2w^T @ position]
    R = rotation  # C2W rotation
    T = -R.T @ position  # W2C translation

    FoVx = focal2fov(fx, width)
    FoVy = focal2fov(fy, height)

    # Dummy image (we only need rendering, not GT comparison)
    dummy_image = torch.zeros(3, height, width)

    cam = Camera(
        colmap_id=cam_entry.get('id', 0),
        R=R, T=T,
        FoVx=FoVx, FoVy=FoVy,
        image=dummy_image, gt_alpha_mask=None,
        image_name=cam_entry.get('img_name', 'reference'),
        uid=cam_entry.get('id', 0),
        data_device='cuda'
    )
    return cam


def main():
    parser = argparse.ArgumentParser(description='Render CUDA reference image')
    parser.add_argument('-m', '--model_path', required=True, help='Model output directory')
    parser.add_argument('--iteration', type=int, default=-1, help='Iteration to load (-1 = latest)')
    parser.add_argument('--camera_idx', type=int, default=0, help='Camera index in cameras.json')
    parser.add_argument('--output_dir', default=None, help='Output directory (default: webviewer/)')
    args = parser.parse_args()

    # Load cfg_args
    cfg_path = os.path.join(args.model_path, 'cfg_args')
    if not os.path.exists(cfg_path):
        print(f'cfg_args not found: {cfg_path}')
        return

    # Parse the saved Namespace
    with open(cfg_path) as f:
        cfg_text = f.read()
    from argparse import Namespace
    cfg = eval(cfg_text)

    # Determine iteration
    if args.iteration == -1:
        iteration = searchForMaxIteration(os.path.join(args.model_path, 'point_cloud'))
    else:
        iteration = args.iteration
    print(f'Using iteration {iteration}')

    # Load Gaussians
    ply_path = os.path.join(args.model_path, 'point_cloud',
                            f'iteration_{iteration}', 'point_cloud.ply')
    if not os.path.exists(ply_path):
        print(f'PLY not found: {ply_path}')
        return

    gaussians = GaussianModel(cfg.sh_degree, cfg.sg_degree)
    gaussians.load_ply(ply_path)
    gaussians.active_sh_degree = cfg.sh_degree
    print(f'Loaded {gaussians.get_xyz.shape[0]} Gaussians (SH={cfg.sh_degree}, SG={cfg.sg_degree})')
    print(f'  filter_3D range: [{gaussians.filter_3D.min().item():.6f}, {gaussians.filter_3D.max().item():.6f}]')

    # Load cameras.json
    cameras_path = os.path.join(args.model_path, 'cameras.json')
    if not os.path.exists(cameras_path):
        print(f'cameras.json not found: {cameras_path}')
        return

    with open(cameras_path) as f:
        cameras_json = json.load(f)
    print(f'Found {len(cameras_json)} cameras')

    if args.camera_idx >= len(cameras_json):
        print(f'Camera index {args.camera_idx} out of range (0-{len(cameras_json)-1})')
        return

    cam_entry = cameras_json[args.camera_idx]
    print(f'Using camera {args.camera_idx}: {cam_entry.get("img_name", "?")} '
          f'({cam_entry["width"]}x{cam_entry["height"]})')

    # Apply same resolution scaling as training
    resolution_scale = cfg.resolution
    cam = camera_from_json(cam_entry, resolution_scale)
    print(f'  Resolution after {resolution_scale}x scaling: {cam.image_width}x{cam.image_height}')
    print(f'  FoVx={math.degrees(cam.FoVx):.1f}° FoVy={math.degrees(cam.FoVy):.1f}°')
    print(f'  Fx={cam.Fx:.1f} Fy={cam.Fy:.1f}')

    # Setup rendering
    bg_color = [1, 1, 1] if cfg.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device='cuda')

    # Create a minimal pipe object
    class PipeObj:
        debug = False
    pipe = PipeObj()

    # Render
    print('Rendering...')
    with torch.no_grad():
        result = render(cam, gaussians, pipe, background, kernel_size=cfg.kernel_size)
    rendered = result['render']  # (3, H, W)
    print(f'  Output shape: {rendered.shape}')

    # Save image
    output_dir = args.output_dir or os.path.join(os.path.dirname(__file__))
    os.makedirs(output_dir, exist_ok=True)

    img_path = os.path.join(output_dir, f'reference_{args.camera_idx}.png')
    torchvision.utils.save_image(rendered, img_path)
    print(f'  Saved: {img_path}')

    # Export camera params for WebGL viewer comparison
    # The view matrix as used by the CUDA rasterizer (column-major float32)
    view_matrix = cam.world_view_transform.cpu().numpy().tolist()
    proj_matrix = cam.projection_matrix.cpu().numpy().tolist()
    camera_center = cam.camera_center.cpu().numpy().tolist()

    cam_info = {
        'camera_idx': args.camera_idx,
        'img_name': cam_entry.get('img_name', ''),
        'width': cam.image_width,
        'height': cam.image_height,
        'fx': cam.Fx,
        'fy': cam.Fy,
        'FoVx': cam.FoVx,
        'FoVy': cam.FoVy,
        'view_matrix_colmajor': view_matrix,
        'projection_matrix_colmajor': proj_matrix,
        'camera_center': camera_center,
        # Also include the original cameras.json entry for the WebGL viewer
        'cameras_json_entry': cam_entry,
    }
    cam_path = os.path.join(output_dir, 'reference_camera.json')
    with open(cam_path, 'w') as f:
        json.dump(cam_info, f, indent=2)
    print(f'  Camera info: {cam_path}')

    print('Done!')


if __name__ == '__main__':
    main()
