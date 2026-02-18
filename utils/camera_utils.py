#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
from PIL import Image

from scene.cameras import Camera
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal

WARNED = False

def _compute_resolution(args, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.width, cam_info.height
    if args.resolution in [1, 2, 4, 8]:
        return round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution
        scale = float(global_down) * float(resolution_scale)
        return (int(orig_w / scale), int(orig_h / scale))

def _load_image(cam_info, resolution):
    """Load and resize image to target resolution (CPU-only, thread-safe)."""
    if cam_info.image is None:
        image = Image.open(cam_info.image_path)
        if cam_info.image_path.lower().endswith(('.jpg', '.jpeg')):
            image.draft('RGB', resolution)
        image = image.resize(resolution)
        image = image.convert('RGB')
        resized_image = torch.from_numpy(np.array(image)) / 255.0
        if len(resized_image.shape) == 3:
            gt_image = resized_image.permute(2, 0, 1)
        else:
            gt_image = resized_image.unsqueeze(dim=-1).permute(2, 0, 1)
        return gt_image, None
    else:
        if len(cam_info.image.split()) > 3:
            gt_image = torch.cat([PILtoTorch(im, resolution) for im in cam_info.image.split()[:3]], dim=0)
            loaded_mask = PILtoTorch(cam_info.image.split()[3], resolution)
            return gt_image, loaded_mask
        else:
            return PILtoTorch(cam_info.image, resolution), None

def loadCam(args, id, cam_info, resolution_scale):
    resolution = _compute_resolution(args, cam_info, resolution_scale)
    gt_image, loaded_mask = _load_image(cam_info, resolution)
    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T,
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY,
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device)

def cameraList_from_camInfos(cam_infos, resolution_scale, args, load_images=True):
    total = len(cam_infos)
    if total == 0:
        return []

    resolutions = [_compute_resolution(args, c, resolution_scale) for c in cam_infos]

    if not load_images:
        # Lightweight mode: skip image I/O, reuse a single zero tensor per resolution
        import math
        camera_list = []
        res_cache = {}
        for id, c in enumerate(cam_infos):
            w, h = resolutions[id]
            if (w, h) not in res_cache:
                res_cache[(w, h)] = torch.zeros(3, h, w)
            dummy = res_cache[(w, h)]
            cam = Camera(colmap_id=c.uid, R=c.R, T=c.T,
                         FoVx=c.FovX, FoVy=c.FovY,
                         image=dummy, gt_alpha_mask=None,
                         image_name=c.image_name, uid=id, data_device=args.data_device)
            cam.original_image = None
            cam.gray_image = None
            camera_list.append(cam)
        sys.stdout.write(f'Created {total} cameras (no image loading)\n')
        return camera_list

    num_workers = min(8, total)

    # Phase 1: multithread image I/O + decode + resize (CPU-only, no CUDA)
    image_data = [None] * total

    def _load_one(idx):
        return idx, _load_image(cam_infos[idx], resolutions[idx])

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_load_one, i) for i in range(total)]
        for done_count, future in enumerate(as_completed(futures), 1):
            idx, data = future.result()
            image_data[idx] = data
            sys.stdout.write(f'\rLoading images {done_count}/{total}')
            sys.stdout.flush()
    sys.stdout.write('\n')

    # Phase 2: create Camera objects sequentially (CUDA ops)
    camera_list = []
    for id, c in enumerate(cam_infos):
        gt_image, loaded_mask = image_data[id]
        cam = Camera(colmap_id=c.uid, R=c.R, T=c.T,
                     FoVx=c.FovX, FoVy=c.FovY,
                     image=gt_image, gt_alpha_mask=loaded_mask,
                     image_name=c.image_name, uid=id, data_device=args.data_device)
        camera_list.append(cam)
    image_data.clear()

    return camera_list

def camera_to_JSON(id, camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry
