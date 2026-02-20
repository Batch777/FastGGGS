#!/usr/bin/env python3
"""Render instance-colored images from a trained Gaussian model.

Workflow:
1. Load a pre-trained Gaussian model and scene (with SAM masks).
2. Run Trace3D-style 3D instance segmentation:
   build_instance_weight_matrix → majority_vote_merge → assign_labels.
3. Re-color each Gaussian by its instance ID, then render each view.
4. Save instance-colored renderings and (optionally) side-by-side comparisons.

Usage:
    python render_instances.py -m output/cuhk_lower --iteration 30000
    python render_instances.py -m output/cuhk_lower --skip_test --mask_dir masks
"""

import os
import sys
from argparse import ArgumentParser

import cv2
import numpy as np
import torch
import torchvision
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render
from scene import Scene
from utils.instance_utils import assign_instance_labels, assign_semantic_labels
from utils.general_utils import safe_state
from utils.sh_utils import RGB2SH


# ---- Instance color generation ----

def generate_instance_colors(max_id: int) -> torch.Tensor:
    """Generate distinct colors for instance IDs 0..max_id.

    Returns (max_id+1, 3) float tensor in [0, 1] (RGB).
    ID 0 (background) → black.
    """
    colors = torch.zeros(max_id + 1, 3)
    for i in range(1, max_id + 1):
        hue = (i * 137.508) % 360.0  # golden angle
        # HSV → RGB
        c = 0.9  # saturation
        v = 0.95  # value
        h = hue / 60.0
        x_val = c * (1 - abs(h % 2 - 1))
        hi = int(h) % 6
        if hi == 0:
            r, g, b = c, x_val, 0
        elif hi == 1:
            r, g, b = x_val, c, 0
        elif hi == 2:
            r, g, b = 0, c, x_val
        elif hi == 3:
            r, g, b = 0, x_val, c
        elif hi == 4:
            r, g, b = x_val, 0, c
        else:
            r, g, b = c, 0, x_val
        m = v - c
        colors[i] = torch.tensor([r + m, g + m, b + m])
    return colors


def render_instance_set(
    model_path: str,
    name: str,
    iteration: int,
    views,
    gaussians: GaussianModel,
    pipeline,
    background: torch.Tensor,
    kernel_size: float,
    save_sidebyside: bool = True,
):
    """Render instance-colored images for a set of views."""
    inst_path = os.path.join(model_path, name, f"ours_{iteration}", "instances")
    os.makedirs(inst_path, exist_ok=True)
    if save_sidebyside:
        sbs_path = os.path.join(model_path, name, f"ours_{iteration}", "sidebyside")
        os.makedirs(sbs_path, exist_ok=True)

    # Determine max instance ID
    max_id = int(gaussians._instance_id.max().item())
    if max_id == 0:
        print("Warning: all instance IDs are 0 (no instances assigned)")
        max_id = 1
    print(f"Max instance ID: {max_id}")
    instance_colors = generate_instance_colors(max_id).cuda()  # (max_id+1, 3)

    # Save original SH parameters
    orig_features_dc = gaussians._features_dc.data.clone()
    orig_features_rest = gaussians._features_rest.data.clone()
    # Save original SG parameters (if any)
    has_sg = gaussians._sg_color.numel() > 0
    if has_sg:
        orig_sg_color = gaussians._sg_color.data.clone()

    # Set Gaussian colors to instance colors
    # SH DC: color = SH_C0 * features_dc + 0.5, so features_dc = (color - 0.5) / SH_C0
    inst_ids = gaussians._instance_id.clamp(0, max_id)
    target_rgb = instance_colors[inst_ids]  # (N, 3)
    gaussians._features_dc.data[:] = RGB2SH(target_rgb).unsqueeze(1)  # (N, 1, 3)
    gaussians._features_rest.data.zero_()
    if has_sg:
        gaussians._sg_color.data.zero_()

    # Also temporarily bump up active SH/SG degrees to 0 only
    orig_sh_degree = gaussians.active_sh_degree
    orig_sg_degree = gaussians.active_sg_degree
    gaussians.active_sh_degree = 0
    gaussians.active_sg_degree = 0

    for idx, view in enumerate(tqdm(views, desc=f"Rendering instances ({name})")):
        result = render(view, gaussians, pipeline, background, kernel_size=kernel_size)
        inst_image = torch.clamp(result["render"], 0.0, 1.0)

        torchvision.utils.save_image(
            inst_image,
            os.path.join(inst_path, f"{idx:05d}.png"),
        )

        if save_sidebyside:
            # Render GT (original image)
            gt = view.original_image[0:3, :, :]
            # Concat horizontally: GT | Instance
            sbs = torch.cat([gt.cpu(), inst_image.cpu()], dim=2)
            torchvision.utils.save_image(
                sbs,
                os.path.join(sbs_path, f"{idx:05d}.png"),
            )

    # Restore original parameters
    gaussians._features_dc.data[:] = orig_features_dc
    gaussians._features_rest.data[:] = orig_features_rest
    if has_sg:
        gaussians._sg_color.data[:] = orig_sg_color
    gaussians.active_sh_degree = orig_sh_degree
    gaussians.active_sg_degree = orig_sg_degree

    print(f"Instance renderings saved to {inst_path}")
    if save_sidebyside:
        print(f"Side-by-side comparisons saved to {sbs_path}")


def render_semantic_set(
    model_path: str,
    name: str,
    iteration: int,
    views,
    gaussians: GaussianModel,
    pipeline,
    background: torch.Tensor,
    kernel_size: float,
    category_names: list,
    category_colors_rgb: list,
):
    """Render category-colored and instance-colored images for semantic segmentation."""
    cat_path = os.path.join(model_path, name, f"ours_{iteration}", "semantic_category")
    inst_path = os.path.join(model_path, name, f"ours_{iteration}", "semantic_instance")
    os.makedirs(cat_path, exist_ok=True)
    os.makedirs(inst_path, exist_ok=True)

    # Build category color LUT: cat_id → RGB [0,1]
    n_cats = len(category_names)
    cat_color_lut = torch.zeros(n_cats + 1, 3)  # +1 for index safety
    for i, rgb in enumerate(category_colors_rgb):
        if i < n_cats + 1:
            cat_color_lut[i] = torch.tensor([c / 255.0 for c in rgb])

    # Instance color LUT
    max_id = int(gaussians._instance_id.max().item())
    inst_color_lut = generate_instance_colors(max(max_id, 1)).cuda()

    # Save original parameters
    orig_features_dc = gaussians._features_dc.data.clone()
    orig_features_rest = gaussians._features_rest.data.clone()
    has_sg = gaussians._sg_color.numel() > 0
    if has_sg:
        orig_sg_color = gaussians._sg_color.data.clone()
    orig_sh_degree = gaussians.active_sh_degree
    orig_sg_degree = gaussians.active_sg_degree

    sem_cats = gaussians._semantic_category.clamp(0, n_cats)
    inst_ids = gaussians._instance_id.clamp(0, max_id)

    # --- Render category-colored ---
    target_rgb = cat_color_lut.cuda()[sem_cats]
    gaussians._features_dc.data[:] = RGB2SH(target_rgb).unsqueeze(1)
    gaussians._features_rest.data.zero_()
    if has_sg:
        gaussians._sg_color.data.zero_()
    gaussians.active_sh_degree = 0
    gaussians.active_sg_degree = 0

    for idx, view in enumerate(tqdm(views, desc=f"Category-colored ({name})")):
        result = render(view, gaussians, pipeline, background, kernel_size=kernel_size)
        img = torch.clamp(result["render"], 0.0, 1.0)
        torchvision.utils.save_image(img, os.path.join(cat_path, f"{idx:05d}.png"))

    # --- Render instance-colored ---
    target_rgb = inst_color_lut[inst_ids]
    gaussians._features_dc.data[:] = RGB2SH(target_rgb).unsqueeze(1)
    gaussians._features_rest.data.zero_()
    if has_sg:
        gaussians._sg_color.data.zero_()

    for idx, view in enumerate(tqdm(views, desc=f"Instance-colored ({name})")):
        result = render(view, gaussians, pipeline, background, kernel_size=kernel_size)
        img = torch.clamp(result["render"], 0.0, 1.0)
        torchvision.utils.save_image(img, os.path.join(inst_path, f"{idx:05d}.png"))

    # Restore
    gaussians._features_dc.data[:] = orig_features_dc
    gaussians._features_rest.data[:] = orig_features_rest
    if has_sg:
        gaussians._sg_color.data[:] = orig_sg_color
    gaussians.active_sh_degree = orig_sh_degree
    gaussians.active_sg_degree = orig_sg_degree

    print(f"Category-colored renders saved to {cat_path}")
    print(f"Instance-colored renders saved to {inst_path}")


def _find_best_views(gaussians, views, instance_id, n_best=5):
    """Find the n_best views where instance_id has the most visible Gaussians."""
    from utils.instance_utils import _project_gaussians

    xyz = gaussians.get_xyz
    mask = gaussians._instance_id == instance_id
    inst_xyz = xyz[mask]

    scores = []
    for vi, view in enumerate(views):
        valid, uv = _project_gaussians(inst_xyz, view)
        scores.append((valid.sum().item(), vi))

    scores.sort(key=lambda x: -x[0])
    return [vi for _, vi in scores[:n_best] if _ > 0]


def render_isolated_instances(
    model_path: str,
    name: str,
    iteration: int,
    views,
    gaussians: GaussianModel,
    pipeline,
    kernel_size: float,
    top_k: int = 20,
    n_best_views: int = 5,
):
    """Render each instance isolated with original appearance on black background.

    For each of the top_k largest instances, finds the n_best_views cameras
    where the instance is most visible, and renders only those views.
    """
    base_path = os.path.join(model_path, name, f"ours_{iteration}", "isolated")
    os.makedirs(base_path, exist_ok=True)

    background = torch.zeros(3, dtype=torch.float32, device="cuda")  # black bg

    inst_ids = gaussians._instance_id
    max_id = int(inst_ids.max().item())

    # Find top-K instances by Gaussian count (skip ID 0 = background)
    counts = torch.bincount(inst_ids, minlength=max_id + 1)
    counts[0] = 0  # exclude background
    topk_ids = counts.argsort(descending=True)[:top_k]
    topk_ids = topk_ids[counts[topk_ids] > 0]  # filter out empty
    print(f"Rendering {len(topk_ids)} largest instances, {n_best_views} best views each")
    for rank, iid in enumerate(topk_ids):
        print(f"  #{rank+1}: instance {iid.item()} — {counts[iid].item()} Gaussians")

    # Save original opacity
    orig_opacity = gaussians._opacity.data.clone()
    view_list = list(views)

    for rank, iid in enumerate(topk_ids):
        iid_val = iid.item()
        inst_dir = os.path.join(base_path, f"instance_{iid_val:04d}")
        os.makedirs(inst_dir, exist_ok=True)

        # Find best views for this instance
        best_vis = _find_best_views(gaussians, view_list, iid_val, n_best=n_best_views)
        if not best_vis:
            print(f"  Instance {iid_val}: no visible views, skipping")
            continue

        # Mask: keep only this instance's Gaussians visible
        inst_mask = inst_ids == iid_val
        gaussians._opacity.data[~inst_mask] = -20.0

        for vi in tqdm(
            best_vis,
            desc=f"Isolated #{rank+1}/{len(topk_ids)} (inst {iid_val})",
            leave=False,
        ):
            view = view_list[vi]
            result = render(view, gaussians, pipeline, background, kernel_size=kernel_size)
            img = torch.clamp(result["render"], 0.0, 1.0)
            torchvision.utils.save_image(img, os.path.join(inst_dir, f"{vi:05d}.png"))

        # Restore opacity
        gaussians._opacity.data[:] = orig_opacity

    print(f"Isolated instance renders saved to {base_path}")


def render_foreground_only(
    model_path: str,
    name: str,
    iteration: int,
    views,
    gaussians: GaussianModel,
    pipeline,
    kernel_size: float,
):
    """Render all foreground instances (original appearance) on black background.

    Hides background Gaussians (instance_id == 0), keeps all building instances.
    """
    fg_path = os.path.join(model_path, name, f"ours_{iteration}", "foreground")
    os.makedirs(fg_path, exist_ok=True)

    background = torch.zeros(3, dtype=torch.float32, device="cuda")

    # Hide background Gaussians
    orig_opacity = gaussians._opacity.data.clone()
    bg_mask = gaussians._instance_id == 0
    gaussians._opacity.data[bg_mask] = -20.0

    n_fg = (~bg_mask).sum().item()
    n_bg = bg_mask.sum().item()
    print(f"Foreground: {n_fg} Gaussians, Background (hidden): {n_bg}")

    for idx, view in enumerate(tqdm(views, desc=f"Rendering foreground ({name})")):
        result = render(view, gaussians, pipeline, background, kernel_size=kernel_size)
        img = torch.clamp(result["render"], 0.0, 1.0)

        gt = view.original_image[0:3, :, :]  # (3, H, W)

        # SAM 2D ground truth: mask GT image by instance_mask > 0
        mask_2d = getattr(view, "instance_mask", None)
        if mask_2d is not None:
            # Resize mask to match GT resolution if needed
            H_gt, W_gt = gt.shape[1], gt.shape[2]
            if mask_2d.shape[0] != H_gt or mask_2d.shape[1] != W_gt:
                mask_2d = torch.nn.functional.interpolate(
                    mask_2d.float().unsqueeze(0).unsqueeze(0),
                    size=(H_gt, W_gt), mode="nearest",
                ).squeeze().long()
            fg_mask_2d = (mask_2d > 0).float().cpu()  # (H, W)
            gt_masked = gt.cpu() * fg_mask_2d.unsqueeze(0)  # black out background
        else:
            gt_masked = torch.zeros_like(gt.cpu())

        # Side-by-side: GT | SAM 2D mask | 3D foreground
        sbs = torch.cat([gt.cpu(), gt_masked, img.cpu()], dim=2)
        torchvision.utils.save_image(sbs, os.path.join(fg_path, f"{idx:05d}.png"))

    # Restore
    gaussians._opacity.data[:] = orig_opacity
    print(f"Foreground renders saved to {fg_path}")


def run(
    dataset: ModelParams,
    iteration: int,
    pipeline: PipelineParams,
    skip_train: bool,
    skip_test: bool,
    mask_dir: str,
    tau_affinity: float,
    render_isolated: bool = False,
    isolated_topk: int = 20,
    isolated_best_views: int = 5,
    semantic: bool = False,
    prompts: str = "building,tree,road",
):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, dataset.sg_degree)

        # Enable instance mask loading
        dataset.use_instance_seg = True
        dataset.instance_mask_dir = mask_dir

        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # --- Stage 1: 3D Instance Segmentation ---
        train_cameras = scene.getTrainCameras()
        print(f"\n=== Running 3D Instance Segmentation ===")
        print(f"Number of Gaussians: {gaussians.get_xyz.shape[0]}")
        print(f"Number of training cameras: {len(train_cameras)}")

        if semantic:
            print(f"Mode: SEMANTIC (multi-category)")
            assign_semantic_labels(
                train_cameras, gaussians,
                tau_aff=tau_affinity,
                max_cameras=150,
            )
        else:
            assign_instance_labels(
                train_cameras, gaussians,
                tau_aff=tau_affinity,
                max_cameras=150,
            )

        # --- Semantic: save segments.json and re-save PLY ---
        if semantic:
            category_names_list = [p.strip() for p in prompts.split(",") if p.strip()]
            # Default colors (RGB) matching generate_masks_semantic.py
            default_colors = {
                "building": [200, 0, 0], "tree": [0, 180, 0], "road": [0, 120, 180],
                "car": [200, 200, 0], "person": [200, 0, 200], "sky": [0, 200, 200],
            }
            all_names = ["background"] + category_names_list
            all_colors = [[0, 0, 0]] + [default_colors.get(n, [128, 128, 128]) for n in category_names_list]

            # Save segments.json
            ply_dir = os.path.join(dataset.model_path, "point_cloud", f"iteration_{scene.loaded_iter}")
            ply_path = os.path.join(ply_dir, "point_cloud.ply")
            gaussians.save_segments_json(ply_path, all_names, all_colors)

            # Re-save PLY with semantic_category populated
            gaussians.save_ply(ply_path)
            print(f"Re-saved PLY with semantic_category to {ply_path}")

        # --- Stage 2: Render ---
        print(f"\n=== Rendering Instance Views ===")
        if not skip_train:
            if semantic:
                category_names_list = [p.strip() for p in prompts.split(",") if p.strip()]
                default_colors = {
                    "building": [200, 0, 0], "tree": [0, 180, 0], "road": [0, 120, 180],
                    "car": [200, 200, 0], "person": [200, 0, 200], "sky": [0, 200, 200],
                }
                all_names = ["background"] + category_names_list
                all_colors = [[0, 0, 0]] + [default_colors.get(n, [128, 128, 128]) for n in category_names_list]
                render_semantic_set(
                    dataset.model_path,
                    "train",
                    scene.loaded_iter,
                    train_cameras,
                    gaussians,
                    pipeline,
                    background,
                    dataset.kernel_size,
                    all_names,
                    all_colors,
                )
            else:
                render_instance_set(
                    dataset.model_path,
                    "train",
                    scene.loaded_iter,
                    train_cameras,
                    gaussians,
                    pipeline,
                    background,
                    dataset.kernel_size,
                )
            print(f"\n=== Rendering Foreground Only ===")
            render_foreground_only(
                dataset.model_path,
                "train",
                scene.loaded_iter,
                train_cameras,
                gaussians,
                pipeline,
                dataset.kernel_size,
            )
            if render_isolated:
                print(f"\n=== Rendering Isolated Instances ===")
                render_isolated_instances(
                    dataset.model_path,
                    "train",
                    scene.loaded_iter,
                    train_cameras,
                    gaussians,
                    pipeline,
                    dataset.kernel_size,
                    top_k=isolated_topk,
                    n_best_views=isolated_best_views,
                )

        if not skip_test:
            test_cameras = scene.getTestCameras()
            if len(test_cameras) > 0:
                render_instance_set(
                    dataset.model_path,
                    "test",
                    scene.loaded_iter,
                    test_cameras,
                    gaussians,
                    pipeline,
                    background,
                    dataset.kernel_size,
                )
            else:
                print("No test cameras found, skipping test render.")


if __name__ == "__main__":
    parser = ArgumentParser(description="Instance segmentation rendering")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--mask_dir", default="masks", type=str,
                        help="Mask subdirectory under source_path (default: masks)")
    parser.add_argument("--tau_affinity", default=0.3, type=float,
                        help="Jaccard affinity threshold for cross-view merge (default: 0.3)")
    parser.add_argument("--render_isolated", action="store_true",
                        help="Also render each instance isolated with original appearance")
    parser.add_argument("--isolated_topk", default=20, type=int,
                        help="Number of largest instances to render isolated (default: 20)")
    parser.add_argument("--isolated_best_views", default=5, type=int,
                        help="Number of best views per instance (default: 5)")
    parser.add_argument("--semantic", action="store_true",
                        help="Use multi-category semantic segmentation mode")
    parser.add_argument("--prompts", default="building,tree,road", type=str,
                        help="Comma-separated category names for semantic mode (default: building,tree,road)")

    args = get_combined_args(parser)
    print("Rendering instances for " + args.model_path)

    safe_state(args.quiet)

    run(
        dataset=model.extract(args),
        iteration=args.iteration,
        pipeline=pipeline.extract(args),
        skip_train=args.skip_train,
        skip_test=args.skip_test,
        mask_dir=args.mask_dir,
        tau_affinity=args.tau_affinity,
        render_isolated=args.render_isolated,
        isolated_topk=args.isolated_topk,
        isolated_best_views=args.isolated_best_views,
        semantic=args.semantic,
        prompts=args.prompts,
    )
