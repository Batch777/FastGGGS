import random

import torch

from gaussian_renderer import render_for_metric
from utils.loss_utils import l1_loss, ssim


def sampling_cameras(my_viewpoint_stack, num_cams=10):
    """Randomly sample a given number of cameras from the viewpoint stack."""
    camlist = []
    for _ in range(num_cams):
        loc = random.randint(0, len(my_viewpoint_stack) - 1)
        camlist.append(my_viewpoint_stack.pop(loc))
    return camlist


def get_loss(reconstructed_image, original_image):
    """Normalized L1 loss map -> (loss - min) / (max - min)."""
    l1 = torch.mean(torch.abs(reconstructed_image - original_image), 0).detach()
    l1_norm = (l1 - torch.min(l1)) / (torch.max(l1) - torch.min(l1) + 1e-8)
    return l1_norm


def compute_photometric_loss(viewpoint_cam, image):
    """Scalar photometric loss for a view."""
    gt_image = viewpoint_cam.original_image.cuda()
    Ll1 = l1_loss(image, gt_image)
    loss = (1.0 - 0.2) * Ll1 + 0.2 * (1.0 - ssim(image.unsqueeze(0), gt_image.unsqueeze(0)))
    return loss


def compute_gaussian_score_fastgs(camlist, gaussians, pipe, bg, args, kernel_size, DENSIFY=False):
    """Multi-view importance/pruning score computation for FastGS VCD/VCP.

    For each view: render -> L1 loss map -> metric_map (>loss_thresh) ->
    re-render with get_flag=True -> accum per-Gaussian counts.

    Returns:
        importance_score: per-Gaussian integer counts (floor-avg across views). None if not DENSIFY.
        pruning_score: normalized [0,1] per-Gaussian photometric consistency score.
    """
    full_metric_counts = None
    full_metric_score = None

    for view in range(len(camlist)):
        my_viewpoint_cam = camlist[view]

        # First render: compute loss map
        render_pkg = render_for_metric(my_viewpoint_cam, gaussians, pipe, bg, kernel_size)
        render_image = render_pkg["render"]
        photometric_loss = compute_photometric_loss(my_viewpoint_cam, render_image)

        gt_image = my_viewpoint_cam.original_image.cuda()
        l1_loss_norm = get_loss(render_image, gt_image)
        metric_map = (l1_loss_norm > args.loss_thresh).int()

        # Second render: with metric accumulation
        render_pkg = render_for_metric(
            my_viewpoint_cam, gaussians, pipe, bg, kernel_size,
            get_flag=True, metric_map=metric_map,
        )
        accum_loss_counts = render_pkg["accum_metric_counts"]

        if DENSIFY:
            if full_metric_counts is None:
                full_metric_counts = accum_loss_counts.clone()
            else:
                full_metric_counts += accum_loss_counts

        if full_metric_score is None:
            full_metric_score = photometric_loss * accum_loss_counts.clone().float()
        else:
            full_metric_score += photometric_loss * accum_loss_counts.float()

    pruning_score = (full_metric_score - torch.min(full_metric_score)) / (
        torch.max(full_metric_score) - torch.min(full_metric_score) + 1e-8
    )

    if DENSIFY:
        importance_score = torch.div(full_metric_counts, len(camlist), rounding_mode="floor")
    else:
        importance_score = None
    return importance_score, pruning_score
