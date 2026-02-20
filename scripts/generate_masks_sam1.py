#!/usr/bin/env python3
"""Generate instance masks using Ultralytics SAM1 (Segment Anything Model 1).

Naive approach: run SAM1's automatic "segment everything" mode on each image,
convert overlapping binary masks to non-overlapping instance label maps, then
feed into the existing Trace3D pipeline.

Key difference from SAM3:
- SAM1 has no semantic understanding (no text prompts)
- It segments purely by visual boundaries → may over-segment buildings
  into roof/facade/windows fragments
- Trace3D cross-view Jaccard merge compensates by merging fragments that
  correspond to the same physical object across viewpoints

Usage:
    # Basic (SAM1-Base, ~375MB auto-download)
    python scripts/generate_masks_sam1.py -s data/CUHK_LOWER

    # SAM1-Large for higher quality
    python scripts/generate_masks_sam1.py -s data/CUHK_LOWER --model sam_l.pt

    # Custom output dir + visualization
    python scripts/generate_masks_sam1.py -s data/CUHK_LOWER --mask_dir masks_sam1 --save_vis

    # Then run Trace3D pipeline
    python render_instances.py -m output/cuhk_lower --mask_dir masks_sam1
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm


def masks_to_instance_map(
    masks: np.ndarray,
    scores: np.ndarray,
    min_area: int = 100,
    iou_merge_thresh: float = 0.5,
) -> np.ndarray:
    """Convert overlapping binary masks (K, H, W) to non-overlapping instance map.

    Masks are sorted by score (descending). Higher-score masks get priority.
    A mask is skipped if it has high IoU overlap with an already-painted instance.

    Returns (H, W) int32 array where 0=background, 1,2,...=instance IDs.
    """
    if masks.ndim == 4:
        masks = masks.squeeze(1)
    K, H, W = masks.shape
    label_map = np.zeros((H, W), dtype=np.int32)

    # Sort by descending score
    order = np.argsort(-scores)
    cur_id = 1

    for idx in order:
        m = masks[idx].astype(bool)
        area = m.sum()
        if area < min_area:
            continue  # skip tiny fragments

        # Check overlap with already-painted pixels
        overlap = (label_map[m] > 0).sum()
        if overlap > 0:
            iou_approx = overlap / area
            if iou_approx > iou_merge_thresh:
                continue  # too much overlap, skip

        # Paint only unlabeled pixels
        label_map[m & (label_map == 0)] = cur_id
        cur_id += 1

    return label_map


def find_images(source_path: str, images_dir: str = "images") -> list[tuple[str, str]]:
    """Return list of (image_path, image_stem) for all training images."""
    img_dir = os.path.join(source_path, images_dir)
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"Images directory not found: {img_dir}")

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    pairs = []
    for f in sorted(os.listdir(img_dir)):
        if Path(f).suffix.lower() in exts:
            pairs.append((os.path.join(img_dir, f), Path(f).stem))
    return pairs


def colorize_label_map(label_map: np.ndarray) -> np.ndarray:
    """Convert integer label map to colorized BGR image for visualization."""
    max_id = label_map.max()
    if max_id == 0:
        return np.zeros((*label_map.shape, 3), dtype=np.uint8)

    colors = np.zeros((max_id + 1, 3), dtype=np.uint8)
    for i in range(1, max_id + 1):
        hue = int((i * 137.508) % 180)
        colors[i] = [hue, 200, 230]

    vis_hsv = colors[label_map]
    vis_bgr = cv2.cvtColor(vis_hsv, cv2.COLOR_HSV2BGR)
    vis_bgr[label_map == 0] = 0
    return vis_bgr


def main():
    parser = argparse.ArgumentParser(
        description="Generate instance masks using Ultralytics SAM1"
    )
    parser.add_argument(
        "-s", "--source_path", required=True,
        help="Path to the dataset (contains images/)",
    )
    parser.add_argument(
        "--images", default="images",
        help="Images subdirectory name (default: images)",
    )
    parser.add_argument(
        "--mask_dir", default="masks_sam1",
        help="Output subdirectory under source_path (default: masks_sam1)",
    )
    parser.add_argument(
        "--model", default="sam_b.pt",
        help="SAM1 model checkpoint: sam_b.pt (base, 375MB) or sam_l.pt (large, 1.2GB)",
    )
    parser.add_argument(
        "--min_area", type=int, default=100,
        help="Minimum mask area in pixels (default: 100)",
    )
    parser.add_argument(
        "--iou_merge", type=float, default=0.5,
        help="IoU overlap threshold to skip a mask (default: 0.5)",
    )
    parser.add_argument(
        "--device", default="cuda",
        help="Device (default: cuda)",
    )
    parser.add_argument(
        "--save_vis", action="store_true",
        help="Save colorized visualization images",
    )
    parser.add_argument(
        "-r", "--downsample", type=int, default=1,
        help="Downsample input images (default: 1, no downsampling)",
    )
    args = parser.parse_args()

    # Discover images
    image_pairs = find_images(args.source_path, args.images)
    if not image_pairs:
        print(f"No images found in {os.path.join(args.source_path, args.images)}")
        sys.exit(1)
    print(f"Found {len(image_pairs)} images")

    # Output directories
    out_dir = os.path.join(args.source_path, args.mask_dir)
    os.makedirs(out_dir, exist_ok=True)
    if args.save_vis:
        vis_dir = os.path.join(args.source_path, args.mask_dir + "_vis")
        os.makedirs(vis_dir, exist_ok=True)

    # Load SAM1 model via ultralytics
    print(f"Loading SAM1 model: {args.model} ...")
    from ultralytics import SAM
    model = SAM(args.model)
    print("Model loaded.")

    stats = {"total_masks": 0, "total_instances": 0}

    for img_path, img_stem in tqdm(image_pairs, desc="Generating masks (SAM1)"):
        # Read image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: cannot read {img_path}, skipping")
            continue

        if args.downsample > 1:
            h, w = img.shape[:2]
            img = cv2.resize(
                img, (w // args.downsample, h // args.downsample),
                interpolation=cv2.INTER_AREA,
            )

        # Run SAM1 automatic segment-everything (no prompts)
        results = model(img, device=args.device, verbose=False)

        if results[0].masks is None or len(results[0].masks) == 0:
            # No masks detected — save empty label map
            h, w = img.shape[:2]
            label_map = np.zeros((h, w), dtype=np.int32)
        else:
            # masks.data: (K, H, W) bool tensor on GPU
            masks_tensor = results[0].masks.data.cpu().numpy().astype(np.uint8)
            scores = results[0].boxes.conf.cpu().numpy()

            stats["total_masks"] += len(masks_tensor)

            label_map = masks_to_instance_map(
                masks_tensor, scores,
                min_area=args.min_area,
                iou_merge_thresh=args.iou_merge,
            )

        n_inst = label_map.max()
        stats["total_instances"] += n_inst

        # Save label map
        if n_inst <= 255:
            out = label_map.astype(np.uint8)
        else:
            out = label_map.astype(np.uint16)

        save_path = os.path.join(out_dir, f"{img_stem}.png")
        cv2.imwrite(save_path, out)

        # Optional visualization
        if args.save_vis:
            vis = colorize_label_map(label_map)
            vis_path = os.path.join(vis_dir, f"{img_stem}.png")
            cv2.imwrite(vis_path, vis)

    print(f"\nDone. Masks saved to {out_dir}")
    print(f"Total raw masks across all images: {stats['total_masks']}")
    print(f"Total instances (after overlap filtering): {stats['total_instances']}")
    print(f"Average instances per image: {stats['total_instances'] / len(image_pairs):.1f}")


if __name__ == "__main__":
    main()
