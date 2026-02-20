#!/usr/bin/env python3
"""Generate instance masks using Ultralytics SAM3 with text prompt "building".

Uses the SAM3SemanticPredictor from ultralytics with a local checkpoint.

Usage:
    python scripts/generate_masks_building.py -s data/CUHK_LOWER
    python scripts/generate_masks_building.py -s data/CUHK_LOWER --prompt "building,road,tree" --save_vis
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

SAM3_CKPT = "submodules/sam3/weights/sam3.pt"


def masks_to_instance_map(
    masks: np.ndarray,
    scores: np.ndarray,
    min_area: int = 50,
    iou_merge_thresh: float = 0.5,
) -> np.ndarray:
    """Convert overlapping binary masks (K, H, W) to non-overlapping instance map."""
    if masks.ndim == 4:
        masks = masks.squeeze(1)
    K, H, W = masks.shape
    label_map = np.zeros((H, W), dtype=np.int32)
    order = np.argsort(-scores)
    cur_id = 1
    for idx in order:
        m = masks[idx].astype(bool)
        area = m.sum()
        if area < min_area:
            continue
        overlap = (label_map[m] > 0).sum()
        if overlap > 0 and overlap / area > iou_merge_thresh:
            continue
        label_map[m & (label_map == 0)] = cur_id
        cur_id += 1
    return label_map


def colorize_label_map(label_map: np.ndarray) -> np.ndarray:
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


def find_images(source_path: str, images_dir: str = "images") -> list[tuple[str, str]]:
    img_dir = os.path.join(source_path, images_dir)
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"Images directory not found: {img_dir}")
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    pairs = []
    for f in sorted(os.listdir(img_dir)):
        if Path(f).suffix.lower() in exts:
            pairs.append((os.path.join(img_dir, f), Path(f).stem))
    return pairs


def main():
    parser = argparse.ArgumentParser(description="SAM3 text-prompt mask generation via ultralytics")
    parser.add_argument("-s", "--source_path", required=True)
    parser.add_argument("--images", default="images")
    parser.add_argument("--mask_dir", default="masks_building")
    parser.add_argument("--prompt", default="building", help="Comma-separated text prompts")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--checkpoint", default=SAM3_CKPT, help="SAM3 checkpoint path")
    parser.add_argument("--save_vis", action="store_true")
    parser.add_argument("-r", "--downsample", type=int, default=1)
    args = parser.parse_args()

    image_pairs = find_images(args.source_path, args.images)
    if not image_pairs:
        print(f"No images found in {os.path.join(args.source_path, args.images)}")
        sys.exit(1)
    print(f"Found {len(image_pairs)} images")

    out_dir = os.path.join(args.source_path, args.mask_dir)
    os.makedirs(out_dir, exist_ok=True)
    if args.save_vis:
        vis_dir = os.path.join(args.source_path, args.mask_dir + "_vis")
        os.makedirs(vis_dir, exist_ok=True)

    # Load SAM3 via ultralytics
    from ultralytics.models.sam import SAM3SemanticPredictor

    overrides = dict(
        conf=args.conf,
        task="segment",
        mode="predict",
        model=args.checkpoint,
        half=True,
        save=False,
        verbose=False,
    )
    predictor = SAM3SemanticPredictor(overrides=overrides)
    print(f"SAM3 loaded from {args.checkpoint}")

    prompts = [p.strip() for p in args.prompt.split(",") if p.strip()]
    print(f"Text prompts: {prompts}")

    stats = {"total_instances": 0}

    for img_path, img_stem in tqdm(image_pairs, desc="Generating masks"):
        img = cv2.imread(img_path)
        if img is None:
            continue
        if args.downsample > 1:
            h, w = img.shape[:2]
            img = cv2.resize(img, (w // args.downsample, h // args.downsample), interpolation=cv2.INTER_AREA)

        H, W = img.shape[:2]

        # Set image (encodes features once)
        predictor.set_image(img)

        # Query with text prompts
        results = predictor(text=prompts)

        if results and results[0].masks is not None and len(results[0].masks) > 0:
            masks_data = results[0].masks.data.cpu().numpy().astype(np.uint8)
            scores = results[0].boxes.conf.cpu().numpy()
            label_map = masks_to_instance_map(masks_data, scores)
        else:
            label_map = np.zeros((H, W), dtype=np.int32)

        n_inst = label_map.max()
        stats["total_instances"] += n_inst

        # Save
        if n_inst <= 255:
            out = label_map.astype(np.uint8)
        else:
            out = label_map.astype(np.uint16)
        cv2.imwrite(os.path.join(out_dir, f"{img_stem}.png"), out)

        if args.save_vis:
            vis = colorize_label_map(label_map)
            cv2.imwrite(os.path.join(vis_dir, f"{img_stem}.png"), vis)

    print(f"\nDone. Masks saved to {out_dir}")
    print(f"Total instances: {stats['total_instances']}")
    print(f"Average per image: {stats['total_instances'] / len(image_pairs):.1f}")


if __name__ == "__main__":
    main()
