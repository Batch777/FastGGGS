#!/usr/bin/env python3
"""Generate multi-category semantic instance masks using SAM3.

Each pixel is encoded as uint16: (category_id << 8) | instance_within_category.
- 0 = background
- 0x0101-0x01FF = category 1 instances
- 0x0201-0x02FF = category 2 instances
- etc.

Saves masks as uint16 PNG and a meta.json with category definitions.

Usage:
    python scripts/generate_masks_semantic.py -s data/CUHK_LOWER --prompt "building,tree,road" --save_vis
    python scripts/generate_masks_semantic.py -s data/CUHK_LOWER -r 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

SAM3_CKPT = "submodules/sam3/weights/sam3.pt"

# Default category colors (BGR for OpenCV visualization)
DEFAULT_COLORS = {
    "building": [0, 0, 200],      # red
    "tree": [0, 180, 0],          # green
    "road": [180, 120, 0],        # blue-ish
    "car": [0, 200, 200],         # yellow
    "person": [200, 0, 200],      # magenta
    "sky": [200, 200, 0],         # cyan
}


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


def masks_to_semantic_label_map(
    masks: np.ndarray,
    scores: np.ndarray,
    cls_ids: np.ndarray,
    category_names: list[str],
    min_area: int = 50,
    iou_merge_thresh: float = 0.5,
) -> np.ndarray:
    """Convert overlapping binary masks to semantic instance label map.

    Returns uint16 map: (category_id << 8) | instance_within_category.
    """
    if masks.ndim == 4:
        masks = masks.squeeze(1)
    K, H, W = masks.shape
    label_map = np.zeros((H, W), dtype=np.uint16)

    # Track per-category instance counters
    n_categories = len(category_names)
    instance_counters = [0] * (n_categories + 1)  # 1-indexed categories

    # Process masks in descending score order (greedy painting)
    order = np.argsort(-scores)
    for idx in order:
        m = masks[idx].astype(bool)
        area = m.sum()
        if area < min_area:
            continue

        # Check overlap with already-assigned pixels
        overlap = (label_map[m] > 0).sum()
        if overlap > 0 and overlap / area > iou_merge_thresh:
            continue

        # cls_ids from SAM3 are 0-indexed prompt indices → +1 for 1-based category
        cat_id = int(cls_ids[idx]) + 1
        if cat_id < 1 or cat_id > n_categories:
            continue

        instance_counters[cat_id] += 1
        inst_id = instance_counters[cat_id]
        if inst_id > 255:
            continue  # max 255 instances per category in uint8 lower byte

        encoded = (cat_id << 8) | inst_id
        label_map[m & (label_map == 0)] = encoded

    return label_map


def colorize_semantic_map(label_map: np.ndarray, category_names: list[str]) -> np.ndarray:
    """Create a colored visualization of the semantic label map."""
    H, W = label_map.shape
    vis = np.zeros((H, W, 3), dtype=np.uint8)

    for cat_idx, name in enumerate(category_names):
        cat_id = cat_idx + 1
        cat_mask = (label_map >> 8) == cat_id
        if not cat_mask.any():
            continue

        base_color = DEFAULT_COLORS.get(name, [128, 128, 128])

        # Tint instances slightly differently
        instances = np.unique(label_map[cat_mask] & 0xFF)
        for inst in instances:
            inst_mask = cat_mask & ((label_map & 0xFF) == inst)
            # Slight brightness variation per instance
            factor = 0.7 + 0.3 * ((inst * 137) % 100) / 100.0
            color = np.clip(np.array(base_color) * factor, 0, 255).astype(np.uint8)
            vis[inst_mask] = color

    return vis


def main():
    parser = argparse.ArgumentParser(description="SAM3 multi-category semantic mask generation")
    parser.add_argument("-s", "--source_path", required=True)
    parser.add_argument("--images", default="images")
    parser.add_argument("--mask_dir", default="masks_semantic")
    parser.add_argument("--prompt", default="building,tree,road",
                        help="Comma-separated category prompts (default: building,tree,road)")
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

    category_names = [p.strip() for p in args.prompt.split(",") if p.strip()]
    print(f"Categories: {category_names}")

    out_dir = os.path.join(args.source_path, args.mask_dir)
    os.makedirs(out_dir, exist_ok=True)
    if args.save_vis:
        vis_dir = os.path.join(args.source_path, args.mask_dir + "_vis")
        os.makedirs(vis_dir, exist_ok=True)

    # Save meta.json with category definitions
    category_colors_rgb = []
    meta = {"encoding": "uint16: (category_id << 8) | instance_id", "categories": []}
    for i, name in enumerate(category_names):
        bgr = DEFAULT_COLORS.get(name, [128, 128, 128])
        rgb = [bgr[2], bgr[1], bgr[0]]
        category_colors_rgb.append(rgb)
        meta["categories"].append({
            "id": i + 1,
            "name": name,
            "color_rgb": rgb,
        })
    # Category 0 = background
    meta["categories"].insert(0, {"id": 0, "name": "background", "color_rgb": [0, 0, 0]})

    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved meta.json to {out_dir}")

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

    stats = {name: 0 for name in category_names}

    for img_path, img_stem in tqdm(image_pairs, desc="Generating semantic masks"):
        img = cv2.imread(img_path)
        if img is None:
            continue
        if args.downsample > 1:
            h, w = img.shape[:2]
            img = cv2.resize(img, (w // args.downsample, h // args.downsample),
                             interpolation=cv2.INTER_AREA)

        H, W = img.shape[:2]

        # Set image (encodes features once)
        predictor.set_image(img)

        # Single call with all category prompts
        results = predictor(text=category_names)

        if results and results[0].masks is not None and len(results[0].masks) > 0:
            masks_data = results[0].masks.data.cpu().numpy().astype(np.uint8)
            scores = results[0].boxes.conf.cpu().numpy()
            cls_ids = results[0].boxes.cls.cpu().numpy()

            label_map = masks_to_semantic_label_map(
                masks_data, scores, cls_ids, category_names,
            )
        else:
            label_map = np.zeros((H, W), dtype=np.uint16)

        # Count instances per category
        for i, name in enumerate(category_names):
            cat_id = i + 1
            cat_insts = np.unique(label_map[((label_map >> 8) == cat_id)] & 0xFF)
            cat_insts = cat_insts[cat_insts > 0]
            stats[name] += len(cat_insts)

        # Save as uint16 PNG
        cv2.imwrite(os.path.join(out_dir, f"{img_stem}.png"), label_map)

        if args.save_vis:
            vis = colorize_semantic_map(label_map, category_names)
            cv2.imwrite(os.path.join(vis_dir, f"{img_stem}.png"), vis)

    print(f"\nDone. Masks saved to {out_dir}")
    for name, count in stats.items():
        print(f"  {name}: {count} total instances across all images")
    print(f"  Average per image: {sum(stats.values()) / max(1, len(image_pairs)):.1f}")


if __name__ == "__main__":
    main()
