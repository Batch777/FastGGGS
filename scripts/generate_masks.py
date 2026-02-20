#!/usr/bin/env python3
"""Generate SAM3 instance segmentation masks for Trace3D integration.

Scans training images from a COLMAP or Blender dataset, runs SAM3 inference
on each image, and saves per-pixel instance label maps to ``<source>/masks/``.

Two modes are supported:

1. **text** (default) — open-vocabulary detection via ``set_text_prompt``.
   Pass one or more comma-separated prompts (e.g. ``"object,person,car"``).
   Each detected instance receives a unique global ID.

2. **grid** — SAM1-style automatic mask generation.  A regular grid of
   foreground points is placed over the image, each point predicts up to 3
   masks (``multimask_output=True``), and overlapping masks are merged via
   NMS on IoU.

Usage
-----
    # text-prompt mode (recommended)
    python scripts/generate_masks.py -s data/dtu_scan24 --prompt "object"

    # grid-point automatic mode
    python scripts/generate_masks.py -s data/dtu_scan24 --mode grid --grid_size 32

    # custom output directory, lower confidence threshold
    python scripts/generate_masks.py -s data/dtu_scan24 --mask_dir masks \
        --confidence 0.3 --prompt "object,thing"

    # specify images subdirectory
    python scripts/generate_masks.py -s data/dtu_scan24 --images images_2
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


# ---------------------------------------------------------------------------
# SAM3 helpers
# ---------------------------------------------------------------------------

def _find_default_checkpoint() -> str | None:
    """Try to find the SAM3 checkpoint in the submodules directory."""
    script_dir = Path(__file__).resolve().parent.parent
    candidates = [
        script_dir / "submodules" / "sam3" / "weights" / "sam3.pt",
        script_dir / "submodules" / "sam3" / "weights" / "model.safetensors",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


def build_model(device: str = "cuda", checkpoint: str | None = None):
    """Build and return the SAM3 image model + processor."""
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    if checkpoint is None:
        checkpoint = _find_default_checkpoint()

    kwargs = dict(
        device=device,
        enable_inst_interactivity=True,  # needed for grid-point mode
    )
    if checkpoint is not None:
        kwargs["checkpoint_path"] = checkpoint
        kwargs["load_from_HF"] = False
        print(f"Using local checkpoint: {checkpoint}")

    model = build_sam3_image_model(**kwargs)
    processor = Sam3Processor(model, device=device)
    return model, processor


def masks_to_instance_map(
    masks: np.ndarray,
    scores: np.ndarray,
    start_id: int = 1,
    iou_merge_thresh: float = 0.5,
) -> tuple[np.ndarray, int]:
    """Convert a set of binary masks (K, H, W) into a single (H, W) instance
    label map.  Masks are sorted by score (descending) and painted in order;
    later (lower-score) masks do not overwrite earlier ones unless overlap is
    below ``iou_merge_thresh``.

    Returns ``(label_map, next_free_id)``.
    """
    if masks.ndim == 4:
        masks = masks.squeeze(1)
    K, H, W = masks.shape
    label_map = np.zeros((H, W), dtype=np.int32)

    # Sort by descending score
    order = np.argsort(-scores)
    cur_id = start_id

    for idx in order:
        m = masks[idx].astype(bool)
        if m.sum() < 10:
            continue  # skip tiny fragments

        # Check overlap with existing labels
        existing = label_map[m]
        if existing.any():
            overlap_ids, counts = np.unique(existing[existing > 0], return_counts=True)
            max_overlap = counts.max() if len(counts) > 0 else 0
            iou_approx = max_overlap / m.sum()
            if iou_approx > iou_merge_thresh:
                # High overlap with existing instance — skip
                continue

        label_map[m & (label_map == 0)] = cur_id
        cur_id += 1

    return label_map, cur_id


# ---------------------------------------------------------------------------
# Text-prompt mode
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_masks_text(
    model,
    processor,
    image: Image.Image,
    prompts: list[str],
    confidence: float = 0.5,
    start_id: int = 1,
) -> tuple[np.ndarray, int]:
    """Run SAM3 text-prompt detection and return an instance label map."""
    all_masks = []
    all_scores = []

    for prompt in prompts:
        state = processor.set_image(image)
        processor.reset_all_prompts(state)
        old_thresh = processor.confidence_threshold
        processor.confidence_threshold = confidence
        state = processor.set_text_prompt(prompt=prompt, state=state)
        processor.confidence_threshold = old_thresh

        if "masks" in state and state["masks"] is not None:
            masks = state["masks"].float().cpu().numpy()  # (K, 1, H, W)
            scores = state["scores"].float().cpu().numpy()  # (K,)
            if masks.ndim == 4:
                masks = masks.squeeze(1)
            if len(masks) > 0:
                all_masks.append(masks)
                all_scores.append(scores)

    if not all_masks:
        H, W = image.height, image.width
        return np.zeros((H, W), dtype=np.int32), start_id

    all_masks = np.concatenate(all_masks, axis=0)
    all_scores = np.concatenate(all_scores, axis=0)

    label_map, next_id = masks_to_instance_map(
        all_masks, all_scores, start_id=start_id
    )
    return label_map, next_id


# ---------------------------------------------------------------------------
# Grid-point mode (SAM1-style automatic)
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_masks_grid(
    model,
    processor,
    image: Image.Image,
    grid_size: int = 32,
    start_id: int = 1,
    score_thresh: float = 0.7,
) -> tuple[np.ndarray, int]:
    """SAM1-style automatic mask generation with a grid of point prompts."""
    W_img, H_img = image.size

    # Build the inference state (shares backbone features for the image)
    state = processor.set_image(image)

    # Generate a regular grid of foreground points
    xs = np.linspace(0, W_img - 1, grid_size, dtype=np.float32)
    ys = np.linspace(0, H_img - 1, grid_size, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)  # (N, 2)

    all_masks = []
    all_scores = []

    # Process each point independently through the interactive predictor
    for pt in points:
        point_coords = pt.reshape(1, 2)
        point_labels = np.array([1])  # foreground
        try:
            masks, scores, _ = model.predict_inst(
                state,
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,
            )
        except Exception:
            continue

        if masks is not None and len(masks) > 0:
            # Take the best mask (highest score)
            best = np.argmax(scores)
            if scores[best] >= score_thresh:
                m = masks[best]
                if m.ndim == 3:
                    m = m.squeeze(0)
                all_masks.append(m)
                all_scores.append(scores[best])

    if not all_masks:
        return np.zeros((H_img, W_img), dtype=np.int32), start_id

    all_masks = np.stack(all_masks, axis=0)  # (K, H, W)
    all_scores = np.array(all_scores)

    # Deduplicate overlapping masks via greedy NMS
    all_masks, all_scores = _nms_masks(all_masks, all_scores, iou_thresh=0.5)

    label_map, next_id = masks_to_instance_map(
        all_masks, all_scores, start_id=start_id
    )
    return label_map, next_id


def _nms_masks(
    masks: np.ndarray,
    scores: np.ndarray,
    iou_thresh: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Simple mask NMS: suppress masks with IoU above threshold."""
    order = np.argsort(-scores)
    keep = []
    areas = masks.sum(axis=(1, 2)).astype(np.float64)

    suppressed = np.zeros(len(masks), dtype=bool)
    for i in order:
        if suppressed[i]:
            continue
        keep.append(i)
        for j in order:
            if j <= i or suppressed[j]:
                continue
            inter = (masks[i] & masks[j]).sum()
            union = areas[i] + areas[j] - inter
            if union > 0 and inter / union > iou_thresh:
                suppressed[j] = True

    keep = np.array(keep)
    return masks[keep], scores[keep]


# ---------------------------------------------------------------------------
# Image discovery
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate SAM3 instance masks for Trace3D integration"
    )
    parser.add_argument(
        "-s", "--source_path", required=True,
        help="Path to the dataset (contains images/ and sparse/)",
    )
    parser.add_argument(
        "--images", default="images",
        help="Name of the images subdirectory (default: images)",
    )
    parser.add_argument(
        "--mask_dir", default="masks",
        help="Output subdirectory name under source_path (default: masks)",
    )
    parser.add_argument(
        "--mode", choices=["text", "grid"], default="text",
        help="Mask generation mode: 'text' (open-vocab) or 'grid' (SAM1-style)",
    )
    parser.add_argument(
        "--prompt", default="object",
        help="Comma-separated text prompts for text mode (default: 'object')",
    )
    parser.add_argument(
        "--confidence", type=float, default=0.5,
        help="Confidence threshold for text mode (default: 0.5)",
    )
    parser.add_argument(
        "--grid_size", type=int, default=32,
        help="Grid resolution for grid mode (default: 32)",
    )
    parser.add_argument(
        "--score_thresh", type=float, default=0.7,
        help="Score threshold for grid mode (default: 0.7)",
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Optional SAM3 checkpoint path (default: auto-download from HF)",
    )
    parser.add_argument(
        "--device", default="cuda",
        help="Device (default: cuda)",
    )
    parser.add_argument(
        "--save_vis", action="store_true",
        help="Also save colorized visualization PNGs",
    )
    parser.add_argument(
        "-r", "--downsample", type=int, default=1,
        help="Downsample factor for input images (default: 1, no downsampling)",
    )
    args = parser.parse_args()

    # Discover images
    image_pairs = find_images(args.source_path, args.images)
    if not image_pairs:
        print(f"No images found in {os.path.join(args.source_path, args.images)}")
        sys.exit(1)
    print(f"Found {len(image_pairs)} images")

    # Output directory
    out_dir = os.path.join(args.source_path, args.mask_dir)
    os.makedirs(out_dir, exist_ok=True)
    if args.save_vis:
        vis_dir = os.path.join(args.source_path, args.mask_dir + "_vis")
        os.makedirs(vis_dir, exist_ok=True)

    # Set up CUDA
    if args.device == "cuda" and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

    # Build model
    print("Loading SAM3 model...")
    model, processor = build_model(device=args.device, checkpoint=args.checkpoint)
    print("Model loaded.")

    prompts = [p.strip() for p in args.prompt.split(",") if p.strip()]

    # Process each image
    for img_path, img_stem in tqdm(image_pairs, desc="Generating masks"):
        image = Image.open(img_path).convert("RGB")
        if args.downsample > 1:
            w, h = image.size
            image = image.resize((w // args.downsample, h // args.downsample), Image.LANCZOS)

        if args.mode == "text":
            label_map, _ = generate_masks_text(
                model, processor, image, prompts,
                confidence=args.confidence,
            )
        else:
            label_map, _ = generate_masks_grid(
                model, processor, image,
                grid_size=args.grid_size,
                score_thresh=args.score_thresh,
            )

        # Determine save format based on max instance count
        max_id = label_map.max()
        if max_id <= 255:
            out = label_map.astype(np.uint8)
        else:
            out = label_map.astype(np.uint16)

        save_path = os.path.join(out_dir, f"{img_stem}.png")
        cv2.imwrite(save_path, out)

        # Optional colorized visualization
        if args.save_vis:
            vis = colorize_label_map(label_map)
            vis_path = os.path.join(vis_dir, f"{img_stem}.png")
            cv2.imwrite(vis_path, vis)

    n_instances_last = label_map.max() if len(image_pairs) > 0 else 0
    print(f"\nDone. Masks saved to {out_dir}")
    print(f"Last image had {n_instances_last} instances detected.")


def colorize_label_map(label_map: np.ndarray) -> np.ndarray:
    """Convert an integer label map to a colorized BGR image for visualization."""
    max_id = label_map.max()
    if max_id == 0:
        return np.zeros((*label_map.shape, 3), dtype=np.uint8)

    # Generate distinct colors via golden-ratio hue spacing
    colors = np.zeros((max_id + 1, 3), dtype=np.uint8)
    for i in range(1, max_id + 1):
        hue = int((i * 137.508) % 180)  # golden angle in degrees / 2
        colors[i] = [hue, 200, 230]

    vis_hsv = colors[label_map]
    vis_bgr = cv2.cvtColor(vis_hsv, cv2.COLOR_HSV2BGR)
    # Set background to black
    vis_bgr[label_map == 0] = 0
    return vis_bgr


if __name__ == "__main__":
    main()
