"""Trace3D-style 3D instance segmentation for Gaussian Splatting.

Stage A: project Gaussian centers to each view, read SAM mask label.
Stage B: cross-view Jaccard affinity + Union-Find merge.
Stage C: assign global labels to Gaussians via majority vote.

The loss functions (CE proxy, depth-edge) are cheap Python-only helpers that
do NOT modify the CUDA rasterizer.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Sequence

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from scene.cameras import Camera
    from scene.gaussian_model import GaussianModel


# ---------------------------------------------------------------------------
# Projection helper (shared by weight matrix + loss)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _project_gaussians(xyz: torch.Tensor, cam) -> tuple[torch.Tensor, torch.Tensor]:
    """Project Gaussian centers to pixel coords in *cam*.

    Returns (valid_mask (N,), uv (N, 2) as long).
    """
    mask = getattr(cam, "instance_mask", None)
    if mask is None:
        N = xyz.shape[0]
        return torch.zeros(N, dtype=torch.bool, device=xyz.device), torch.zeros(N, 2, dtype=torch.long, device=xyz.device)

    H, W_img = mask.shape
    R, T = cam.R, cam.T
    xyz_cam = torch.addmm(T.unsqueeze(0), xyz, R)
    z = xyz_cam[:, 2]

    u = (xyz_cam[:, 0] / z * cam.Fx + cam.Cx).long()
    v = (xyz_cam[:, 1] / z * cam.Fy + cam.Cy).long()

    valid = (z > 0.2) & (u >= 0) & (u < W_img) & (v >= 0) & (v < H)
    uv = torch.stack([u, v], dim=1)
    return valid, uv


# ---------------------------------------------------------------------------
# Stage A+B+C — memory-efficient all-in-one pipeline
# ---------------------------------------------------------------------------

class _UnionFind:
    """Simple Union-Find (Disjoint Set Union) for merging instance labels."""

    def __init__(self) -> None:
        self.parent: dict[int, int] = {}
        self.rank: dict[int, int] = {}

    def find(self, x: int) -> int:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


@torch.no_grad()
def assign_instance_labels(
    cameras: Sequence[Camera],
    gaussians: GaussianModel,
    *,
    tau_aff: float = 0.3,
    max_cameras: int = 150,
    fg_bg_ratio: float = 1.5,
) -> None:
    """Memory-efficient 3D instance label assignment.

    Replaces the old three-step pipeline (build_instance_weight_matrix →
    majority_vote_merge → assign_labels) with a streaming approach that
    avoids huge Python dicts.

    Algorithm
    ---------
    Phase 1 — Per-camera projection:
        For each (subsampled) camera, project all Gaussians, read mask labels.
        Store per-Gaussian best-so-far label and weight (O(N) memory).
        Also build a compact inverted index: global_label → sampled prim set
        (for Jaccard merge, capped at 2000 prims per label).

    Phase 2 — Cross-view Jaccard merge:
        For label pairs that share Gaussians, compute Jaccard and merge
        via Union-Find.

    Phase 3 — Final assignment:
        Second pass through cameras. For each Gaussian, accumulate
        opacity-weighted votes into *merged* labels (after Union-Find remap).
        Pick argmax per Gaussian.
    """
    xyz = gaussians.get_xyz  # (N, 3)
    N = xyz.shape[0]
    opacity = gaussians.get_opacity.squeeze(-1)  # (N,)
    device = xyz.device

    # Subsample cameras to limit memory
    cam_list = list(cameras)
    if len(cam_list) > max_cameras:
        step = max(1, len(cam_list) // max_cameras)
        cam_list = cam_list[::step]
    print(f"  Using {len(cam_list)}/{len(list(cameras))} cameras for instance assignment")

    # ------------------------------------------------------------------
    # Phase 1: Per-camera projection → build inverted index for merge
    # ------------------------------------------------------------------
    # For each global label, keep a SAMPLED set of Gaussian IDs (cap 2000)
    label_to_prims: dict[int, set[int]] = defaultdict(set)
    SAMPLE_CAP = 2000

    # Also track which global labels appear in which camera (for pair generation)
    cam_to_labels: list[set[int]] = []
    cam_nearest: list[list[int]] = []

    label_offset = 0
    cam_offsets: list[int] = []  # offset per camera for Phase 3

    for ci, cam in enumerate(cam_list):
        mask = getattr(cam, "instance_mask", None)
        if mask is None:
            cam_to_labels.append(set())
            cam_nearest.append([])
            cam_offsets.append(label_offset)
            continue

        cam_max = int(mask.max().item())
        cam_offsets.append(label_offset)

        valid, uv = _project_gaussians(xyz, cam)
        if not valid.any():
            cam_to_labels.append(set())
            cam_nearest.append(getattr(cam, "nearest_id", []))
            label_offset += cam_max + 1
            continue

        idx = torch.where(valid)[0]
        raw = mask[uv[idx, 1], uv[idx, 0]]
        fg = raw > 0
        idx = idx[fg]
        raw = raw[fg]

        # Global labels for this camera
        global_labels_gpu = raw.long() + label_offset

        # Build inverted index (sampled)
        gl_cpu = global_labels_gpu.cpu().tolist()
        idx_cpu = idx.cpu().tolist()
        labels_in_cam = set()
        for gi, gl in zip(idx_cpu, gl_cpu):
            labels_in_cam.add(gl)
            s = label_to_prims[gl]
            if len(s) < SAMPLE_CAP:
                s.add(gi)

        cam_to_labels.append(labels_in_cam)
        cam_nearest.append(getattr(cam, "nearest_id", []))
        label_offset += cam_max + 1

    total_labels = len(label_to_prims)
    print(f"  Total global labels (pre-merge): {total_labels}")

    # ------------------------------------------------------------------
    # Phase 2: Cross-view Jaccard merge via Union-Find
    # ------------------------------------------------------------------
    # Build candidate pairs from nearby cameras
    uf = _UnionFind()
    merge_count = 0

    for ci in range(len(cam_list)):
        labels_i = cam_to_labels[ci]
        if not labels_i:
            continue
        # Find neighbors in our subsampled list
        for raw_nid in cam_nearest[ci]:
            # raw_nid is index into full camera list; map to subsampled index
            step = max(1, len(list(cameras)) // max_cameras) if len(list(cameras)) > max_cameras else 1
            cj = raw_nid // step
            if cj >= len(cam_list) or cj == ci:
                continue
            labels_j = cam_to_labels[cj]
            if not labels_j:
                continue
            # Check all (la, lb) pairs where la ∈ cam_i, lb ∈ cam_j
            for la in labels_i:
                sa = label_to_prims.get(la)
                if not sa:
                    continue
                for lb in labels_j:
                    if la == lb:
                        continue
                    # Quick check: already merged?
                    if uf.find(la) == uf.find(lb):
                        continue
                    sb = label_to_prims.get(lb)
                    if not sb:
                        continue
                    inter = len(sa & sb)
                    if inter == 0:
                        continue
                    union = len(sa | sb)
                    jaccard = inter / union
                    if jaccard > tau_aff:
                        uf.union(la, lb)
                        merge_count += 1

    # If no neighbor info produced merges, fall back to comparing labels
    # that share any Gaussians (more expensive but correct)
    if merge_count == 0 and total_labels < 5000:
        print("  No neighbor-based merges; falling back to pairwise Jaccard")
        all_labels = sorted(label_to_prims.keys())
        # Build prim→labels index for co-occurrence
        prim_to_labels: dict[int, list[int]] = defaultdict(list)
        for gl, prims in label_to_prims.items():
            for p in prims:
                prim_to_labels[p].append(gl)
        # Collect co-occurring pairs
        pairs: set[tuple[int, int]] = set()
        for p, lbls in prim_to_labels.items():
            for i in range(len(lbls)):
                for j in range(i + 1, len(lbls)):
                    pairs.add((min(lbls[i], lbls[j]), max(lbls[i], lbls[j])))
        for la, lb in pairs:
            if uf.find(la) == uf.find(lb):
                continue
            sa = label_to_prims.get(la, set())
            sb = label_to_prims.get(lb, set())
            inter = len(sa & sb)
            union = len(sa | sb)
            if union > 0 and inter / union > tau_aff:
                uf.union(la, lb)
                merge_count += 1

    # Build remap
    merged_ids = set()
    for gl in label_to_prims:
        merged_ids.add(uf.find(gl))
    print(f"  Merged: {total_labels} → {len(merged_ids)} labels ({merge_count} merges)")

    # Free inverted index
    del label_to_prims

    # ------------------------------------------------------------------
    # Phase 3: Final assignment — second pass with merged labels
    # ------------------------------------------------------------------
    # For each Gaussian, accumulate opacity-weighted votes per MERGED label.
    # Use a fixed-size tensor: (N, K) for top-K labels and their weights.
    K = 8  # keep top-K candidates per Gaussian
    topk_labels = torch.zeros(N, K, dtype=torch.long, device=device)
    topk_weights = torch.zeros(N, K, dtype=torch.float, device=device)
    bg_weight = torch.zeros(N, dtype=torch.float, device=device)  # background votes

    for ci, cam in enumerate(cam_list):
        mask = getattr(cam, "instance_mask", None)
        if mask is None:
            continue

        valid, uv = _project_gaussians(xyz, cam)
        if not valid.any():
            continue

        idx = torch.where(valid)[0]
        raw = mask[uv[idx, 1], uv[idx, 0]]

        # Accumulate background votes for label==0
        bg_mask = raw == 0
        if bg_mask.any():
            bg_idx = idx[bg_mask]
            bg_weight[bg_idx] += opacity[bg_idx]

        fg = raw > 0
        idx = idx[fg]
        raw = raw[fg]

        if idx.shape[0] == 0:
            continue

        offset = cam_offsets[ci]
        global_labels = raw.long() + offset

        # Remap through Union-Find (vectorized via lookup)
        gl_cpu = global_labels.cpu().tolist()
        merged_labels = torch.tensor(
            [uf.find(g) for g in gl_cpu], dtype=torch.long, device=device
        )

        w = opacity[idx]

        # Read current top-K state for these Gaussians (COPIES, shape M×K)
        cur_labels = topk_labels[idx]    # (M, K)
        cur_weights = topk_weights[idx]  # (M, K)

        # Check which slots match the incoming merged label
        merged_exp = merged_labels.unsqueeze(1)  # (M, 1)
        match_any = (cur_labels == merged_exp)    # (M, K) bool

        # Add weight to matching slots
        cur_weights += w.unsqueeze(1) * match_any.float()

        # For Gaussians where the label is NOT yet in any slot, insert it
        has_match = match_any.any(dim=1)  # (M,)
        no_match = ~has_match

        if no_match.any():
            nm_local = torch.where(no_match)[0]  # indices into M (local)
            nm_labels = merged_labels[nm_local]
            nm_w = w[nm_local]
            nm_cur_w = cur_weights[nm_local]  # (M_new, K) — local index!

            # Find the slot with minimum weight for each
            min_vals, min_slots = nm_cur_w.min(dim=1)  # (M_new,)

            # Only replace if new weight > min existing weight
            replace = nm_w > min_vals
            if replace.any():
                rep_local = nm_local[replace]       # local indices into M
                rep_slots = min_slots[replace]
                rep_labels = nm_labels[replace]
                rep_w = nm_w[replace]
                # Update the copies using advanced indexing
                cur_labels[rep_local, rep_slots] = rep_labels
                cur_weights[rep_local, rep_slots] = rep_w

        # Write everything back at once (no conflict)
        topk_labels[idx] = cur_labels
        topk_weights[idx] = cur_weights

    # Pick argmax per Gaussian, competing with background weight
    best_fg_weight, best_slot = topk_weights.max(dim=1)  # (N,)
    instance_id = topk_labels[torch.arange(N, device=device), best_slot]

    # Foreground must exceed background * fg_bg_ratio to win
    bg_wins = best_fg_weight < bg_weight * fg_bg_ratio
    instance_id[bg_wins] = 0
    n_bg = bg_wins.sum().item()
    n_fg = N - n_bg
    print(f"  Background wins: {n_bg} Gaussians ({100*n_bg/N:.1f}%)")
    print(f"  Foreground wins: {n_fg} Gaussians ({100*n_fg/N:.1f}%)")

    # Compact label IDs to consecutive range
    unique_labels, inverse = instance_id.unique(return_inverse=True)
    gaussians._instance_id = inverse  # 0-based consecutive IDs

    n_unique = unique_labels.shape[0]
    print(f"  Unique instance IDs assigned: {n_unique}")
    print(f"  ID range: [0, {n_unique - 1}]")


# ---------------------------------------------------------------------------
# Semantic multi-category instance assignment
# ---------------------------------------------------------------------------

@torch.no_grad()
def assign_semantic_labels(
    cameras: Sequence[Camera],
    gaussians: GaussianModel,
    *,
    tau_aff: float = 0.3,
    max_cameras: int = 150,
    fg_bg_ratio: float = 1.5,
) -> None:
    """Multi-category 3D instance label assignment.

    Like assign_instance_labels(), but decodes semantic category from the mask
    encoding: raw_value = (category_id << 8) | instance_within_category.
    After assignment, sets both gaussians._instance_id (compacted) and
    gaussians._semantic_category.
    """
    xyz = gaussians.get_xyz  # (N, 3)
    N = xyz.shape[0]
    opacity = gaussians.get_opacity.squeeze(-1)  # (N,)
    device = xyz.device

    # Subsample cameras
    cam_list = list(cameras)
    if len(cam_list) > max_cameras:
        step = max(1, len(cam_list) // max_cameras)
        cam_list = cam_list[::step]
    print(f"  [Semantic] Using {len(cam_list)}/{len(list(cameras))} cameras")

    # Phase 1: Per-camera projection → build inverted index
    label_to_prims: dict[int, set[int]] = defaultdict(set)
    global_label_to_category: dict[int, int] = {}  # global_label → category_id
    SAMPLE_CAP = 2000

    cam_to_labels: list[set[int]] = []
    cam_nearest: list[list[int]] = []

    label_offset = 0
    cam_offsets: list[int] = []

    for ci, cam in enumerate(cam_list):
        mask = getattr(cam, "instance_mask", None)
        if mask is None:
            cam_to_labels.append(set())
            cam_nearest.append([])
            cam_offsets.append(label_offset)
            continue

        cam_max = int(mask.max().item())
        cam_offsets.append(label_offset)

        valid, uv = _project_gaussians(xyz, cam)
        if not valid.any():
            cam_to_labels.append(set())
            cam_nearest.append(getattr(cam, "nearest_id", []))
            label_offset += cam_max + 1
            continue

        idx = torch.where(valid)[0]
        raw = mask[uv[idx, 1], uv[idx, 0]]
        fg = raw > 0
        idx = idx[fg]
        raw = raw[fg]

        global_labels_gpu = raw.long() + label_offset

        gl_cpu = global_labels_gpu.cpu().tolist()
        raw_cpu = raw.cpu().tolist()
        idx_cpu = idx.cpu().tolist()
        labels_in_cam = set()
        for gi, gl, rv in zip(idx_cpu, gl_cpu, raw_cpu):
            labels_in_cam.add(gl)
            s = label_to_prims[gl]
            if len(s) < SAMPLE_CAP:
                s.add(gi)
            # Decode category from the raw mask value
            if gl not in global_label_to_category:
                global_label_to_category[gl] = int(rv) >> 8

        cam_to_labels.append(labels_in_cam)
        cam_nearest.append(getattr(cam, "nearest_id", []))
        label_offset += cam_max + 1

    total_labels = len(label_to_prims)
    print(f"  [Semantic] Total global labels (pre-merge): {total_labels}")

    # Phase 2: Cross-view Jaccard merge via Union-Find
    uf = _UnionFind()
    merge_count = 0

    for ci in range(len(cam_list)):
        labels_i = cam_to_labels[ci]
        if not labels_i:
            continue
        for raw_nid in cam_nearest[ci]:
            step = max(1, len(list(cameras)) // max_cameras) if len(list(cameras)) > max_cameras else 1
            cj = raw_nid // step
            if cj >= len(cam_list) or cj == ci:
                continue
            labels_j = cam_to_labels[cj]
            if not labels_j:
                continue
            for la in labels_i:
                sa = label_to_prims.get(la)
                if not sa:
                    continue
                for lb in labels_j:
                    if la == lb:
                        continue
                    if uf.find(la) == uf.find(lb):
                        continue
                    sb = label_to_prims.get(lb)
                    if not sb:
                        continue
                    inter = len(sa & sb)
                    if inter == 0:
                        continue
                    union = len(sa | sb)
                    jaccard = inter / union
                    if jaccard > tau_aff:
                        uf.union(la, lb)
                        merge_count += 1

    # Fallback pairwise if no neighbor-based merges
    if merge_count == 0 and total_labels < 5000:
        print("  [Semantic] No neighbor-based merges; falling back to pairwise Jaccard")
        prim_to_labels: dict[int, list[int]] = defaultdict(list)
        for gl, prims in label_to_prims.items():
            for p in prims:
                prim_to_labels[p].append(gl)
        pairs: set[tuple[int, int]] = set()
        for p, lbls in prim_to_labels.items():
            for i in range(len(lbls)):
                for j in range(i + 1, len(lbls)):
                    pairs.add((min(lbls[i], lbls[j]), max(lbls[i], lbls[j])))
        for la, lb in pairs:
            if uf.find(la) == uf.find(lb):
                continue
            sa = label_to_prims.get(la, set())
            sb = label_to_prims.get(lb, set())
            inter = len(sa & sb)
            union = len(sa | sb)
            if union > 0 and inter / union > tau_aff:
                uf.union(la, lb)
                merge_count += 1

    merged_ids = set()
    for gl in label_to_prims:
        merged_ids.add(uf.find(gl))
    print(f"  [Semantic] Merged: {total_labels} -> {len(merged_ids)} labels ({merge_count} merges)")

    del label_to_prims

    # Phase 3: Final assignment with category tracking
    K = 8
    topk_labels = torch.zeros(N, K, dtype=torch.long, device=device)
    topk_weights = torch.zeros(N, K, dtype=torch.float, device=device)
    bg_weight = torch.zeros(N, dtype=torch.float, device=device)

    for ci, cam in enumerate(cam_list):
        mask = getattr(cam, "instance_mask", None)
        if mask is None:
            continue

        valid, uv = _project_gaussians(xyz, cam)
        if not valid.any():
            continue

        idx = torch.where(valid)[0]
        raw = mask[uv[idx, 1], uv[idx, 0]]

        bg_mask = raw == 0
        if bg_mask.any():
            bg_idx = idx[bg_mask]
            bg_weight[bg_idx] += opacity[bg_idx]

        fg = raw > 0
        idx = idx[fg]
        raw = raw[fg]

        if idx.shape[0] == 0:
            continue

        offset = cam_offsets[ci]
        global_labels = raw.long() + offset

        gl_cpu = global_labels.cpu().tolist()
        merged_labels = torch.tensor(
            [uf.find(g) for g in gl_cpu], dtype=torch.long, device=device
        )

        w = opacity[idx]

        cur_labels = topk_labels[idx]
        cur_weights = topk_weights[idx]

        merged_exp = merged_labels.unsqueeze(1)
        match_any = (cur_labels == merged_exp)

        cur_weights += w.unsqueeze(1) * match_any.float()

        has_match = match_any.any(dim=1)
        no_match = ~has_match

        if no_match.any():
            nm_local = torch.where(no_match)[0]
            nm_labels = merged_labels[nm_local]
            nm_w = w[nm_local]
            nm_cur_w = cur_weights[nm_local]

            min_vals, min_slots = nm_cur_w.min(dim=1)

            replace = nm_w > min_vals
            if replace.any():
                rep_local = nm_local[replace]
                rep_slots = min_slots[replace]
                rep_labels = nm_labels[replace]
                rep_w = nm_w[replace]
                cur_labels[rep_local, rep_slots] = rep_labels
                cur_weights[rep_local, rep_slots] = rep_w

        topk_labels[idx] = cur_labels
        topk_weights[idx] = cur_weights

    # Pick argmax, competing with background
    best_fg_weight, best_slot = topk_weights.max(dim=1)
    instance_id = topk_labels[torch.arange(N, device=device), best_slot]

    # Foreground must exceed background * fg_bg_ratio to win
    bg_wins = best_fg_weight < bg_weight * fg_bg_ratio
    instance_id[bg_wins] = 0
    n_bg = bg_wins.sum().item()
    print(f"  [Semantic] Background: {n_bg} ({100*n_bg/N:.1f}%), Foreground: {N-n_bg}")

    # Compact label IDs
    unique_labels, inverse = instance_id.unique(return_inverse=True)
    gaussians._instance_id = inverse

    # Assign semantic category via Union-Find root lookup
    semantic_cat = torch.zeros(N, dtype=torch.long, device=device)
    # Build root→category mapping: for each merged root, pick the most common category
    root_to_cat: dict[int, int] = {}
    for gl, cat in global_label_to_category.items():
        root = uf.find(gl)
        if root not in root_to_cat:
            root_to_cat[root] = cat
        # If conflict, keep the first (most labels share the same category
        # since different categories' instances rarely merge)

    # Map unique_labels back to their roots
    unique_labels_cpu = unique_labels.cpu().tolist()
    for compact_id, orig_label in enumerate(unique_labels_cpu):
        if orig_label == 0:
            continue  # background stays 0
        cat = root_to_cat.get(orig_label, 0)
        mask_for_id = inverse == compact_id
        semantic_cat[mask_for_id] = cat

    gaussians._semantic_category = semantic_cat

    # Print per-category stats
    n_unique = unique_labels.shape[0]
    print(f"  [Semantic] Unique instance IDs: {n_unique}")
    cats_cpu = semantic_cat.cpu()
    for cat_id in sorted(cats_cpu.unique().tolist()):
        count = (cats_cpu == cat_id).sum().item()
        n_inst = inverse[cats_cpu == cat_id].unique().numel()
        print(f"    Category {cat_id}: {count} Gaussians, {n_inst} instances")


# ---------------------------------------------------------------------------
# Legacy API wrappers (for train.py compatibility)
# ---------------------------------------------------------------------------

@torch.no_grad()
def build_instance_weight_matrix(
    cameras: Sequence[Camera],
    gaussians: GaussianModel,
    *,
    opacity_weighted: bool = True,
) -> dict:
    """Legacy wrapper — returns empty dict. Use assign_instance_labels() instead."""
    return {}


def majority_vote_merge(W_sparse, cameras, tau_aff=0.3):
    """Legacy wrapper — returns empty dict. Use assign_instance_labels() instead."""
    return {}


@torch.no_grad()
def assign_labels(gaussians, W_sparse, remap):
    """Legacy wrapper — no-op. Use assign_instance_labels() instead."""
    pass


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def instance_seg_loss_ce(
    viewpoint_cam: Camera,
    gaussians: GaussianModel,
    render_pkg: dict,
) -> torch.Tensor:
    """Cross-entropy proxy: project visible Gaussians to 2D, compare their
    ``_instance_id`` to the mask label at the projected pixel.  Return
    mismatch fraction (non-differentiable but usable as a penalty signal)."""
    mask = getattr(viewpoint_cam, "instance_mask", None)
    if mask is None:
        return torch.tensor(0.0, device=gaussians.get_xyz.device)

    H, W = mask.shape
    vis = render_pkg.get("visibility_filter", None)
    if vis is None:
        return torch.tensor(0.0, device=gaussians.get_xyz.device)

    xyz = gaussians.get_xyz  # (N, 3)
    inst_id = gaussians._instance_id  # (N,)

    R = viewpoint_cam.R
    T = viewpoint_cam.T
    xyz_cam = torch.addmm(T.unsqueeze(0), xyz, R)
    z = xyz_cam[:, 2]

    u = (xyz_cam[:, 0] / z * viewpoint_cam.Fx + viewpoint_cam.Cx).long()
    v = (xyz_cam[:, 1] / z * viewpoint_cam.Fy + viewpoint_cam.Cy).long()

    valid = vis & (z > 0.2) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if not valid.any():
        return torch.tensor(0.0, device=xyz.device)

    idx = torch.where(valid)[0]
    gt_labels = mask[v[idx], u[idx]]
    pred_labels = inst_id[idx]

    mismatch = (pred_labels != gt_labels).float().mean()
    return mismatch


def instance_seg_loss_contrastive(
    viewpoint_cam: Camera,
    gaussians: GaussianModel,
    render_pkg: dict,
) -> torch.Tensor:
    """Placeholder for contrastive loss on ``_instance_features``.
    Not yet implemented — returns zero."""
    return torch.tensor(0.0, device=gaussians.get_xyz.device)


def depth_edge_loss(
    render_pkg: dict,
    viewpoint_cam: Camera,
) -> torch.Tensor:
    """Encourage depth discontinuities to align with instance mask edges.

    Applies Sobel filters to both the rendered median depth and the instance
    mask, then penalizes depth locations where mask edges exist but depth
    edges are weak.
    """
    mask = getattr(viewpoint_cam, "instance_mask", None)
    if mask is None:
        return torch.tensor(0.0, device="cuda")

    depth = render_pkg.get("median_depth", None)
    if depth is None:
        return torch.tensor(0.0, device="cuda")

    # depth: (1, H, W) or (H, W)
    if depth.dim() == 2:
        depth = depth.unsqueeze(0).unsqueeze(0)
    elif depth.dim() == 3:
        depth = depth.unsqueeze(0)

    # Sobel kernels
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=torch.float32,
        device=depth.device,
    ).reshape(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        dtype=torch.float32,
        device=depth.device,
    ).reshape(1, 1, 3, 3)

    # Depth edges
    depth_norm = depth / (depth.max() + 1e-7)
    edge_dx = F.conv2d(depth_norm, sobel_x, padding=1)
    edge_dy = F.conv2d(depth_norm, sobel_y, padding=1)
    depth_edge = (edge_dx.abs() + edge_dy.abs()).squeeze()

    # Mask edges
    mask_f = mask.float().unsqueeze(0).unsqueeze(0)
    mask_dx = F.conv2d(mask_f, sobel_x, padding=1)
    mask_dy = F.conv2d(mask_f, sobel_y, padding=1)
    mask_edge = (mask_dx.abs() + mask_dy.abs()).squeeze()
    mask_edge = (mask_edge > 0).float()

    # Loss: where mask has edges, depth should also have edges
    # Penalize low depth gradient at mask boundaries
    if mask_edge.sum() < 1:
        return torch.tensor(0.0, device=depth.device)

    loss = (mask_edge * (1.0 - depth_edge.clamp(0, 1))).sum() / mask_edge.sum()
    return loss
