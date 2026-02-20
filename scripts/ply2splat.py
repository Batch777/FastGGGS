#!/usr/bin/env python3
"""Convert Gaussian Splatting PLY files to .splat v2 binary format.

Precomputes all activations (exp, sigmoid, filter_3D, quaternion→covariance,
softplus, normalization) in Python, matching the JS parsePLY() math exactly.
The resulting .splat file uses float16 for most data (positions stay float32)
and SoA layout for fast bulk reads in the browser.

Usage:
    python scripts/ply2splat.py <input.ply> [output.splat]
    python scripts/ply2splat.py <input.ply> --full-precision
"""

import argparse
import os
import struct
import sys

import numpy as np
from plyfile import PlyData

C0 = 0.28209479177387814

HEADER_SIZE = 128
MAGIC = b"SPLT"
VERSION = 2


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softplus(x):
    return np.log1p(np.exp(x))


def quaternion_to_rotmat(q):
    """Convert (N, 4) quaternions [w, x, y, z] to (N, 3, 3) rotation matrices.

    Matches the JS quaternionToRotationMatrix() exactly.
    """
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.zeros((len(q), 3, 3), dtype=np.float64)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def compute_covariance(scale, rotation):
    """Compute 3D covariance from scale (N,3) and rotation (N,3,3).

    M = S * R^T  →  Sigma = M^T * M = R * S^2 * R^T
    Returns (N, 6) upper-triangular: [s00, s01, s02, s11, s12, s22]
    """
    # M[i][j] = s_i * R[j][i]  (i.e. M = S * R^T)
    # M shape: (N, 3, 3) where M[:, i, j] = scale[:, i] * R[:, j, i]
    M = scale[:, :, None] * np.swapaxes(rotation, 1, 2)  # (N, 3, 3)

    # Sigma = M^T * M
    Sigma = np.einsum("nij,nik->njk", M, M)  # (N, 3, 3)

    cov6 = np.stack(
        [
            Sigma[:, 0, 0],
            Sigma[:, 0, 1],
            Sigma[:, 0, 2],
            Sigma[:, 1, 1],
            Sigma[:, 1, 2],
            Sigma[:, 2, 2],
        ],
        axis=1,
    )
    return cov6


def read_ply(path):
    """Read PLY and return structured data dict."""
    print(f"Reading {path}...")
    plydata = PlyData.read(path)
    vertex = plydata["vertex"]
    n = len(vertex.data)
    props = set(vertex.data.dtype.names)
    print(f"  {n:,} Gaussians, {len(props)} properties")

    data = {}
    data["count"] = n

    # Positions
    data["xyz"] = np.stack(
        [vertex["x"], vertex["y"], vertex["z"]], axis=1
    ).astype(np.float64)

    # DC color
    data["f_dc"] = np.stack(
        [vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]], axis=1
    ).astype(np.float64)

    # Opacity (raw)
    data["opacity_raw"] = vertex["opacity"].astype(np.float64)

    # Scale (raw)
    data["scale_raw"] = np.stack(
        [vertex["scale_0"], vertex["scale_1"], vertex["scale_2"]], axis=1
    ).astype(np.float64)

    # Rotation (raw quaternion: w, x, y, z)
    data["rot_raw"] = np.stack(
        [vertex["rot_0"], vertex["rot_1"], vertex["rot_2"], vertex["rot_3"]],
        axis=1,
    ).astype(np.float64)

    # filter_3D
    if "filter_3D" in props:
        data["filter_3D"] = vertex["filter_3D"].astype(np.float64)
    else:
        data["filter_3D"] = np.zeros(n, dtype=np.float64)

    # SH degree
    sh_degree = 0
    if "f_rest_0" in props:
        rest_count = sum(1 for p in props if p.startswith("f_rest_"))
        coeffs_per_ch = rest_count // 3
        if coeffs_per_ch >= 15:
            sh_degree = 3
        elif coeffs_per_ch >= 8:
            sh_degree = 2
        elif coeffs_per_ch >= 3:
            sh_degree = 1
    data["sh_degree"] = sh_degree

    # SH rest coefficients (always store 45 for simplicity)
    if sh_degree > 0:
        sh_rest = np.zeros((n, 45), dtype=np.float64)
        for j in range(45):
            key = f"f_rest_{j}"
            if key in props:
                sh_rest[:, j] = vertex[key].astype(np.float64)
        data["sh_rest"] = sh_rest
    else:
        data["sh_rest"] = None

    # SG degree
    sg_degree = 0
    if "sg_sharpness_0" in props:
        sg_degree = sum(1 for p in props if p.startswith("sg_sharpness_"))
    data["sg_degree"] = sg_degree

    if sg_degree > 0:
        # Axis: (N, sg_degree, 3)
        sg_axis = np.zeros((n, sg_degree, 3), dtype=np.float64)
        for g in range(sg_degree):
            for k in range(3):
                key = f"sg_axis_{g * 3 + k}"
                if key in props:
                    sg_axis[:, g, k] = vertex[key].astype(np.float64)
        data["sg_axis_raw"] = sg_axis

        # Sharpness: (N, sg_degree)
        sg_sharp = np.zeros((n, sg_degree), dtype=np.float64)
        for g in range(sg_degree):
            key = f"sg_sharpness_{g}"
            if key in props:
                sg_sharp[:, g] = vertex[key].astype(np.float64)
        data["sg_sharpness_raw"] = sg_sharp

        # Color: (N, sg_degree, 3)
        sg_color = np.zeros((n, sg_degree, 3), dtype=np.float64)
        for g in range(sg_degree):
            for k in range(3):
                key = f"sg_color_{g * 3 + k}"
                if key in props:
                    sg_color[:, g, k] = vertex[key].astype(np.float64)
        data["sg_color"] = sg_color
    else:
        data["sg_axis_raw"] = None
        data["sg_sharpness_raw"] = None
        data["sg_color"] = None

    # Segments
    has_seg = "instance_id" in props and "semantic_category" in props
    if has_seg:
        instance_ids = vertex["instance_id"].astype(np.float64)
        semantic_cats = vertex["semantic_category"].astype(np.float64)
        if np.any(semantic_cats > 0):
            data["instance_id"] = instance_ids
            data["semantic_category"] = semantic_cats
            data["has_segments"] = True
        else:
            data["has_segments"] = False
    else:
        data["has_segments"] = False

    return data


def activate(data):
    """Apply all activations matching JS parsePLY() exactly."""
    n = data["count"]
    print("Applying activations...")

    # DC color: C0 * f_dc + 0.5 (NO clamp)
    colors = C0 * data["f_dc"] + 0.5

    # Scale: exp
    scale = np.exp(data["scale_raw"])

    # filter_3D correction
    f3d = data["filter_3D"]
    f3d2 = f3d * f3d
    has_f3d = f3d2 > 0

    scale_sq = scale * scale
    det1 = np.prod(scale_sq, axis=1)

    adjusted_sq = scale_sq + f3d2[:, None]
    det2 = np.prod(adjusted_sq, axis=1)

    # Opacity: sigmoid(raw) * coef where applicable
    opacity = sigmoid(data["opacity_raw"])
    coef = np.where(has_f3d, np.sqrt(det1 / det2), 1.0)
    opacity *= coef

    # Scale adjusted for filter_3D
    scale_adj = np.where(
        has_f3d[:, None], np.sqrt(adjusted_sq), scale
    )

    # Quaternion normalize
    q = data["rot_raw"].copy()
    qlen = np.linalg.norm(q, axis=1, keepdims=True)
    qlen = np.where(qlen > 0, qlen, 1.0)
    q /= qlen

    # Rotation matrix
    R = quaternion_to_rotmat(q)

    # Covariance
    cov6 = compute_covariance(scale_adj, R)

    result = {
        "count": n,
        "positions": data["xyz"],
        "covariances": cov6,
        "colors": colors,
        "opacities": opacity,
        "sh_degree": data["sh_degree"],
        "sh_rest": data["sh_rest"],
        "sg_degree": data["sg_degree"],
        "has_segments": data["has_segments"],
    }

    # SG activations
    if data["sg_degree"] > 0:
        sg_deg = data["sg_degree"]

        # Normalize axis
        axis = data["sg_axis_raw"].copy()  # (N, sg_deg, 3)
        axis_len = np.linalg.norm(axis, axis=2, keepdims=True)
        axis_len = np.where(axis_len > 0, axis_len, 1.0)
        axis /= axis_len

        # Softplus sharpness
        sharpness = softplus(data["sg_sharpness_raw"])  # (N, sg_deg)

        # Pack axis+sharpness: (N, sg_deg, 4)
        sg_axis_sharp = np.concatenate(
            [axis, sharpness[:, :, None]], axis=2
        )
        result["sg_axis_sharp"] = sg_axis_sharp
        result["sg_color"] = data["sg_color"]  # raw, no activation
    else:
        result["sg_axis_sharp"] = None
        result["sg_color"] = None

    # Segments
    if data["has_segments"]:
        result["semantic_category"] = data["semantic_category"]
        result["instance_id"] = data["instance_id"]

    return result


def float32_to_float16_bytes(arr):
    """Convert float64/float32 array to float16 bytes (little-endian)."""
    return arr.astype(np.float16).tobytes()


def write_splat(result, output_path, full_precision=False):
    """Write .splat v2 binary file."""
    n = result["count"]
    sh_deg = result["sh_degree"]
    sg_deg = result["sg_degree"]
    has_seg = 1 if result["has_segments"] else 0

    float_type = np.float32 if full_precision else np.float16

    print(f"Writing {output_path}...")
    print(f"  {n:,} Gaussians, SH degree {sh_deg}, SG degree {sg_deg}, segments: {bool(has_seg)}")
    print(f"  Precision: {'float32' if full_precision else 'float16'} (positions always float32)")

    with open(output_path, "wb") as f:
        # Header (128 bytes)
        header = bytearray(HEADER_SIZE)
        header[0:4] = MAGIC
        struct.pack_into("<I", header, 4, VERSION)
        struct.pack_into("<I", header, 8, n)
        struct.pack_into("<I", header, 12, sh_deg)
        struct.pack_into("<I", header, 16, sg_deg)
        struct.pack_into("<I", header, 20, has_seg)
        f.write(header)

        # Positions: N × 3 × float32 (always f32 for sort precision)
        f.write(result["positions"].astype(np.float32).tobytes())

        # Covariances: N × 6 × float_type
        f.write(result["covariances"].astype(float_type).tobytes())

        # Colors: N × 3 × float_type
        f.write(result["colors"].astype(float_type).tobytes())

        # Opacities: N × 1 × float_type
        f.write(result["opacities"].astype(float_type).tobytes())

        # SH rest: N × 45 × float_type (only if sh_degree > 0)
        if sh_deg > 0 and result["sh_rest"] is not None:
            f.write(result["sh_rest"].astype(float_type).tobytes())

        # SG axis+sharpness: N × sg_degree × 4 × float_type (only if sg_degree > 0)
        if sg_deg > 0 and result["sg_axis_sharp"] is not None:
            f.write(result["sg_axis_sharp"].astype(float_type).tobytes())

        # SG color: N × sg_degree × 3 × float_type (only if sg_degree > 0)
        if sg_deg > 0 and result["sg_color"] is not None:
            f.write(result["sg_color"].astype(float_type).tobytes())

        # Segments: N × 2 × uint16 (only if has_segments)
        if has_seg:
            seg = np.stack(
                [result["semantic_category"], result["instance_id"]], axis=1
            ).astype(np.uint16)
            f.write(seg.tobytes())

    file_size = os.path.getsize(output_path)
    print(f"  Output: {file_size / 1024 / 1024:.1f} MB")


def copy_segments_json(input_ply, output_splat):
    """Copy segments.json alongside output if it exists next to input PLY."""
    seg_src = os.path.join(os.path.dirname(input_ply), "segments.json")
    seg_dst = os.path.join(os.path.dirname(output_splat), "segments.json")
    if os.path.exists(seg_src) and seg_src != seg_dst:
        import shutil
        shutil.copy2(seg_src, seg_dst)
        print(f"  Copied segments.json to {seg_dst}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Gaussian Splatting PLY to .splat v2 binary format"
    )
    parser.add_argument("input", help="Input .ply file path")
    parser.add_argument("output", nargs="?", help="Output .splat file path (default: alongside input)")
    parser.add_argument(
        "--full-precision",
        action="store_true",
        help="Use float32 instead of float16 (larger but exact)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found", file=sys.stderr)
        sys.exit(1)

    output = args.output
    if output is None:
        output = os.path.join(
            os.path.dirname(args.input), "point_cloud.splat"
        )

    # Read PLY
    data = read_ply(args.input)

    # Apply activations
    result = activate(data)

    # Write .splat
    write_splat(result, output, full_precision=args.full_precision)

    # Copy segments.json if present
    copy_segments_json(args.input, output)

    # Size comparison
    ply_size = os.path.getsize(args.input)
    splat_size = os.path.getsize(output)
    ratio = (1 - splat_size / ply_size) * 100
    print(f"\nSize: {ply_size / 1024 / 1024:.1f} MB → {splat_size / 1024 / 1024:.1f} MB ({ratio:.0f}% smaller)")


if __name__ == "__main__":
    main()
