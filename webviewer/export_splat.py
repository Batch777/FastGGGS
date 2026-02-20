#!/usr/bin/env python3
"""Convert GGGS PLY output to compact .splat binary format for the web viewer.

Usage:
    python webviewer/export_splat.py -m <output_dir> [--iteration 30000] [--sh_degree 0]

Output: <output_dir>/point_cloud/iteration_<N>/point_cloud.splat
"""

import argparse
import os
import struct
import shutil
import numpy as np
from plyfile import PlyData


C0 = 0.28209479177387814


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def quaternion_to_rotation_matrix(quats):
    """Convert (N, 4) quaternions [w,x,y,z] to (N, 3, 3) rotation matrices."""
    w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    # Matching CUDA rasterizer convention (render_forward.cu:152-155)
    R = np.zeros((len(quats), 3, 3), dtype=np.float32)
    R[:, 0, 0] = 1 - 2*(y*y + z*z)
    R[:, 0, 1] = 2*(x*y - w*z)
    R[:, 0, 2] = 2*(x*z + w*y)
    R[:, 1, 0] = 2*(x*y + w*z)
    R[:, 1, 1] = 1 - 2*(x*x + z*z)
    R[:, 1, 2] = 2*(y*z - w*x)
    R[:, 2, 0] = 2*(x*z - w*y)
    R[:, 2, 1] = 2*(y*z + w*x)
    R[:, 2, 2] = 1 - 2*(x*x + y*y)
    return R


def compute_covariance(scales, rotations):
    """Compute 3D covariance matrices from scales and quaternion rotations.

    Returns (N, 6) array of upper-triangle symmetric covariance values:
    [s00, s01, s02, s11, s12, s22]
    """
    N = len(scales)
    R = quaternion_to_rotation_matrix(rotations)  # (N, 3, 3)

    # M = S * R where S = diag(scale)
    # Sigma = M^T * M
    S = scales[:, :, np.newaxis] * np.eye(3)[np.newaxis, :, :]  # (N, 3, 3)
    M = np.einsum('nij,njk->nik', S, R)  # (N, 3, 3)
    Sigma = np.einsum('nji,njk->nik', M, M)  # (N, 3, 3) = M^T * M

    cov = np.zeros((N, 6), dtype=np.float32)
    cov[:, 0] = Sigma[:, 0, 0]
    cov[:, 1] = Sigma[:, 0, 1]
    cov[:, 2] = Sigma[:, 0, 2]
    cov[:, 3] = Sigma[:, 1, 1]
    cov[:, 4] = Sigma[:, 1, 2]
    cov[:, 5] = Sigma[:, 2, 2]
    return cov


def main():
    parser = argparse.ArgumentParser(description='Convert GGGS PLY to .splat format')
    parser.add_argument('-m', '--model_path', required=True, help='Model output directory')
    parser.add_argument('--iteration', type=int, default=30000, help='Iteration to export')
    parser.add_argument('--sh_degree', type=int, default=0, help='SH degree to include (0-3)')
    parser.add_argument('--min_opacity', type=float, default=0.004,
                        help='Minimum opacity threshold (after sigmoid)')
    args = parser.parse_args()

    # Locate PLY file
    ply_path = os.path.join(args.model_path, 'point_cloud',
                            f'iteration_{args.iteration}', 'point_cloud.ply')
    if not os.path.exists(ply_path):
        print(f'PLY file not found: {ply_path}')
        return

    print(f'Loading {ply_path}...')
    plydata = PlyData.read(ply_path)
    vertex = plydata['vertex']
    N = len(vertex)
    print(f'  {N} Gaussians')

    # Extract properties
    xyz = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=1).astype(np.float32)

    # Scale: exp activation
    scale = np.stack([vertex['scale_0'], vertex['scale_1'], vertex['scale_2']],
                     axis=1).astype(np.float32)
    scale = np.exp(scale)

    # Rotation: normalize quaternion (w,x,y,z)
    rot = np.stack([vertex['rot_0'], vertex['rot_1'], vertex['rot_2'], vertex['rot_3']],
                   axis=1).astype(np.float32)
    rot_norm = np.linalg.norm(rot, axis=1, keepdims=True)
    rot_norm = np.maximum(rot_norm, 1e-8)
    rot = rot / rot_norm

    # GGGS filter_3D: adjust scale and opacity for anti-aliasing
    opacity_raw = vertex['opacity'].astype(np.float32)
    opacity = sigmoid(opacity_raw)

    if 'filter_3D' in vertex.data.dtype.names:
        filter_3D = vertex['filter_3D'].astype(np.float32)
        f3d2 = filter_3D[:, np.newaxis] ** 2  # (N, 1)
        scale_sq = scale ** 2
        det1 = np.prod(scale_sq, axis=1)
        scale_sq_filtered = scale_sq + f3d2
        det2 = np.prod(scale_sq_filtered, axis=1)
        coef = np.sqrt(det1 / np.maximum(det2, 1e-30))
        scale = np.sqrt(scale_sq_filtered)
        opacity = opacity * coef
        print(f'  Applied filter_3D (GGGS anti-aliasing)')
    else:
        print(f'  No filter_3D found, using standard 3DGS')

    # DC color: C0 * f_dc + 0.5
    f_dc = np.stack([vertex['f_dc_0'], vertex['f_dc_1'], vertex['f_dc_2']],
                    axis=1).astype(np.float32)
    color = np.clip(C0 * f_dc + 0.5, 0, 1)

    # Prune low-opacity Gaussians
    mask = opacity > args.min_opacity
    print(f'  Pruning: {N - mask.sum()} Gaussians below opacity {args.min_opacity}')
    xyz = xyz[mask]
    opacity = opacity[mask]
    scale = scale[mask]
    rot = rot[mask]
    color = color[mask]
    N = len(xyz)
    print(f'  {N} Gaussians after pruning')

    # Compute 3D covariance
    print('Computing covariance matrices...')
    cov = compute_covariance(scale, rot)

    # Write .splat binary
    out_path = ply_path.replace('.ply', '.splat')
    print(f'Writing {out_path}...')

    with open(out_path, 'wb') as f:
        # Header (128 bytes)
        header = bytearray(128)
        header[0:4] = b'SPLT'                                    # magic
        struct.pack_into('<I', header, 4, 1)                      # version
        struct.pack_into('<I', header, 8, N)                      # gaussian count
        struct.pack_into('<I', header, 12, args.sh_degree)        # SH degree
        f.write(header)

        # Per-Gaussian data: 40 bytes each (vectorized)
        # position(12) + covariance(24) + color_rgb(3) + opacity(1)
        # Build structured array for efficient binary write
        dtype = np.dtype([
            ('pos', '<f4', 3),
            ('cov', '<f4', 6),
            ('color', 'u1', 3),
            ('opacity', 'u1'),
        ])
        data = np.zeros(N, dtype=dtype)
        data['pos'] = xyz
        data['cov'] = cov
        data['color'] = np.clip(color * 255, 0, 255).astype(np.uint8)
        data['opacity'] = np.clip(opacity * 255, 0, 255).astype(np.uint8)
        f.write(data.tobytes())

    ply_size = os.path.getsize(ply_path)
    splat_size = os.path.getsize(out_path)
    print(f'  PLY: {ply_size / 1024 / 1024:.1f} MB')
    print(f'  .splat: {splat_size / 1024 / 1024:.1f} MB')
    print(f'  Compression: {ply_size / splat_size:.1f}x')

    # Copy cameras.json if available
    cameras_src = os.path.join(args.model_path, 'cameras.json')
    cameras_dst = os.path.join(os.path.dirname(out_path), 'cameras.json')
    if os.path.exists(cameras_src):
        shutil.copy2(cameras_src, cameras_dst)
        print(f'  Copied cameras.json')

    print('Done!')


if __name__ == '__main__':
    main()
