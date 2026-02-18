#!/usr/bin/env python3
"""Compare CUDA vs WebGL 2D covariance math for a few Gaussians."""
import numpy as np
import struct
import json
import math

def quaternion_to_rotation_matrix_standard(q):
    """Standard quaternion to rotation matrix (q = [w,x,y,z])."""
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]
    ])

def quaternion_to_rotation_matrix_cuda(q):
    """CUDA rasterizer quaternion convention (column-major glm::mat3)."""
    w, x, y, z = q
    # GLM column-major constructor fills col0, col1, col2
    # Reading the CUDA source, R = glm::mat3(
    #   1-2(yy+zz), 2(xy-wz), 2(xz+wy),    // col0
    #   2(xy+wz), 1-2(xx+zz), 2(yz-wx),      // col1
    #   2(xz-wy), 2(yz+wx), 1-2(xx+yy));     // col2
    # GLM: mat[col][row], so as a numpy array (row,col):
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y+w*z), 2*(x*z-w*y)],
        [2*(x*y-w*z), 1-2*(x*x+z*z), 2*(y*z+w*x)],
        [2*(x*z+w*y), 2*(y*z-w*x), 1-2*(x*x+y*y)]
    ])

def compute_cov3d_loader(scale, quaternion):
    """WebGL loader (BUGGY): M = diag(S) * R_standard, Sigma = M^T * M.
    This gives R^T * S^2 * R, but CUDA gives R * S^2 * R^T."""
    R = quaternion_to_rotation_matrix_standard(quaternion)
    S = np.diag(scale)
    M = S @ R
    Sigma = M.T @ M
    return Sigma

def compute_cov3d_loader_fixed(scale, quaternion):
    """WebGL loader (FIXED): M = diag(S) * R^T, Sigma = M^T * M.
    This gives R * S^2 * R^T, matching CUDA."""
    R = quaternion_to_rotation_matrix_standard(quaternion)
    S = np.diag(scale)
    M = S @ R.T  # Use R^T to match CUDA's GLM column-major convention
    Sigma = M.T @ M
    return Sigma

def compute_cov3d_cuda(scale, quaternion):
    """CUDA computeCov3D: M = S * R_cuda, Sigma = M^T * M."""
    R = quaternion_to_rotation_matrix_cuda(quaternion)
    S = np.diag(scale)
    M = S @ R  # In GLM: S * R where R uses CUDA convention
    Sigma = M.T @ M
    return Sigma

def compute_cov2d_cuda(mean, scale, quaternion, viewmatrix, fx, fy, tan_fovx, tan_fovy, kernel_size):
    """CUDA computeCov2D (if scale branch): M = S * R * T, cov = M^T * M."""
    # Transform to camera space
    t = viewmatrix[:3,:3] @ mean + viewmatrix[:3,3]

    # Frustum clamp
    limx = 1.3 * tan_fovx
    limy = 1.3 * tan_fovy
    u = t[0] / t[2]
    v = t[1] / t[2]
    t[0] = min(limx, max(-limx, u)) * t[2]
    t[1] = min(limy, max(-limy, v)) * t[2]

    # Jacobian
    J = np.array([
        [fx/t[2], 0, 0],
        [0, fy/t[2], 0],
        [-(fx*t[0])/(t[2]**2), -(fy*t[1])/(t[2]**2), 0]
    ])

    # W = transpose of rotation part of view matrix
    # In CUDA: W = glm::mat3(vm[0],vm[4],vm[8], vm[1],vm[5],vm[9], vm[2],vm[6],vm[10])
    # Which is rows of the view matrix as columns of W → W = view_rot^T
    W = viewmatrix[:3,:3].T

    T = W @ J

    R_cuda = quaternion_to_rotation_matrix_cuda(quaternion)
    S = np.diag(scale)
    M = S @ R_cuda @ T
    cov = M.T @ M

    # Add kernel_size and compute coef
    cov2d_raw = np.array([cov[0,0], cov[0,1], cov[1,1]])
    det_0 = max(1e-6, cov[0,0]*cov[1,1] - cov[0,1]**2)
    cov2d = np.array([cov[0,0]+kernel_size, cov[0,1], cov[1,1]+kernel_size])
    det_1 = max(1e-6, cov2d[0]*cov2d[2] - cov2d[1]**2)
    coef = math.sqrt(det_0 / det_1)

    return cov2d, coef, t

def compute_cov2d_webgl(mean, cov3d_6, viewmatrix, fx, fy, viewport_w, viewport_h, kernel_size):
    """WebGL shader math: T^T * Vrk * T."""
    # Transform to camera space
    t = viewmatrix[:3,:3] @ mean + viewmatrix[:3,3]

    # Frustum clamp (FIXED version with tanFov)
    tan_fovx = viewport_w / (2.0 * fx)
    tan_fovy = viewport_h / (2.0 * fy)
    limx = 1.3 * tan_fovx * t[2]
    limy = 1.3 * tan_fovy * t[2]
    t[0] = min(limx, max(-limx, t[0]))
    t[1] = min(limy, max(-limy, t[1]))

    # Jacobian
    z2 = t[2]**2
    J = np.zeros((3,3))
    J[0,0] = fx / t[2]
    J[1,1] = fy / t[2]
    J[2,0] = -(fx * t[0]) / z2
    J[2,1] = -(fy * t[1]) / z2

    # W = transpose of rotation part of view matrix
    W = viewmatrix[:3,:3].T

    T_mat = W @ J

    # Reconstruct 3D covariance
    s00,s01,s02,s11,s12,s22 = cov3d_6
    Vrk = np.array([
        [s00, s01, s02],
        [s01, s11, s12],
        [s02, s12, s22]
    ])

    cov2d_full = T_mat.T @ Vrk @ T_mat

    a_raw = cov2d_full[0,0]
    b = cov2d_full[0,1]
    c_raw = cov2d_full[1,1]
    a = a_raw + kernel_size
    c = c_raw + kernel_size

    det_before = max(1e-6, a_raw * c_raw - b*b)
    det_after = max(1e-6, a * c - b*b)
    coef = math.sqrt(det_before / det_after)

    cov2d = np.array([a, b, c])
    return cov2d, coef, t

def compute_cov2d_webgl_old(mean, cov3d_6, viewmatrix, fx, fy, viewport_w, viewport_h, kernel_size):
    """Old WebGL shader math (before fixes): limx = 1.3 * t.z, no coef."""
    t = viewmatrix[:3,:3] @ mean + viewmatrix[:3,3]

    # OLD frustum clamp (bug: assumes tanFov=1.0)
    limx = 1.3 * t[2]
    limy = 1.3 * t[2]
    t[0] = min(limx, max(-limx, t[0]))
    t[1] = min(limy, max(-limy, t[1]))

    z2 = t[2]**2
    J = np.zeros((3,3))
    J[0,0] = fx / t[2]
    J[1,1] = fy / t[2]
    J[2,0] = -(fx * t[0]) / z2
    J[2,1] = -(fy * t[1]) / z2

    W = viewmatrix[:3,:3].T
    T_mat = W @ J

    s00,s01,s02,s11,s12,s22 = cov3d_6
    Vrk = np.array([[s00,s01,s02],[s01,s11,s12],[s02,s12,s22]])

    cov2d_full = T_mat.T @ Vrk @ T_mat
    a = cov2d_full[0,0] + kernel_size
    b = cov2d_full[0,1]
    c = cov2d_full[1,1] + kernel_size

    # OLD: no opacity coef correction
    coef = 1.0

    cov2d = np.array([a, b, c])
    return cov2d, coef, t

# ── Load data ──
ply_path = 'output/dtu_baseline/scan24/point_cloud/iteration_30000/point_cloud.ply'
cam_path = 'output/dtu_baseline/scan24/cameras.json'

with open(ply_path, 'rb') as f:
    header = b''
    while b'end_header' not in header:
        header += f.readline()
    header_text = header.decode()

    props = []
    for line in header_text.split('\n'):
        if line.strip().startswith('property float '):
            props.append(line.strip().split()[-1])

    vertex_size = len(props) * 4
    prop_idx = {name: i for i, name in enumerate(props)}

    # Read a few vertices
    n_test = 100
    vertices = []
    for i in range(n_test):
        data = f.read(vertex_size)
        vals = struct.unpack(f'<{len(props)}f', data)
        vertices.append({name: vals[prop_idx[name]] for name in props})

with open(cam_path) as f:
    cams = json.load(f)

cam = cams[0]

# Build view matrix
R_c2w = np.array(cam['rotation'])
pos = np.array(cam['position'])
R_w2c = R_c2w.T
t_w2c = -R_w2c @ pos

viewmatrix = np.eye(4)
viewmatrix[:3,:3] = R_w2c
viewmatrix[:3,3] = t_w2c

fx = cam['fx']
fy = cam['fy']
w = cam['width']
h = cam['height']
tan_fovx = w / (2.0 * fx)
tan_fovy = h / (2.0 * fy)
kernel_size = 0.0

print(f"Camera: {w}x{h}, fx={fx:.2f}, fy={fy:.2f}")
print(f"tanFovx={tan_fovx:.6f}, tanFovy={tan_fovy:.6f}")
print(f"kernel_size={kernel_size}")
print()

# Compare for each test vertex
max_cov_diff = 0
max_cov_diff_fixed = 0
max_coef_diff = 0
max_t_diff = 0
big_diffs = []

for vi, v in enumerate(vertices):
    mean = np.array([v['x'], v['y'], v['z']])

    sx = math.exp(v['scale_0'])
    sy = math.exp(v['scale_1'])
    sz = math.exp(v['scale_2'])

    f3d = v.get('filter_3D', 0)
    f3d2 = f3d * f3d
    if f3d2 > 0:
        ax2 = sx*sx + f3d2
        ay2 = sy*sy + f3d2
        az2 = sz*sz + f3d2
        scale = [math.sqrt(ax2), math.sqrt(ay2), math.sqrt(az2)]
    else:
        scale = [sx, sy, sz]

    qw, qx, qy, qz = v['rot_0'], v['rot_1'], v['rot_2'], v['rot_3']
    qlen = math.sqrt(qw**2+qx**2+qy**2+qz**2)
    if qlen > 0:
        qw /= qlen; qx /= qlen; qy /= qlen; qz /= qlen
    quaternion = [qw, qx, qy, qz]

    # Compute 3D covariance (WebGL loader style - BUGGY)
    cov3d = compute_cov3d_loader(scale, quaternion)
    cov3d_6 = [cov3d[0,0], cov3d[0,1], cov3d[0,2], cov3d[1,1], cov3d[1,2], cov3d[2,2]]

    # Compute 3D covariance (FIXED)
    cov3d_fix = compute_cov3d_loader_fixed(scale, quaternion)
    cov3d_6_fix = [cov3d_fix[0,0], cov3d_fix[0,1], cov3d_fix[0,2], cov3d_fix[1,1], cov3d_fix[1,2], cov3d_fix[2,2]]

    # CUDA 2D covariance (ground truth)
    cov2d_cuda, coef_cuda, t_cuda = compute_cov2d_cuda(
        mean.copy(), scale, quaternion, viewmatrix, fx, fy, tan_fovx, tan_fovy, kernel_size)

    # WebGL 2D covariance (buggy 3D cov)
    cov2d_webgl, coef_webgl, t_webgl = compute_cov2d_webgl(
        mean.copy(), cov3d_6, viewmatrix, fx, fy, w, h, kernel_size)

    # WebGL 2D covariance (FIXED 3D cov)
    cov2d_fixed, coef_fixed, t_fixed = compute_cov2d_webgl(
        mean.copy(), cov3d_6_fix, viewmatrix, fx, fy, w, h, kernel_size)

    cov_diff = np.max(np.abs(cov2d_cuda - cov2d_webgl))
    cov_diff_fixed = np.max(np.abs(cov2d_cuda - cov2d_fixed))
    t_diff = np.max(np.abs(t_cuda - t_webgl))

    if cov_diff > max_cov_diff:
        max_cov_diff = cov_diff
    if cov_diff_fixed > max_cov_diff_fixed:
        max_cov_diff_fixed = cov_diff_fixed
    if abs(coef_cuda - coef_fixed) > max_coef_diff:
        max_coef_diff = abs(coef_cuda - coef_fixed)
    if t_diff > max_t_diff:
        max_t_diff = t_diff

    if cov_diff_fixed > 0.01 or t_diff > 0.001:
        big_diffs.append((vi, cov_diff, cov_diff_fixed, t_diff))

    if vi < 3:
        print(f"── Vertex {vi} ──")
        print(f"  t_cuda:  {t_cuda}")
        print(f"  t_fixed: {t_fixed}")
        print(f"  cov2d_cuda:  {cov2d_cuda}")
        print(f"  cov2d_buggy: {cov2d_webgl}")
        print(f"  cov2d_fixed: {cov2d_fixed}")
        print(f"  coef_cuda={coef_cuda:.6f} coef_fixed={coef_fixed:.6f}")
        print(f"  buggy_diff={cov_diff:.2e}  fixed_diff={cov_diff_fixed:.2e}")
        print()

print(f"── Summary ({n_test} vertices) ──")
print(f"Max cov2d diff (CUDA vs buggy): {max_cov_diff:.2e}")
print(f"Max cov2d diff (CUDA vs FIXED): {max_cov_diff_fixed:.2e}")
print(f"Max coef diff (CUDA vs FIXED): {max_coef_diff:.2e}")
print(f"Max t diff: {max_t_diff:.2e}")
print(f"Vertices with big fixed_diff: {len(big_diffs)}/{n_test}")
if big_diffs:
    print("Big diffs (idx, buggy_diff, fixed_diff, t_diff):")
    for d in big_diffs[:10]:
        print(f"  v{d[0]}: buggy={d[1]:.2e} fixed={d[2]:.2e} t_diff={d[3]:.2e}")
