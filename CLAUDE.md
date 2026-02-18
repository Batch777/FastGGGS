# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Geometry-Grounded Gaussian Splatting (GGGS) — a 3D reconstruction framework combining Gaussian Splatting with geometry grounding for surface reconstruction (mesh extraction) and novel view synthesis. Extends standard 3DGS with multi-view consistency constraints, spherical Gaussian appearance models, and mesh extraction via marching tetrahedra.

## Build & Installation

Requires Ubuntu 20.04+, CUDA 12.8+, Python 3.12, conda.

```bash
# Clone with submodules
git clone --recursive <repo-url>

# Create environment
conda create -n gggs python=3.12
conda activate gggs

# PyTorch (adjust CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# Python deps
pip install -r requirements.txt

# Submodules (order matters, all need --no-build-isolation)
pip install submodules/diff-gaussian-rasterization --no-build-isolation
pip install submodules/warp-patch-ncc --no-build-isolation
pip install submodules/simple-knn --no-build-isolation
pip install git+https://github.com/rahul-goel/fused-ssim/ --no-build-isolation

# For marching tetrahedra mesh extraction
conda install -y conda-forge::cgal
pip install submodules/tetra_triangulation --no-build-isolation
```

## Key Commands

```bash
# Training
python train.py -s <dataset_path> -m <output_dir> -r 2 --use_decoupled_appearance 3

# Mesh extraction (two methods)
python mesh_extract.py -m <output_dir>                          # TSDF (faster)
python mesh_extract_tetrahedra.py -m <output_dir>               # Tetrahedra (more accurate)
python mesh_extract_tetrahedra.py -m <output_dir> --move_cpu    # Lower GPU memory

# Rendering
python render.py -m <output_dir> --iteration 30000

# Evaluation
python metric.py -m <output_dir>                                # NVS metrics (PSNR/SSIM/LPIPS)
python evaluate_dtu_mesh.py -m <output_dir>                     # DTU geometry eval
python eval_tnt/run.py --dataset-dir <gt> --traj-path <sfm_log> --ply-path <mesh> --out-dir <out>

# Batch scripts for specific datasets
bash scripts/dtu.sh
bash scripts/tnt.sh
bash scripts/mip360.sh
```

## Architecture

### Core Pipeline
`train.py` → `GaussianModel` + `Scene` → `render()` → losses → optimize → `mesh_extract*.py`

### Key Modules

- **`train.py`** — Main training loop: initializes model, runs optimization (30k iters default), handles densification/pruning, saves checkpoints.

- **`gaussian_renderer/__init__.py`** (~9700 lines) — Core rendering engine. Key functions:
  - `render()` — Forward rendering via CUDA rasterizer, returns image/depth/normals/alpha
  - `integrate()` — Multi-view opacity integration for mesh extraction
  - `evaluate_sdf()` — SDF estimation from Gaussians (used by tetrahedra extraction)

- **`scene/gaussian_model.py`** — `GaussianModel` class holding all splat parameters (_xyz, _scaling, _rotation, _opacity, SH/SG coefficients), optimizer setup, densification logic, 3D filter computation, PLY I/O.

- **`scene/dataset_readers.py`** — Loads Colmap (binary sparse) or Blender (transforms JSON) datasets into `SceneInfo`.

- **`arguments/__init__.py`** — Three param groups: `ModelParams` (model config), `PipelineParams` (rendering flags), `OptimizationParams` (LRs, loss weights, densification schedule). Saved as `cfg_args` in output dir and reloaded by render/extract scripts.

- **`utils/loss_utils.py`** — Loss functions including L1, SSIM, appearance-aware L1, and `PatchMatch` class for multi-view NCC consistency loss.

### Appearance Models (`--use_decoupled_appearance`)
- `0` (NO): No appearance modeling
- `1` (GS): Per-view 4x4 exposure correction matrix
- `2` (GOF): Learned appearance network (`scene/appearance_network.py`)
- `3` (PGSR): Per-view embeddings + shared appearance network

### CUDA Submodules (`submodules/`)
- `diff-gaussian-rasterization` — Differentiable Gaussian rasterizer (core CUDA kernel)
- `simple-knn` — Fast CUDA KNN for Gaussian initialization
- `warp-patch-ncc` — Patch warping for multi-view NCC loss
- `tetra_triangulation` — Marching tetrahedra (requires CGAL)

### Key Loss Weights (in OptimizationParams)
- `lambda_dssim`: 0.2 (SSIM loss)
- `lambda_depth_normal`: 0.05 (depth-normal consistency)
- `lambda_multi_view_ncc`: 0.6 (multi-view patch NCC)
- `lambda_multi_view_geo`: 0.02 (geometric consistency)

### Output Directory Structure
```
<output_dir>/
├── cfg_args                    # Saved config
├── point_cloud/iteration_*/    # Gaussian PLY snapshots
├── cameras.json, multi_view.json
├── test/ours_*/renders/, gt/   # Rendered vs ground truth
├── recon.ply, recon_post.ply   # Extracted meshes
└── results.json                # PSNR/SSIM/LPIPS metrics
```

## Important Patterns

- Training config is persisted to `cfg_args` in the model directory; `render.py`, `mesh_extract.py`, and `metric.py` reload it automatically. Command-line args override saved config.
- Gaussians are densified from iter 500–15000 (every 100 iters) and pruned by opacity. Opacity is reset every 3000 iters.
- Multi-view neighbors are precomputed and cached in `multi_view.json` per scene.
- The `--eval` flag enables train/test split for novel view synthesis evaluation.
- Resolution scaling (`-r 2` = half resolution) affects camera generation, not the Gaussian model itself.
- Conda environment is based on `gspl`, do not use other environment.

## WebGL Viewer (`webviewer/`)

### Running
```bash
python webviewer/server.py [--port 8080] [--output_dir output]
```

### Key Rendering Pitfalls (CUDA vs WebGL)

1. **3D Covariance rotation convention**: CUDA uses GLM column-major matrices. `glm::mat3 R = glm::mat3(a,b,c, d,e,f, g,h,i)` fills columns, so the effective row-major matrix is R_std^T (transpose of standard quaternion rotation). When computing `Sigma = M^T * M` where `M = S * R`, the JS loader must use `M = S * R^T` (i.e. `M[i][j] = s_i * R[j*3+i]`) to match CUDA's `R * S^2 * R^T`. Using `S * R` directly gives `R^T * S^2 * R` which is wrong for anisotropic Gaussians.

2. **Frustum clamping**: CUDA uses FoV-dependent limits `limx = 1.3 * tan_fovx * t.z`. The WebGL shader must compute `tanFovx = viewport.x / (2.0 * focal.x)` and use it — not just `1.3 * t.z` (which assumes 90-degree FoV).

3. **DC color clamping**: Do NOT clamp DC SH color to [0,1] in the loader. CUDA only clamps the final result after `SH_DC + SH_rest + SG + 0.5` via `max(0)`.

4. **kernel_size opacity correction**: When adding `kernel_size` to the 2D covariance diagonal, compute `opacity *= sqrt(det_before / det_after)` to match CUDA.

5. **File format preference**: Always prefer `.ply` over `.splat` — PLY contains SH coefficients, SG lobes, and `filter_3D` data that `.splat` lacks.

### Files
- `shaders.js` — GLSL vertex/fragment shaders (EWA splatting, SH/SG evaluation)
- `splat-loader.js` — PLY/splat parser (covariance computation, SH/SG extraction)
- `viewer.js` — WebGL2 renderer (texture upload, sorting, draw loop)
- `camera-controls.js` — Orbit/direct camera with JSON camera loading
- `server.py` — Dev server with model browsing API
- `test_math.py` — Numerical comparison of CUDA vs WebGL 2D covariance math
