# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Geometry-Grounded Gaussian Splatting (GGGS) — a 3D reconstruction framework combining Gaussian Splatting with geometry grounding for surface reconstruction (mesh extraction) and novel view synthesis. Extends standard 3DGS with multi-view consistency constraints, spherical Gaussian appearance models, mesh extraction via marching tetrahedra, FastGS view-consistent densification/pruning, and Trace3D-style instance segmentation.

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

# Python deps (open3d, trimesh, scikit-image, opencv-python, plyfile, tqdm)
pip install -r requirements.txt

# Submodules (order matters, all need --no-build-isolation)
pip install submodules/diff-gaussian-rasterization --no-build-isolation
pip install submodules/warp-patch-ncc --no-build-isolation
pip install submodules/simple-knn --no-build-isolation
pip install git+https://github.com/rahul-goel/fused-ssim/ --no-build-isolation

# For marching tetrahedra mesh extraction
conda install -y conda-forge::cgal
pip install submodules/tetra_triangulation --no-build-isolation

# For Viser-based viewer (optional)
pip install viser scipy
```

## Key Commands

```bash
# Training (standard GGGS)
python train.py -s <dataset_path> -m <output_dir> -r 2 --use_decoupled_appearance 3

# Training with FastGS (view-consistent densification/pruning)
python train.py -s <dataset_path> -m <output_dir> --use_fastgs

# Training with instance segmentation (requires masks in <dataset>/masks/)
python train.py -s <dataset_path> -m <output_dir> --use_instance_seg

# Mesh extraction (two methods)
python mesh_extract.py -m <output_dir>                          # TSDF (faster)
python mesh_extract_tetrahedra.py -m <output_dir>               # Tetrahedra (more accurate)
python mesh_extract_tetrahedra.py -m <output_dir> --move_cpu    # Lower GPU memory

# Rendering
python render.py -m <output_dir> --iteration 30000

# Instance-colored rendering (Trace3D pipeline)
python render_instances.py -m <output_dir> --iteration 30000
python render_instances.py -m <output_dir> --skip_test --mask_dir masks

# Evaluation
python metric.py -m <output_dir>                                # NVS metrics (PSNR/SSIM/LPIPS)
python evaluate_dtu_mesh.py -m <output_dir>                     # DTU geometry eval
python eval_tnt/run.py --dataset-dir <gt> --traj-path <sfm_log> --ply-path <mesh> --out-dir <out>

# Instance mask generation
python scripts/generate_masks.py -s <dataset_path>                              # SAM3 text-prompt "object"
python scripts/generate_masks.py -s <dataset_path> --prompt "object,person"     # multiple prompts
python scripts/generate_masks.py -s <dataset_path> --mode grid --grid_size 32   # SAM1-style grid
python scripts/generate_masks.py -s <dataset_path> --save_vis                   # + colorized vis
python scripts/generate_masks_building.py -s <dataset_path>                     # SAM3 "building" prompt
python scripts/generate_masks_sam1.py -s <dataset_path>                         # SAM1 auto-segment
python scripts/generate_masks_semantic.py -s <dataset_path> --prompt "building,tree,road"  # multi-category semantic

# PLY to splat conversion (for web viewer)
python scripts/ply2splat.py <input.ply> [output.splat]

# Batch scripts for specific datasets
bash scripts/dtu.sh
bash scripts/tnt.sh
bash scripts/mip360.sh
```

## Architecture

### Core Pipeline
`train.py` → `GaussianModel` + `Scene` → `render()` → losses → optimize → `mesh_extract*.py`

### Key Modules

- **`train.py`** (~483 lines) — Main training loop: initializes model, runs optimization (30k iters default), handles densification/pruning (standard or FastGS), instance segmentation updates, saves checkpoints. Supports TensorBoard logging.

- **`gaussian_renderer/__init__.py`** (~340 lines) — Core rendering engine wrapping the CUDA rasterizer. Key functions:
  - `render()` — Forward rendering via CUDA rasterizer, returns image/depth/normals/alpha/metric_counts
  - `render_for_metric()` — Lightweight render for FastGS metric accumulation (no grad, no depth)
  - `integrate()` — Multi-view opacity integration for mesh extraction (from GOF)
  - `evaluate_sdf()` — SDF estimation from Gaussians (used by tetrahedra extraction)
  - `sample_depth()` — Depth sampling at 3D query points

- **`scene/gaussian_model.py`** (~1286 lines) — `GaussianModel` class holding all splat parameters (_xyz, _scaling, _rotation, _opacity, SH/SG coefficients), optimizer setup, densification logic, 3D filter computation, PLY I/O. Includes:
  - Standard GGGS densification (`densify_and_prune`)
  - FastGS VCD/VCP methods (`densify_and_prune_fastgs`, `final_prune_fastgs`)
  - FastGS dual-optimizer setup (`training_setup_fastgs`, `optimizer_step`, shoptimizer for SH rest)
  - Instance-aware densification (`densify_and_prune_instance`)
  - Instance fields: `_instance_id` (long), `_instance_entropy` (float), `_semantic_category` (long)

- **`scene/__init__.py`** (~181 lines) — `Scene` class: loads datasets, computes multi-view neighbors, loads instance masks, manages train/test camera splits.

- **`scene/dataset_readers.py`** (~320 lines) — Loads Colmap (binary sparse) or Blender (transforms JSON) datasets into `SceneInfo`.

- **`scene/cameras.py`** — `Camera` class with intrinsics/extrinsics, `MiniCam` for lightweight viewer cameras.

- **`scene/appearance_network.py`** — MLP-based appearance network for GOF/PGSR appearance models.

- **`arguments/__init__.py`** (~165 lines) — Three param groups: `ModelParams` (model config), `PipelineParams` (rendering flags), `OptimizationParams` (LRs, loss weights, densification/FastGS/instance schedules). Saved as `cfg_args` in output dir and reloaded by render/extract scripts.

- **`utils/loss_utils.py`** (~270 lines) — Loss functions: L1, SSIM, appearance-aware L1 (`L1_loss_appearance`), and `PatchMatch` class for multi-view NCC consistency loss.

- **`utils/fast_utils.py`** (~85 lines) — FastGS utilities: `sampling_cameras()`, `get_loss()`, `compute_photometric_loss()`, `compute_gaussian_score_fastgs()` (multi-view importance/pruning score computation).

- **`utils/instance_utils.py`** (~779 lines) — Instance segmentation algorithms: `build_instance_weight_matrix` (project Gaussians → read mask label), `majority_vote_merge` (Jaccard + Union-Find), `assign_labels`/`assign_instance_labels`/`assign_semantic_labels`, loss functions (`instance_seg_loss_ce`, `instance_seg_loss_contrastive`, `depth_edge_loss`).

- **`render_instances.py`** (~552 lines) — Render instance-colored images from a trained Gaussian model. Runs Trace3D 3D instance segmentation, re-colors Gaussians by instance ID, renders each view.

### Appearance Models (`--use_decoupled_appearance`)
- `0` (NO): No appearance modeling
- `1` (GS): Per-view 4x4 exposure correction matrix
- `2` (GOF): Learned appearance network (`scene/appearance_network.py`)
- `3` (PGSR): Per-view embeddings + shared appearance network

### FastGS (`--use_fastgs`)

View-Consistent Densification (VCD) and View-Consistent Pruning (VCP) strategy that uses multi-view photometric consistency to guide Gaussian management.

**How it works:**
1. For each densification step, sample 10 random cameras
2. For each camera: render → compute L1 loss map → threshold (`loss_thresh`) → re-render with `get_flag=True` to accumulate per-Gaussian metric counts
3. Only densify Gaussians that pass both gradient thresholds AND multi-view metric filter (`importance_score > 5`)
4. VCP: every 3000 iters between 15k–30k, prune Gaussians with high photometric inconsistency (`pruning_score > 0.9`)

**Dual optimizer (shoptimizer):** Separates SH rest coefficients into a second Adam optimizer with scheduled stepping:
- 0–15k iters: main optimizer every iter, shoptimizer every 16 iters
- 15k–20k: both every 32 iters
- 20k+: both every 64 iters

**Key args:** `use_fastgs`, `loss_thresh=0.1`, `grad_thresh=0.0002`, `grad_abs_thresh=0.0012`, `dense=0.001`, `highfeature_lr=0.005`, `lowfeature_lr=0.0025`, `mult=0.5`.

### Instance Segmentation (`--use_instance_seg`, Trace3D-style)

- **Mask generation scripts** (in `scripts/`):
  - `generate_masks.py` — SAM3-based, text-prompt or grid-point mode
  - `generate_masks_building.py` — SAM3 with "building" prompt, uses `ultralytics` SAM3SemanticPredictor
  - `generate_masks_sam1.py` — SAM1 auto-segment (no text prompts, purely visual boundaries)
  - `generate_masks_semantic.py` — Multi-category semantic masks as uint16 PNG: `(category_id << 8) | instance_id`
- **`utils/instance_utils.py`** — Core algorithms: build weight matrix, majority vote merge, assign labels, loss functions
- Instance fields on `GaussianModel`: `_instance_id` (long), `_instance_entropy` (float), `_semantic_category` (long) — non-optimizable, synced during densify/prune
- Key args: `instance_seg_from_iter=7000`, `instance_update_interval=500`, `lambda_seg=0.1`, `lambda_depth_edge=0.05`, `tau_affinity=0.3`, `seg_loss_mode="ce"` (or "contrastive"), `instance_mask_dir="masks"`

### CUDA Submodules (`submodules/`)
- `diff-gaussian-rasterization` — Differentiable Gaussian rasterizer (core CUDA kernel, includes GLM)
- `simple-knn` — Fast CUDA KNN for Gaussian initialization
- `warp-patch-ncc` — Patch warping for multi-view NCC loss
- `tetra_triangulation` — Marching tetrahedra (requires CGAL)

Note: `sam3` (SAM 3) is referenced by mask generation scripts but is gitignored and not a registered git submodule. Checkpoint expected at `submodules/sam3/weights/sam3.pt`.

### Utility Modules (`utils/`)
- `camera_utils.py` — Camera list construction from `CameraInfo`, JSON serialization
- `colmap_read_model.py` — Binary Colmap model reader (cameras, images, points3D)
- `colmap_wrapper.py` — Shell wrapper to invoke Colmap CLI
- `general_utils.py` — Misc: `inverse_sigmoid`, `build_rotation`, `build_scaling_rotation`, `get_expon_lr_func`, `safe_state`, `strip_symmetric`
- `graphics_utils.py` — `BasicPointCloud`, projection matrices, focal↔FoV conversions, `depth_to_normal`
- `image_utils.py` — PSNR computation
- `pose_utils.py` — Camera pose utilities
- `render_utils.py` — Rendering helpers (from 2DGS), requires `mediapy`
- `sh_utils.py` — Spherical harmonics evaluation, `RGB2SH` conversion
- `system_utils.py` — `mkdir_p`, `searchForMaxIteration`
- `tetmesh.py` — Tetrahedra mesh utilities for marching tetrahedra
- `vis_utils.py` — Depth/colormap visualization (from GOF/nerfstudio)

### Evaluation Modules
- `dtu_eval/eval.py` — DTU dataset mesh evaluation (chamfer distance)
- `eval_tnt/` — Tanks and Temples evaluation pipeline: `run.py` (main), `evaluation.py`, `cull_mesh.py`, `compute_bbox_for_mesh.py`, `registration.py`, `config.py`, `trajectory_io.py`
- `lpipsPyTorch/` — LPIPS perceptual loss wrapper

### Key Loss Weights (in OptimizationParams)
- `lambda_dssim`: 0.2 (SSIM loss)
- `lambda_depth_normal`: 0.05 (depth-normal consistency, active from `regularization_from_iter=7000`)
- `lambda_multi_view_ncc`: 0.6 (multi-view patch NCC)
- `lambda_multi_view_geo`: 0.02 (geometric consistency)
- `lambda_seg`: 0.1 (instance segmentation CE/contrastive loss)
- `lambda_depth_edge`: 0.05 (depth edge loss at instance boundaries)

### Output Directory Structure
```
<output_dir>/
├── cfg_args                    # Saved config (Python Namespace repr)
├── input.ply                   # Copy of initial point cloud
├── point_cloud/iteration_*/    # Gaussian PLY snapshots
│   └── point_cloud.ply
├── cameras.json                # Camera parameters
├── multi_view.json             # Precomputed multi-view neighbors
├── chkpnt*.pth                 # Training checkpoints (iter 15000 default)
├── chkpnt*.txt                 # Test PSNR at checkpoint iterations
├── debug/                      # Debug outputs during training
├── train/ours_*/renders/, gt/  # Train set rendered vs ground truth
├── test/ours_*/renders/, gt/   # Test set rendered vs ground truth
├── recon.ply, recon_post.ply   # Extracted meshes
├── results.json                # PSNR/SSIM/LPIPS metrics
└── per_view.json               # Per-image metrics
```

## Important Patterns

- Training config is persisted to `cfg_args` in the model directory; `render.py`, `mesh_extract.py`, and `metric.py` reload it automatically via `get_combined_args()`. Command-line args override saved config.
- Gaussians are densified from iter 500–15000 (every 100 iters) and pruned by opacity. Opacity is reset every 3000 iters.
- Multi-view neighbors are precomputed and cached in `multi_view.json` per scene (in `Scene.__init__`).
- The `--eval` flag enables train/test split for novel view synthesis evaluation.
- Resolution scaling (`-r 2` = half resolution) affects camera generation, not the Gaussian model itself.
- Conda environment is based on `gspl`, do not use other environment.
- SH degree is incremented every 1000 iterations up to `max_sh_degree`. SG degree is unlocked at iteration 100.
- Regularization losses (depth-normal, multi-view NCC/geo) kick in at `regularization_from_iter` (default 7000).
- Instance labels are recomputed every `instance_update_interval` (500) iterations starting from `instance_seg_from_iter` (7000).
- The `GaussianModel` has a `shoptimizer` attribute (secondary optimizer for SH rest) only when `use_fastgs=True`. Code that manipulates optimizers must handle both paths.
- The diff-gaussian-rasterization CUDA kernel supports `get_flag` + `metric_map` parameters for FastGS metric accumulation during rasterization.

## Scripts (`scripts/`)

- `generate_masks.py` — Primary SAM3 mask generator (text-prompt or grid mode)
- `generate_masks_building.py` — SAM3 with "building" text prompt, uses Ultralytics SAM3SemanticPredictor
- `generate_masks_sam1.py` — SAM1 automatic segmentation (no text prompts)
- `generate_masks_semantic.py` — Multi-category semantic masks (uint16 encoding: `category << 8 | instance`)
- `ply2splat.py` — Convert PLY to `.splat` v2 binary format for web viewer (precomputes all activations, float16 SoA layout)
- `dtu.sh`, `tnt.sh`, `mip360.sh` — Batch training/evaluation for DTU, Tanks and Temples, Mip-NeRF 360 datasets

## WebGL Viewer (`webviewer/`)

### Running
```bash
# Client-side WebGL viewer (static files served by Python)
python webviewer/server.py [--port 8080] [--output_dir output]

# Server-side CUDA rendering via Viser (pixel-perfect match with render.py)
python webviewer/viser_viewer.py --output_dir output [--port 8080]
```

### Key Rendering Pitfalls (CUDA vs WebGL)

1. **3D Covariance rotation convention**: CUDA uses GLM column-major matrices. `glm::mat3 R = glm::mat3(a,b,c, d,e,f, g,h,i)` fills columns, so the effective row-major matrix is R_std^T (transpose of standard quaternion rotation). When computing `Sigma = M^T * M` where `M = S * R`, the JS loader must use `M = S * R^T` (i.e. `M[i][j] = s_i * R[j*3+i]`) to match CUDA's `R * S^2 * R^T`. Using `S * R` directly gives `R^T * S^2 * R` which is wrong for anisotropic Gaussians.

2. **Frustum clamping**: CUDA uses FoV-dependent limits `limx = 1.3 * tan_fovx * t.z`. The WebGL shader must compute `tanFovx = viewport.x / (2.0 * focal.x)` and use it — not just `1.3 * t.z` (which assumes 90-degree FoV).

3. **DC color clamping**: Do NOT clamp DC SH color to [0,1] in the loader. CUDA only clamps the final result after `SH_DC + SH_rest + SG + 0.5` via `max(0)`.

4. **kernel_size opacity correction**: When adding `kernel_size` to the 2D covariance diagonal, compute `opacity *= sqrt(det_before / det_after)` to match CUDA.

5. **File format preference**: Always prefer `.ply` over `.splat` — PLY contains SH coefficients, SG lobes, and `filter_3D` data that `.splat` lacks.

### Files
- `index.html` — Main viewer page
- `shaders.js` — GLSL vertex/fragment shaders (EWA splatting, SH/SG evaluation)
- `splat-loader.js` — PLY/splat parser (covariance computation, SH/SG extraction)
- `viewer.js` (~1059 lines) — WebGL2 renderer (texture upload, sorting, draw loop)
- `camera-controls.js` — Orbit/direct camera with JSON camera loading
- `camera-animator.js` — Camera animation: prev/next stepping, smooth Catmull-Rom flythrough
- `sort-worker.js` — Web Worker for Gaussian depth sorting
- `server.py` — Dev server with model browsing API
- `viser_viewer.py` (~344 lines) — Viser-based viewer with server-side CUDA rendering (pixel-perfect)
- `export_splat.py` — Convert GGGS PLY to compact `.splat` binary for web viewer
- `render_reference.py` — Render CUDA reference image + camera JSON for WebGL comparison
- `test_math.py` — Numerical comparison of CUDA vs WebGL 2D covariance math

## Other Directories

- **`SIBR_viewers/`** — SIBR interactive viewer (C++ OpenGL, CMake build). Not used by the Python pipeline but provides real-time interactive viewing of Gaussian splat models.
- **`assets/`** — Contains `teaser.jpg` (project teaser image for README).
