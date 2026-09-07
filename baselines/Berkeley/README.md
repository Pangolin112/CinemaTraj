# Cinematographic Camera Trajectory Generation in 3D Scenes

Implementation of the pipeline described in Wu (2025), UC Berkeley MS Thesis.

## Pipeline Overview

```
Input Images + NL Prompt
         │
         ▼
┌─────────────────────┐
│  Stage 1: COLMAP    │  Structure-from-Motion → camera poses + sparse point cloud
│  Scene Reconstruction│
└─────────┬───────────┘
         │
         ▼
┌─────────────────────┐
│  Stage 2: CLIP      │  Match prompt segments to training images via cosine similarity
│  Keyframe Selection  │  Constrained to training views for quality guarantee
└─────────┬───────────┘
         │
         ▼
┌─────────────────────┐
│  Stage 3: Trajectory │  Cubic spline + SLERP interpolation between keyframes
│  Generation + Refine │  NeRF density-based collision avoidance (6-ray casting)
└─────────┬───────────┘
         │
         ▼
  Camera Trajectory + Rendered Video
```

## Project Structure

```
trajectory_generation/
├── main.py                    # Pipeline orchestrator
├── scene_reconstruction.py    # COLMAP wrapper + binary file readers
├── keyframe_selection.py      # CLIP-based keyframe selection
├── trajectory_generation.py   # Trajectory interpolation + collision refinement
├── rendering.py               # NeRF rendering + Nerfstudio export
├── utils.py                   # Prompt parsing, I/O, visualization
└── README.md                  # This file
```

## Usage

### Full Pipeline
```bash
python main.py \
  --data_dir /path/to/scene/images \
  --colmap_model_dir /path/to/colmap/sparse/0 \
  --nerfstudio_model_dir /path/to/nerfstudio/outputs \
  --prompt "Overview of the table, zoom in on the mug, pan to the bear" \
  --output_dir ./output \
  --render
```

### Without NeRF (COLMAP + CLIP only)
```bash
python main.py \
  --data_dir /path/to/images \
  --colmap_model_dir /path/to/colmap/sparse/0 \
  --prompt "Close up of the mug, wide shot of the room" \
  --output_dir ./output
```

### Demo with Synthetic Data
```bash
python demo.py
```

### Export for Nerfstudio Rendering
After generating a trajectory, render with Nerfstudio CLI:
```bash
ns-render camera-path \
  --load-config /path/to/config.yml \
  --camera-path-filename output/nerfstudio_camera_path.json \
  --output-path output/video.mp4
```

## Key Components

### Keyframe Selection (Eq. 3.1)
```
similarity(I, T) = CLIP(I) · CLIP(T) / (||CLIP(I)|| · ||CLIP(T)||)
```
Images with highest cosine similarity to each prompt segment become keyframes.

### Collision Avoidance
From each trajectory point, 6 orthogonal rays (±X, ±Y, ±Z) are cast.
NeRF density serves as collision probability proxy. Points below the safety
threshold are shifted away from obstacles proportional to penetration depth.

### Trajectory Smoothing
- **Positions**: Cubic spline interpolation (clamped boundary conditions)
- **Rotations**: Spherical linear interpolation (SLERP)
- **Post-refinement**: Gaussian smoothing preserving keyframe positions
