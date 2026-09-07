"""
Trajectory Optimization & Combination — WITH OCCLUSION COST
============================================================

Key addition over the previous version:
  - **Occlusion cost**: penalizes camera poses where the target object is
    occluded by scene geometry.  Uses differentiable ray-marching through
    the precomputed SDF grid so gradients flow back to trajectory parameters.

The occlusion cost works as follows:
  1. For each sampled camera position, compute the ray direction toward the
     target object center (the object the camera is supposed to look at).
  2. March along the ray in fixed steps, querying the differentiable SDF grid.
  3. If any sample along the ray has SDF < threshold (i.e. hits geometry)
     *before* reaching the target, accumulate a penalty.
  4. The penalty is differentiable w.r.t. camera position because both the
     ray origin and the SDF query (trilinear interpolation) are differentiable.

Integration points (marked with # >>> OCCLUSION):
  - ParametricOptimizationConfig: new fields occlusion_weight, occlusion_*
  - ParametricTrajectoryOptimizer._occlusion_cost(): new method
  - ParametricTrajectoryOptimizer.optimize(): includes occlusion in total cost
  - optimize_trajectory_result(): passes target_center to optimizer

UPDATED: Supports new scene graph JSON format with string object IDs and
  OBB-based bounding boxes (quaternion rotation). Mesh is loaded from PLY
  (built by build_obb_mesh.py) instead of USD.
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import numpy as np
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Union
from scipy.spatial.transform import Rotation, Slerp
import trimesh
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.anchor_selector.anchor_selector import CameraAnchor
from src.parametric_trajectory_builder.trajectory_executor import TrajectoryExecutor, TrajectorySegment
from src.utils.room_connectivity import (
    build_room_graph,
    get_cross_room_info,
    RoomGraph,
)
import time

# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class CameraPose:
    position: np.ndarray
    rotation: np.ndarray
    timestamp: float = 0.0


@dataclass
class CameraIntrinsics:
    width: float
    height: float
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass
class CameraTrajectory:
    poses: List[CameraPose] = field(default_factory=list)
    timestamps: np.ndarray = field(default_factory=lambda: np.array([]))
    positions: np.ndarray = field(default_factory=lambda: np.array([]))
    rotations: np.ndarray = field(default_factory=lambda: np.array([]))
    intrinsics: Optional[CameraIntrinsics] = None
    focal_multipliers: Optional[np.ndarray] = None

    def __len__(self):
        return len(self.poses)

    def get_c2w_matrices(self) -> np.ndarray:
        n = len(self.poses)
        if n == 0:
            return np.zeros((0, 4, 4))
        matrices = np.zeros((n, 4, 4))
        matrices[:, :3, :3] = self.rotations
        matrices[:, :3, 3] = self.positions
        matrices[:, 3, 3] = 1.0
        return matrices


@dataclass
class SegmentFrameMapping:
    segment_index: int
    movement_type: str
    frame_start: int
    frame_end: int
    is_transition: bool
    object_label: str = ""


# =============================================================================
# Optimization Config  — >>> OCCLUSION fields added
# =============================================================================

@dataclass
class ParametricOptimizationConfig:
    """Configuration for trajectory optimization."""
    # --- Existing costs ---
    collision_weight: float = 5.0
    mesh_sdf_weight: float = 5.0
    smoothness_weight: float = 0.5
    parameter_regularization: float = 0.1
    boundary_weight: float = 5.0
    density_threshold: float = 0.2
    safety_margin: float = 0.2
    learning_rate: float = 1.0
    n_iterations: int = 1000
    convergence_threshold: float = 1e-10
    n_samples: int = 100
    k_neighbors: int = 32
    density_scale: float = 1.0

    # >>> OCCLUSION: new fields
    occlusion_weight: float = 3.0       # weight of occlusion cost in total loss
    occlusion_n_steps: int = 32         # number of ray-march steps per ray
    occlusion_margin: float = 0.05      # SDF threshold below which geometry is "hit"
    occlusion_near_skip: float = 0.1    # skip this fraction of ray near camera (avoid self-hit)
    occlusion_target_shrink: float = 0.3  # stop ray at (1 - shrink) of total distance to avoid hitting target object itself


# =============================================================================
# Differentiable Mesh SDF  (unchanged)
# =============================================================================

class DifferentiableMeshSDF(nn.Module):
    def __init__(self, mesh, resolution=64, padding=0.5, device='cuda',
                 chunk_size=10000, cache_dir=None, scene_name=None):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.mesh = mesh
        self.resolution = resolution

        mesh_min = mesh.bounds[0] - padding
        mesh_max = mesh.bounds[1] + padding
        self.register_buffer('grid_min', torch.tensor(mesh_min, dtype=torch.float32))
        self.register_buffer('grid_max', torch.tensor(mesh_max, dtype=torch.float32))

        cache_path = None
        if cache_dir and scene_name:
            cache_path = Path(cache_dir) / f"{scene_name}_sdf_res{resolution}.npz"
            if cache_path.exists():
                print(f"Loading cached SDF from {cache_path}")
                cached = np.load(cache_path)
                sdf_tensor = torch.from_numpy(cached['sdf_grid']).float().unsqueeze(0).unsqueeze(0)
                self.register_buffer('sdf_grid', sdf_tensor)
                if 'grid_min' in cached and 'grid_max' in cached:
                    self.grid_min.copy_(torch.tensor(cached['grid_min'], dtype=torch.float32))
                    self.grid_max.copy_(torch.tensor(cached['grid_max'], dtype=torch.float32))
                self.to(self.device)
                return

        print(f"Building SDF grid ({resolution}³)...")
        x = np.linspace(mesh_min[0], mesh_max[0], resolution)
        y = np.linspace(mesh_min[1], mesh_max[1], resolution)
        z = np.linspace(mesh_min[2], mesh_max[2], resolution)
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        grid_points = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3).astype(np.float32)

        sdf_values = self._compute_sdf_chunked(mesh, grid_points, chunk_size)
        sdf_grid = sdf_values.reshape(resolution, resolution, resolution)

        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_path, sdf_grid=sdf_grid, grid_min=mesh_min,
                                grid_max=mesh_max, resolution=resolution)
            print(f"  Cached SDF to {cache_path}")

        sdf_tensor = torch.from_numpy(sdf_grid).float().unsqueeze(0).unsqueeze(0)
        self.register_buffer('sdf_grid', sdf_tensor)
        print(f"  SDF range: [{sdf_values.min():.3f}, {sdf_values.max():.3f}]")
        self.to(self.device)

    @classmethod
    def load_from_cache(cls, cache_path, device='cuda'):
        cached = np.load(cache_path)
        instance = cls.__new__(cls)
        nn.Module.__init__(instance)
        instance.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        instance.mesh = None
        instance.resolution = int(cached['resolution'])
        instance.register_buffer('grid_min', torch.tensor(cached['grid_min'], dtype=torch.float32))
        instance.register_buffer('grid_max', torch.tensor(cached['grid_max'], dtype=torch.float32))
        instance.register_buffer('sdf_grid', torch.from_numpy(cached['sdf_grid']).float().unsqueeze(0).unsqueeze(0))
        instance.to(instance.device)
        return instance

    def _compute_sdf_chunked(self, mesh, points, chunk_size):
        n = len(points)
        sdf = np.zeros(n, dtype=np.float32)
        for i in range(0, n, chunk_size):
            end = min(i + chunk_size, n)
            chunk = points[i:end]
            _, dists, _ = trimesh.proximity.closest_point(mesh, chunk)
            try:
                inside = mesh.contains(chunk)
                signs = np.where(inside, 1.0, -1.0).astype(np.float32)
            except:
                signs = np.ones(len(chunk), dtype=np.float32)
            sdf[i:end] = signs * dists.astype(np.float32)
        return sdf

    def query_sdf(self, points):
        points = points.to(self.device)
        if points.dim() == 1:
            points = points.unsqueeze(0)
        N = points.shape[0]
        normalized = 2.0 * (points - self.grid_min) / (self.grid_max - self.grid_min + 1e-8) - 1.0
        grid_coords = normalized.view(1, 1, 1, N, 3)[..., [2, 1, 0]]
        sampled = F.grid_sample(self.sdf_grid, grid_coords, mode='bilinear',
                                padding_mode='border', align_corners=True)
        return sampled.view(N)

    def query_collision(self, points, margin=0.1):
        return self.query_sdf(points) < margin


# =============================================================================
# Gaussian Density Field  (unchanged)
# =============================================================================

def load_ply_3dgs(ply_path):
    from plyfile import PlyData
    plydata = PlyData.read(ply_path)
    v = plydata['vertex']
    positions = np.stack([v['x'], v['y'], v['z']], axis=-1).astype(np.float32)
    try:
        scales = np.exp(np.stack([v['scale_0'], v['scale_1'], v['scale_2']], axis=-1).astype(np.float32))
    except (ValueError, KeyError):
        scales = np.ones((len(positions), 3), dtype=np.float32) * 0.01
    try:
        opacities = 1.0 / (1.0 + np.exp(-v['opacity'].astype(np.float32)))
    except (ValueError, KeyError):
        opacities = np.ones(len(positions), dtype=np.float32)
    print(f"Loaded 3DGS: {len(positions)} Gaussians")
    return positions, scales, opacities


class GaussianDensityField(nn.Module):
    def __init__(self, positions, scales, opacities, config, device='cuda',
                 mesh=None, sdf_resolution=64, sdf_cache_dir=None, scene_name=None):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.config = config
        self.register_buffer('gs_positions', torch.from_numpy(positions).float())
        self.register_buffer('gs_scales', torch.from_numpy(scales).float())
        self.register_buffer('gs_opacities', torch.from_numpy(opacities).float())
        self.scene_min = positions.min(axis=0)
        self.scene_max = positions.max(axis=0)
        self.diff_mesh_sdf = None
        if mesh is not None:
            self.diff_mesh_sdf = DifferentiableMeshSDF(
                mesh, resolution=sdf_resolution, padding=0.5, device=device,
                cache_dir=sdf_cache_dir, scene_name=scene_name)
            self.scene_min = self.diff_mesh_sdf.grid_min.cpu().numpy()
            self.scene_max = self.diff_mesh_sdf.grid_max.cpu().numpy()
        self.to(self.device)

    def query_density(self, points, max_gaussians=500_000):
        points = points.to(self.device)
        if points.dim() == 1:
            points = points.unsqueeze(0)

        gs_pos = self.gs_positions
        gs_scales = self.gs_scales
        gs_opacities = self.gs_opacities

        # Subsample Gaussians if too many
        N_gs = gs_pos.shape[0]
        if N_gs > max_gaussians:
            indices = torch.randperm(N_gs, device=self.device)[:max_gaussians]
            gs_pos = gs_pos[indices]
            gs_scales = gs_scales[indices]
            gs_opacities = gs_opacities[indices]

        N = points.shape[0]
        k = min(self.config.k_neighbors, gs_pos.shape[0])
        # Budget ~2GB per chunk
        chunk_size = max(1, min(512, int(2e9 / (gs_pos.shape[0] * 3 * 4))))

        result = torch.zeros(N, device=self.device)

        for i in range(0, N, chunk_size):
            end = min(i + chunk_size, N)
            pts_chunk = points[i:end]
            diffs = pts_chunk.unsqueeze(1) - gs_pos.unsqueeze(0)
            dists_sq = (diffs ** 2).sum(dim=-1)
            _, idx = torch.topk(dists_sq, k=k, dim=-1, largest=False)
            neighbor_scales = gs_scales[idx]
            neighbor_opacities = gs_opacities[idx]
            diffs_knn = pts_chunk.unsqueeze(1) - gs_pos[idx]
            exponents = -0.5 * ((diffs_knn / (neighbor_scales + 1e-8)) ** 2).sum(dim=-1)
            result[i:end] = (neighbor_opacities * torch.exp(exponents)).sum(dim=-1) * self.config.density_scale

        return result

    def query_mesh_sdf(self, points):
        if self.diff_mesh_sdf is None:
            return torch.zeros(points.shape[0], device=self.device)
        return self.diff_mesh_sdf.query_sdf(points)

    def query_mesh_collision_count(self, points, margin=0.1):
        if self.diff_mesh_sdf is None:
            return 0
        with torch.no_grad():
            return int(self.diff_mesh_sdf.query_collision(points, margin).sum().item())


# =============================================================================
# Differentiable Trajectory Wrapper  (unchanged)
# =============================================================================

class DifferentiableTrajectory(nn.Module):
    def __init__(self, trajectory_class_name, fixed_params, free_params,
                 free_param_names, param_bounds, optimize_only=None, device='cuda'):
        super().__init__()
        self.trajectory_class_name = trajectory_class_name
        self.fixed_params = fixed_params
        self.free_param_names = free_param_names
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        all_params = torch.from_numpy(free_params.astype(np.float64)).float()
        lower = torch.from_numpy(param_bounds[0].astype(np.float64)).float()
        upper = torch.from_numpy(param_bounds[1].astype(np.float64)).float()

        self.register_buffer('all_original_params', all_params.clone().to(self.device))

        if optimize_only is not None:
            self.opt_indices = [i for i, n in enumerate(free_param_names) if n in optimize_only]
        else:
            self.opt_indices = list(range(len(free_param_names)))
        self.frozen_indices = [i for i in range(len(free_param_names)) if i not in self.opt_indices]

        if self.opt_indices:
            opt_vals = all_params[self.opt_indices].clone()
            self.opt_params = nn.Parameter(opt_vals.to(self.device))
            self.register_buffer('opt_lower', lower[self.opt_indices].to(self.device))
            self.register_buffer('opt_upper', upper[self.opt_indices].to(self.device))
            self.register_buffer('opt_original', all_params[self.opt_indices].clone().to(self.device))
        else:
            self.opt_params = nn.Parameter(torch.empty(0, device=self.device))
            self.register_buffer('opt_lower', torch.empty(0, device=self.device))
            self.register_buffer('opt_upper', torch.empty(0, device=self.device))
            self.register_buffer('opt_original', torch.empty(0, device=self.device))

        if self.frozen_indices:
            self.register_buffer('frozen_values', all_params[self.frozen_indices].clone().to(self.device))
        else:
            self.register_buffer('frozen_values', torch.empty(0, device=self.device))
        self.to(self.device)

    def get_clamped_params(self):
        full = torch.empty(len(self.free_param_names), device=self.device)
        if self.opt_indices:
            clamped_opt = torch.clamp(self.opt_params, self.opt_lower, self.opt_upper)
            for i, idx in enumerate(self.opt_indices):
                full[idx] = clamped_opt[i]
        for i, idx in enumerate(self.frozen_indices):
            full[idx] = self.frozen_values[i]
        return full

    def get_all_final_params(self):
        with torch.no_grad():
            return self.get_clamped_params().cpu().numpy()

    def _yup_to_zup(self, pos):
        return torch.stack([pos[:, 0], pos[:, 2], pos[:, 1]], dim=-1)

    def evaluate_zup(self, t):
        return self._yup_to_zup(self.evaluate_yup(t))

    def evaluate_yup(self, t):
        params = self.get_clamped_params()
        if 'CircularOrbit' in self.trajectory_class_name:
            return self._eval_circular_orbit(t, params)
        elif 'CraneShot' in self.trajectory_class_name:
            return self._eval_crane_shot(t, params)
        elif 'DollyMove' in self.trajectory_class_name:
            return self._eval_dolly_move(t, params)
        elif 'StationaryPan' in self.trajectory_class_name:
            return self._eval_stationary_pan(t, params)
        elif 'StationaryTilt' in self.trajectory_class_name:
            return self._eval_stationary_tilt(t, params)
        elif 'ZoomLens' in self.trajectory_class_name:
            return self._eval_zoom_lens(t, params)
        elif 'TransitionalArc' in self.trajectory_class_name:
            return self._eval_arc(t, params)
        else:
            raise ValueError(f"Unknown trajectory: {self.trajectory_class_name}")

    def _get_param(self, name, params):
        if name in self.free_param_names:
            return params[self.free_param_names.index(name)]
        val = self.fixed_params[name]
        if isinstance(val, (list, tuple, np.ndarray)):
            return torch.tensor(val, device=self.device, dtype=torch.float32)
        return torch.tensor(val, device=self.device, dtype=torch.float32)

    def _eval_circular_orbit(self, t, params):
        radius = self._get_param('radius', params)
        height = self._get_param('height', params)
        start_angle = self._get_param('start_angle', params)
        end_angle = self._get_param('end_angle', params)
        center = self._get_param('object_center', params)
        angles = torch.deg2rad(start_angle + t * (end_angle - start_angle))
        n = len(t)
        pos = torch.zeros(n, 3, device=self.device)
        pos[:, 0] = center[0] + radius * torch.cos(angles)
        pos[:, 2] = center[2] + radius * torch.sin(angles)
        pos[:, 1] = center[1] + height
        return pos

    def _eval_dolly_move(self, t, params):
        start_radius = self._get_param('start_radius', params)
        end_radius = self._get_param('end_radius', params)
        height = self._get_param('height', params)
        center = self._get_param('object_center', params)
        approach_angle_deg = self._get_param('approach_angle', params)

        n = len(t)
        t_eased = torch.where(t < 0.5, 2 * t * t, 1 - 0.5 * (2 - 2 * t) ** 2)

        dist = start_radius + t_eased * (end_radius - start_radius)

        angle_rad = torch.deg2rad(approach_angle_deg)
        dir_x = torch.cos(angle_rad)
        dir_z = torch.sin(angle_rad)

        raw_x = start_radius * dir_x
        raw_y = height
        raw_z = start_radius * dir_z
        ray_len = torch.sqrt(raw_x ** 2 + raw_y ** 2 + raw_z ** 2 + 1e-16)

        unit_x = raw_x / ray_len
        unit_y = raw_y / ray_len
        unit_z = raw_z / ray_len

        pos = torch.zeros(n, 3, device=self.device)
        pos[:, 0] = center[0] + dist * unit_x
        pos[:, 1] = center[1] + dist * unit_y
        pos[:, 2] = center[2] + dist * unit_z

        return pos

    def _eval_stationary_pan(self, t, params):
        camera_pos = self._get_param('camera_position', params)
        return camera_pos.unsqueeze(0).expand(len(t), -1).clone()

    def _eval_stationary_tilt(self, t, params):
        camera_pos = self._get_param('camera_position', params)
        return camera_pos.unsqueeze(0).expand(len(t), -1).clone()

    def _eval_zoom_lens(self, t, params):
        camera_pos = self._get_param('camera_position', params)
        return camera_pos.unsqueeze(0).expand(len(t), -1).clone()

    def _eval_crane_shot(self, t, params):
        radius = self._get_param('radius', params)
        start_elev = self._get_param('start_elevation', params)
        end_elev = self._get_param('end_elevation', params)
        center = self._get_param('object_center', params)
        approach_val = self.fixed_params.get('approach_angle', 0.0)
        approach_angle = torch.tensor(float(approach_val), device=self.device)
        n = len(t)
        t_eased = torch.where(t < 0.5, 2 * t * t, 1 - 0.5 * (2 - 2 * t) ** 2)
        elevation = start_elev + t_eased * (end_elev - start_elev)
        elevation_rad = torch.deg2rad(elevation)
        approach_rad = torch.deg2rad(approach_angle)
        dir_x = torch.cos(approach_rad)
        dir_z = torch.sin(approach_rad)
        h_dist = radius * torch.cos(elevation_rad)
        v_dist = radius * torch.sin(elevation_rad)
        pos = torch.zeros(n, 3, device=self.device)
        pos[:, 0] = center[0] + h_dist * dir_x
        pos[:, 2] = center[2] + h_dist * dir_z
        pos[:, 1] = center[1] + v_dist
        return pos

    def _eval_arc(self, t, params):
        arc_angle_deg = self._get_param('arc_angle', params)
        start_pos = self._get_param('start_position', params)
        end_pos = self._get_param('end_position', params)
        n = len(t)
        t_eased = torch.where(t < 0.5, 2 * t * t, 1 - 0.5 * (2 - 2 * t) ** 2)
        pos = (1 - t_eased).unsqueeze(-1) * start_pos + t_eased.unsqueeze(-1) * end_pos
        chord_xz = torch.stack([end_pos[0] - start_pos[0], end_pos[2] - start_pos[2]])
        chord_len = torch.norm(chord_xz) + 1e-8
        chord_dir = chord_xz / chord_len
        perp_dir = torch.stack([-chord_dir[1], chord_dir[0]])
        amplitude = torch.tan(torch.deg2rad(arc_angle_deg)) * chord_len * 0.5
        bulge = amplitude * torch.sin(np.pi * t_eased)
        pos = pos.clone()
        pos[:, 0] = pos[:, 0] + bulge * perp_dir[0]
        pos[:, 2] = pos[:, 2] + bulge * perp_dir[1]
        return pos


# =============================================================================
# Trajectory Optimizer  — >>> OCCLUSION cost added
# =============================================================================

class ParametricTrajectoryOptimizer:
    def __init__(self, density_field, config=None, device='cuda'):
        self.density_field = density_field
        self.config = config or ParametricOptimizationConfig()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.scene_min = torch.from_numpy(density_field.scene_min).float().to(self.device)
        self.scene_max = torch.from_numpy(density_field.scene_max).float().to(self.device)

        self._target_center_zup: Optional[torch.Tensor] = None
        self._target_obb_center: Optional[torch.Tensor] = None
        self._target_obb_axes: Optional[torch.Tensor] = None
        self._target_obb_half_extents: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Existing costs (unchanged)
    # ------------------------------------------------------------------

    def _collision_cost(self, pos_zup):
        densities = self.density_field.query_density(pos_zup)
        return F.softplus(densities - self.config.density_threshold, beta=10.0).mean() * self.config.collision_weight

    def _mesh_sdf_cost(self, pos_zup):
        if self.density_field.diff_mesh_sdf is None:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        sdf = self.density_field.query_mesh_sdf(pos_zup)
        violation = F.relu(self.config.safety_margin - sdf)
        cost = (violation + violation ** 2).mean()
        return cost * self.config.mesh_sdf_weight

    def _smoothness_cost(self, pos):
        vel = pos[1:] - pos[:-1]
        acc = vel[1:] - vel[:-1]
        jerk = acc[1:] - acc[:-1]
        return (jerk ** 2).sum(dim=-1).mean() * self.config.smoothness_weight

    def _regularization_cost(self, diff_traj):
        if len(diff_traj.opt_indices) == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        clamped_opt = torch.clamp(diff_traj.opt_params, diff_traj.opt_lower, diff_traj.opt_upper)
        rng = diff_traj.opt_upper - diff_traj.opt_lower + 1e-6
        return (((clamped_opt - diff_traj.opt_original) / rng) ** 2).mean() * self.config.parameter_regularization

    def _boundary_cost(self, pos_zup):
        margin = self.config.safety_margin
        lower_v = self.scene_min + margin - pos_zup
        upper_v = pos_zup - self.scene_max + margin
        lower_c = torch.where(lower_v > 0, lower_v ** 2 + torch.exp(lower_v) - 1, torch.zeros_like(lower_v))
        upper_c = torch.where(upper_v > 0, upper_v ** 2 + torch.exp(upper_v) - 1, torch.zeros_like(upper_v))
        return (lower_c.sum(-1) + upper_c.sum(-1)).mean() * self.config.boundary_weight

    # ------------------------------------------------------------------
    # >>> OCCLUSION cost (unchanged)
    # ------------------------------------------------------------------

    def _point_obb_distance(self, points: torch.Tensor) -> torch.Tensor:
        obb_center = self._target_obb_center if self._target_obb_center is not None else self._target_center_zup
        obb_axes = self._target_obb_axes
        half_ext = self._target_obb_half_extents
        local = (points - obb_center.unsqueeze(0)) @ obb_axes
        q = torch.abs(local) - half_ext.unsqueeze(0)
        outside_dist = torch.norm(torch.clamp(q, min=0.0), dim=-1)
        inside_dist = torch.clamp(q.max(dim=-1).values, max=0.0)
        return outside_dist + inside_dist

    def _occlusion_cost(self, pos_zup: torch.Tensor) -> torch.Tensor:
        if self._target_center_zup is None:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        # Need either SDF or density field for ray marching
        has_sdf = self.density_field.diff_mesh_sdf is not None
        if not has_sdf and self.density_field.gs_positions is None:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        target = self._target_center_zup
        N = pos_zup.shape[0]
        n_steps = self.config.occlusion_n_steps
        margin = self.config.occlusion_margin
        near_skip = self.config.occlusion_near_skip

        ray_vec = target.unsqueeze(0) - pos_zup

        if has_sdf:
            t_far = self._ray_obb_t_enter(pos_zup, ray_vec)
        else:
            t_far = self._ray_obb_t_enter(pos_zup, ray_vec)
        t_far = t_far.clamp(min=near_skip + 0.05, max=1.0)

        t_uniform = torch.linspace(0, 1, n_steps, device=self.device)
        t_vals = near_skip + t_uniform.unsqueeze(0) * (t_far - near_skip)
        t_expand = t_vals.unsqueeze(-1)
        ray_points = (
            pos_zup.unsqueeze(1) * (1.0 - t_expand)
            + target.unsqueeze(0).unsqueeze(0) * t_expand
        )

        ray_points_flat = ray_points.reshape(N * n_steps, 3)

        if has_sdf:
            # SDF-based: negative SDF = inside geometry
            sdf_vals = self.density_field.diff_mesh_sdf.query_sdf(ray_points_flat)
            sdf_vals = sdf_vals.reshape(N, n_steps)

            if self._target_obb_axes is not None and self._target_obb_half_extents is not None:
                obb_dist = self._point_obb_distance(ray_points_flat)
                obb_dist = obb_dist.reshape(N, n_steps)
                exclusion_margin = self._target_obb_half_extents.max().item() * 0.5
                target_mask = (obb_dist < exclusion_margin).float()
                sdf_vals = sdf_vals + target_mask * 10.0

            per_step_penalty = F.softplus(margin - sdf_vals, beta=20.0)
        else:
            # Density-based: chunked to avoid OOM with large point clouds
            chunk_size = 256
            densities_flat = torch.zeros(N * n_steps, device=self.device)
            for ci in range(0, N * n_steps, chunk_size):
                ce = min(ci + chunk_size, N * n_steps)
                densities_flat[ci:ce] = self.density_field.query_density(ray_points_flat[ci:ce])
            densities = densities_flat.reshape(N, n_steps)

            if self._target_obb_axes is not None and self._target_obb_half_extents is not None:
                obb_dist = self._point_obb_distance(ray_points_flat)
                obb_dist = obb_dist.reshape(N, n_steps)
                exclusion_margin = self._target_obb_half_extents.max().item() * 0.5
                target_mask = (obb_dist < exclusion_margin).float()
                densities = densities * (1.0 - target_mask)

            per_step_penalty = F.softplus(densities - self.config.density_threshold, beta=10.0)

        cumulative = torch.cumsum(per_step_penalty, dim=1)
        shifted_cumulative = torch.cat([
            torch.zeros(N, 1, device=self.device),
            cumulative[:, :-1]
        ], dim=1)
        transmittance = torch.exp(-shifted_cumulative)
        occlusion_per_camera = (transmittance * per_step_penalty).sum(dim=1)

        return occlusion_per_camera.mean() * self.config.occlusion_weight

    def _ray_obb_t_enter(self, ray_origins: torch.Tensor, ray_vecs: torch.Tensor) -> torch.Tensor:
        if self._target_obb_axes is None or self._target_obb_half_extents is None:
            return (1.0 - self.config.occlusion_target_shrink) * torch.ones(
                ray_origins.shape[0], 1, device=self.device)

        obb_center = self._target_obb_center if self._target_obb_center is not None else self._target_center_zup
        obb_axes = self._target_obb_axes
        half_ext = self._target_obb_half_extents
        N = ray_origins.shape[0]

        origin_local = (ray_origins - obb_center.unsqueeze(0)) @ obb_axes
        dir_local = ray_vecs @ obb_axes

        inv_dir = 1.0 / (dir_local + 1e-10)
        t1 = (-half_ext.unsqueeze(0) - origin_local) * inv_dir
        t2 = ( half_ext.unsqueeze(0) - origin_local) * inv_dir

        t_slab_min = torch.minimum(t1, t2)
        t_slab_max = torch.maximum(t1, t2)

        t_enter = t_slab_min.max(dim=-1, keepdim=True).values
        t_exit  = t_slab_max.min(dim=-1, keepdim=True).values

        valid = (t_enter < t_exit) & (t_enter > 0)
        t_enter_safe = t_enter - 0.02
        fallback = 1.0 - self.config.occlusion_target_shrink
        t_far = torch.where(valid, t_enter_safe, torch.full_like(t_enter, fallback))

        return t_far

    def _count_occluded(self, pos_zup: torch.Tensor, verbose_diag: bool = False) -> int:
        if self._target_center_zup is None:
            return 0
        has_sdf = self.density_field.diff_mesh_sdf is not None
        if not has_sdf and self.density_field.gs_positions is None:
            return 0

        target = self._target_center_zup
        N = pos_zup.shape[0]
        n_steps = self.config.occlusion_n_steps
        near_skip = self.config.occlusion_near_skip

        ray_vec = target.unsqueeze(0) - pos_zup
        t_far = self._ray_obb_t_enter(pos_zup, ray_vec).clamp(min=near_skip + 0.05, max=1.0)

        t_uniform = torch.linspace(0, 1, n_steps, device=self.device)
        t_vals = near_skip + t_uniform.unsqueeze(0) * (t_far - near_skip)
        t_expand = t_vals.unsqueeze(-1)
        ray_points = (
            pos_zup.unsqueeze(1) * (1.0 - t_expand)
            + target.unsqueeze(0).unsqueeze(0) * t_expand
        )
        ray_points_flat = ray_points.reshape(N * n_steps, 3)

        with torch.no_grad():
            if has_sdf:
                vals = self.density_field.diff_mesh_sdf.query_sdf(ray_points_flat).reshape(N, n_steps)

                if self._target_obb_axes is not None and self._target_obb_half_extents is not None:
                    obb_dist = self._point_obb_distance(ray_points_flat).reshape(N, n_steps)
                    exclusion_margin = self._target_obb_half_extents.max().item() * 0.5
                    near_target = (obb_dist < exclusion_margin)
                    vals = vals.clone()
                    vals[near_target] = 1.0

                occluded_mask = (vals < 0).any(dim=1)
            else:
                chunk_size = 256
                densities_flat = torch.zeros(N * n_steps, device=self.device)
                for ci in range(0, N * n_steps, chunk_size):
                    ce = min(ci + chunk_size, N * n_steps)
                    densities_flat[ci:ce] = self.density_field.query_density(ray_points_flat[ci:ce])
                densities = densities_flat.reshape(N, n_steps)

                if self._target_obb_axes is not None and self._target_obb_half_extents is not None:
                    obb_dist = self._point_obb_distance(ray_points_flat).reshape(N, n_steps)
                    exclusion_margin = self._target_obb_half_extents.max().item() * 0.5
                    near_target = (obb_dist < exclusion_margin)
                    densities = densities.clone()
                    densities[near_target] = 0.0

                occluded_mask = (densities > self.config.density_threshold).any(dim=1)

            n_occluded = int(occluded_mask.sum().item())

            if verbose_diag and has_sdf:
                # existing SDF diagnostic printing...
                t_far_np = t_far.cpu().numpy().flatten()
                valid_obb = (t_far_np > near_skip + 0.06)
                n_obb_valid = int(valid_obb.sum())
                sdf_min_per_ray = vals.min(dim=1).values.cpu().numpy()
                excl_str = ""
                if self._target_obb_half_extents is not None:
                    excl_str = f", exclusion_margin={self._target_obb_half_extents.max().item() * 0.5:.3f}"
                print(f"      [diag] OBB valid hits: {n_obb_valid}/{N}, "
                      f"t_far range: [{t_far_np.min():.3f}, {t_far_np.max():.3f}]{excl_str}")
                print(f"      [diag] SDF min per ray (after exclusion): [{sdf_min_per_ray.min():.4f}, {sdf_min_per_ray.max():.4f}], "
                      f"margin={self.config.occlusion_margin}")
            elif verbose_diag and not has_sdf:
                max_density_per_ray = densities.max(dim=1).values.cpu().numpy()
                print(f"      [diag-density] Max density per ray: "
                      f"[{max_density_per_ray.min():.4f}, {max_density_per_ray.max():.4f}], "
                      f"threshold={self.config.density_threshold}")

            return n_occluded

    # ------------------------------------------------------------------
    # Optimize (unchanged)
    # ------------------------------------------------------------------

    def optimize(self, diff_traj, verbose=True, target_center_zup=None,
                 target_obb_center=None, target_obb_axes=None, target_obb_half_extents=None):
        if target_center_zup is not None:
            if isinstance(target_center_zup, np.ndarray):
                target_center_zup = torch.from_numpy(target_center_zup).float()
            self._target_center_zup = target_center_zup.to(self.device)
            if target_obb_center is not None:
                if isinstance(target_obb_center, np.ndarray):
                    target_obb_center = torch.from_numpy(target_obb_center).float()
                self._target_obb_center = target_obb_center.to(self.device)
            else:
                self._target_obb_center = None
            if target_obb_axes is not None and target_obb_half_extents is not None:
                if isinstance(target_obb_axes, np.ndarray):
                    target_obb_axes = torch.from_numpy(target_obb_axes).float()
                if isinstance(target_obb_half_extents, np.ndarray):
                    target_obb_half_extents = torch.from_numpy(target_obb_half_extents).float()
                self._target_obb_axes = target_obb_axes.to(self.device)
                self._target_obb_half_extents = target_obb_half_extents.to(self.device)
            else:
                self._target_obb_axes = None
                self._target_obb_half_extents = None
        else:
            self._target_center_zup = None
            self._target_obb_center = None
            self._target_obb_axes = None
            self._target_obb_half_extents = None

        if len(diff_traj.opt_indices) == 0:
            if verbose:
                print("    No optimizable parameters, skipping")
            return diff_traj.get_all_final_params(), {'final_cost': 0.0, 'n_iterations': 0}

        t_samples = torch.linspace(0, 1, self.config.n_samples, device=self.device)

        # ── FIX 1: Initial collision check with density fallback ──
        with torch.no_grad():
            pos_zup = diff_traj.evaluate_zup(t_samples)
            n_collisions = self.density_field.query_mesh_collision_count(pos_zup, self.config.safety_margin)
            n_occluded = self._count_occluded(pos_zup, verbose_diag=verbose)

            # When SDF is unavailable, use Gaussian density as collision signal
            if n_collisions == 0 and self.density_field.diff_mesh_sdf is None:
                densities = self.density_field.query_density(pos_zup)
                n_density_violations = int((densities > self.config.density_threshold).sum().item())
                if verbose and n_density_violations > 0:
                    print(f"    No SDF — density check: {n_density_violations}/{self.config.n_samples} above threshold")
                n_collisions = n_density_violations

        if n_collisions == 0 and n_occluded == 0:
            if verbose:
                print("    Already collision-free and unoccluded, skipping optimization")
            return diff_traj.get_all_final_params(), {'final_cost': 0.0, 'n_iterations': 0}

        if verbose and n_occluded > 0:
            bbox_info = ""
            if self._target_obb_half_extents is not None:
                he = self._target_obb_half_extents.cpu().numpy()
                bbox_info = f" (OBB half-extents: [{he[0]:.2f}, {he[1]:.2f}, {he[2]:.2f}])"
            print(f"    Initial occlusion: {n_occluded}/{self.config.n_samples} cameras occluded{bbox_info}")

        optimizer = torch.optim.Adam([diff_traj.opt_params], lr=self.config.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=30)

        best_cost = float('inf')
        best_opt_params = diff_traj.opt_params.data.clone()
        history = []

        for it in range(self.config.n_iterations):
            optimizer.zero_grad()
            pos_zup = diff_traj.evaluate_zup(t_samples)
            c_smooth = self._smoothness_cost(pos_zup)
            c_reg = self._regularization_cost(diff_traj)
            c_bound = self._boundary_cost(pos_zup)
            c_mesh = self._mesh_sdf_cost(pos_zup)
            c_occl = self._occlusion_cost(pos_zup)

            total = c_mesh + c_smooth + c_reg + c_bound + c_occl

            # When no SDF, also add density collision cost for extra signal
            # (_mesh_sdf_cost returns 0 without SDF, so density fills that role)
            if self.density_field.diff_mesh_sdf is None:
                c_density = self._collision_cost(pos_zup)
                total = total + c_density

            total.backward()
            torch.nn.utils.clip_grad_norm_([diff_traj.opt_params], max_norm=1.0)
            optimizer.step()
            scheduler.step(total)

            with torch.no_grad():
                diff_traj.opt_params.data = torch.clamp(
                    diff_traj.opt_params.data, diff_traj.opt_lower, diff_traj.opt_upper)

            cost_val = total.item()
            history.append(cost_val)
            if cost_val < best_cost:
                best_cost = cost_val
                best_opt_params = diff_traj.opt_params.data.clone()

            # ── FIX 2: Iteration logging with density fallback ──
            if verbose and it % 50 == 0:
                with torch.no_grad():
                    eval_pos = diff_traj.evaluate_zup(t_samples)
                    n_violations = self.density_field.query_mesh_collision_count(
                        eval_pos, self.config.safety_margin)
                    n_occ = self._count_occluded(eval_pos)

                    if n_violations == 0 and self.density_field.diff_mesh_sdf is None:
                        densities = self.density_field.query_density(eval_pos)
                        n_violations = int((densities > self.config.density_threshold).sum().item())

                if self.density_field.diff_mesh_sdf is None:
                    print(f"    Iter {it:4d}: total={cost_val:.4f}, "
                          f"density={c_density.item():.4f}, "
                          f"smooth={c_smooth.item():.4f}, "
                          f"reg={c_reg.item():.4f}, bound={c_bound.item():.4f}, "
                          f"density_coll={n_violations}/{self.config.n_samples}")
                else:
                    print(f"    Iter {it:4d}: total={cost_val:.4f}, "
                          f"mesh_sdf={c_mesh.item():.4f}, occl={c_occl.item():.4f}, "
                          f"smooth={c_smooth.item():.4f}, "
                          f"reg={c_reg.item():.4f}, bound={c_bound.item():.4f}, "
                          f"out_mesh={n_violations}/{self.config.n_samples}, "
                          f"occluded={n_occ}/{self.config.n_samples}")

            # ── FIX 3: Convergence check with density fallback ──
            with torch.no_grad():
                eval_pos = diff_traj.evaluate_zup(t_samples)
                n_out = self.density_field.query_mesh_collision_count(
                    eval_pos, self.config.safety_margin)
                n_occ = self._count_occluded(eval_pos)

                if n_out == 0 and self.density_field.diff_mesh_sdf is None:
                    densities = self.density_field.query_density(eval_pos)
                    n_out = int((densities > self.config.density_threshold).sum().item())

            if n_out == 0 and n_occ == 0:
                if verbose:
                    print(f"    Converged with 0 collisions & 0 occlusions at iter {it}")
                break
            elif n_out == 0 and len(history) > 20 and max(history[-20:]) - min(history[-20:]) < 1e-6:
                if verbose:
                    print(f"    Collision-free, occlusion plateaued ({n_occ}/{self.config.n_samples}) at iter {it}")
                break

            if len(history) > 10 and max(history[-10:]) - min(history[-10:]) < self.config.convergence_threshold:
                MAX_MESH_SDF_WEIGHT = 1e3
                if n_out > self.config.n_samples * 0.05:
                    if self.density_field.diff_mesh_sdf is not None and self.config.mesh_sdf_weight < MAX_MESH_SDF_WEIGHT:
                        self.config.mesh_sdf_weight *= 2.0
                        history.clear()
                        continue
                    elif self.density_field.diff_mesh_sdf is None and self.config.collision_weight < MAX_MESH_SDF_WEIGHT:
                        self.config.collision_weight *= 2.0
                        history.clear()
                        continue
                    else:
                        if verbose:
                            print(f"    Reached max weight, accepting {n_out} collisions")
                        break
                else:
                    if verbose:
                        print(f"    Cost plateaued, accepting {n_out} collisions & {n_occ} occlusions at iter {it}")
                    break

        with torch.no_grad():
            diff_traj.opt_params.data = best_opt_params
        return diff_traj.get_all_final_params(), {'final_cost': best_cost, 'n_iterations': it + 1}


# =============================================================================
# Integration Helpers
# =============================================================================

def extract_trajectory_params(trajectory):
    class_name = type(trajectory).__name__
    free_specs = trajectory.get_free_parameter_specs()
    fixed_params = {}
    for attr in ['object_center', 'camera_position', 'start_position',
                 'end_position', 'start_rotation', 'end_rotation']:
        if hasattr(trajectory, attr):
            fixed_params[attr] = tuple(getattr(trajectory, attr))
    for attr in ['approach_angle', 'end_height']:
        if hasattr(trajectory, attr):
            fixed_params[attr] = float(getattr(trajectory, attr))
    free_params = trajectory.get_free_parameters()
    free_names = [s.name for s in free_specs]
    lower = np.array([s.min_bound for s in free_specs])
    upper = np.array([s.max_bound for s in free_specs])
    return class_name, fixed_params, free_params, free_names, (lower, upper)


def create_differentiable_trajectory(trajectory, device='cuda', optimize_only=None):
    cls_name, fixed, free, names, bounds = extract_trajectory_params(trajectory)
    return DifferentiableTrajectory(cls_name, fixed, free, names, bounds,
                                    optimize_only=optimize_only, device=device)


def optimize_and_regenerate(trajectory, optimizer, n_frames, fps=30.0,
                            verbose=True, optimize_only=None,
                            target_center_zup=None,
                            target_obb_center=None,
                            target_obb_axes=None,
                            target_obb_half_extents=None):
    diff_traj = create_differentiable_trajectory(trajectory, optimizer.device, optimize_only)
    if verbose:
        print(f"  Optimizing {type(trajectory).__name__}")
        if optimize_only:
            print(f"    Only: {optimize_only}")
    opt_params, results = optimizer.optimize(diff_traj, verbose,
                                             target_center_zup=target_center_zup,
                                             target_obb_center=target_obb_center,
                                             target_obb_axes=target_obb_axes,
                                             target_obb_half_extents=target_obb_half_extents)
    trajectory.set_free_parameters(opt_params)
    if verbose:
        orig = diff_traj.all_original_params.cpu().numpy()
        for i, name in enumerate(diff_traj.free_param_names):
            frozen = "(frozen)" if i in diff_traj.frozen_indices else ""
            print(f"      {name}: {orig[i]:.3f} → {opt_params[i]:.3f} (Δ={opt_params[i]-orig[i]:+.3f}) {frozen}")
    return trajectory.generate(n_frames, fps=int(fps))


# =============================================================================
# Pose Extraction Helpers  (unchanged)
# =============================================================================

def _extract_pose_from_c2w_zup(c2w_zup, executor):
    position_zup = c2w_zup[:3, 3].copy()
    rotation_zup = c2w_zup[:3, :3].copy()
    position_yup = executor._zup_to_yup(position_zup)
    forward_zup = rotation_zup[:, 2].copy()
    forward_yup = executor._zup_to_yup(forward_zup)
    forward_norm = np.linalg.norm(forward_yup)
    if forward_norm > 1e-6:
        forward_yup = forward_yup / forward_norm
    else:
        forward_yup = np.array([0.0, 0.0, 1.0])
    yaw = np.rad2deg(np.arctan2(forward_yup[0], forward_yup[2]))
    pitch = np.rad2deg(np.arcsin(np.clip(-forward_yup[1], -1.0, 1.0)))
    return {
        'position': position_yup,
        'position_zup': position_zup,
        'rotation': np.array([pitch, yaw, 0.0]),
        'look_at_yup': position_yup + forward_yup,
    }


def _get_last_pose_from_output(traj_output, executor):
    return _extract_pose_from_c2w_zup(traj_output['c2w'][-1], executor)


def _get_first_pose_from_output(traj_output, executor):
    return _extract_pose_from_c2w_zup(traj_output['c2w'][0], executor)


# =============================================================================
# Arc Optimization Helpers
# =============================================================================

def find_best_arc_init(diff_traj, density_field, config, device, candidates=None, n_eval=50):
    if candidates is None:
        candidates = list(range(-45, 46, 5))
    dev = torch.device(device if torch.cuda.is_available() else 'cpu')
    t = torch.linspace(0, 1, n_eval, device=dev)
    if 'arc_angle' not in diff_traj.free_param_names:
        return 0.0, 0
    idx = diff_traj.free_param_names.index('arc_angle')
    if idx not in diff_traj.opt_indices:
        return 0.0, 0
    local = diff_traj.opt_indices.index(idx)
    best_angle = 0.0
    best_count = float('inf')
    orig = diff_traj.opt_params.data[local].item()
    with torch.no_grad():
        for angle in candidates:
            diff_traj.opt_params.data[local] = angle
            pos = diff_traj.evaluate_zup(t)
            count = density_field.query_mesh_collision_count(pos, config.safety_margin)
            # Density fallback when no SDF
            if count == 0 and density_field.diff_mesh_sdf is None:
                densities = density_field.query_density(pos)
                count = int((densities > config.density_threshold).sum().item())
            if count < best_count or (count == best_count and abs(angle) < abs(best_angle)):
                best_count = count
                best_angle = angle
    diff_traj.opt_params.data[local] = orig
    return best_angle, best_count


def optimize_arc_segment(trajectory, optimizer, density_field, config,
                         n_frames, fps, device, verbose=True,
                         optimize_only=None, max_arc_angle=45.0, init_candidates=None,
                         target_center_zup=None):
    diff_traj = create_differentiable_trajectory(trajectory, device, optimize_only)
    if 'arc_angle' in diff_traj.free_param_names:
        angle_idx = diff_traj.free_param_names.index('arc_angle')
        if angle_idx in diff_traj.opt_indices:
            local_idx = diff_traj.opt_indices.index(angle_idx)
            diff_traj.opt_lower[local_idx] = max(diff_traj.opt_lower[local_idx].item(), -max_arc_angle)
            diff_traj.opt_upper[local_idx] = min(diff_traj.opt_upper[local_idx].item(), max_arc_angle)
            if verbose:
                print(f"    Arc angle bounds: [{diff_traj.opt_lower[local_idx].item():.0f}°, {diff_traj.opt_upper[local_idx].item():.0f}°]")

    best_init, best_count = find_best_arc_init(diff_traj, density_field, config, device, candidates=init_candidates)
    if 'arc_angle' in diff_traj.free_param_names:
        angle_idx = diff_traj.free_param_names.index('arc_angle')
        if angle_idx in diff_traj.opt_indices:
            local_idx = diff_traj.opt_indices.index(angle_idx)
            current_val = diff_traj.opt_params.data[local_idx].item()
            diff_traj.opt_params.data[local_idx] = best_init
            diff_traj.opt_original[local_idx] = best_init
            if verbose:
                print(f"    Arc angle init: {current_val:.0f}° → {best_init:.0f}° ({best_count}/{config.n_samples} collisions)")

    if best_count == 0:
        if verbose:
            print(f"    Zero collisions at init, skipping optimization")
        trajectory.set_free_parameters(diff_traj.get_all_final_params())
        return trajectory.generate(n_frames, fps=int(fps))

    if verbose:
        print(f"  Optimizing {type(trajectory).__name__}")
        if optimize_only:
            print(f"    Only: {optimize_only}")
    opt_params, results = optimizer.optimize(diff_traj, verbose,
                                             target_center_zup=target_center_zup)
    trajectory.set_free_parameters(opt_params)
    if verbose:
        orig = diff_traj.all_original_params.cpu().numpy()
        for i, name in enumerate(diff_traj.free_param_names):
            frozen = "(frozen)" if i in diff_traj.frozen_indices else ""
            print(f"      {name}: {orig[i]:.3f} → {opt_params[i]:.3f} (Δ={opt_params[i]-orig[i]:+.3f}) {frozen}")
    return trajectory.generate(n_frames, fps=int(fps))


# =============================================================================
# OBB Extraction — UPDATED for both formats
# =============================================================================

def _extract_obb_from_bbox_data(obj_data: dict):
    """
    Extract OBB (center, axes, half_extents) from bbox_data dict.

    Supports both:
    - New scene graph format: has 'extent' (local OBB sizes) and we can
      reconstruct axes from the original quaternion if available, or use
      axis-aligned approximation from corners.
    - Legacy format: has 'corners' (8 points) from labels.json.

    Returns:
        (center, axes, half_extents) or (None, None, None)
    """
    if obj_data is None:
        return None, None, None

    corners = obj_data.get('corners')
    if corners is None:
        return None, None, None

    corners = np.array(corners, dtype=np.float64)
    if corners.shape != (8, 3):
        return None, None, None

    center = corners.mean(axis=0)

    # --- New format: 'extent' holds local OBB sizes (sx, sy, sz) ---
    # The corners are from obb_to_corners() with a consistent sign ordering.
    # We can recover axes from the corner geometry.
    extent = obj_data.get('extent')
    if extent is not None:
        extent = np.array(extent, dtype=np.float64)
        half_extents = extent / 2.0

        # obb_to_corners uses sign ordering:
        #   [0] = (-1,-1,-1), [4] = (1,-1,-1), [2] = (-1,1,-1), [1] = (-1,-1,1)
        # So edges from corner[0]:
        #   corner[4] - corner[0] = local +x direction * sx
        #   corner[2] - corner[0] = local +y direction * sy
        #   corner[1] - corner[0] = local +z direction * sz
        e_x = corners[4] - corners[0]
        e_y = corners[2] - corners[0]
        e_z = corners[1] - corners[0]

        len_x = np.linalg.norm(e_x)
        len_y = np.linalg.norm(e_y)
        len_z = np.linalg.norm(e_z)

        ax_x = e_x / len_x if len_x > 1e-8 else np.array([1.0, 0.0, 0.0])
        ax_y = e_y / len_y if len_y > 1e-8 else np.array([0.0, 1.0, 0.0])
        ax_z = e_z / len_z if len_z > 1e-8 else np.array([0.0, 0.0, 1.0])

        # Orthonormalize
        ax_y = ax_y - np.dot(ax_y, ax_x) * ax_x
        n_y = np.linalg.norm(ax_y)
        ax_y = ax_y / n_y if n_y > 1e-8 else np.cross(ax_z, ax_x)
        ax_y = ax_y / (np.linalg.norm(ax_y) + 1e-10)
        ax_z = np.cross(ax_x, ax_y)
        ax_z = ax_z / (np.linalg.norm(ax_z) + 1e-10)

        if np.linalg.det(np.column_stack([ax_x, ax_y, ax_z])) < 0:
            ax_z = -ax_z

        axes = np.column_stack([ax_x, ax_y, ax_z])
        padding = 0.05
        half_extents = half_extents + padding
        return center, axes, half_extents

    # --- Legacy format: use edge-based extraction ---
    return _extract_obb_from_8_corners_legacy(corners)


def _extract_obb_from_8_corners_legacy(corners: np.ndarray):
    """
    Extract OBB from 8 corner points as stored in legacy labels.json.

    Convention: corners[0:4] are the bottom face, corners[4:8] are the top face.
    """
    center = corners.mean(axis=0)

    e1 = corners[1] - corners[0]
    e2 = corners[3] - corners[0]
    e3 = corners[4] - corners[0]

    h1 = np.linalg.norm(e1)
    h2 = np.linalg.norm(e2)
    h3 = np.linalg.norm(e3)

    ax1 = e1 / h1 if h1 > 1e-8 else np.array([1.0, 0.0, 0.0])
    ax2 = e2 / h2 if h2 > 1e-8 else np.array([0.0, 1.0, 0.0])
    ax3 = e3 / h3 if h3 > 1e-8 else np.array([0.0, 0.0, 1.0])

    ax2 = ax2 - np.dot(ax2, ax1) * ax1
    n2 = np.linalg.norm(ax2)
    ax2 = ax2 / n2 if n2 > 1e-8 else np.cross(ax3, ax1)
    ax2 = ax2 / (np.linalg.norm(ax2) + 1e-10)

    ax3 = np.cross(ax1, ax2)
    ax3 = ax3 / (np.linalg.norm(ax3) + 1e-10)

    if np.linalg.det(np.column_stack([ax1, ax2, ax3])) < 0:
        ax3 = -ax3

    axes = np.column_stack([ax1, ax2, ax3])
    half_extents = np.array([h1 * 0.5, h2 * 0.5, h3 * 0.5])
    padding = 0.05
    half_extents = half_extents + padding

    return center, axes, half_extents


# Keep old name as alias for backward compatibility
_extract_obb_from_8_corners = _extract_obb_from_8_corners_legacy


def _get_target_center_zup(segment, executor):
    """
    Extract the target object's look-at point, OBB center, and OBB geometry.

    Works with both new scene graph format (string IDs, OBB arrays) and
    legacy labels.json format (integer IDs, 8-corner bounding boxes).

    Returns:
        (look_at_zup, obb_center_zup, obb_axes, obb_half_extents) or (None, None, None, None)
    """
    anchor = segment.start_anchor
    if anchor is None:
        return None, None, None, None
    if not (hasattr(anchor, 'look_at') and anchor.look_at is not None):
        return None, None, None, None

    look_at = anchor.look_at.copy()
    obb_center = None
    obb_axes = None
    obb_half_extents = None

    # --- Try to get OBB data from executor's anchor_determinator.bbox_data ---
    obj_data = None

    if hasattr(executor, 'anchor_determinator'):
        ad = executor.anchor_determinator
        if hasattr(ad, 'bbox_data'):
            # Try string ID first (new format), then str(int) (legacy)
            obj_data = ad.bbox_data.get(str(anchor.object_id))
            if obj_data is None and isinstance(anchor.object_id, int):
                obj_data = ad.bbox_data.get(anchor.object_id)

    # Fallback: try executor.bbox_data directly
    if obj_data is None and hasattr(executor, 'bbox_data') and executor.bbox_data is not None:
        obj_id = anchor.object_id
        if isinstance(executor.bbox_data, dict):
            obj_data = executor.bbox_data.get(str(obj_id))
            if obj_data is None:
                obj_data = executor.bbox_data.get(obj_id)
        elif isinstance(executor.bbox_data, list):
            for obj in executor.bbox_data:
                if str(obj.get('ins_id', '')) == str(obj_id) or obj.get('id') == obj_id:
                    if 'bounding_box' in obj and obj['bounding_box']:
                        bb = obj['bounding_box']
                        obj_data = {
                            'corners': np.array([[p['x'], p['y'], p['z']] for p in bb], dtype=np.float64),
                        }
                    elif 'corners' in obj:
                        obj_data = obj
                    break

    # --- Extract OBB from bbox_data ---
    if obj_data is not None:
        obb_center, obb_axes, obb_half_extents = _extract_obb_from_bbox_data(obj_data)

    # Fallback: if we still don't have OBB data, try anchor's bbox_corners attribute
    if obb_center is None and hasattr(anchor, 'bbox_corners') and anchor.bbox_corners is not None:
        corners_8 = np.array(anchor.bbox_corners, dtype=np.float64)
        if corners_8.shape == (8, 3):
            obb_center, obb_axes, obb_half_extents = _extract_obb_from_8_corners_legacy(corners_8)

    # Final fallback: identity OBB at look_at
    if obb_center is None:
        obb_center = look_at.copy()
        obb_axes = np.eye(3)
        obb_half_extents = np.array([0.3, 0.3, 0.3])

    return look_at, obb_center, obb_axes, obb_half_extents


# =============================================================================
# High-Level: Optimize TrajectoryResult
# =============================================================================

def optimize_trajectory_result(
    result, executor, ply_path, mesh=None, config=None, device='cuda',
    verbose=True, sdf_resolution=64, sdf_cache_dir=None, scene_name=None,
    scene_json=None,
):
    """Two-pass optimization with collision + occlusion costs."""
    config = config or ParametricOptimizationConfig()
    if verbose:
        print("\n" + "=" * 60)
        print("PARAMETRIC TRAJECTORY OPTIMIZATION (with occlusion)")
        print("=" * 60)

    positions, scales, opacities = load_ply_3dgs(ply_path)
    density_field = GaussianDensityField(
        positions, scales, opacities, config, device,
        mesh=mesh, sdf_resolution=sdf_resolution,
        sdf_cache_dir=sdf_cache_dir, scene_name=scene_name)
    optimizer = ParametricTrajectoryOptimizer(density_field, config, device)

    room_graph = None
    scene_objects = None
    if scene_json is not None and "rooms" in scene_json:
        rooms = scene_json.get("rooms", {})
        if len(rooms) > 1:
            room_graph = build_room_graph(scene_json, verbose=verbose)
            scene_objects = scene_json.get("objects", {})
        elif verbose:
            print("  Single room detected — cross-room routing disabled")

    # ==================================================================
    # PASS 1: Optimize all object-level segments
    # ==================================================================
    if verbose:
        print("\n" + "-" * 60)
        print("PASS 1: Optimizing object-level segments (collision + occlusion)")
        print("-" * 60)

    for i, segment in enumerate(result.segments):
        if segment.movement_category == 'transitional':
            continue

        optimizer.config.mesh_sdf_weight = config.mesh_sdf_weight

        if verbose:
            label = getattr(segment.start_anchor, 'object_label', '?') if segment.start_anchor else '?'
            print(f"\n--- [{i}] {segment.movement_type} @ {label} ---")

        config_entry = executor.MOVEMENT_CONFIGS.get(segment.movement_type)
        if config_entry is None:
            if verbose:
                print("  Unknown movement, skipping")
            continue

        start_pose = executor._anchor_to_pose(segment.start_anchor)
        if segment.trajectory_output is not None:
            n_frames = segment.trajectory_output.get('n_frames', 90)
        else:
            n_frames = 90

        trajectory = executor._create_object_centric_trajectory(
            config_entry, segment.start_anchor, start_pose, segment.movement_type)

        target_look_at, target_obb_center, target_obb_axes, target_obb_half_extents = _get_target_center_zup(segment, executor)
        if verbose and target_look_at is not None:
            ext_str = ""
            if target_obb_half_extents is not None:
                he = target_obb_half_extents
                ext_str = f", OBB half-extents: [{he[0]:.2f}, {he[1]:.2f}, {he[2]:.2f}]"
            if target_obb_center is not None:
                oc = target_obb_center
                offset = np.linalg.norm(target_look_at - oc)
                if offset > 0.05:
                    ext_str += f", OBB center offset: {offset:.2f}"
            print(f"  Target look-at (Z-up): [{target_look_at[0]:.2f}, {target_look_at[1]:.2f}, {target_look_at[2]:.2f}]{ext_str}")

        optimize_only = None
        cls_name = type(trajectory).__name__

        if cls_name in ('StationaryPan', 'StationaryTilt', 'ZoomLens'):
            if segment.trajectory_output is None:
                try:
                    new_output = trajectory.generate(n_frames, fps=int(executor.fps))
                    c2w_zup = executor._convert_c2w_yup_to_zup(new_output['c2w'])
                    new_output['c2w'] = c2w_zup
                    new_output['positions'] = c2w_zup[:, :3, 3]
                    new_output['rotations_matrix'] = c2w_zup[:, :3, :3]
                    new_output['coordinate_system'] = 'z-up'
                    segment.trajectory_output = new_output
                except Exception as e:
                    if verbose:
                        print(f"  ✗ Failed to generate {cls_name}: {e}")
            if verbose:
                print(f"  {cls_name} — no positional optimization needed")
            continue

        if 'CircularOrbit' in cls_name:
            optimize_only = ['end_angle', 'radius']
        elif 'CraneShot' in cls_name:
            optimize_only = ['radius']
        elif 'DollyMove' in cls_name:
            optimize_only = ['start_radius', 'end_radius']

        try:
            new_output = optimize_and_regenerate(
                trajectory, optimizer, n_frames, executor.fps, verbose, optimize_only,
                target_center_zup=target_look_at,
                target_obb_center=target_obb_center,
                target_obb_axes=target_obb_axes,
                target_obb_half_extents=target_obb_half_extents)
            c2w_zup = executor._convert_c2w_yup_to_zup(new_output['c2w'])
            new_output['c2w'] = c2w_zup
            new_output['positions'] = c2w_zup[:, :3, 3]
            new_output['rotations_matrix'] = c2w_zup[:, :3, :3]
            new_output['coordinate_system'] = 'z-up'
            new_output['optimized'] = True
            segment.trajectory_output = new_output
            if verbose:
                first = c2w_zup[0, :3, 3]
                last = c2w_zup[-1, :3, 3]
                print(f"  ✓ Optimized {n_frames} frames")
                print(f"    Actual start: [{first[0]:.2f}, {first[1]:.2f}, {first[2]:.2f}]")
                print(f"    Actual end:   [{last[0]:.2f}, {last[1]:.2f}, {last[2]:.2f}]")
        except Exception as e:
            if verbose:
                print(f"  ✗ Failed: {e}")
            import traceback; traceback.print_exc()

    # ==================================================================
    # PASS 2: Rebuild and optimize transitional arcs
    # ==================================================================
    if verbose:
        print("\n" + "-" * 60)
        print("PASS 2: Building & optimizing transitional arcs")
        print("-" * 60)
 
    # We may need to INSERT new segments, so iterate over a copy of indices
    i = 0
    while i < len(result.segments):
        segment = result.segments[i]
        if segment.movement_category != 'transitional':
            i += 1
            continue
        
        optimizer.config.mesh_sdf_weight = config.mesh_sdf_weight
 
        # Find neighbouring object-level segments
        prev_obj_seg = None
        next_obj_seg = None
        for j in range(i - 1, -1, -1):
            if result.segments[j].movement_category == 'object-level':
                prev_obj_seg = result.segments[j]
                break
        for j in range(i + 1, len(result.segments)):
            if result.segments[j].movement_category == 'object-level':
                next_obj_seg = result.segments[j]
                break
 
        if verbose:
            prev_label = getattr(prev_obj_seg.start_anchor, 'object_label', '?') if prev_obj_seg else '?'
            next_label = getattr(next_obj_seg.start_anchor, 'object_label', '?') if next_obj_seg else '?'
            print(f"\n--- [{i}] arc: {prev_label} → {next_label} ---")
 
        # Get end/start poses from optimized neighbours
        if prev_obj_seg is not None and prev_obj_seg.trajectory_output is not None:
            prev_end_pose = _get_last_pose_from_output(prev_obj_seg.trajectory_output, executor)
        else:
            prev_end_pose = executor._anchor_to_pose(segment.start_anchor)
 
        if next_obj_seg is not None and next_obj_seg.trajectory_output is not None:
            next_start_pose = _get_first_pose_from_output(next_obj_seg.trajectory_output, executor)
        else:
            next_start_pose = executor._anchor_to_pose(segment.end_anchor) if segment.end_anchor else prev_end_pose
 
        if verbose:
            p1 = prev_end_pose.get('position_zup', prev_end_pose.get('position', [0,0,0]))
            p2 = next_start_pose.get('position_zup', next_start_pose.get('position', [0,0,0]))
            print(f"  From: [{p1[0]:.2f}, {p1[1]:.2f}, {p1[2]:.2f}]")
            print(f"  To:   [{p2[0]:.2f}, {p2[1]:.2f}, {p2[2]:.2f}]")
 
        # >>> NEW: CROSS-ROOM — Check if this is a cross-room transition
        cross_room_waypoints = None
        if room_graph is not None and prev_obj_seg is not None and next_obj_seg is not None:
            cross_room_waypoints = get_cross_room_info(
                prev_obj_seg, next_obj_seg, room_graph, scene_objects,
                executor, verbose=verbose,
            )
 
        if cross_room_waypoints is not None and len(cross_room_waypoints) > 0:
            # ── CROSS-ROOM: Split into sub-arcs through door/window waypoints ──
            if verbose:
                n_wp = len(cross_room_waypoints)
                print(f"  Cross-room transition: {n_wp} waypoint(s) via "
                      f"{[wp.get('door_window_id', '?') for wp in cross_room_waypoints]}")
 
            # Build list of poses: [prev_end, wp1, wp2, ..., next_start]
            all_poses = [prev_end_pose]
            for wp in cross_room_waypoints:
                all_poses.append({
                    'position': wp['position_yup'],
                    'position_zup': wp['position_zup'],
                    'rotation': wp['rotation'],
                    'look_at_yup': wp['look_at_yup'],
                })
            all_poses.append(next_start_pose)
 
            # Remove the original single-arc segment
            result.segments.pop(i)
 
            # Create sub-arc segments
            n_sub = len(all_poses) - 1
            n_frames_per_sub = max(30, 60 // n_sub)  # distribute frames
 
            inserted_count = 0
            for k in range(n_sub):
                sub_start = all_poses[k]
                sub_end = all_poses[k + 1]
 
                config_entry = executor.MOVEMENT_CONFIGS.get('arc')
                if config_entry is None:
                    continue
 
                arc_angle = 0.0  # straight through door — minimize curve
                trajectory = executor._create_transitional_trajectory(
                    config_entry, sub_start, sub_end, arc_angle,
                )
 
                try:
                    new_output = optimize_arc_segment(
                        trajectory, optimizer, density_field, config,
                        n_frames_per_sub, executor.fps, device, verbose,
                        optimize_only=['arc_angle'],
                        target_center_zup=None,
                    )
                    c2w_zup = executor._convert_c2w_yup_to_zup(new_output['c2w'])
                    new_output['c2w'] = c2w_zup
                    new_output['positions'] = c2w_zup[:, :3, 3]
                    new_output['rotations_matrix'] = c2w_zup[:, :3, :3]
                    new_output['coordinate_system'] = 'z-up'
                    new_output['optimized'] = True
                except Exception as e:
                    if verbose:
                        print(f"  ✗ Sub-arc {k} optimization failed: {e}")
                    new_output = None
 
                # Create segment for this sub-arc
                sub_seg = TrajectorySegment(
                    start_anchor=segment.start_anchor if k == 0 else segment.end_anchor,
                    end_anchor=segment.end_anchor if k == n_sub - 1 else segment.start_anchor,
                    movement_type='arc',
                    movement_category='transitional',
                    arc_angle=arc_angle,
                    trajectory_output=new_output,
                )
 
                result.segments.insert(i + inserted_count, sub_seg)
                inserted_count += 1
 
                if verbose and new_output is not None:
                    first = c2w_zup[0, :3, 3]
                    last = c2w_zup[-1, :3, 3]
                    door_id = cross_room_waypoints[min(k, len(cross_room_waypoints)-1)].get('door_window_id', '?')
                    print(f"  ✓ Sub-arc {k} ({n_frames_per_sub} frames) "
                          f"[via {door_id}]")
                    print(f"    [{first[0]:.2f}, {first[1]:.2f}, {first[2]:.2f}] → "
                          f"[{last[0]:.2f}, {last[1]:.2f}, {last[2]:.2f}]")
 
            # Advance past all inserted sub-arcs
            i += inserted_count
            continue
        # <<< END CROSS-ROOM
 
        # ── Original single-arc logic (same room or no waypoints) ──
        config_entry = executor.MOVEMENT_CONFIGS.get('arc')
        if config_entry is None:
            i += 1
            continue
 
        arc_angle = getattr(segment, 'arc_angle', 0.0)
        n_frames = 60
        trajectory = executor._create_transitional_trajectory(
            config_entry, prev_end_pose, next_start_pose, arc_angle,
        )
 
        try:
            new_output = optimize_arc_segment(
                trajectory, optimizer, density_field, config,
                n_frames, executor.fps, device, verbose,
                optimize_only=['arc_angle'],
                target_center_zup=None,
            )
            c2w_zup = executor._convert_c2w_yup_to_zup(new_output['c2w'])
            new_output['c2w'] = c2w_zup
            new_output['positions'] = c2w_zup[:, :3, 3]
            new_output['rotations_matrix'] = c2w_zup[:, :3, :3]
            new_output['coordinate_system'] = 'z-up'
            new_output['optimized'] = True
            segment.trajectory_output = new_output
            if verbose:
                first = c2w_zup[0, :3, 3]
                last = c2w_zup[-1, :3, 3]
                print(f"  ✓ Optimized arc {n_frames} frames")
                print(f"    Arc start: [{first[0]:.2f}, {first[1]:.2f}, {first[2]:.2f}]")
                print(f"    Arc end:   [{last[0]:.2f}, {last[1]:.2f}, {last[2]:.2f}]")
        except Exception as e:
            if verbose:
                print(f"  ✗ Arc optimization failed: {e}")
            import traceback; traceback.print_exc()
 
        i += 1
 
    return result


# =============================================================================
# TrajectoryCombiner  (unchanged except save_anchors handles string IDs)
# =============================================================================

class TrajectoryCombiner:
    def __init__(self, transition_frames=15, blend_frames=5, fps=30.0, smoothing_factor=0.5):
        self.transition_frames = transition_frames
        self.blend_frames = blend_frames
        self.fps = fps
        self.smoothing_factor = smoothing_factor

    @staticmethod
    def reorder_segments_nearest_neighbour(result, long_hop_weight=2.0, long_hop_threshold=None):
        segments = result.segments
        if len(segments) <= 3:
            return result
        obj_segments = [seg for seg in segments if getattr(seg, 'movement_category', '') == 'object-level']
        if len(obj_segments) <= 2:
            return result
        positions = []
        for seg in obj_segments:
            a = seg.start_anchor
            if a is not None and hasattr(a, 'look_at') and a.look_at is not None:
                positions.append(a.look_at)
            elif a is not None:
                positions.append(a.position)
            else:
                positions.append(np.zeros(3))
        positions = np.array(positions)
        n = len(obj_segments)
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dists[i, j] = np.linalg.norm(positions[i] - positions[j])
        if long_hop_threshold is None:
            upper = dists[np.triu_indices(n, k=1)]
            long_hop_threshold = np.median(upper) if len(upper) > 0 else 1.0
        def hop_cost(d):
            if d <= long_hop_threshold:
                return d
            excess = d - long_hop_threshold
            return d + long_hop_weight * excess ** 2
        visited = [False] * n
        order = [0]
        visited[0] = True
        for _ in range(n - 1):
            last = order[-1]
            best_cost = float('inf')
            best_idx = -1
            for j in range(n):
                if not visited[j]:
                    c = hop_cost(dists[last][j])
                    if c < best_cost:
                        best_cost = c
                        best_idx = j
            order.append(best_idx)
            visited[best_idx] = True
        reordered_obj = [obj_segments[i] for i in order]
        old_labels = [getattr(s.start_anchor, 'object_label', '?') for s in obj_segments]
        new_labels = [getattr(s.start_anchor, 'object_label', '?') for s in reordered_obj]
        if old_labels != new_labels:
            old_dist = sum(dists[i][i+1] for i in range(n-1))
            new_dist = sum(dists[order[i]][order[i+1]] for i in range(n-1))
            max_hop = max(dists[order[i]][order[i+1]] for i in range(n-1))
            print(f"  Reordered: {' → '.join(old_labels)}")
            print(f"         →   {' → '.join(new_labels)}")
            print(f"  Distance:  {old_dist:.2f} → {new_dist:.2f} (saved {old_dist - new_dist:.2f})")
            print(f"  Max hop:   {max_hop:.2f}  (threshold: {long_hop_threshold:.2f})")
        
        new_segments = []
        for i, obj_seg in enumerate(reordered_obj):
            new_segments.append(obj_seg)
            if i < len(reordered_obj) - 1:
                next_seg = reordered_obj[i + 1]
                arc_seg = TrajectorySegment(
                    start_anchor=obj_seg.start_anchor,
                    end_anchor=next_seg.start_anchor,
                    movement_type='arc',
                    movement_category='transitional',
                    trajectory_output=None)
                arc_seg.arc_angle = np.random.uniform(-15.0, 15.0)
                new_segments.append(arc_seg)
        result.segments = new_segments
        return result

    @staticmethod
    def _estimate_velocity(positions, fps, end='last'):
        if len(positions) < 2:
            return np.zeros(3)
        if end == 'last':
            if len(positions) >= 3:
                v = (3*positions[-1] - 4*positions[-2] + positions[-3]) / 2.0
            else:
                v = positions[-1] - positions[-2]
        else:
            if len(positions) >= 3:
                v = (-3*positions[0] + 4*positions[1] - positions[2]) / 2.0
            else:
                v = positions[1] - positions[0]
        return v * fps

    def combine_trajectories(self, result, add_transitions=True):
        all_pos, all_rot, all_ts, all_focal = [], [], [], []
        current_time = 0.0
        intrinsics = CameraIntrinsics(512, 512, 256, 256, 256, 256)
        frame_mappings: List[SegmentFrameMapping] = []
        seg_arrays = []
        for seg in result.segments:
            traj_out = seg.trajectory_output
            if traj_out is not None and 'c2w' in traj_out:
                c2w = traj_out['c2w']
                fm = traj_out.get('focal_multiplier', None)
                seg_arrays.append((c2w[:, :3, 3].copy(), c2w[:, :3, :3].copy(), fm))
            else:
                pose = self.anchor_to_pose(seg.start_anchor)
                p = pose.position[np.newaxis]
                r = pose.rotation[np.newaxis]
                seg_arrays.append((p, r, None))
        n_segs = len(seg_arrays)
        for i in range(n_segs):
            pos_i, rot_i, fm_i = seg_arrays[i]
            n_i = len(pos_i)
            segment_frame_start = len(all_pos)
            for j in range(n_i):
                all_pos.append(pos_i[j].copy())
                all_rot.append(rot_i[j].copy())
                all_ts.append(current_time)
                all_focal.append(float(fm_i[j]) if fm_i is not None else 1.0)
                current_time += 1.0 / self.fps
            segment_frame_end = len(all_pos)
            seg = result.segments[i]
            obj_label = ""
            if hasattr(seg, 'start_anchor') and seg.start_anchor is not None:
                obj_label = getattr(seg.start_anchor, 'object_label', '')
            frame_mappings.append(SegmentFrameMapping(
                segment_index=i, movement_type=getattr(seg, 'movement_type', 'unknown'),
                frame_start=segment_frame_start, frame_end=segment_frame_end,
                is_transition=False, object_label=obj_label))
        if not all_pos:
            return CameraTrajectory(intrinsics=intrinsics), frame_mappings
        all_pos = np.array(all_pos)
        all_rot = np.array(all_rot)
        all_ts = np.array(all_ts)
        all_focal = np.array(all_focal, dtype=np.float64)
        poses = [CameraPose(all_pos[i], all_rot[i], all_ts[i]) for i in range(len(all_pos))]
        print(f"Combined: {len(poses)} poses, {all_ts[-1]:.1f}s")
        zoom_frames = np.sum(np.abs(all_focal - 1.0) > 1e-6)
        if zoom_frames > 0:
            print(f"  Zoom frames: {zoom_frames}/{len(all_focal)} "
                  f"(multiplier range [{all_focal.min():.2f}, {all_focal.max():.2f}])")
        traj = CameraTrajectory(poses, all_ts, all_pos, all_rot, intrinsics,
                                focal_multipliers=all_focal)
        return traj, frame_mappings

    def anchor_to_pose(self, anchor):
        pos = anchor.position
        forward = anchor.look_at - pos
        fn = np.linalg.norm(forward)
        forward = forward / fn if fn > 1e-6 else np.array([0.0, 1.0, 0.0])
        up = np.array(anchor.up, dtype=np.float64)
        if np.abs(np.dot(forward, up)) > 0.999:
            up = np.array([0, 0, 1.0]) if np.abs(forward[2]) < 0.9 else np.array([0, 1, 0.0])
        right = np.cross(forward, up)
        rn = np.linalg.norm(right)
        right = right / rn if rn > 1e-6 else np.cross(forward, np.array([1, 0, 0.0]))
        right = right / (np.linalg.norm(right) + 1e-8)
        up_c = np.cross(right, forward)
        up_c = up_c / (np.linalg.norm(up_c) + 1e-8)
        rot = np.column_stack([right, -up_c, forward])
        if np.linalg.det(rot) < 0:
            rot[:, 0] *= -1
        if np.abs(np.linalg.det(rot) - 1.0) > 0.01:
            U, _, Vt = np.linalg.svd(rot)
            rot = U @ Vt
            if np.linalg.det(rot) < 0:
                U[:, -1] *= -1
                rot = U @ Vt
        return CameraPose(pos.copy(), rot, 0.0)

    def smooth_trajectory(self, traj, window_size=5):
        if len(traj) < window_size:
            return traj
        pos = traj.positions.copy()
        kernel = np.ones(window_size) / window_size
        hw = window_size // 2
        for d in range(3):
            pos[:, d] = np.convolve(pos[:, d], kernel, mode='same')
        pos[:hw] = traj.positions[:hw]
        pos[-hw:] = traj.positions[-hw:]
        poses = [CameraPose(pos[i], traj.rotations[i], traj.timestamps[i]) for i in range(len(pos))]
        return CameraTrajectory(poses, traj.timestamps, pos, traj.rotations,
                                traj.intrinsics, traj.focal_multipliers)

    def save_anchors(self, anchors, output_path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data = {'anchors': []}
        for a in anchors:
            p = self.anchor_to_pose(a)
            T = np.eye(4); T[:3, :3] = p.rotation; T[:3, 3] = p.position
            data['anchors'].append({
                # UPDATED: use str() to handle both int and string IDs
                'object_id': str(a.object_id) if a.object_id is not None else None,
                'object_label': a.object_label,
                'score': float(a.score), 'position': a.position.tolist(),
                'look_at': a.look_at.tolist(), 'up': a.up.tolist(),
                'transform_matrix': T.tolist()})
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved {len(anchors)} anchors to {output_path}")

    def save_trajectory(self, traj, output_path, format='json'):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if len(traj) == 0:
            print("WARNING: Empty trajectory, not saving")
            return
        if format == 'npy':
            np.save(output_path, traj.get_c2w_matrices())
        elif format == 'json':
            data = {
                'w': traj.intrinsics.width if traj.intrinsics else 512,
                'h': traj.intrinsics.height if traj.intrinsics else 512,
                'fl_x': traj.intrinsics.fx if traj.intrinsics else 256,
                'fl_y': traj.intrinsics.fy if traj.intrinsics else 256,
                'cx': traj.intrinsics.cx if traj.intrinsics else 256,
                'cy': traj.intrinsics.cy if traj.intrinsics else 256,
                'frames': [],
            }
            has_focal = (traj.focal_multipliers is not None and len(traj.focal_multipliers) == len(traj.poses))
            for i, pose in enumerate(traj.poses):
                T = np.eye(4); T[:3, :3] = pose.rotation; T[:3, 3] = pose.position
                frame_data = {
                    'transform_matrix': T.tolist(),
                    'frame_id': i,
                    'timestamp': pose.timestamp,
                }
                if has_focal:
                    frame_data['focal_multiplier'] = float(traj.focal_multipliers[i])
                data['frames'].append(frame_data)
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=4)
        elif format == 'txt':
            with open(output_path, 'w') as f:
                for pose in traj.poses:
                    q = Rotation.from_matrix(pose.rotation).as_quat()
                    f.write(f"{pose.timestamp:.6f} "
                            f"{' '.join(f'{v:.6f}' for v in pose.position)} "
                            f"{' '.join(f'{v:.6f}' for v in q)}\n")
        print(f"Saved {len(traj)} poses to {output_path}")

    def save_frame_mappings(self, mappings, output_path, fps=None):
        fps = fps or self.fps
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data = []
        for m in mappings:
            data.append({
                'segment_index': m.segment_index,
                'movement_type': m.movement_type,
                'frame_start': m.frame_start,
                'frame_end': m.frame_end,
                'time_start': m.frame_start / fps,
                'time_end': m.frame_end / fps,
                'is_transition': m.is_transition,
                'object_label': m.object_label})
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved {len(data)} frame mappings to {output_path}")

    def visualize_trajectory(self, traj, output_path, title="Trajectory"):
        if len(traj) == 0:
            return
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            c2ws = traj.get_c2w_matrices()
            pos = c2ws[:, :3, 3]
            colors = np.linspace(0, 1, len(pos))
            has_focal = (traj.focal_multipliers is not None and
                         len(traj.focal_multipliers) == len(pos) and
                         np.any(np.abs(traj.focal_multipliers - 1.0) > 1e-6))
            n_cols = 4 if has_focal else 3
            fig = plt.figure(figsize=(5 * n_cols, 5))
            fig.suptitle(title, fontsize=14, fontweight='bold')
            ax1 = fig.add_subplot(1, n_cols, 1, projection='3d')
            for i in range(len(pos) - 1):
                ax1.plot3D(pos[i:i+2, 0], pos[i:i+2, 1], pos[i:i+2, 2],
                           color=plt.cm.viridis(colors[i]), linewidth=2)
            ax1.scatter(*pos[0], c='green', s=100, marker='o', label='Start')
            ax1.scatter(*pos[-1], c='red', s=100, marker='s', label='End')
            ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
            ax1.legend()
            ax2 = fig.add_subplot(1, n_cols, 2)
            ax2.scatter(pos[:, 0], pos[:, 1], c=colors, cmap='viridis', s=5)
            ax2.set_xlabel('X'); ax2.set_ylabel('Y')
            ax2.set_title('Top (XY)'); ax2.axis('equal'); ax2.grid(True, alpha=0.3)
            ax3 = fig.add_subplot(1, n_cols, 3)
            ax3.scatter(pos[:, 0], pos[:, 2], c=colors, cmap='viridis', s=5)
            ax3.set_xlabel('X'); ax3.set_ylabel('Z')
            ax3.set_title('Side (XZ)'); ax3.axis('equal'); ax3.grid(True, alpha=0.3)
            if has_focal:
                ax4 = fig.add_subplot(1, n_cols, 4)
                frames = np.arange(len(traj.focal_multipliers))
                ax4.plot(frames, traj.focal_multipliers, 'm-', linewidth=2)
                ax4.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
                ax4.set_xlabel('Frame'); ax4.set_ylabel('Focal multiplier (×)')
                ax4.set_title('Zoom'); ax4.grid(True, alpha=0.3)
            plt.tight_layout()
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved visualization to {output_path}")
        except Exception as e:
            print(f"Visualization failed: {e}")


# =============================================================================
# Utilities
# =============================================================================

def load_anchors_json(path):
    with open(path) as f:
        data = json.load(f)
    anchors = []
    for a in data.get('anchors', []):
        # Handle both string and int object_id
        obj_id = a['object_id']
        if obj_id is not None:
            try:
                obj_id = int(obj_id)
            except (ValueError, TypeError):
                pass  # Keep as string
        anchors.append(CameraAnchor(
            position=np.array(a['position']), look_at=np.array(a['look_at']),
            up=np.array(a['up']), score=a['score'],
            object_id=obj_id, object_label=a['object_label']))
    return anchors


def cast_out_outbounds(trajectory, scene_mesh):
    proximity = trimesh.proximity.ProximityQuery(scene_mesh)
    dists = proximity.signed_distance(trajectory.positions)
    mask = dists > 0.1
    print(f"Kept {mask.sum()}/{len(dists)} poses inside mesh")
    return CameraTrajectory(
        poses=[p for p, k in zip(trajectory.poses, mask) if k],
        timestamps=trajectory.timestamps[mask],
        positions=trajectory.positions[mask],
        rotations=trajectory.rotations[mask],
        intrinsics=trajectory.intrinsics,
        focal_multipliers=trajectory.focal_multipliers[mask] if trajectory.focal_multipliers is not None else None)


def _remap_subtitle_timings(sub_track, result, frame_mappings, fps):
    if not hasattr(sub_track, 'entries') or not sub_track.entries:
        return
    if not frame_mappings:
        return
    transition_frames = 0
    for e in sub_track.entries:
        if e.movement_type == '_transition_interp':
            transition_frames = e.end_frame - e.start_frame
            break
    transition_mappings = [m for m in frame_mappings if m.is_transition]
    segment_mappings = [m for m in frame_mappings if not m.is_transition]
    trans_idx = 0
    seg_idx = 0
    for entry in sub_track.entries:
        if entry.movement_type == '_transition_interp':
            if trans_idx < len(transition_mappings):
                m = transition_mappings[trans_idx]
                entry.start_frame = m.frame_start
                entry.end_frame = m.frame_end
                trans_idx += 1
        else:
            if seg_idx < len(segment_mappings):
                m = segment_mappings[seg_idx]
                entry.start_frame = m.frame_start
                entry.end_frame = m.frame_end
                seg_idx += 1
    if frame_mappings:
        sub_track.total_frames = max(m.frame_end for m in frame_mappings)


# =============================================================================
# Main  — UPDATED: uses PLY mesh instead of USD
# =============================================================================

def main():
    llm_output = {
        "atomic_trajectories": (
            "1. Call Anchor Determinator with 'door' (id: door_0). 2. Call AtomTraj with 'pan_right' (object-level). 3. Call Anchor Determinator with 'kitchen_counter' (id: kitchen_counter_0). 4. Call AtomTraj with 'arc', angle=0 (transitional). 5. Call AtomTraj with 'orbit_quarter' (object-level). 6. Call Anchor Determinator with 'refrigerator' (id: refrigerator_0). 7. Call AtomTraj with 'arc', angle=30 (transitional). 8. Call AtomTraj with 'orbit_quarter' (object-level). 9. Call Anchor Determinator with 'window' (id: window_0). 10. Call AtomTraj with 'arc', angle=45 (transitional). 11. Call AtomTraj with 'pan_left' (object-level). 12. Call Anchor Determinator with 'sofa' (id: sofa_0). 13. Call AtomTraj with 'arc', angle=60 (transitional). 14. Call AtomTraj with 'orbit_quarter' (object-level). 15. Call Anchor Determinator with 'table' (id: table_0). 16. Call AtomTraj with 'arc', angle=-30 (transitional). 17. Call AtomTraj with 'orbit_quarter' (object-level). 18. Call Anchor Determinator with 'tv' (id: tv_0). 19. Call AtomTraj with 'arc', angle=45 (transitional). 20. Call AtomTraj with 'pan_right' (object-level). 21. Call Anchor Determinator with 'picture' (id: picture_3). 22. Call AtomTraj with 'arc', angle=30 (transitional). 23. Call AtomTraj with 'zoom_in_out' (object-level). 24. Call traj_compose. 25. Render video."
        )
    }

    scene_id = "09c1414f1b"
    output_dir = Path(f"outputs/scannetpp/{scene_id}")
    output_dir.mkdir(parents=True, exist_ok=True)

    bbox_path = f"data/ScanNetpp/scenes/{scene_id}/dslr/sg/{scene_id}-simple.json"
    # UPDATED: use PLY mesh built by build_obb_mesh.py instead of USD
    mesh_path = f"outputs/scannetpp/{scene_id}/obb_mesh.ply"

    executor = TrajectoryExecutor(
        project_root=".",
        bbox_path=bbox_path, mesh_path=mesh_path)

    combiner = TrajectoryCombiner(transition_frames=30, fps=30.0)

    result = executor.execute_from_llm_output(llm_output, base_name="executor_results")

    ply_path = f"data/ScanNetpp/scenes/{scene_id}/dslr/ply/point_cloud.ply"

    result = optimize_trajectory_result(
        result, executor, ply_path, executor.mesh,
        sdf_resolution=128, sdf_cache_dir=str(output_dir), scene_name=scene_id, verbose=True)

    combined, frame_mappings = combiner.combine_trajectories(result)
    smoothed = combiner.smooth_trajectory(combined, window_size=5)

    combiner.save_trajectory(smoothed, output_dir / "combined_trajectory.json")
    combiner.save_anchors(result.all_anchors, output_dir / "anchors.json")
    combiner.save_frame_mappings(frame_mappings, output_dir / "frame_mappings.json")
    combiner.visualize_trajectory(smoothed, output_dir / "combined_trajectory.png", title="Living Room Tour")

    print(f"\nFinal: {len(smoothed)} poses, {smoothed.timestamps[-1]:.1f}s")
    if smoothed.focal_multipliers is not None:
        zoom_count = np.sum(np.abs(smoothed.focal_multipliers - 1.0) > 1e-6)
        print(f"  Zoom frames: {zoom_count}/{len(smoothed)}")


if __name__ == "__main__":
    main()