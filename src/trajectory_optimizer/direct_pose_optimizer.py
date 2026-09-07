"""
Direct Pose Optimizer (Non-Parametric)
=======================================

Optimizes raw c2w camera trajectories using the same SDF-based collision
and occlusion costs as the parametric optimizer, but WITHOUT re-creating
trajectories through parametric classes (CircularOrbit, DollyMove, etc.).

This is used for the "w/o Parametric Trajectories" ablation where GenDoP
produces raw 6-DoF pose sequences. The parametric optimizer cannot be used
because there is no parametric object to extract/modify parameters from.

Approach:
  - Treat the N camera positions as free parameters (nn.Parameter)
  - Apply the same cost functions: mesh SDF, occlusion, smoothness, boundary
  - Optimize via Adam with gradient clipping
  - Rotations are NOT optimized (kept fixed) — only positions move

This preserves the SDF-based optimization quality while being compatible
with any trajectory source (parametric, GenDoP, or hand-crafted).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
from pathlib import Path


class DirectPoseOptimizer:
    """
    Optimizes camera positions directly using SDF collision + occlusion costs.

    Unlike ParametricTrajectoryOptimizer which optimizes a few parametric
    variables (radius, angle, height), this optimizes all N×3 position
    coordinates simultaneously while keeping rotations fixed.
    """

    def __init__(self, density_field, config=None, device='cuda'):
        """
        Args:
            density_field: GaussianDensityField instance (same as parametric optimizer).
            config: ParametricOptimizationConfig (reuses the same config dataclass).
            device: torch device string.
        """
        from src.trajectory_optimizer.trajectory_optimizer import (
            ParametricOptimizationConfig,
        )

        self.density_field = density_field
        self.config = config or ParametricOptimizationConfig()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.scene_min = torch.from_numpy(density_field.scene_min).float().to(self.device)
        self.scene_max = torch.from_numpy(density_field.scene_max).float().to(self.device)

        # Occlusion target (set per-segment)
        self._target_center_zup: Optional[torch.Tensor] = None
        self._target_obb_center: Optional[torch.Tensor] = None
        self._target_obb_axes: Optional[torch.Tensor] = None
        self._target_obb_half_extents: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Cost functions (same as ParametricTrajectoryOptimizer)
    # ------------------------------------------------------------------

    def _mesh_sdf_cost(self, pos_zup):
        if self.density_field.diff_mesh_sdf is None:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        sdf = self.density_field.query_mesh_sdf(pos_zup)
        violation = F.relu(self.config.safety_margin - sdf)
        cost = (violation + violation ** 2).mean()
        return cost * self.config.mesh_sdf_weight

    def _smoothness_cost(self, pos):
        if len(pos) < 4:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        vel = pos[1:] - pos[:-1]
        acc = vel[1:] - vel[:-1]
        jerk = acc[1:] - acc[:-1]
        return (jerk ** 2).sum(dim=-1).mean() * self.config.smoothness_weight

    def _boundary_cost(self, pos_zup):
        margin = self.config.safety_margin
        lower_v = self.scene_min + margin - pos_zup
        upper_v = pos_zup - self.scene_max + margin
        lower_c = torch.where(lower_v > 0, lower_v ** 2 + torch.exp(lower_v) - 1, torch.zeros_like(lower_v))
        upper_c = torch.where(upper_v > 0, upper_v ** 2 + torch.exp(upper_v) - 1, torch.zeros_like(upper_v))
        return (lower_c.sum(-1) + upper_c.sum(-1)).mean() * self.config.boundary_weight

    def _regularization_cost(self, pos, original_pos):
        """Penalize deviation from original positions."""
        diff = pos - original_pos
        return (diff ** 2).sum(dim=-1).mean() * self.config.parameter_regularization

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
        if self.density_field.diff_mesh_sdf is None:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        target = self._target_center_zup
        N = pos_zup.shape[0]
        n_steps = self.config.occlusion_n_steps
        margin = self.config.occlusion_margin
        near_skip = self.config.occlusion_near_skip

        ray_vec = target.unsqueeze(0) - pos_zup
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
        sdf_vals = self.density_field.diff_mesh_sdf.query_sdf(ray_points_flat)
        sdf_vals = sdf_vals.reshape(N, n_steps)

        if self._target_obb_axes is not None and self._target_obb_half_extents is not None:
            obb_dist = self._point_obb_distance(ray_points_flat)
            obb_dist = obb_dist.reshape(N, n_steps)
            exclusion_margin = self._target_obb_half_extents.max().item() * 0.5
            target_mask = (obb_dist < exclusion_margin).float()
            sdf_vals = sdf_vals + target_mask * 10.0

        per_step_penalty = F.softplus(margin - sdf_vals, beta=20.0)

        cumulative = torch.cumsum(per_step_penalty, dim=1)
        shifted_cumulative = torch.cat([
            torch.zeros(N, 1, device=self.device),
            cumulative[:, :-1]
        ], dim=1)
        transmittance = torch.exp(-shifted_cumulative)
        occlusion_per_camera = (transmittance * per_step_penalty).sum(dim=1)

        return occlusion_per_camera.mean() * self.config.occlusion_weight

    def _ray_obb_t_enter(self, ray_origins, ray_vecs):
        if self._target_obb_axes is None or self._target_obb_half_extents is None:
            return (1.0 - self.config.occlusion_target_shrink) * torch.ones(
                ray_origins.shape[0], 1, device=self.device)

        obb_center = self._target_obb_center if self._target_obb_center is not None else self._target_center_zup
        obb_axes = self._target_obb_axes
        half_ext = self._target_obb_half_extents

        origin_local = (ray_origins - obb_center.unsqueeze(0)) @ obb_axes
        dir_local = ray_vecs @ obb_axes

        inv_dir = 1.0 / (dir_local + 1e-10)
        t1 = (-half_ext.unsqueeze(0) - origin_local) * inv_dir
        t2 = (half_ext.unsqueeze(0) - origin_local) * inv_dir

        t_slab_min = torch.minimum(t1, t2)
        t_slab_max = torch.maximum(t1, t2)

        t_enter = t_slab_min.max(dim=-1, keepdim=True).values
        t_exit = t_slab_max.min(dim=-1, keepdim=True).values

        valid = (t_enter < t_exit) & (t_enter > 0)
        t_enter_safe = t_enter - 0.02
        fallback = 1.0 - self.config.occlusion_target_shrink
        return torch.where(valid, t_enter_safe, torch.full_like(t_enter, fallback))

    def _count_occluded(self, pos_zup):
        if self._target_center_zup is None or self.density_field.diff_mesh_sdf is None:
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
            sdf_vals = self.density_field.diff_mesh_sdf.query_sdf(ray_points_flat).reshape(N, n_steps)
            if self._target_obb_axes is not None and self._target_obb_half_extents is not None:
                obb_dist = self._point_obb_distance(ray_points_flat).reshape(N, n_steps)
                exclusion_margin = self._target_obb_half_extents.max().item() * 0.5
                near_target = (obb_dist < exclusion_margin)
                sdf_vals = sdf_vals.clone()
                sdf_vals[near_target] = 1.0
            occluded_mask = (sdf_vals < 0).any(dim=1)
            return int(occluded_mask.sum().item())

    # ------------------------------------------------------------------
    # Set target for occlusion
    # ------------------------------------------------------------------

    def set_target(self, target_center_zup=None, target_obb_center=None,
                   target_obb_axes=None, target_obb_half_extents=None):
        def _to_tensor(v):
            if v is None:
                return None
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v).float()
            return v.to(self.device)

        self._target_center_zup = _to_tensor(target_center_zup)
        self._target_obb_center = _to_tensor(target_obb_center)
        self._target_obb_axes = _to_tensor(target_obb_axes)
        self._target_obb_half_extents = _to_tensor(target_obb_half_extents)

    # ------------------------------------------------------------------
    # Main optimize method
    # ------------------------------------------------------------------

    def optimize_positions(
        self,
        positions_zup: np.ndarray,
        verbose: bool = True,
        pin_endpoints: bool = True,
    ) -> np.ndarray:
        """
        Optimize N×3 camera positions using SDF collision + occlusion costs.

        Args:
            positions_zup: (N, 3) camera positions in Z-up scene coordinates.
            verbose: Print iteration logs.
            pin_endpoints: If True, first and last positions are frozen.

        Returns:
            (N, 3) optimized positions.
        """
        N = len(positions_zup)
        if N < 2:
            return positions_zup.copy()

        original = torch.from_numpy(positions_zup.astype(np.float32)).to(self.device)

        # Check if optimization is needed
        with torch.no_grad():
            n_collisions = self.density_field.query_mesh_collision_count(
                original, self.config.safety_margin)
            n_occluded = self._count_occluded(original)

        if n_collisions == 0 and n_occluded == 0:
            if verbose:
                print("    Already collision-free and unoccluded, skipping")
            return positions_zup.copy()

        if verbose:
            print(f"    Initial: {n_collisions}/{N} collisions, "
                  f"{n_occluded}/{N} occluded")

        # Set up optimizable positions
        # Subsample for efficiency (optimize every k-th, interpolate rest)
        n_samples = min(N, self.config.n_samples)
        if n_samples < N:
            sample_indices = np.linspace(0, N - 1, n_samples).astype(int)
        else:
            sample_indices = np.arange(N)

        sampled_pos = original[sample_indices].clone()
        opt_pos = nn.Parameter(sampled_pos.clone())

        # Pin endpoints
        if pin_endpoints:
            endpoint_indices = [0, len(sample_indices) - 1]
            endpoint_values = sampled_pos[endpoint_indices].clone()

        optimizer = torch.optim.Adam([opt_pos], lr=self.config.learning_rate * 0.1)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=30)

        best_cost = float('inf')
        best_pos = opt_pos.data.clone()
        history = []

        for it in range(self.config.n_iterations):
            optimizer.zero_grad()

            # Enforce pinned endpoints
            if pin_endpoints:
                with torch.no_grad():
                    opt_pos.data[endpoint_indices] = endpoint_values

            pos = opt_pos

            c_mesh = self._mesh_sdf_cost(pos)
            c_smooth = self._smoothness_cost(pos)
            c_reg = self._regularization_cost(pos, sampled_pos)
            c_bound = self._boundary_cost(pos)
            c_occl = self._occlusion_cost(pos)

            total = c_mesh + c_smooth + c_reg + c_bound + c_occl

            total.backward()
            torch.nn.utils.clip_grad_norm_([opt_pos], max_norm=0.5)
            optimizer.step()
            scheduler.step(total)

            # Enforce pinned endpoints after step
            if pin_endpoints:
                with torch.no_grad():
                    opt_pos.data[endpoint_indices] = endpoint_values

            cost_val = total.item()
            history.append(cost_val)
            if cost_val < best_cost:
                best_cost = cost_val
                best_pos = opt_pos.data.clone()

            if verbose and it % 50 == 0:
                with torch.no_grad():
                    n_coll = self.density_field.query_mesh_collision_count(
                        opt_pos, self.config.safety_margin)
                    n_occ = self._count_occluded(opt_pos)
                print(f"    Iter {it:4d}: total={cost_val:.4f}, "
                      f"mesh_sdf={c_mesh.item():.4f}, occl={c_occl.item():.4f}, "
                      f"smooth={c_smooth.item():.4f}, reg={c_reg.item():.4f}, "
                      f"bound={c_bound.item():.4f}, "
                      f"coll={n_coll}/{n_samples}, occ={n_occ}/{n_samples}")

            # Early stopping
            with torch.no_grad():
                n_coll = self.density_field.query_mesh_collision_count(
                    opt_pos, self.config.safety_margin)
                n_occ = self._count_occluded(opt_pos)
            if n_coll == 0 and n_occ == 0:
                if verbose:
                    print(f"    Converged at iter {it}")
                break

            if len(history) > 20 and max(history[-20:]) - min(history[-20:]) < self.config.convergence_threshold:
                if verbose:
                    print(f"    Plateaued at iter {it}, accepting {n_coll} coll, {n_occ} occ")
                break

        # Use best result
        with torch.no_grad():
            opt_pos.data = best_pos

        # Interpolate back to full N if subsampled
        optimized_sampled = opt_pos.detach().cpu().numpy()
        if n_samples < N:
            optimized_full = positions_zup.copy()
            # Place optimized samples
            for i, idx in enumerate(sample_indices):
                optimized_full[idx] = optimized_sampled[i]
            # Linearly interpolate between samples
            for i in range(len(sample_indices) - 1):
                start_idx = sample_indices[i]
                end_idx = sample_indices[i + 1]
                if end_idx - start_idx > 1:
                    for j in range(start_idx + 1, end_idx):
                        t = (j - start_idx) / (end_idx - start_idx)
                        optimized_full[j] = (
                            (1 - t) * optimized_full[start_idx]
                            + t * optimized_full[end_idx]
                        )
            return optimized_full
        else:
            return optimized_sampled


# =============================================================================
# High-level: optimize a TrajectoryResult with direct pose optimization
# =============================================================================

def optimize_trajectory_result_direct(
    result,
    executor,
    ply_path: str,
    mesh=None,
    config=None,
    device='cuda',
    verbose=True,
    sdf_resolution=64,
    sdf_cache_dir=None,
    scene_name=None,
    scene_json=None,
):
    """
    Optimize trajectories using direct pose optimization (no parametric re-creation).

    Uses the same SDF collision + occlusion costs as the parametric optimizer,
    but treats camera positions as free variables instead of re-parameterizing
    through CircularOrbit/DollyMove/etc.

    Drop-in replacement for optimize_trajectory_result() when trajectories
    come from GenDoP or other non-parametric sources.
    """
    from src.trajectory_optimizer.trajectory_optimizer import (
        ParametricOptimizationConfig,
        GaussianDensityField,
        load_ply_3dgs,
        _get_target_center_zup,
        _get_last_pose_from_output,
        _get_first_pose_from_output,
    )

    config = config or ParametricOptimizationConfig()

    if verbose:
        print("\n" + "=" * 60)
        print("DIRECT POSE OPTIMIZATION (SDF collision + occlusion)")
        print("  (non-parametric — positions as free variables)")
        print("=" * 60)

    positions, scales, opacities = load_ply_3dgs(ply_path)
    density_field = GaussianDensityField(
        positions, scales, opacities, config, device,
        mesh=mesh, sdf_resolution=sdf_resolution,
        sdf_cache_dir=sdf_cache_dir, scene_name=scene_name,
    )

    optimizer = DirectPoseOptimizer(density_field, config, device)

    # ==================================================================
    # Optimize all segments (object-level and transitional)
    # ==================================================================

    for i, segment in enumerate(result.segments):
        traj_out = segment.trajectory_output
        if traj_out is None or "c2w" not in traj_out:
            continue

        c2w = traj_out["c2w"]
        positions_zup = c2w[:, :3, 3].copy()
        rotations = c2w[:, :3, :3].copy()
        n_frames = len(positions_zup)

        label = ""
        if segment.start_anchor:
            label = getattr(segment.start_anchor, "object_label", "")

        if verbose:
            cat = segment.movement_category
            print(f"\n--- [{i}] {segment.movement_type} ({cat}) @ {label} ---")

        # Set occlusion target for object-level segments
        if segment.movement_category == 'object-level':
            target_look_at, target_obb_center, target_obb_axes, target_obb_half_extents = \
                _get_target_center_zup(segment, executor)
            optimizer.set_target(
                target_center_zup=target_look_at,
                target_obb_center=target_obb_center,
                target_obb_axes=target_obb_axes,
                target_obb_half_extents=target_obb_half_extents,
            )
            if verbose and target_look_at is not None:
                print(f"  Target: [{target_look_at[0]:.2f}, "
                      f"{target_look_at[1]:.2f}, {target_look_at[2]:.2f}]")
        else:
            # Transitional: no specific occlusion target
            optimizer.set_target(None)

        # Skip stationary segments (pan, tilt, zoom — single position)
        pos_range = positions_zup.max(axis=0) - positions_zup.min(axis=0)
        if pos_range.max() < 1e-4:
            if verbose:
                print(f"  Stationary segment — skipping position optimization")
            continue

        # Optimize
        optimized_positions = optimizer.optimize_positions(
            positions_zup,
            verbose=verbose,
            pin_endpoints=True,
        )

        # Update c2w with optimized positions (keep rotations unchanged)
        new_c2w = c2w.copy()
        new_c2w[:, :3, 3] = optimized_positions
        traj_out["c2w"] = new_c2w
        traj_out["positions"] = optimized_positions
        traj_out["rotations_matrix"] = rotations
        traj_out["optimized"] = True
        traj_out["optimization_method"] = "direct_pose_sdf"

        if verbose:
            max_shift = np.linalg.norm(optimized_positions - positions_zup, axis=1).max()
            print(f"  ✓ Max position shift: {max_shift:.3f}m")

    return result