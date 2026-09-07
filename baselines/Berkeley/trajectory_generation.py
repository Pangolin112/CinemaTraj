"""
Trajectory Generation and Refinement Module
=============================================
Generates smooth camera trajectories between keyframes via interpolation,
then refines them using point cloud collision avoidance (KD-tree).

No nerfstudio dependency. Collision checking uses COLMAP/3DGS point clouds
or mesh vertices loaded via trimesh/plyfile.

Reference: Section 3.4 of Wu (2025)
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import CubicSpline


class PointCloudCollisionChecker:
    """
    Collision checker using a scene point cloud (from COLMAP, 3DGS, or mesh).
    Uses a KD-tree for efficient nearest-neighbor distance queries.
    """

    def __init__(self, points, safety_margin=0.3):
        self.points = points
        self.safety_margin = safety_margin
        self._tree = None

    def build_kdtree(self):
        from scipy.spatial import KDTree
        if len(self.points) > 0:
            self._tree = KDTree(self.points)
            print(f"    Built KD-tree with {len(self.points)} points")

    def query_clearance(self, position):
        if self._tree is None:
            self.build_kdtree()
        if self._tree is None or len(self.points) == 0:
            return float("inf")
        dist, _ = self._tree.query(position)
        return float(dist)

    def query_clearance_batch(self, positions):
        if self._tree is None:
            self.build_kdtree()
        if self._tree is None or len(self.points) == 0:
            return np.full(len(positions), float("inf"))
        dists, _ = self._tree.query(positions)
        return dists

    def check_trajectory(self, positions):
        clearances = self.query_clearance_batch(positions)
        violations = clearances < self.safety_margin
        return clearances, violations

    def find_nearest_direction(self, position, k=10):
        if self._tree is None:
            self.build_kdtree()
        if self._tree is None or len(self.points) == 0:
            return np.array([0, 0, 0], dtype=float)
        k = min(k, len(self.points))
        dists, indices = self._tree.query(position, k=k)
        nearest_points = self.points[indices]
        weights = 1.0 / (dists + 1e-8)
        direction = np.sum(
            (nearest_points - position) * weights[:, np.newaxis], axis=0
        )
        norm = np.linalg.norm(direction)
        if norm > 1e-8:
            direction /= norm
        return direction


class TrajectoryGenerator:
    """
    Generates and refines camera trajectories between keyframes.

    Pipeline:
        1. Interpolate positions (cubic spline) and rotations (SLERP)
        2. Check clearance using point cloud KD-tree
        3. Iteratively push positions away from nearby geometry
        4. Re-smooth the trajectory after corrections

    Keyframe positions and orientations are always preserved exactly.
    Only interpolated frames between keyframes are modified during refinement.
    """

    def __init__(
        self,
        point_cloud=None,
        safety_threshold=0.3,
        max_refinement_iters=50,
        num_interpolation_steps=30,
        device="cuda",
        # Legacy kwargs for backward compat (ignored)
        nerf_model_dir=None,
        **kwargs,
    ):
        self.safety_threshold = safety_threshold
        self.max_refinement_iters = max_refinement_iters
        self.num_interpolation_steps = num_interpolation_steps
        self.device = device
        self.pc_checker = None
        self.density_querier = None  # kept as None for API compat

        if nerf_model_dir is not None:
            print("    Note: nerf_model_dir is ignored (nerfstudio removed). "
                  "Use point_cloud or --point_cloud_path instead.")

        if point_cloud is not None and len(point_cloud) > 0:
            self.pc_checker = PointCloudCollisionChecker(point_cloud, safety_threshold)

    def interpolate_trajectory(self, positions, rotations, num_steps=None):
        """
        Generate a smooth trajectory by interpolating between keyframes.
        Uses cubic spline for positions and SLERP for rotations.
        """
        if num_steps is None:
            num_steps = self.num_interpolation_steps

        K = len(positions)
        if K < 2:
            return {
                "positions": positions,
                "rotations": rotations,
                "keyframe_indices": [0],
            }

        t_keyframes = np.arange(K, dtype=float)
        total_frames = (K - 1) * num_steps + 1
        t_interp = np.linspace(0, K - 1, total_frames)

        cs_x = CubicSpline(t_keyframes, positions[:, 0], bc_type="clamped")
        cs_y = CubicSpline(t_keyframes, positions[:, 1], bc_type="clamped")
        cs_z = CubicSpline(t_keyframes, positions[:, 2], bc_type="clamped")

        interp_positions = np.stack([
            cs_x(t_interp), cs_y(t_interp), cs_z(t_interp),
        ], axis=-1)

        key_rots = Rotation.from_matrix(rotations)
        slerp = Slerp(t_keyframes, key_rots)
        interp_rots = slerp(t_interp)
        interp_rotmats = interp_rots.as_matrix()

        keyframe_indices = [int(round(t * num_steps)) for t in range(K)]

        return {
            "positions": interp_positions,
            "rotations": interp_rotmats,
            "keyframe_indices": keyframe_indices,
        }

    def refine_trajectory(self, trajectory, points3d=None):
        """
        Refine trajectory for collision avoidance using point cloud KD-tree.

        For each non-keyframe camera position, checks distance to nearest
        scene geometry. If below safety_threshold, pushes position away
        from the obstacle.

        Keyframe positions and orientations are never modified.
        """
        positions = trajectory["positions"].copy()
        rotations = trajectory["rotations"].copy()
        keyframe_indices = trajectory.get("keyframe_indices", [])
        keyframe_set = set(keyframe_indices)
        N = len(positions)

        # Save original keyframe poses to guarantee they are never changed
        keyframe_positions_orig = {i: positions[i].copy() for i in keyframe_set}
        keyframe_rotations_orig = {i: rotations[i].copy() for i in keyframe_set}

        if self.pc_checker is None and points3d is not None:
            self.pc_checker = PointCloudCollisionChecker(points3d, self.safety_threshold)

        if self.pc_checker is None:
            print("    No collision geometry available, skipping refinement")
            return {
                "positions": positions,
                "rotations": rotations,
                "keyframe_indices": keyframe_indices,
                "num_corrections": 0,
            }

        if self.pc_checker._tree is None:
            self.pc_checker.build_kdtree()

        total_corrections = 0

        for iteration in range(self.max_refinement_iters):
            corrections_this_iter = 0
            clearances = self.pc_checker.query_clearance_batch(positions)

            for i in range(N):
                # Never modify keyframe positions
                if i in keyframe_set:
                    continue

                if clearances[i] < self.safety_threshold:
                    obstacle_dir = self.pc_checker.find_nearest_direction(positions[i])
                    penetration_depth = self.safety_threshold - clearances[i]
                    correction_vector = -obstacle_dir * penetration_depth * 1.2
                    positions[i] = positions[i] + correction_vector
                    corrections_this_iter += 1
                    total_corrections += 1

            if corrections_this_iter == 0:
                print(f"    Converged after {iteration + 1} iterations")
                break

            if (iteration + 1) % 10 == 0:
                print(f"    Iteration {iteration + 1}: {corrections_this_iter} corrections")

            # Smooth only non-keyframe positions
            positions = self._smooth_positions(
                positions,
                keyframe_indices=keyframe_indices,
                window_size=5,
            )

        # Smooth rotations for non-keyframe frames only
        if total_corrections > 0:
            rotations = self._smooth_rotations(
                positions, rotations,
                keyframe_indices=keyframe_indices,
            )

        # Restore keyframe poses exactly (safety guarantee)
        for i in keyframe_set:
            positions[i] = keyframe_positions_orig[i]
            rotations[i] = keyframe_rotations_orig[i]

        return {
            "positions": positions,
            "rotations": rotations,
            "keyframe_indices": keyframe_indices,
            "num_corrections": total_corrections,
        }

    def _smooth_positions(self, positions, keyframe_indices, window_size=5):
        """Gaussian-weighted smoothing of positions, skipping keyframes."""
        smoothed = positions.copy()
        N = len(positions)
        half_w = window_size // 2
        keyframe_set = set(keyframe_indices)

        for i in range(N):
            if i in keyframe_set:
                continue
            start = max(0, i - half_w)
            end = min(N, i + half_w + 1)
            indices = np.arange(start, end)
            weights = np.exp(-0.5 * ((indices - i) / (half_w / 2.0)) ** 2)
            weights /= weights.sum()
            smoothed[i] = np.sum(positions[start:end] * weights[:, np.newaxis], axis=0)

        return smoothed

    def _smooth_rotations(self, positions, rotations, keyframe_indices=None):
        """
        Smooth rotations based on trajectory direction, blended with
        keyframe-interpolated rotations for continuity.

        For non-keyframe frames:
          1. Compute a direction-based rotation (look along local trajectory)
          2. SLERP between the two enclosing keyframe rotations at this t
          3. Blend the two using a cosine weight: near keyframes the SLERP
             result dominates, at the segment midpoint the direction-based
             rotation dominates.

        This eliminates the abrupt orientation snap at keyframe boundaries
        that occurred when direction-based rotations were applied uniformly.

        Keyframe orientations are never modified.
        """
        N = len(positions)
        smoothed_rots = rotations.copy()
        kf_list = sorted(keyframe_indices or [])
        keyframe_set = set(kf_list)

        if len(kf_list) == 0:
            kf_list = [0, N - 1]

        for i in range(1, N - 1):
            if i in keyframe_set:
                continue

            # --- compute direction-based rotation ---
            forward = positions[i + 1] - positions[i - 1]
            forward_norm = np.linalg.norm(forward)
            if forward_norm < 1e-8:
                continue
            forward = forward / forward_norm

            original_up = rotations[i][:, 1]
            right = np.cross(forward, original_up)
            right_norm = np.linalg.norm(right)
            if right_norm < 1e-8:
                continue
            right = right / right_norm
            up = np.cross(right, forward)
            up = up / np.linalg.norm(up)
            direction_rot = np.stack([right, up, -forward], axis=-1)

            # --- find enclosing keyframe segment ---
            prev_kf = kf_list[0]
            next_kf = kf_list[-1]
            for k in kf_list:
                if k <= i:
                    prev_kf = k
                if k >= i:
                    next_kf = k
                    break

            seg_len = next_kf - prev_kf
            if seg_len <= 0:
                smoothed_rots[i] = direction_rot
                continue

            # t in [0, 1] within this keyframe segment
            t = (i - prev_kf) / seg_len

            # SLERP between the two enclosing keyframe rotations
            r_prev = Rotation.from_matrix(rotations[prev_kf])
            r_next = Rotation.from_matrix(rotations[next_kf])
            kf_slerp = Slerp([0.0, 1.0], Rotation.concatenate([r_prev, r_next]))
            kf_rot = kf_slerp(t).as_matrix()

            # Blend weight: 0 at keyframes (use keyframe SLERP), 1 at midpoint
            # (use direction-based). Smooth cosine ramp: 0 → 1 → 0 over segment.
            blend = 0.5 * (1.0 - np.cos(2.0 * np.pi * t))

            # SLERP between keyframe-interpolated and direction-based rotation
            r_kf = Rotation.from_matrix(kf_rot)
            r_dir = Rotation.from_matrix(direction_rot)
            blend_slerp = Slerp([0.0, 1.0], Rotation.concatenate([r_kf, r_dir]))
            smoothed_rots[i] = blend_slerp(blend).as_matrix()

        return smoothed_rots