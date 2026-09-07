from __future__ import annotations

"""
Trajectory Composer Module
============================
Combines atomic trajectories from CineGPT with anchor points from the 
Anchor Determinator to form the final camera trajectory.

Algorithm:
    1. Validate input: trajectories and anchors must alternate (no adjacent anchors).
    2. Merge adjacent trajectories via Euclidean (6-DoF) transforms so each
       merged trajectory's start coincides with the previous one's end.
    3. Align trajectories to their neighboring anchors:
       - Two adjacent anchors → similarity transform (7-DoF: rotation + translation + uniform scale)
       - One adjacent anchor  → Euclidean transform (6-DoF: rotation + translation)

Reference: Section 3.3 (Final Trajectory Composition) of Liu et al. (2024)
"""

import numpy as np
from typing import Dict, List, Optional, Union
from scipy.spatial.transform import Rotation, Slerp


class TrajectoryComposer:
    """
    Composes final camera trajectories from an interleaved sequence of
    sub-trajectories and anchor points.
    """

    def __init__(
        self,
        blend_frames: int = 15,
        total_frames: int = 120,
    ):
        self.blend_frames = blend_frames
        self.total_frames = total_frames

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compose(
        self,
        elements: List[dict],
    ) -> dict:
        """
        Compose a final trajectory from an interleaved list of trajectory
        segments and anchor points.

        Args:
            elements: Ordered list where each item is either
                - a *trajectory*  dict with key ``'c2ws'`` of shape (N, 4, 4)
                - an *anchor* dict with key ``'c2w'`` of shape (4, 4)
              The two types must not have two consecutive anchors.

        Returns:
            Dict with:
                - c2ws:       (N, 4, 4) composed trajectory
                - intrinsics: from the first trajectory found (or None)
                - num_frames: total number of frames
        """
        if not elements:
            return {"c2ws": None, "intrinsics": None, "num_frames": 0}

        # ----- Step 0: classify each element -----
        typed: List[dict] = []
        for elem in elements:
            if "c2ws" in elem:
                typed.append({"type": "traj", "data": elem})
            elif "c2w" in elem:
                typed.append({"type": "anchor", "data": elem})
            else:
                raise ValueError(
                    "Each element must contain either 'c2ws' (trajectory) "
                    "or 'c2w' (anchor)."
                )

        # ----- Step 1: validate – no adjacent anchors -----
        self._validate_no_adjacent_anchors(typed)

        # ----- Step 2: merge adjacent trajectories -----
        merged = self._merge_adjacent_trajectories(typed)

        # ----- Step 3: align trajectories to neighboring anchors -----
        final_c2ws = self._align_to_anchors(merged)

        # Collect intrinsics from the first trajectory
        intrinsics = None
        for elem in elements:
            if "c2ws" in elem and "intrinsics" in elem:
                intrinsics = elem["intrinsics"]
                break

        return {
            "c2ws": final_c2ws,
            "intrinsics": intrinsics,
            "num_frames": len(final_c2ws),
        }

    # ------------------------------------------------------------------
    # Step 1 – validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_no_adjacent_anchors(typed: List[dict]) -> None:
        """Raise if two anchors appear consecutively."""
        for i in range(len(typed) - 1):
            if typed[i]["type"] == "anchor" and typed[i + 1]["type"] == "anchor":
                raise ValueError(
                    f"Illegal input: adjacent anchors at positions {i} and {i+1}. "
                    "Anchors must be separated by at least one trajectory."
                )

    # ------------------------------------------------------------------
    # Step 2 – merge adjacent trajectories
    # ------------------------------------------------------------------

    def _merge_adjacent_trajectories(self, typed: List[dict]) -> List[dict]:
        """
        Walk through the typed element list and merge consecutive trajectory
        entries into a single trajectory using 6-DoF Euclidean alignment.

        Returns a new list where no two consecutive elements are trajectories.
        """
        merged: List[dict] = []

        for elem in typed:
            if elem["type"] == "traj":
                if merged and merged[-1]["type"] == "traj":
                    # Merge into the previous trajectory
                    prev_c2ws = merged[-1]["c2ws"]
                    curr_c2ws = np.array(elem["data"]["c2ws"], dtype=np.float64)
                    assert curr_c2ws.ndim == 3 and curr_c2ws.shape[1:] == (4, 4)

                    # Euclidean transform: align curr start -> prev end
                    prev_end = prev_c2ws[-1]
                    aligned = self._apply_euclidean_align(
                        curr_c2ws, source_c2w=curr_c2ws[0], target_c2w=prev_end
                    )
                    merged[-1]["c2ws"] = np.concatenate(
                        [prev_c2ws, aligned], axis=0
                    )
                else:
                    c2ws = np.array(elem["data"]["c2ws"], dtype=np.float64)
                    assert c2ws.ndim == 3 and c2ws.shape[1:] == (4, 4)
                    merged.append({"type": "traj", "c2ws": c2ws})
            else:
                # anchor – just pass through
                c2w = np.array(elem["data"]["c2w"], dtype=np.float64)
                assert c2w.shape == (4, 4)
                merged.append({"type": "anchor", "c2w": c2w})

        return merged

    # ------------------------------------------------------------------
    # Step 3 – align trajectories to neighboring anchors
    # ------------------------------------------------------------------

    def _align_to_anchors(self, merged: List[dict]) -> np.ndarray:
        """
        For every trajectory in *merged*, align it to its neighboring anchors.

        The alignment strategy preserves anchor positions AND orientations:

        **Positions**: The trajectory's positions are re-parameterized so
        that the start/end match the anchor positions exactly.  For
        two-anchor segments, intermediate positions are obtained by
        linearly blending between:
          (a) a straight lerp from anchor_start → anchor_end, and
          (b) the offset pattern of the original GenDoP trajectory
              (scaled and shifted to fit between the anchors).
        This keeps the trajectory's "shape" (e.g. arc, S-curve) while
        ensuring it stays between the two anchor positions.

        **Rotations**: A full SLERP between the two anchor orientations
        provides the base rotation at each frame.  The GenDoP trajectory's
        *relative* rotation changes are blended in using a smooth cosine
        weight that is strongest in the middle of the segment and zero at
        the anchor endpoints.

        For one-anchor segments, a 6-DoF Euclidean transform aligns the
        relevant endpoint, then the anchor rotation is blended into the
        nearby frames.
        """
        segments: List[np.ndarray] = []

        for i, elem in enumerate(merged):
            if elem["type"] != "traj":
                continue

            c2ws = elem["c2ws"].copy()
            N = len(c2ws)

            left_anchor = None
            right_anchor = None
            if i > 0 and merged[i - 1]["type"] == "anchor":
                left_anchor = merged[i - 1]["c2w"]
            if i < len(merged) - 1 and merged[i + 1]["type"] == "anchor":
                right_anchor = merged[i + 1]["c2w"]

            if left_anchor is not None and right_anchor is not None:
                c2ws = self._align_two_anchors(c2ws, left_anchor, right_anchor)
            elif left_anchor is not None:
                c2ws = self._align_one_anchor(c2ws, left_anchor, side="left")
            elif right_anchor is not None:
                c2ws = self._align_one_anchor(c2ws, right_anchor, side="right")

            segments.append(c2ws.astype(np.float64))

        if not segments:
            return np.empty((0, 4, 4), dtype=np.float64)

        return np.concatenate(segments, axis=0)

    # ------------------------------------------------------------------

    @staticmethod
    def _align_two_anchors(
        c2ws: np.ndarray,
        left_anchor: np.ndarray,
        right_anchor: np.ndarray,
    ) -> np.ndarray:
        """
        Align a trajectory segment between two anchor points.

        Positions:  Blend the GenDoP trajectory shape with a straight
                    line between the anchors.
        Rotations:  SLERP between anchor orientations, with GenDoP's
                    relative rotation changes mixed in via a cosine bell.
        """
        N = len(c2ws)
        result = np.zeros_like(c2ws)

        src_start_pos = c2ws[0, :3, 3].copy()
        src_end_pos = c2ws[-1, :3, 3].copy()
        dst_start_pos = left_anchor[:3, 3]
        dst_end_pos = right_anchor[:3, 3]

        # --- Positions ---
        # Compute normalized offsets of the GenDoP trajectory from its
        # own start→end line, then apply them to the anchor start→end line.
        src_span = src_end_pos - src_start_pos
        src_len = np.linalg.norm(src_span)
        dst_span = dst_end_pos - dst_start_pos
        dst_len = np.linalg.norm(dst_span)

        scale = dst_len / max(src_len, 1e-8)

        for j in range(N):
            t = j / max(N - 1, 1)

            # Position on the straight anchor→anchor line
            lerp_pos = (1 - t) * dst_start_pos + t * dst_end_pos

            # GenDoP's offset from its own straight line at this frame
            src_lerp = (1 - t) * src_start_pos + t * src_end_pos
            offset = c2ws[j, :3, 3] - src_lerp

            # Scale the offset to match the anchor-to-anchor distance
            result[j] = np.eye(4)
            result[j, :3, 3] = lerp_pos + offset * scale

        # --- Rotations ---
        R_left = Rotation.from_matrix(left_anchor[:3, :3])
        R_right = Rotation.from_matrix(right_anchor[:3, :3])

        try:
            anchor_slerp = Slerp([0, 1], Rotation.concatenate([R_left, R_right]))
        except ValueError:
            anchor_slerp = None

        for j in range(N):
            t = j / max(N - 1, 1)

            # Base: SLERP between anchor orientations
            if anchor_slerp is not None:
                R_base = anchor_slerp(t)
            else:
                R_base = R_left if t < 0.5 else R_right

            # GenDoP's relative rotation from its start pose
            R_src_start = Rotation.from_matrix(c2ws[0, :3, :3])
            R_src_curr = Rotation.from_matrix(c2ws[j, :3, :3])
            R_relative = R_src_start.inv() * R_src_curr

            # Blend weight: cosine bell, 0 at endpoints, 1 in the middle
            # This lets GenDoP's rotation influence the middle but not
            # override the anchor orientations at the boundaries.
            blend = 0.5 * (1 - np.cos(2 * np.pi * t)) * 0.5  # max 0.5

            # Compose: apply a fraction of the relative rotation on top of base
            if blend > 0.01:
                # Scale the relative rotation by blend factor
                angle = R_relative.magnitude()
                if angle > 1e-6:
                    axis = R_relative.as_rotvec() / angle
                    R_partial = Rotation.from_rotvec(axis * angle * blend)
                    R_final = R_base * R_partial
                else:
                    R_final = R_base
            else:
                R_final = R_base

            result[j, :3, :3] = R_final.as_matrix()

        return result

    @staticmethod
    def _align_one_anchor(
        c2ws: np.ndarray,
        anchor: np.ndarray,
        side: str = "left",
        blend_fraction: float = 0.4,
    ) -> np.ndarray:
        """
        Align one end of a trajectory to a single anchor point.

        Uses Euclidean alignment for position, then blends the anchor's
        rotation into the nearby frames with a smooth cosine ramp.
        """
        N = len(c2ws)
        if N == 0:
            return c2ws

        # Positional alignment via Euclidean transform
        if side == "left":
            source = c2ws[0]
        else:
            source = c2ws[-1]

        T = anchor @ np.linalg.inv(source)
        result = np.array([T @ c2w for c2w in c2ws])

        # Rotation blending: smoothly transition from anchor rotation
        # into the (transformed) trajectory rotation
        blend_len = max(2, int(N * blend_fraction))
        anchor_R = anchor[:3, :3]
        R_anchor = Rotation.from_matrix(anchor_R)

        if side == "left":
            result[0, :3, :3] = anchor_R
            for j in range(1, min(blend_len, N)):
                # Cosine ramp: 0 at anchor (use anchor R), 1 at blend_len (use traj R)
                t = j / blend_len
                alpha = 0.5 * (1 - np.cos(np.pi * t))  # smooth 0→1
                R_traj = Rotation.from_matrix(result[j, :3, :3])
                try:
                    slerp = Slerp([0, 1], Rotation.concatenate([R_anchor, R_traj]))
                    result[j, :3, :3] = slerp(alpha).as_matrix()
                except ValueError:
                    R_blend = (1 - alpha) * anchor_R + alpha * result[j, :3, :3]
                    U, _, Vt = np.linalg.svd(R_blend)
                    result[j, :3, :3] = U @ Vt
        else:
            result[-1, :3, :3] = anchor_R
            for j in range(max(0, N - blend_len), N - 1):
                t = (N - 1 - j) / blend_len
                alpha = 0.5 * (1 - np.cos(np.pi * t))  # smooth 0→1
                R_traj = Rotation.from_matrix(result[j, :3, :3])
                try:
                    slerp = Slerp([0, 1], Rotation.concatenate([R_traj, R_anchor]))
                    result[j, :3, :3] = slerp(1 - alpha).as_matrix()
                except ValueError:
                    R_blend = (1 - alpha) * result[j, :3, :3] + alpha * anchor_R
                    U, _, Vt = np.linalg.svd(R_blend)
                    result[j, :3, :3] = U @ Vt

        return result

    # ------------------------------------------------------------------
    # Transforms
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_euclidean_align(
        c2ws: np.ndarray,
        source_c2w: np.ndarray,
        target_c2w: np.ndarray,
    ) -> np.ndarray:
        """
        Compute and apply a 6-DoF Euclidean (rigid) transform T such that
        ``T @ source_c2w == target_c2w``, then apply T to every pose.

        T = target @ source^{-1}
        """
        T = target_c2w @ np.linalg.inv(source_c2w)
        return np.array([T @ c2w for c2w in c2ws])

    @staticmethod
    def _apply_similarity_align(
        c2ws: np.ndarray,
        src_start: np.ndarray,
        src_end: np.ndarray,
        dst_start: np.ndarray,
        dst_end: np.ndarray,
    ) -> np.ndarray:
        """
        Compute and apply a 7-DoF similarity transform (rotation +
        translation + uniform scale) so that:
            - the trajectory's starting position matches ``dst_start``
            - the trajectory's ending position matches ``dst_end``

        Strategy
        --------
        1. Compute uniform scale from the ratio of endpoint distances.
        2. Apply an intermediate Euclidean transform that aligns the start
           exactly to ``dst_start``.
        3. Compute a corrective rotation (around ``dst_start``) that swings
           the scaled endpoint onto ``dst_end``.
        4. Compose everything into a single per-frame transform.

        The rotation components of each frame's 3×3 sub-matrix are kept
        orthonormal (no shear); only the *positions* are scaled.
        """
        # --- positions ---
        src_start_t = src_start[:3, 3]
        src_end_t = src_end[:3, 3]
        dst_start_t = dst_start[:3, 3]
        dst_end_t = dst_end[:3, 3]

        src_span = np.linalg.norm(src_end_t - src_start_t)
        dst_span = np.linalg.norm(dst_end_t - dst_start_t)

        if src_span < 1e-8:
            # Degenerate: start == end in source → fall back to Euclidean
            T = dst_start @ np.linalg.inv(src_start)
            return np.array([T @ c2w for c2w in c2ws])

        scale = dst_span / src_span

        # Step 1: Euclidean alignment of start → dst_start
        T_rigid = dst_start @ np.linalg.inv(src_start)

        # Apply rigid + scale *positions only* relative to the new start
        N = len(c2ws)
        aligned = np.empty_like(c2ws)
        for i in range(N):
            posed = T_rigid @ c2ws[i]
            # Scale position offset from the start
            offset = posed[:3, 3] - dst_start_t
            posed[:3, 3] = dst_start_t + scale * offset
            aligned[i] = posed

        # Step 2: corrective rotation around dst_start to swing the end
        #         position onto dst_end
        current_end_t = aligned[-1][:3, 3]
        v_cur = current_end_t - dst_start_t
        v_dst = dst_end_t - dst_start_t

        v_cur_norm = np.linalg.norm(v_cur)
        v_dst_norm = np.linalg.norm(v_dst)

        if v_cur_norm < 1e-8 or v_dst_norm < 1e-8:
            return aligned

        v_cur_u = v_cur / v_cur_norm
        v_dst_u = v_dst / v_dst_norm

        # Rotation that takes v_cur_u → v_dst_u
        R_corr = TrajectoryComposer._rotation_between_vectors(v_cur_u, v_dst_u)

        # Apply corrective rotation around dst_start to all frames
        for i in range(N):
            # Rotate position offset
            offset = aligned[i][:3, 3] - dst_start_t
            aligned[i][:3, 3] = dst_start_t + R_corr @ offset
            # Rotate orientation
            aligned[i][:3, :3] = R_corr @ aligned[i][:3, :3]

        return aligned

    @staticmethod
    def _rotation_between_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Compute the rotation matrix that rotates unit vector *a* onto
        unit vector *b* using Rodrigues' formula.
        """
        v = np.cross(a, b)
        c = np.dot(a, b)

        if np.linalg.norm(v) < 1e-10:
            if c > 0:
                return np.eye(3)
            else:
                # 180-degree rotation: find an arbitrary perpendicular axis
                perp = np.array([1, 0, 0], dtype=np.float64)
                if abs(np.dot(a, perp)) > 0.9:
                    perp = np.array([0, 1, 0], dtype=np.float64)
                axis = np.cross(a, perp)
                axis /= np.linalg.norm(axis)
                # R = 2 * outer(axis,axis) - I
                return 2.0 * np.outer(axis, axis) - np.eye(3)

        vx = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ])
        R = np.eye(3) + vx + vx @ vx / (1.0 + c)
        return R

    # ------------------------------------------------------------------
    # Utility – pose interpolation (kept for potential future use)
    # ------------------------------------------------------------------

    @staticmethod
    def _interpolate_c2w(
        c2w_a: np.ndarray,
        c2w_b: np.ndarray,
        alpha: float,
    ) -> np.ndarray:
        """
        Interpolate between two c2w matrices.

        Translation: linear interpolation
        Rotation: SLERP via scipy
        """
        t_interp = (1 - alpha) * c2w_a[:3, 3] + alpha * c2w_b[:3, 3]

        R_a = Rotation.from_matrix(c2w_a[:3, :3])
        R_b = Rotation.from_matrix(c2w_b[:3, :3])

        try:
            slerp = Slerp([0, 1], Rotation.concatenate([R_a, R_b]))
            R_interp = slerp(alpha).as_matrix()
        except ValueError:
            R_blend = (1 - alpha) * c2w_a[:3, :3] + alpha * c2w_b[:3, :3]
            U, _, Vt = np.linalg.svd(R_blend)
            R_interp = U @ Vt

        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :3] = R_interp
        c2w[:3, 3] = t_interp
        return c2w