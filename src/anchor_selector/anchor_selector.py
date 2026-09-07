"""
Camera Anchor Determinator
==========================

Determines optimal camera viewpoints for objects in a 3D scene.

Key improvements over the original:
  1. Uses median bounding-box extent (not diagonal) for distance estimation —
     wide/flat objects no longer push candidates outside the room.
  2. Face-normal-biased sampling: candidates are concentrated in front of
     accessible bbox faces instead of uniformly around the object.
  3. Adaptive max_distance scales with room size.
  4. Smart fallback projects toward the room centre (open space) instead of
     an arbitrary +X offset.
  5. **LLM viewing preferences**: elevation and distance can be steered by
     per-object hints from the trajectory planner (elevation, distance).
     When no hints are provided, a geometry-based heuristic is used as
     fallback (aspect-ratio of the bounding box).
  6. Supports both legacy labels.json (int IDs, 8-corner bboxes) and
     new scene graph JSON (string IDs like "table_0", OBB arrays with
     quaternion rotations).
  7. **SDF-based collision checking**: Uses precomputed SDF grid with
     trilinear interpolation instead of trimesh proximity queries.
     Much more reliable for non-watertight OBB meshes.
  8. **Candidate visualization**: Exports all candidates as colored PLY
     files for debugging (green=valid, red=collision, blue=selected).
"""

import numpy as np
import trimesh
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
from scipy.spatial.transform import Rotation


# =========================================================================
# Data Structures
# =========================================================================

@dataclass
class CameraAnchor:
    """Camera anchor point with position and orientation."""
    position: np.ndarray
    look_at: np.ndarray
    up: np.ndarray
    score: float
    object_id: Union[int, str]
    object_label: str


# =========================================================================
# Viewing-preference constants
# =========================================================================

# Maps the LLM's symbolic elevation names to (min_deg, max_deg) ranges.
ELEVATION_MAP: Dict[str, Tuple[float, float]] = {
    "low":      (10.0, 20.0),
    "medium":   (25.0, 35.0),
    "high":     (40.0, 55.0),
    "overhead": (60.0, 80.0),
}

# Maps the LLM's symbolic distance names to a multiplier applied on top of
# the geometry-derived preferred_distance.
DISTANCE_MAP: Dict[str, float] = {
    "close":  0.7,
    "medium": 1.0,
    "far":    1.4,
}

# Default elevation range when nothing is specified (original behaviour).
DEFAULT_ELEVATION_RANGE: Tuple[float, float] = (15.0, 30.0)


# =========================================================================
# SDF Collision Checker
# =========================================================================

class SDFCollisionChecker:
    """
    Collision checking via precomputed SDF grid with trilinear interpolation.

    Convention: SDF < 0 means INSIDE an obstacle (collision).
                SDF > 0 means free space.

    Points outside the grid domain are treated as collisions (conservative).
    """

    def __init__(
        self,
        sdf_grid: np.ndarray,
        grid_min: np.ndarray,
        grid_max: np.ndarray,
        collision_margin: float = 0.05,
    ):
        """
        Args:
            sdf_grid:  (R, R, R) float array — signed distance values.
            grid_min:  (3,) world-space minimum corner of the grid.
            grid_max:  (3,) world-space maximum corner of the grid.
            collision_margin: Minimum SDF value to consider "free".
                              Points with SDF < collision_margin are collisions.
        """
        self.sdf_grid = sdf_grid.astype(np.float32)
        self.grid_min = grid_min.astype(np.float32)
        self.grid_max = grid_max.astype(np.float32)
        self.collision_margin = collision_margin

        res = sdf_grid.shape[0]
        self.resolution = res
        self.voxel_size = (self.grid_max - self.grid_min) / (res - 1)

        print(f"  [SDFCollisionChecker] grid {res}³, "
              f"bounds [{grid_min[0]:.2f}..{grid_max[0]:.2f}, "
              f"{grid_min[1]:.2f}..{grid_max[1]:.2f}, "
              f"{grid_min[2]:.2f}..{grid_max[2]:.2f}], "
              f"margin={collision_margin:.3f}")

    @classmethod
    def from_npz(cls, npz_path: str, collision_margin: float = 0.05) -> "SDFCollisionChecker":
        """Load from the .npz format produced by the pipeline."""
        data = np.load(npz_path)
        return cls(
            sdf_grid=data["sdf_grid"],
            grid_min=data["grid_min"],
            grid_max=data["grid_max"],
            collision_margin=collision_margin,
        )

    def query_sdf(self, positions: np.ndarray) -> np.ndarray:
        """
        Trilinear interpolation of SDF values at arbitrary world-space positions.

        Args:
            positions: (N, 3) world-space coordinates.

        Returns:
            (N,) SDF values.  Points outside the grid get -1.0 (collision).
        """
        positions = np.asarray(positions, dtype=np.float32)
        if positions.ndim == 1:
            positions = positions[None, :]

        # Normalize to grid coordinates [0, resolution-1]
        grid_coords = (positions - self.grid_min) / self.voxel_size

        res = self.resolution
        sdf_values = np.full(len(positions), -1.0, dtype=np.float32)

        # Check bounds (with small epsilon for edge voxels)
        eps = 0.01
        in_bounds = np.all(grid_coords >= eps, axis=1) & np.all(
            grid_coords <= res - 1 - eps, axis=1
        )

        if not np.any(in_bounds):
            return sdf_values

        gc = grid_coords[in_bounds]

        # Floor indices
        i0 = np.floor(gc).astype(np.int32)
        i0 = np.clip(i0, 0, res - 2)  # ensure i0+1 is valid

        # Fractional parts
        frac = gc - i0.astype(np.float32)
        fx, fy, fz = frac[:, 0], frac[:, 1], frac[:, 2]

        # 8 corner values
        ix, iy, iz = i0[:, 0], i0[:, 1], i0[:, 2]
        c000 = self.sdf_grid[ix,     iy,     iz    ]
        c001 = self.sdf_grid[ix,     iy,     iz + 1]
        c010 = self.sdf_grid[ix,     iy + 1, iz    ]
        c011 = self.sdf_grid[ix,     iy + 1, iz + 1]
        c100 = self.sdf_grid[ix + 1, iy,     iz    ]
        c101 = self.sdf_grid[ix + 1, iy,     iz + 1]
        c110 = self.sdf_grid[ix + 1, iy + 1, iz    ]
        c111 = self.sdf_grid[ix + 1, iy + 1, iz + 1]

        # Trilinear interpolation
        c00 = c000 * (1 - fx) + c100 * fx
        c01 = c001 * (1 - fx) + c101 * fx
        c10 = c010 * (1 - fx) + c110 * fx
        c11 = c011 * (1 - fx) + c111 * fx

        c0 = c00 * (1 - fy) + c10 * fy
        c1 = c01 * (1 - fy) + c11 * fy

        result = c0 * (1 - fz) + c1 * fz

        sdf_values[in_bounds] = result
        return sdf_values

    def check_collisions(self, positions: np.ndarray) -> np.ndarray:
        """
        Args:
            positions: (N, 3) world-space coordinates.

        Returns:
            (N,) bool — True means COLLISION (SDF < margin).
        """
        sdf_vals = self.query_sdf(positions)
        return sdf_vals < self.collision_margin


# =========================================================================
# Candidate Visualization (Debug)
# =========================================================================

def save_candidates_debug_ply(
    all_candidates: np.ndarray,
    collision_mask: np.ndarray,
    selected_idx: Optional[int],
    scores: Optional[np.ndarray],
    target_center: np.ndarray,
    output_path: str,
    object_label: str = "",
    sdf_values: Optional[np.ndarray] = None,
):
    """
    Export candidate camera positions as a colored PLY for debugging.

    Color scheme:
      - Green (0, 255, 0):   valid candidate (no collision)
      - Red (255, 0, 0):     collision
      - Blue (0, 100, 255):  selected best candidate
      - Yellow (255, 255, 0): target object center

    If scores are provided, valid candidates are shaded from dark-green
    (low score) to bright-green (high score).

    If sdf_values are provided, they are stored as a scalar vertex attribute.
    """
    n = len(all_candidates)
    # Include the target center as an extra point
    points = np.vstack([all_candidates, target_center[None, :]])
    colors = np.zeros((n + 1, 3), dtype=np.uint8)

    for i in range(n):
        if collision_mask[i]:
            colors[i] = [255, 0, 0]  # red = collision
        else:
            if scores is not None:
                # Shade valid by score: dark green (low) → bright green (high)
                s = scores[i] if i < len(scores) else 0.0
                brightness = int(80 + 175 * np.clip(s, 0, 1))
                colors[i] = [0, brightness, 0]
            else:
                colors[i] = [0, 200, 0]

    if selected_idx is not None and 0 <= selected_idx < n:
        colors[selected_idx] = [0, 100, 255]  # blue = selected

    # Target center = yellow
    colors[-1] = [255, 255, 0]

    # Build trimesh point cloud
    cloud = trimesh.PointCloud(points, colors=colors)

    # Attach SDF values as metadata in a comment if available
    cloud.export(str(output_path))

    # Also write a companion CSV with detailed per-candidate info
    csv_path = Path(output_path).with_suffix('.csv')
    with open(csv_path, 'w') as f:
        f.write("idx,x,y,z,collision,sdf,score,selected\n")
        for i in range(n):
            sdf_val = sdf_values[i] if sdf_values is not None else float('nan')
            score_val = scores[i] if scores is not None and i < len(scores) else float('nan')
            is_sel = 1 if (selected_idx is not None and i == selected_idx) else 0
            f.write(f"{i},{all_candidates[i,0]:.4f},{all_candidates[i,1]:.4f},"
                    f"{all_candidates[i,2]:.4f},{int(collision_mask[i])},"
                    f"{sdf_val:.4f},{score_val:.4f},{is_sel}\n")
        # target center row
        f.write(f"{n},{target_center[0]:.4f},{target_center[1]:.4f},"
                f"{target_center[2]:.4f},0,nan,nan,0\n")

    print(f"  [Debug] Saved {n} candidates → {output_path}")
    print(f"  [Debug] Details CSV → {csv_path}")
    if sdf_values is not None:
        n_coll = collision_mask.sum()
        sdf_at_coll = sdf_values[collision_mask] if n_coll > 0 else np.array([])
        sdf_at_valid = sdf_values[~collision_mask] if n_coll < n else np.array([])
        print(f"  [Debug] {n_coll}/{n} collisions "
              f"(collision SDF range: [{sdf_at_coll.min():.3f}, {sdf_at_coll.max():.3f}])"
              if len(sdf_at_coll) > 0 else f"  [Debug] {n_coll}/{n} collisions")
        if len(sdf_at_valid) > 0:
            print(f"  [Debug] Valid SDF range: [{sdf_at_valid.min():.3f}, {sdf_at_valid.max():.3f}]")


# =========================================================================
# OBB Helpers
# =========================================================================

def obb_to_corners(center: np.ndarray, size: np.ndarray, qxyzw: np.ndarray) -> np.ndarray:
    """
    Compute 8 oriented bounding box corners from center, size, and quaternion.

    Args:
        center: [cx, cy, cz]
        size:   [sx, sy, sz] — full extents along local axes
        qxyzw:  [qx, qy, qz, qw] — rotation quaternion

    Returns:
        (8, 3) array of corner positions in world space
    """
    R = Rotation.from_quat(qxyzw).as_matrix()  # scipy uses (x,y,z,w)
    half = size / 2.0

    # 8 corners in local space
    signs = np.array([
        [-1, -1, -1],
        [-1, -1,  1],
        [-1,  1, -1],
        [-1,  1,  1],
        [ 1, -1, -1],
        [ 1, -1,  1],
        [ 1,  1, -1],
        [ 1,  1,  1],
    ], dtype=np.float64)

    local_corners = signs * half  # (8, 3)
    world_corners = (R @ local_corners.T).T + center  # (8, 3)
    return world_corners


# =========================================================================
# Anchor Determinator
# =========================================================================

class AnchorDeterminator:
    def __init__(
        self,
        bboxes,
        mesh: trimesh.Trimesh,
        min_distance: float = 0.3,
        max_distance: float = 3.0,
        preferred_distance_factor: float = 1.2,
        height_offset: float = 0.1,
        num_candidate_angles: int = 60,
        collision_margin: float = 0.05,
        scene_bounds_margin: float = 0.3,
        sdf_collision_checker: Optional[SDFCollisionChecker] = None,
        debug_output_dir: Optional[str] = None,
    ):
        """
        Args:
            bboxes: Either:
                - List[Dict] with keys 'ins_id', 'label', 'bounding_box' (legacy format)
                - Dict (scene graph JSON) with key 'objects' mapping obj_id → {"obb": [...], ...}
                  Can also be just the 'objects' sub-dict directly.
            mesh: Combined trimesh for collision/visibility checks.
            sdf_collision_checker: Optional SDFCollisionChecker instance.
                If provided, uses SDF-based collision checking (recommended).
                If None, falls back to trimesh proximity queries (legacy).
            debug_output_dir: If set, saves candidate visualizations as
                PLY+CSV files to this directory for every object.
        """
        self.mesh = mesh
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.preferred_distance_factor = preferred_distance_factor
        self.height_offset = height_offset
        self.num_candidate_angles = num_candidate_angles
        self.collision_margin = collision_margin
        self.scene_bounds_margin = scene_bounds_margin

        # SDF-based collision (preferred) vs trimesh proximity (fallback)
        self.sdf_checker = sdf_collision_checker
        if self.sdf_checker is not None:
            print("  [AnchorDeterminator] Using SDF-based collision checking")
        else:
            print("  [AnchorDeterminator] Using trimesh proximity collision checking (legacy)")

        # Debug visualization
        self.debug_output_dir = Path(debug_output_dir) if debug_output_dir else None
        if self.debug_output_dir:
            self.debug_output_dir.mkdir(parents=True, exist_ok=True)
            print(f"  [AnchorDeterminator] Debug output → {self.debug_output_dir}")

        # Build bbox index (auto-detects format)
        self._build_bbox_index(bboxes)

        # Ray intersector for visibility checks
        self.ray_intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)

        # Proximity query — kept for fallback collision checks
        self.proximity = trimesh.proximity.ProximityQuery(mesh)

        # Scene bounds
        self.scene_bounds = {
            "min": mesh.bounds[0],
            "max": mesh.bounds[1],
            "center": mesh.centroid,
        }

        # Adaptive max distance: don't exceed 40% of the smallest horizontal
        # room dimension so cameras stay well inside the room.
        room_extent = mesh.bounds[1] - mesh.bounds[0]
        room_horiz = np.min(room_extent[:2])
        self.max_distance = min(self.max_distance, room_horiz * 0.4)

    # ------------------------------------------------------------------
    # Bbox index
    # ------------------------------------------------------------------

    def _build_bbox_index(self, bboxes):
        """
        Build index of all bounding boxes. Auto-detects format:

        1. New scene graph format (dict):
           {"table_0": {"obb": [cx,cy,cz,sx,sy,sz,qx,qy,qz,qw], ...}, ...}
           or {"objects": {"table_0": {...}, ...}, ...}

        2. Legacy labels.json format (list):
           [{"ins_id": 42, "label": "table", "bounding_box": [{x,y,z},...8]}, ...]
        """
        self.bbox_data = {}

        if isinstance(bboxes, dict):
            # New scene graph format
            if "objects" in bboxes:
                objects_dict = bboxes["objects"]
            else:
                objects_dict = bboxes

            for obj_id, obj_data in objects_dict.items():
                if "obb" not in obj_data:
                    continue

                obb = obj_data["obb"]
                center = np.array(obb[0:3], dtype=np.float64)
                size = np.array(obb[3:6], dtype=np.float64)
                qxyzw = np.array(obb[6:10], dtype=np.float64)

                corners = obb_to_corners(center, size, qxyzw)

                aa_min = corners.min(axis=0)
                aa_max = corners.max(axis=0)

                label = obj_id.rsplit("_", 1)[0]

                self.bbox_data[obj_id] = {
                    "id": obj_id,
                    "label": label,
                    "corners": corners,
                    "center": center,
                    "extent": size,
                    "aa_extent": aa_max - aa_min,
                    "min": aa_min,
                    "max": aa_max,
                    "against_wall": obj_data.get("against_wall", False),
                    "attached_to_ceiling": obj_data.get("attached_to_ceiling", False),
                }

        elif isinstance(bboxes, list):
            for item in bboxes:
                if "bounding_box" not in item:
                    continue
                obj_id = item.get("ins_id")
                corners = np.array(
                    [[p["x"], p["y"], p["z"]] for p in item["bounding_box"]]
                )
                self.bbox_data[str(obj_id)] = {
                    "id": obj_id,
                    "label": item.get("label", ""),
                    "corners": corners,
                    "center": corners.mean(axis=0),
                    "extent": corners.max(axis=0) - corners.min(axis=0),
                    "aa_extent": corners.max(axis=0) - corners.min(axis=0),
                    "min": corners.min(axis=0),
                    "max": corners.max(axis=0),
                }
        else:
            raise TypeError(
                f"bboxes must be a dict (scene graph) or list (legacy labels), "
                f"got {type(bboxes).__name__}"
            )

        print(f"  [AnchorDeterminator] Indexed {len(self.bbox_data)} bounding boxes")

    def get_object_bbox(self, object_id: Union[int, str]) -> Optional[Dict]:
        """Look up bbox by ID (supports both string and int keys)."""
        result = self.bbox_data.get(str(object_id))
        if result is None and not isinstance(object_id, str):
            result = self.bbox_data.get(object_id)
        return result

    # ------------------------------------------------------------------
    # Collision & visibility
    # ------------------------------------------------------------------

    def check_collisions_batch(self, positions: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Check for collisions at the given positions.

        Returns:
            (collision_mask, sdf_values)
            - collision_mask: (N,) bool — True = collision.
            - sdf_values: (N,) float or None — raw SDF values (only if SDF checker is used).
        """
        if self.sdf_checker is not None:
            sdf_values = self.sdf_checker.query_sdf(positions)
            collision_mask = sdf_values < self.sdf_checker.collision_margin
            return collision_mask, sdf_values
        else:
            # Legacy: trimesh proximity
            distances = self.proximity.signed_distance(positions)
            collision_mask = distances < self.collision_margin
            return collision_mask, None

    def check_visibility_batch(
        self,
        camera_positions: np.ndarray,
        target_points: np.ndarray,
    ) -> np.ndarray:
        """
        (n_cameras, n_targets) bool array — True = visible.
        """
        n_cameras = len(camera_positions)
        n_targets = len(target_points)

        ray_origins = []
        ray_directions = []
        ray_lengths = []

        for cam_pos in camera_positions:
            for target in target_points:
                d = target - cam_pos
                length = np.linalg.norm(d)
                ray_origins.append(cam_pos)
                ray_directions.append(d / max(length, 1e-8))
                ray_lengths.append(length)

        ray_origins = np.array(ray_origins)
        ray_directions = np.array(ray_directions)
        ray_lengths = np.array(ray_lengths)

        locations, index_ray, _ = self.ray_intersector.intersects_location(
            ray_origins=ray_origins,
            ray_directions=ray_directions,
        )

        visible = np.ones(n_cameras * n_targets, dtype=bool)
        if len(locations) > 0:
            hit_dists = np.linalg.norm(locations - ray_origins[index_ray], axis=1)
            for ray_idx, hit_dist in zip(index_ray, hit_dists):
                if hit_dist < ray_lengths[ray_idx] - 0.01:
                    visible[ray_idx] = False

        return visible.reshape(n_cameras, n_targets)

    # ------------------------------------------------------------------
    # Viewing distance estimation
    # ------------------------------------------------------------------

    @staticmethod
    def _viewing_distance(extent: np.ndarray) -> float:
        """
        Estimate a sensible viewing distance from a bounding-box extent.
        Uses the *median* dimension.
        """
        sorted_ext = np.sort(extent)
        return float(sorted_ext[1])

    # ------------------------------------------------------------------
    # Viewing-preference resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_elevation_range(
        viewing_prefs: Optional[Dict],
        extent: np.ndarray,
    ) -> Tuple[float, float]:
        if viewing_prefs is not None:
            elev_key = viewing_prefs.get("elevation")
            if elev_key and elev_key in ELEVATION_MAP:
                return ELEVATION_MAP[elev_key]

        horizontal_extent = max(extent[0], extent[1])
        vertical_extent = extent[2]
        aspect = horizontal_extent / max(vertical_extent, 0.01)

        if aspect > 2.5:
            return ELEVATION_MAP["high"]
        elif aspect < 0.6:
            return ELEVATION_MAP["low"]
        else:
            return DEFAULT_ELEVATION_RANGE

    @staticmethod
    def _resolve_distance_multiplier(viewing_prefs: Optional[Dict]) -> float:
        if viewing_prefs is not None:
            dist_key = viewing_prefs.get("distance")
            if dist_key and dist_key in DISTANCE_MAP:
                return DISTANCE_MAP[dist_key]
        return 1.0

    # ------------------------------------------------------------------
    # Candidate generation
    # ------------------------------------------------------------------

    def generate_candidate_positions(
        self,
        target_bbox: Dict,
        elevation_range: Optional[Tuple[float, float]] = None,
        distance_multiplier: float = 1.0,
    ) -> np.ndarray:
        """
        Generate candidate camera positions biased toward *accessible*
        bounding-box faces.
        """
        center = target_bbox["center"]
        extent = target_bbox["extent"]
        obj_min = target_bbox["min"]
        obj_max = target_bbox["max"]

        viewing_size = self._viewing_distance(extent)
        preferred_dist = np.clip(
            viewing_size * self.preferred_distance_factor * distance_multiplier,
            self.min_distance,
            self.max_distance,
        )

        bounds_lo = self.scene_bounds["min"] + self.scene_bounds_margin
        bounds_hi = self.scene_bounds["max"] - self.scene_bounds_margin

        if elevation_range is not None:
            lo, hi = elevation_range
            elevations = [lo, (lo + hi) / 2.0, hi]
        else:
            elevations = [10, 20, 35]

        face_normals = [
            np.array([1, 0, 0]),
            np.array([-1, 0, 0]),
            np.array([0, 1, 0]),
            np.array([0, -1, 0]),
            np.array([0, 0, 1]),
            np.array([0, 0, -1]),
        ]
        face_centers = [
            np.array([obj_max[0], center[1], center[2]]),
            np.array([obj_min[0], center[1], center[2]]),
            np.array([center[0], obj_max[1], center[2]]),
            np.array([center[0], obj_min[1], center[2]]),
            np.array([center[0], center[1], obj_max[2]]),
            np.array([center[0], center[1], obj_min[2]]),
        ]

        candidates = []

        for normal, fc in zip(face_normals, face_centers):
            test_point = fc + normal * preferred_dist * 0.5
            if not (np.all(test_point >= bounds_lo) and np.all(test_point <= bounds_hi)):
                continue

            base_theta = np.arctan2(normal[1], normal[0])

            for dist_factor in [0.5, 0.7, 1.0, 1.3]:
                distance = preferred_dist * dist_factor
                for angle_offset_deg in np.linspace(-45, 45, 10):
                    theta = base_theta + np.radians(angle_offset_deg)
                    for elev_deg in elevations:
                        phi = np.radians(elev_deg)
                        pos = fc + distance * np.array([
                            np.cos(theta) * np.cos(phi),
                            np.sin(theta) * np.cos(phi),
                            np.sin(phi),
                        ])
                        candidates.append(pos)

        if len(candidates) == 0:
            candidates = self._uniform_candidates(
                center, preferred_dist, elevations=elevations,
            )
        else:
            candidates.extend(
                self._uniform_candidates(
                    center, preferred_dist, sparse=True, elevations=elevations,
                )
            )

        return np.array(candidates)

    def _uniform_candidates(
        self,
        center: np.ndarray,
        preferred_dist: float,
        sparse: bool = False,
        elevations: Optional[List[float]] = None,
    ) -> List[np.ndarray]:
        n_angles = self.num_candidate_angles if not sparse else max(self.num_candidate_angles // 3, 8)
        if elevations is None:
            elevations = [15, 30, 45] if not sparse else [20, 35]
        elif sparse:
            elevations = [elevations[len(elevations) // 2]]
        dist_factors = [0.8, 1.0, 1.2] if not sparse else [1.0]

        candidates = []
        for i in range(n_angles):
            theta = 2 * np.pi * i / n_angles
            for phi_deg in elevations:
                phi = np.radians(phi_deg)
                for df in dist_factors:
                    d = preferred_dist * df
                    pos = center + d * np.array([
                        np.cos(theta) * np.cos(phi),
                        np.sin(theta) * np.cos(phi),
                        np.sin(phi),
                    ])
                    candidates.append(pos)
        return candidates

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _perpendicularity_scores(self, candidates: np.ndarray, target_center: np.ndarray) -> np.ndarray:
        """
        Score candidates by how perpendicular their viewing angle is to the wall.
        
        Approximation: "perpendicular to wall" ≈ "aligned with object→room_center direction".
        Candidates directly in front of the object (from room interior) score highest.
        """
        # Room-interior direction in XY plane
        to_room = self.scene_bounds["center"][:2] - target_center[:2]
        dist = np.linalg.norm(to_room)
        if dist < 1e-6:
            return np.ones(len(candidates))
        interior_dir = to_room / dist

        # Candidate direction from object in XY plane
        candidate_dirs = candidates[:, :2] - target_center[:2]
        norms = np.linalg.norm(candidate_dirs, axis=1, keepdims=True)
        candidate_dirs_normed = candidate_dirs / np.maximum(norms, 1e-8)

        # dot=1 means candidate is directly in front (perpendicular to wall)
        # dot=0 means parallel to wall, dot=-1 means behind (between object and wall)
        alignment = candidate_dirs_normed @ interior_dir
        return np.clip(alignment, 0.0, 1.0)

    def compute_scores_batch(
        self,
        positions: np.ndarray,
        target_bbox: Dict,
        elevation_range: Optional[Tuple[float, float]] = None,
    ) -> np.ndarray:
        target_center = target_bbox["center"]
        target_corners = target_bbox["corners"]
        target_extent = target_bbox["extent"]

        visibility_matrix = self.check_visibility_batch(positions, target_corners)
        visibility_scores = visibility_matrix.mean(axis=1)

        distances = np.linalg.norm(positions - target_center, axis=1)
        viewing_size = self._viewing_distance(target_extent)
        preferred_dist = np.clip(
            viewing_size * self.preferred_distance_factor,
            self.min_distance,
            self.max_distance,
        )
        distance_scores = np.exp(
            -((distances - preferred_dist) / max(preferred_dist, 0.1)) ** 2
        )
        distance_scores[
            (distances < self.min_distance) | (distances > self.max_distance)
        ] = 0.0

        if elevation_range is not None:
            ideal_min, ideal_max = elevation_range
        else:
            ideal_min, ideal_max = DEFAULT_ELEVATION_RANGE

        ideal_center = (ideal_min + ideal_max) / 2.0
        ideal_half = (ideal_max - ideal_min) / 2.0

        directions = positions - target_center
        horizontal_dist = np.linalg.norm(directions[:, :2], axis=1)
        vertical_dist = directions[:, 2]
        angles = np.degrees(
            np.arctan2(vertical_dist, np.maximum(horizontal_dist, 0.01))
        )

        angle_scores = np.where(
            (angles >= ideal_min) & (angles <= ideal_max),
            1.0,
            np.exp(
                -((angles - ideal_center) / max(ideal_half * 2.0, 1.0)) ** 2
            ) * 0.7,
        )

        room_center_xz = self.scene_bounds["center"][:2]
        cam_xz = positions[:, :2]
        room_radius = np.linalg.norm(
            (self.scene_bounds["max"][:2] - self.scene_bounds["min"][:2]) / 2
        )
        center_dist = np.linalg.norm(cam_xz - room_center_xz, axis=1)
        openness_scores = np.exp(-(center_dist / max(room_radius, 0.1)) ** 2)

        if target_bbox.get("against_wall", False):
            perp_scores = self._perpendicularity_scores(positions, target_center)
            total = (
                visibility_scores * 0.35
                + distance_scores * 0.20
                + angle_scores * 0.10
                + openness_scores * 0.10
                + perp_scores * 0.25
            )
        else:
            total = (
                visibility_scores * 0.45
                + distance_scores * 0.25
                + angle_scores * 0.15
                + openness_scores * 0.15
            )
        return total

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def determine_anchor(
        self,
        object_id: Union[int, str],
        object_label: str = "",
        viewing_prefs: Optional[Dict] = None,
    ) -> CameraAnchor:
        """
        Determine the best camera anchor for viewing an object.

        Now with SDF-based collision checking and optional debug visualization.
        """
        target_bbox = self.get_object_bbox(object_id)
        if target_bbox is None:
            raise ValueError(f"Object {object_id} not found in bboxes")

        target_center = target_bbox["center"]
        if not object_label:
            object_label = target_bbox.get("label", "")

        # Resolve viewing preferences
        elevation_range = self._resolve_elevation_range(
            viewing_prefs, target_bbox["extent"],
        )
        distance_multiplier = self._resolve_distance_multiplier(viewing_prefs)

        if viewing_prefs:
            print(
                f"  [Anchor] {object_label} (id={object_id}): "
                f"elevation={viewing_prefs.get('elevation', '?')} → "
                f"{elevation_range[0]:.0f}°–{elevation_range[1]:.0f}°, "
                f"distance={viewing_prefs.get('distance', '?')} → "
                f"×{distance_multiplier:.1f}"
            )
        else:
            print(
                f"  [Anchor] {object_label} (id={object_id}): "
                f"geometry fallback → {elevation_range[0]:.0f}°–{elevation_range[1]:.0f}°"
            )

        # Generate candidates
        candidates = self.generate_candidate_positions(
            target_bbox,
            elevation_range=elevation_range,
            distance_multiplier=distance_multiplier,
        )

        # Filter by scene bounds
        bounds_lo = self.scene_bounds["min"] + self.scene_bounds_margin
        bounds_hi = self.scene_bounds["max"] - self.scene_bounds_margin
        in_bounds = np.all(candidates >= bounds_lo, axis=1) & np.all(
            candidates <= bounds_hi, axis=1
        )
        candidates = candidates[in_bounds]

        if len(candidates) == 0:
            print(f"Warning: No candidates in bounds for object {object_id}")
            return self._fallback_anchor(target_bbox, object_id, object_label)

        # Collision filter (SDF-based or proximity-based)
        collision_mask, sdf_values = self.check_collisions_batch(candidates)

        n_total = len(candidates)
        n_coll = collision_mask.sum()
        print(f"  [Anchor] {object_label}: {n_coll}/{n_total} candidates collide "
              f"({n_coll/n_total*100:.1f}%)")

        valid_mask = ~collision_mask
        valid = candidates[valid_mask]

        # Score ALL candidates for debug visualization (even colliding ones)
        all_scores = self.compute_scores_batch(
            candidates, target_bbox, elevation_range=elevation_range,
        )

        if len(valid) == 0:
            print(f"Warning: All {n_total} candidates collide for object {object_id}")

            # Debug dump before falling back
            if self.debug_output_dir:
                save_candidates_debug_ply(
                    all_candidates=candidates,
                    collision_mask=collision_mask,
                    selected_idx=None,
                    scores=all_scores,
                    target_center=target_center,
                    output_path=str(self.debug_output_dir / f"candidates_{object_id}_ALL_COLLIDE.ply"),
                    object_label=object_label,
                    sdf_values=sdf_values,
                )

            return self._fallback_anchor(target_bbox, object_id, object_label)

        # Score valid candidates
        valid_scores = all_scores[valid_mask]
        best_valid_idx = np.argmax(valid_scores)

        # Map back to full candidate index for debug
        valid_indices = np.where(valid_mask)[0]
        best_global_idx = valid_indices[best_valid_idx]

        # Debug visualization
        if self.debug_output_dir:
            save_candidates_debug_ply(
                all_candidates=candidates,
                collision_mask=collision_mask,
                selected_idx=int(best_global_idx),
                scores=all_scores,
                target_center=target_center,
                output_path=str(self.debug_output_dir / f"candidates_{object_id}.ply"),
                object_label=object_label,
                sdf_values=sdf_values,
            )

        return CameraAnchor(
            position=valid[best_valid_idx],
            look_at=target_center,
            up=np.array([0.0, 0.0, 1.0]),
            score=float(valid_scores[best_valid_idx]),
            object_id=object_id,
            object_label=object_label,
        )

    # ------------------------------------------------------------------
    # Smart fallback: project toward room center
    # ------------------------------------------------------------------

    def _fallback_anchor(
        self,
        target_bbox: Dict,
        object_id: Union[int, str],
        object_label: str,
    ) -> CameraAnchor:
        target_center = target_bbox["center"]
        target_extent = target_bbox["extent"]

        to_room = self.scene_bounds["center"] - target_center
        to_room[2] = 0.0
        dist = np.linalg.norm(to_room)
        if dist > 1e-6:
            to_room /= dist
        else:
            to_room = np.array([1.0, 0.0, 0.0])

        viewing_size = self._viewing_distance(target_extent)
        view_dist = np.clip(
            viewing_size * self.preferred_distance_factor,
            self.min_distance,
            self.max_distance,
        )

        position = target_center + to_room * view_dist
        position[2] = target_center[2] + self.height_offset

        position = np.clip(
            position,
            self.scene_bounds["min"] + self.collision_margin,
            self.scene_bounds["max"] - self.collision_margin,
        )

        return CameraAnchor(
            position=position,
            look_at=target_center,
            up=np.array([0.0, 0.0, 1.0]),
            score=0.1,
            object_id=object_id,
            object_label=object_label,
        )

    # ------------------------------------------------------------------
    # Multi-object convenience
    # ------------------------------------------------------------------

    def determine_path_anchors(
        self,
        object_ids: List[Union[int, str]],
        object_labels: Optional[List[str]] = None,
        viewing_preferences: Optional[Dict[str, Dict]] = None,
    ) -> List[CameraAnchor]:
        if object_labels is None:
            object_labels = [""] * len(object_ids)
        if viewing_preferences is None:
            viewing_preferences = {}

        anchors = []
        for obj_id, label in zip(object_ids, object_labels):
            try:
                prefs = viewing_preferences.get(str(obj_id))
                if prefs is None:
                    prefs = viewing_preferences.get(label)
                anchor = self.determine_anchor(obj_id, label, viewing_prefs=prefs)
                anchors.append(anchor)
            except ValueError as e:
                print(f"Warning: {e}")
        return anchors


# =========================================================================
# Mesh Loader (legacy USD)
# =========================================================================

def load_mesh(mesh_path: str, scale: float = None) -> trimesh.Trimesh:
    """Load USD mesh with correct transform handling."""
    from pxr import Usd, UsdGeom

    mesh_path = Path(mesh_path)
    stage = Usd.Stage.Open(str(mesh_path))

    meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
    print(f"USD meters per unit: {meters_per_unit}")

    time = Usd.TimeCode.Default()
    xform_cache = UsdGeom.XformCache(time)

    all_meshes = []

    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue

        mesh_prim = UsdGeom.Mesh(prim)
        points = mesh_prim.GetPointsAttr().Get(time)
        if points is None or len(points) == 0:
            continue

        points = np.array(points, dtype=np.float64)

        world_xform = xform_cache.GetLocalToWorldTransform(prim)
        world_matrix = np.array(world_xform).reshape(4, 4).T

        points_h = np.hstack([points, np.ones((len(points), 1))])
        points_world = (world_matrix @ points_h.T).T[:, :3]

        face_counts = mesh_prim.GetFaceVertexCountsAttr().Get(time)
        face_indices = mesh_prim.GetFaceVertexIndicesAttr().Get(time)
        if face_counts is None or face_indices is None:
            continue

        triangles = []
        idx = 0
        for count in face_counts:
            if count == 3:
                triangles.append(
                    [face_indices[idx], face_indices[idx + 1], face_indices[idx + 2]]
                )
            elif count == 4:
                triangles.append(
                    [face_indices[idx], face_indices[idx + 1], face_indices[idx + 2]]
                )
                triangles.append(
                    [face_indices[idx], face_indices[idx + 2], face_indices[idx + 3]]
                )
            elif count > 4:
                for i in range(1, count - 1):
                    triangles.append(
                        [
                            face_indices[idx],
                            face_indices[idx + i],
                            face_indices[idx + i + 1],
                        ]
                    )
            idx += count

        if triangles:
            submesh = trimesh.Trimesh(
                vertices=points_world,
                faces=np.array(triangles),
                process=False,
            )
            all_meshes.append(submesh)

    if not all_meshes:
        raise ValueError("No valid meshes found")

    combined = trimesh.util.concatenate(all_meshes)

    print(f"\nFinal mesh: {len(combined.vertices)} verts, {len(combined.faces)} faces")
    print(f"Bounds: {combined.bounds}")

    return combined