"""
Ablation Study Utilities for GSCinema
======================================

Provides drop-in replacement components for three ablation conditions:

1. w/o Anchor Determinator:
   - Replaces AnchorDeterminator with CLIP-based keyframe selection
   - CLIPAnchorAdapter wraps CLIPKeyframeSelector to produce CameraAnchor objects

2. w/o Parametric Trajectories:
   - Replaces parametric atomic trajectory builder with GenDoP's diffusion model
   - GenDoPTrajectoryExecutor overrides execute_tools() to use CineGPTWrapper

3. w/o SDF-based Pose Optimization:
   - Replaces SDF collision/occlusion optimizer with point-cloud KD-tree refinement
   - optimize_trajectory_result_pointcloud() replaces optimize_trajectory_result()
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import sys
chatcam_dir = str(Path(__file__).resolve().parents[2] / "baselines" / "ChatCam_GenDoP")
if chatcam_dir not in sys.path:
    sys.path.insert(0, chatcam_dir)
cctg_dir = str(Path(__file__).resolve().parents[2] / "baselines" / "Berkeley")
if cctg_dir not in sys.path:
    sys.path.insert(0, cctg_dir)

from src.anchor_selector.anchor_selector import (
    AnchorDeterminator,
    CameraAnchor,
    SDFCollisionChecker,
)


# =============================================================================
# Ablation 1: w/o Anchor Determinator → CLIP-based anchor selection
# =============================================================================

class CLIPAnchorAdapter:
    """
    Adapter that wraps CLIPKeyframeSelector to provide the same interface
    as AnchorDeterminator.determine_anchor().

    Instead of generating candidates around OBB faces and scoring them,
    this selects the training image whose CLIP embedding best matches
    the object label, then uses that image's camera pose as the anchor.
    """

    def __init__(
        self,
        bboxes,
        mesh,
        images_dir: str,
        colmap_model_dir: str,
        clip_model_name: str = "ViT-B/32",
        device: str = "cuda",
        output_dir: str = "",
        # Accept and ignore AnchorDeterminator-specific kwargs
        sdf_collision_checker=None,
        debug_output_dir=None,
        **kwargs,
    ):
        from baselines.Berkeley.keyframe_selection import (
            CLIPKeyframeSelector,
        )
        from baselines.ChatCam_GenDoP.scene_reconstruction import (
            read_images_binary,
            read_images_text,
        )

        self.clip_selector = CLIPKeyframeSelector(
            model_name=clip_model_name, device=device,
        )

        # Save directory for anchor images
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            (self.output_dir / "anchor_images").mkdir(parents=True, exist_ok=True)

        # Load training images
        images_path = Path(images_dir)
        self.image_paths = sorted(
            list(images_path.glob("*.jpg"))
            + list(images_path.glob("*.JPG"))
            + list(images_path.glob("*.png"))
            + list(images_path.glob("*.PNG"))
        )
        print(f"  [CLIPAnchorAdapter] Found {len(self.image_paths)} images in {images_dir}")

        # Load COLMAP poses
        colmap_dir = Path(colmap_model_dir)
        images_bin = colmap_dir / "images.bin"
        images_txt = colmap_dir / "images.txt"
        if images_bin.exists():
            self.camera_poses = read_images_binary(str(images_bin))
        elif images_txt.exists():
            self.camera_poses = read_images_text(str(images_txt))
        else:
            raise FileNotFoundError(
                f"No COLMAP images file found in {colmap_dir}"
            )
        print(f"  [CLIPAnchorAdapter] Loaded {len(self.camera_poses)} COLMAP poses")

        if self.image_paths:
            self._image_features = self.clip_selector.encode_images(self.image_paths)
        else:
            self._image_features = None

        self.bbox_data = {}
        self._build_bbox_index(bboxes)
        self.mesh = mesh

        # Track all selections for summary export
        self._selections: list = []

    def _build_bbox_index(self, bboxes):
        """Bbox index with corners and extent for ray-OBB intersection."""
        if isinstance(bboxes, dict):
            objects_dict = bboxes.get("objects", bboxes)
            for obj_id, obj_data in objects_dict.items():
                if "obb" not in obj_data:
                    continue
                obb = obj_data["obb"]
                center = np.array(obb[0:3], dtype=np.float64)
                size = np.array(obb[3:6], dtype=np.float64)
                qxyzw = np.array(obb[6:10], dtype=np.float64)
                label = obj_id.rsplit("_", 1)[0]

                # Compute corners for ray-OBB intersection
                from src.anchor_selector.anchor_selector import obb_to_corners
                corners = obb_to_corners(center, size, qxyzw)

                self.bbox_data[obj_id] = {
                    "id": obj_id,
                    "label": label,
                    "center": center,
                    "extent": size,
                    "corners": corners,
                }
        elif isinstance(bboxes, list):
            for item in bboxes:
                obj_id = item.get("ins_id")
                if "bounding_box" in item:
                    corners = np.array(
                        [[p["x"], p["y"], p["z"]] for p in item["bounding_box"]]
                    )
                    self.bbox_data[str(obj_id)] = {
                        "id": obj_id,
                        "label": item.get("label", ""),
                        "center": corners.mean(axis=0),
                        "corners": corners,
                    }

    def get_object_bbox(self, object_id):
        return self.bbox_data.get(str(object_id))

    def determine_anchor(
        self,
        object_id,
        object_label: str = "",
        viewing_prefs=None,
    ) -> CameraAnchor:
        import shutil

        if not object_label:
            bbox = self.get_object_bbox(object_id)
            object_label = bbox["label"] if bbox else str(object_id)

        prompt = f"a view of {object_label}"
        print(f"  [CLIPAnchor] Selecting keyframe for '{object_label}' (id={object_id})")
        print(f"  [CLIPAnchor] CLIP prompt: \"{prompt}\"")

        keyframes = self.clip_selector.select_keyframes(
            image_paths=self.image_paths,
            camera_poses=self.camera_poses,
            prompt_segments=[prompt],
            top_k=1,
        )

        if not keyframes:
            print(f"  [CLIPAnchor] WARNING: No keyframes found, using fallback")
            return self._fallback_anchor(object_id, object_label)

        kf = keyframes[0]
        position = kf["position"].astype(np.float64)
        rotation = kf["rotation"].astype(np.float64)  # (3, 3) c2w rotation

        # ── Ray-cast to find look-at target ──
        # Central ray: camera position + forward direction (z-axis of c2w)
        forward = rotation[:, 2]
        forward = forward / (np.linalg.norm(forward) + 1e-8)

        hit_obj_id, hit_point = self._raycast_first_obb(position, forward)

        if hit_point is not None:
            look_at = hit_point
            print(f"  [CLIPAnchor] Ray hit object '{hit_obj_id}' at "
                f"[{hit_point[0]:.2f}, {hit_point[1]:.2f}, {hit_point[2]:.2f}]")
            if hit_obj_id != str(object_id):
                print(f"  [CLIPAnchor] NOTE: Ray hit '{hit_obj_id}', "
                    f"not requested '{object_id}'")
        else:
            # No OBB hit — use forward projection as fallback
            look_at = position + forward * 2.0
            print(f"  [CLIPAnchor] No OBB hit, using forward projection as look-at")

        print(f"  [CLIPAnchor] Selected: {kf['image_name']} "
            f"(similarity={kf['similarity']:.3f})")
        print(f"    Position: {position}, Look at: {look_at}")

        # ── Save anchor image ──
        if self.output_dir:
            src_path = Path(kf["image_path"])
            dst_name = f"anchor_{object_id}_{object_label}_{kf['image_name']}"
            dst_path = self.output_dir / "anchor_images" / dst_name
            try:
                shutil.copy2(src_path, dst_path)
                print(f"    Saved anchor image → {dst_path.name}")
            except Exception as e:
                print(f"    Warning: could not save anchor image: {e}")

        # Track selection
        self._selections.append({
            "object_id": str(object_id),
            "object_label": object_label,
            "image_name": kf["image_name"],
            "similarity": float(kf["similarity"]),
            "position": position.tolist(),
            "look_at": look_at.tolist(),
            "ray_hit_object": hit_obj_id,
        })

        if self.output_dir:
            import json
            sel_path = self.output_dir / "anchor_images" / "selections.json"
            with open(sel_path, "w") as f:
                json.dump(self._selections, f, indent=2)

        return CameraAnchor(
            position=position,
            look_at=look_at.astype(np.float64),
            up=np.array([0.0, 0.0, 1.0]),
            score=float(kf["similarity"]),
            object_id=object_id,
            object_label=object_label,
        )

    def _fallback_anchor(self, object_id, object_label):
        """Fallback when CLIP selection fails."""
        bbox = self.get_object_bbox(object_id)
        if bbox is not None:
            center = bbox["center"]
            position = center + np.array([1.5, 0.0, 0.5])
            return CameraAnchor(
                position=position,
                look_at=center,
                up=np.array([0.0, 0.0, 1.0]),
                score=0.1,
                object_id=object_id,
                object_label=object_label,
            )
        return CameraAnchor(
            position=np.array([0.0, 0.0, 1.5]),
            look_at=np.zeros(3),
            up=np.array([0.0, 0.0, 1.0]),
            score=0.0,
            object_id=object_id,
            object_label=object_label,
        )

    def determine_path_anchors(self, object_ids, object_labels=None, viewing_preferences=None):
        """Multi-object convenience — same interface as AnchorDeterminator."""
        if object_labels is None:
            object_labels = [""] * len(object_ids)
        if viewing_preferences is None:
            viewing_preferences = {}
        anchors = []
        for obj_id, label in zip(object_ids, object_labels):
            try:
                anchor = self.determine_anchor(obj_id, label)
                anchors.append(anchor)
            except Exception as e:
                print(f"  Warning: CLIP anchor failed for {obj_id}: {e}")
        return anchors

    def _raycast_first_obb(
        self,
        ray_origin: np.ndarray,
        ray_dir: np.ndarray,
        max_dist: float = 20.0,
    ) -> tuple:
        """
        Cast a ray and find the first OBB it intersects.

        Uses slab method for ray-OBB intersection against all objects
        in bbox_data.

        Returns:
            (object_id, hit_point) or (None, None) if no intersection.
        """
        best_t = max_dist
        best_obj_id = None

        for obj_id, obj_data in self.bbox_data.items():
            corners = obj_data.get("corners")
            if corners is None:
                continue

            corners = np.array(corners, dtype=np.float64)
            center = obj_data["center"]

            # Get OBB axes and half-extents from corners
            extent = obj_data.get("extent")
            if extent is not None:
                extent = np.array(extent, dtype=np.float64)
                half_ext = extent / 2.0

                # Recover local axes from corners
                # obb_to_corners ordering: [0]=(-1,-1,-1), [4]=(1,-1,-1),
                # [2]=(-1,1,-1), [1]=(-1,-1,1)
                e_x = corners[4] - corners[0]
                e_y = corners[2] - corners[0]
                e_z = corners[1] - corners[0]

                axes = []
                for e, h in zip([e_x, e_y, e_z], half_ext):
                    n = np.linalg.norm(e)
                    if n > 1e-8:
                        axes.append(e / n)
                    else:
                        axes.append(np.array([1, 0, 0]))
                axes = np.array(axes)  # (3, 3) rows are axes
            else:
                # Fallback: axis-aligned
                aa_min = corners.min(axis=0)
                aa_max = corners.max(axis=0)
                half_ext = (aa_max - aa_min) / 2.0
                center = (aa_min + aa_max) / 2.0
                axes = np.eye(3)

            # Ray-OBB intersection via slab method
            t_hit = self._ray_obb_intersect(
                ray_origin, ray_dir, center, axes, half_ext,
            )

            if t_hit is not None and 0.01 < t_hit < best_t:
                best_t = t_hit
                best_obj_id = obj_id

        if best_obj_id is not None:
            hit_point = ray_origin + ray_dir * best_t
            return best_obj_id, hit_point
        return None, None

    @staticmethod
    def _ray_obb_intersect(
        ray_origin: np.ndarray,
        ray_dir: np.ndarray,
        obb_center: np.ndarray,
        obb_axes: np.ndarray,
        obb_half_extents: np.ndarray,
    ) -> float:
        """
        Ray-OBB intersection using the slab method.

        Args:
            ray_origin: (3,) ray start
            ray_dir: (3,) ray direction (unit)
            obb_center: (3,) OBB center
            obb_axes: (3, 3) rows are OBB local axes (unit)
            obb_half_extents: (3,) half-extents along each axis

        Returns:
            t value of first intersection, or None if no hit.
        """
        diff = obb_center - ray_origin

        t_min = -np.inf
        t_max = np.inf

        for i in range(3):
            axis = obb_axes[i]
            h = obb_half_extents[i]

            # Project ray onto this axis
            e = np.dot(axis, diff)
            f = np.dot(axis, ray_dir)

            if abs(f) > 1e-10:
                t1 = (e - h) / f
                t2 = (e + h) / f
                if t1 > t2:
                    t1, t2 = t2, t1
                t_min = max(t_min, t1)
                t_max = min(t_max, t2)
                if t_min > t_max:
                    return None
            else:
                # Ray parallel to slab
                if -e - h > 0 or -e + h < 0:
                    return None

        if t_max < 0:
            return None

        return t_min if t_min > 0 else t_max


# =============================================================================
# Ablation 2: w/o Parametric Trajectories → GenDoP direct 6-DoF generation
# =============================================================================

class GenDoPTrajectoryExecutor:
    """
    Wraps the standard TrajectoryExecutor but overrides trajectory generation
    to use CineGPTWrapper (GenDoP) instead of parametric trajectory classes.

    For each AtomTraj command, instead of instantiating CircularOrbit/DollyMove/etc.,
    this generates a text description and calls CineGPT to produce a 6-DoF pose
    sequence conditioned on the anchor position.
    """

    def __init__(self, base_executor, gendop_config=None):
        """
        Args:
            base_executor: A standard TrajectoryExecutor instance (for parsing,
                           anchors, coordinate conversion, etc.)
            gendop_config: Config dict for CineGPTWrapper (e.g., {"resume": "path/to/ckpt"})
        """
        from baselines.ChatCam_GenDoP.cinegpt_wrapper import CineGPTWrapper

        self.base = base_executor
        self.gendop = CineGPTWrapper(config=gendop_config or {})

        # Movement type to natural language description mapping
        self.movement_descriptions = {
            "orbit_full": "orbit 360 degrees around the object",
            "orbit_half": "orbit 180 degrees around the object",
            "orbit_quarter": "orbit 90 degrees to the right of the object",
            "pan_left": "pan the camera to the left",
            "pan_right": "pan the camera to the right",
            "move_in": "dolly forward toward the object",
            "move_out": "dolly backward away from the object",
            "zoom_in_out": "zoom in then zoom out on the object",
            "zoom_out_in": "zoom out then zoom in on the object",
            "crane": "crane shot rising up above the object",
            "tilt_up": "tilt the camera upward",
            "tilt_down": "tilt the camera downward",
            "static": "hold the camera still looking at the object",
            "arc": "arc transition between two viewpoints",
        }

    def execute_tools(self, cmd, start_anchor, end_anchor=None, n_frames=None):
        """
        Generate trajectory using GenDoP instead of parametric classes.
        
        Coordinate conventions:
        - GenDoP output: OpenGL (X=right, Y=up, Z=backward)
        - Pipeline expects: Z-up scene coordinates
        
        Steps:
        1. Generate raw c2ws from GenDoP (OpenGL convention)
        2. Convert GenDoP → COLMAP/OpenCV (flip Y and Z rotation cols)
        3. Align first frame to anchor pose in scene coordinates
        """
        config = self.base.MOVEMENT_CONFIGS.get(cmd.movement_type)
        if n_frames is None:
            n_frames = config.get("default_frames", 120) if config else 120

        # Build text prompt for GenDoP
        desc = self.movement_descriptions.get(
            cmd.movement_type,
            f"{cmd.movement_type} camera movement",
        )
        object_label = getattr(start_anchor, "object_label", "object")
        text_prompt = f"{desc} around {object_label}"

        print(f"  [GenDoP] Generating: \"{text_prompt}\" ({n_frames} frames)")

        # Step 1: Generate raw trajectory from GenDoP
        gendop_result = self.gendop.generate(text=text_prompt)
        raw_c2ws = gendop_result["c2ws"]  # (N, 4, 4) in GenDoP's OpenGL convention

        # Resample to desired frame count
        if len(raw_c2ws) != n_frames:
            indices = np.linspace(0, len(raw_c2ws) - 1, n_frames).astype(int)
            raw_c2ws = raw_c2ws[indices]

        # Step 2: Convert GenDoP (OpenGL) → COLMAP/OpenCV convention
        # GenDoP: X=right, Y=up, Z=backward
        # COLMAP: X=right, Y=down, Z=forward
        # Flip Y and Z columns of rotation, keep translation unchanged
        conv_rot = np.diag([1.0, -1.0, -1.0]).astype(np.float64)
        for i in range(len(raw_c2ws)):
            raw_c2ws[i][:3, :3] = raw_c2ws[i][:3, :3] @ conv_rot

        # Step 3: Align to scene coordinates via anchor pose
        # Build anchor c2w in scene Z-up coordinates
        anchor_pos_zup = start_anchor.position.copy()
        anchor_look_at = start_anchor.look_at.copy()

        forward = anchor_look_at - anchor_pos_zup
        forward_norm = np.linalg.norm(forward)
        if forward_norm > 1e-6:
            forward = forward / forward_norm
        else:
            forward = np.array([1.0, 0.0, 0.0])

        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, up)
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-6:
            right = np.array([0.0, 1.0, 0.0])
        else:
            right = right / right_norm
        up = np.cross(right, forward)
        up = up / (np.linalg.norm(up) + 1e-8)

        # Anchor c2w: columns are [right, up, forward] with position
        # This matches COLMAP convention where camera looks along +Z
        anchor_c2w = np.eye(4, dtype=np.float64)
        anchor_c2w[:3, 0] = right
        anchor_c2w[:3, 1] = -up       # COLMAP Y = down
        anchor_c2w[:3, 2] = forward   # COLMAP Z = forward
        anchor_c2w[:3, 3] = anchor_pos_zup

        # Compute relative transform: align GenDoP frame 0 to anchor
        first_c2w = raw_c2ws[0].astype(np.float64).copy()
        first_c2w_inv = np.linalg.inv(first_c2w)
        alignment = anchor_c2w @ first_c2w_inv

        # Apply alignment to all frames
        c2w_aligned = np.array(
            [alignment @ c2w.astype(np.float64) for c2w in raw_c2ws]
        )

        # ── For transitional arcs: force endpoints to match both anchors ──
        if cmd.movement_category == "transitional" and end_anchor is not None:
            start_pos = start_anchor.position.copy()
            end_pos = end_anchor.position.copy()
            end_look_at = end_anchor.look_at.copy()

            # Build end anchor c2w
            end_forward = end_look_at - end_pos
            end_forward_norm = np.linalg.norm(end_forward)
            if end_forward_norm > 1e-6:
                end_forward = end_forward / end_forward_norm
            else:
                end_forward = np.array([1.0, 0.0, 0.0])

            end_up = np.array([0.0, 0.0, 1.0])
            end_right = np.cross(end_forward, end_up)
            end_right_norm = np.linalg.norm(end_right)
            if end_right_norm < 1e-6:
                end_right = np.array([0.0, 1.0, 0.0])
            else:
                end_right = end_right / end_right_norm
            end_up = np.cross(end_right, end_forward)
            end_up = end_up / (np.linalg.norm(end_up) + 1e-8)

            end_c2w = np.eye(4, dtype=np.float64)
            end_c2w[:3, 0] = end_right
            end_c2w[:3, 1] = -end_up
            end_c2w[:3, 2] = end_forward
            end_c2w[:3, 3] = end_pos

            # Current first/last frames after alignment
            aligned_first = c2w_aligned[0]
            aligned_last = c2w_aligned[-1]

            # Compute how GenDoP's internal motion progresses (normalized offsets)
            # relative to its own start/end
            aligned_positions = c2w_aligned[:, :3, 3].copy()
            aligned_rotations = c2w_aligned[:, :3, :3].copy()

            first_pos = aligned_positions[0]
            last_pos = aligned_positions[-1]
            internal_span = last_pos - first_pos
            internal_len = np.linalg.norm(internal_span)

            # Target span
            target_span = end_pos - start_pos

            # For each frame, compute blend parameter t based on
            # how far it progressed in GenDoP's own trajectory
            n = len(c2w_aligned)
            for j in range(n):
                if internal_len > 1e-6:
                    # Project onto GenDoP's internal motion direction
                    progress = np.dot(
                        aligned_positions[j] - first_pos, internal_span
                    ) / (internal_len ** 2)
                else:
                    progress = j / max(n - 1, 1)

                t = np.clip(progress, 0.0, 1.0)

                # Smooth easing
                t_smooth = t * t * (3 - 2 * t)  # smoothstep

                # Interpolate position between start and end anchor
                new_pos = start_pos + t_smooth * target_span

                # Add GenDoP's lateral offset (perpendicular to direct path)
                if internal_len > 1e-6:
                    direct_dir = internal_span / internal_len
                    offset = aligned_positions[j] - first_pos
                    along = np.dot(offset, direct_dir) * direct_dir
                    lateral = offset - along
                    # Scale lateral offset by ratio of target/internal distance
                    target_len = np.linalg.norm(target_span)
                    scale = target_len / internal_len if internal_len > 1e-6 else 1.0
                    scale = min(scale, 3.0)  # cap to avoid wild offsets
                    new_pos = new_pos + lateral * scale

                c2w_aligned[j, :3, 3] = new_pos

                # Interpolate rotation via SLERP between start and end
                from scipy.spatial.transform import Rotation, Slerp
                if j == 0:
                    c2w_aligned[j, :3, :3] = anchor_c2w[:3, :3]
                elif j == n - 1:
                    c2w_aligned[j, :3, :3] = end_c2w[:3, :3]
                else:
                    r_start = Rotation.from_matrix(anchor_c2w[:3, :3])
                    r_end = Rotation.from_matrix(end_c2w[:3, :3])
                    slerp = Slerp([0.0, 1.0], Rotation.concatenate([r_start, r_end]))
                    c2w_aligned[j, :3, :3] = slerp(t_smooth).as_matrix()

            print(f"    [GenDoP] Arc endpoints pinned: "
                  f"[{start_pos[0]:.2f},{start_pos[1]:.2f},{start_pos[2]:.2f}] → "
                  f"[{end_pos[0]:.2f},{end_pos[1]:.2f},{end_pos[2]:.2f}]")

        c2w_zup = c2w_aligned

        # Build output dict matching TrajectoryExecutor format
        result = {
            "c2w": c2w_zup,
            "positions": c2w_zup[:, :3, 3],
            "rotations_matrix": c2w_zup[:, :3, :3],
            "n_frames": n_frames,
            "movement_type": cmd.movement_type,
            "movement_category": cmd.movement_category,
            "arc_angle": cmd.arc_angle,
            "start_anchor": start_anchor,
            "end_anchor": end_anchor,
            "coordinate_system": "z-up",
            "generation_method": "gendop",
        }

        # Handle focal multiplier for zoom movements
        if "zoom" in cmd.movement_type:
            intrinsics = gendop_result.get("intrinsics", {})
            if "fx_sequence" in intrinsics:
                base_fx = intrinsics.get("fx", 500.0)
                fx_seq = np.array(intrinsics["fx_sequence"])
                if len(fx_seq) != n_frames:
                    indices = np.linspace(0, len(fx_seq) - 1, n_frames).astype(int)
                    fx_seq = fx_seq[indices]
                result["focal_multiplier"] = fx_seq / base_fx
            else:
                result["focal_multiplier"] = np.ones(n_frames)

        print(f"    [GenDoP] Position range (Z-up): "
            f"{result['positions'].min(axis=0).round(2)} to "
            f"{result['positions'].max(axis=0).round(2)}")

        return result