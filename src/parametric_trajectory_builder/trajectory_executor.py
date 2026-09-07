"""
Trajectory Executor (Simplified)
================================

Executes camera trajectory plans from the LLM dialog system.

Movement vocabulary:
  Object-level:  orbit_full, orbit_half, orbit_quarter, pan_left, pan_right,
                 move_in, move_out, zoom_in_out, zoom_out_in,
                 crane, tilt_up, tilt_down, static
  Transitional:  arc (with angle parameter)

Maps to parametric trajectory classes:
  CircularOrbit  → orbit_full, orbit_half, orbit_quarter
  StationaryPan  → pan_left, pan_right, static (static uses yaw_delta=0)
  StationaryTilt → tilt_up, tilt_down
  DollyMove      → move_in, move_out
  CraneShot      → crane
  ZoomLens       → zoom_in_out, zoom_out_in
  TransitionalArc → arc

NEW: zoom_in_out / zoom_out_in — stationary camera with focal-length ramp.
     The camera stays still; only focal length changes and returns to
     original.  The output includes a 'focal_multiplier' array.

UPDATED: move_in / move_out — DollyMove now translates along a 3D ray
     (dolly on a ramp), so start_radius / end_radius are 3D Euclidean
     distances from object_center.  The executor now computes and passes
     the 3D distance instead of horizontal-only radius.

UPDATED: Supports new scene graph JSON format with string IDs ("table_0",
     "sofa_0") alongside legacy labels.json with integer IDs.

UPDATED: Supports SDF-based collision checking in AnchorDeterminator
     (pass sdf_collision_checker or sdf_npz_path). Candidate debug
     visualization via debug_candidates flag.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import json
import re
import numpy as np
import trimesh
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
from pathlib import Path

from src.anchor_selector.anchor_selector import (
    AnchorDeterminator,
    CameraAnchor,
    SDFCollisionChecker,
    load_mesh,
)
from src.parametric_trajectory_builder.parametric_trajectories import (
    ParametricTrajectoryBase,
    TransitionalArc,
    CircularOrbit,
    StationaryPan,
    StationaryTilt,
    DollyMove,
    CraneShot,
    ZoomLens,
)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class AnchorCommand:
    """Command to determine anchor point for an object."""
    object_label: str
    object_id: Union[int, str]


@dataclass
class AtomTrajCommand:
    """Command to generate camera trajectory."""
    movement_type: str       # e.g., 'orbit_full', 'pan_left', 'arc', 'move_in', 'zoom_in_out', 'static'
    movement_category: str   # 'object-level' or 'transitional'
    arc_angle: float = 0.0   # Only used when movement_type == 'arc'


@dataclass
class TrajectoryPlan:
    """Parsed trajectory plan from LLM output."""
    observation: str
    reasoning: str
    object_sequence: List[str]
    commands: List[Union[AnchorCommand, AtomTrajCommand]] = field(default_factory=list)


@dataclass
class TrajectorySegment:
    """A single trajectory segment."""
    start_anchor: CameraAnchor
    end_anchor: Optional[CameraAnchor]
    movement_type: str
    movement_category: str
    arc_angle: float = 0.0
    trajectory_output: Optional[Dict[str, Any]] = None


@dataclass
class TrajectoryResult:
    """Result of trajectory execution."""
    segments: List[TrajectorySegment] = field(default_factory=list)
    all_anchors: List[CameraAnchor] = field(default_factory=list)
    trajectory_outputs: List[Dict[str, Any]] = field(default_factory=list)
    trajectory_paths: List[Path] = field(default_factory=list)


# =============================================================================
# Executor
# =============================================================================

class TrajectoryExecutor:
    """
    Executes trajectory plans using the simplified parametric vocabulary.
    """
    
    # --- Movement configurations ---
    MOVEMENT_CONFIGS = {
        # Object-level: CircularOrbit
        'orbit_full': {
            'class': CircularOrbit,
            'category': 'object-level',
            'default_params': {'angle_span': 360, 'radius': 2.5, 'height': 0.5, 'pitch_offset': 0},
            'default_frames': 240,
        },
        'orbit_half': {
            'class': CircularOrbit,
            'category': 'object-level',
            'default_params': {'angle_span': 180, 'radius': 2.5, 'height': 0.5, 'pitch_offset': 0},
            'default_frames': 120,
        },
        'orbit_quarter': {
            'class': CircularOrbit,
            'category': 'object-level',
            'default_params': {'angle_span': 90, 'radius': 2.5, 'height': 0.5, 'pitch_offset': 0},
            'default_frames': 60,
        },
        # Object-level: StationaryPan
        'pan_left': {
            'class': StationaryPan,
            'category': 'object-level',
            'default_params': {'yaw_delta': -60},
            'default_frames': 120,
        },
        'pan_right': {
            'class': StationaryPan,
            'category': 'object-level',
            'default_params': {'yaw_delta': 60},
            'default_frames': 120,
        },
        # Object-level: DollyMove
        'move_in': {
            'class': DollyMove,
            'category': 'object-level',
            'default_params': {'radius_ratio': 0.35},
            'default_frames': 120,
        },
        'move_out': {
            'class': DollyMove,
            'category': 'object-level',
            'default_params': {'radius_ratio': 2.5},
            'default_frames': 120,
        },
        # Object-level: ZoomLens
        'zoom_in_out': {
            'class': ZoomLens,
            'category': 'object-level',
            'default_params': {'peak_multiplier': 2.0},
            'default_frames': 120,
        },
        'zoom_out_in': {
            'class': ZoomLens,
            'category': 'object-level',
            'default_params': {'peak_multiplier': 0.5},
            'default_frames': 120,
        },
        # Object-level: CraneShot
        'crane': {
            'class': CraneShot,
            'category': 'object-level',
            'default_params': {
                'end_elevation': 85,
                'pitch_offset': 0,
            },
            'default_frames': 180,
        },
        # Object-level: StationaryTilt
        'tilt_up': {
            'class': StationaryTilt,
            'category': 'object-level',
            'default_params': {'pitch_delta': -45},
            'default_frames': 90,
        },
        'tilt_down': {
            'class': StationaryTilt,
            'category': 'object-level',
            'default_params': {'pitch_delta': 45},
            'default_frames': 90,
        },
        # Object-level: Static
        'static': {
            'class': StationaryPan,
            'category': 'object-level',
            'default_params': {'yaw_delta': 0},
            'default_frames': 60,
        },
        # Transitional: Arc
        'arc': {
            'class': TransitionalArc,
            'category': 'transitional',
            'default_params': {'arc_angle': 0.0},
            'default_frames': 60,
        },
    }
    
    def __init__(
        self,
        project_root: Union[str, Path],
        bbox_path: Union[str, Path],
        mesh_path: Union[str, Path],
        workspace: str = "outputs",
        fps: float = 30.0,
        viewing_preferences: Optional[Dict[str, Dict]] = None,
        # ── NEW: SDF collision checking ──
        sdf_collision_checker: Optional["SDFCollisionChecker"] = None,
        sdf_npz_path: Optional[str] = None,
        collision_margin: float = 0.10,
        # ── NEW: Debug candidate visualization ──
        debug_candidates: bool = False,
        **kwargs,
    ):
        """
        Args:
            project_root:          Project root directory.
            bbox_path:             Path to scene graph JSON or labels.json.
            mesh_path:             Path to collision mesh (PLY/OBJ/USD).
            workspace:             Output workspace directory.
            fps:                   Frames per second.
            viewing_preferences:   Per-object viewing preference hints from LLM.
            sdf_collision_checker: Pre-built SDFCollisionChecker instance.
                                   If provided, used for anchor collision checks.
            sdf_npz_path:          Path to SDF .npz file. If sdf_collision_checker
                                   is None but this is provided, builds one.
            collision_margin:      SDF collision margin (meters). Default 0.10.
            debug_candidates:      If True, saves candidate visualizations per object.
        """
        self.project_root = Path(project_root)
        self.workspace = workspace
        self.fps = fps
        
        self.viewing_preferences = viewing_preferences or {}
        
        with open(bbox_path, 'r') as f:
            self.bboxes = json.load(f)
        
        # Detect format
        if isinstance(self.bboxes, dict) and "objects" in self.bboxes:
            self._bbox_format = "scene_graph"
            print(f"  [Executor] Detected scene graph format with {len(self.bboxes['objects'])} objects")
        elif isinstance(self.bboxes, dict):
            self._bbox_format = "scene_graph"
            print(f"  [Executor] Detected scene graph objects dict with {len(self.bboxes)} entries")
        else:
            self._bbox_format = "legacy"
            print(f"  [Executor] Detected legacy labels format with {len(self.bboxes)} entries")
        
        # Load mesh
        mesh_ext = Path(mesh_path).suffix.lower()
        if mesh_ext in ('.usd', '.usda', '.usdc'):
            self.mesh = load_mesh(str(mesh_path))
        else:
            self.mesh = trimesh.load(str(mesh_path), force='mesh')
            print(f"Loaded mesh: {len(self.mesh.vertices)} verts, {len(self.mesh.faces)} faces")
        
        # ── Build or reuse SDF collision checker ──
        if sdf_collision_checker is not None:
            self._sdf_checker = sdf_collision_checker
        elif sdf_npz_path is not None:
            print(f"  [Executor] Loading SDF from {sdf_npz_path}")
            self._sdf_checker = SDFCollisionChecker.from_npz(
                sdf_npz_path, collision_margin=collision_margin,
            )
        else:
            self._sdf_checker = None
        
        # ── Debug output directory ──
        debug_dir = None
        if debug_candidates:
            debug_dir = str(self.project_root / workspace / "debug_candidates")
            print(f"  [Executor] Candidate debug output → {debug_dir}")
        
        # ── Build AnchorDeterminator with SDF + debug support ──
        self.anchor_determinator = AnchorDeterminator(
            bboxes=self.bboxes,
            mesh=self.mesh,
            sdf_collision_checker=self._sdf_checker,
            debug_output_dir=debug_dir,
        )
        
        self._anchor_cache: Dict[Union[int, str], CameraAnchor] = {}

    # =========================================================================
    # Parsing
    # =========================================================================

    def parse_llm_output(self, llm_output: Union[str, Dict]) -> TrajectoryPlan:
        """Parse LLM JSON output into a TrajectoryPlan."""
        if isinstance(llm_output, str):
            data = json.loads(llm_output)
        else:
            data = llm_output
        
        print(f"\n=== Parsing LLM Output ===")
        
        if not self.viewing_preferences:
            llm_prefs = data.get('viewing_preferences', {})
            if llm_prefs and isinstance(llm_prefs, dict):
                self.viewing_preferences = llm_prefs
                print(f"  Extracted viewing_preferences for {len(llm_prefs)} objects")
        
        atomic_str = None
        for key in ['atomic_trajectories', 'atomic trajectories', 'atomicTrajectories']:
            if key in data:
                atomic_str = data[key]
                break
        
        if atomic_str is None:
            print("WARNING: No atomic trajectories found!")
            atomic_str = ""
        
        commands = self._parse_atomic_trajectories(str(atomic_str))
        
        plan = TrajectoryPlan(
            observation=data.get('observation', ''),
            reasoning=data.get('reasoning', ''),
            object_sequence=data.get('object_sequence', []),
            commands=commands,
        )
        
        print(f"Parsed {len(commands)} commands:")
        for i, cmd in enumerate(commands):
            if isinstance(cmd, AnchorCommand):
                print(f"  {i+1}. Anchor: {cmd.object_label} (id: {cmd.object_id})")
            elif isinstance(cmd, AtomTrajCommand):
                extra = f", angle={cmd.arc_angle}" if cmd.movement_type == 'arc' else ""
                print(f"  {i+1}. AtomTraj: {cmd.movement_type} ({cmd.movement_category}{extra})")
        
        return plan

    def _parse_atomic_trajectories(self, atomic_str: str) -> List[Union[AnchorCommand, AtomTrajCommand]]:
        """Parse the atomic trajectories string into commands."""
        commands = []
        seen_anchor_ids = set()
        
        if not atomic_str.strip():
            return commands
        
        steps = re.split(r'\d+\.\s+', atomic_str)
        steps = [s.strip() for s in steps if s.strip()]
        
        for step in steps:
            step_lower = step.lower()
            
            if any(kw in step_lower for kw in ['traj_compose', 'render', 'connect']):
                continue
            
            if 'anchor' in step_lower:
                cmd = self._parse_anchor_command(step)
                if cmd and cmd.object_id not in seen_anchor_ids:
                    commands.append(cmd)
                    seen_anchor_ids.add(cmd.object_id)
                    
            elif 'atomtraj' in step_lower:
                cmd = self._parse_AtomTraj_command(step)
                if cmd:
                    commands.append(cmd)
        
        return commands

    def _parse_anchor_command(self, step: str) -> Optional[AnchorCommand]:
        """Parse an Anchor Determinator command."""
        # The LLM occasionally closes the quote after the id instead of before
        # it — "with 'sink (id: sink_0)'" rather than "with 'sink' (id: sink_0)".
        # Match that first; the patterns below require the id outside the quotes
        # and would otherwise drop the anchor silently.
        match = re.search(
            r"['\"]([^'\"]+?)\s*\(id:\s*([^\)]+?)\)\s*['\"]",
            step, re.IGNORECASE,
        )
        if match:
            label = match.group(1).strip()
            raw_id = match.group(2).strip()
            obj_id = self._parse_id(raw_id)
            return AnchorCommand(object_label=label, object_id=obj_id)

        match = re.search(
            r"['\"]([^'\"]+)['\"].*?\(id:\s*([^\)]+)\)",
            step, re.IGNORECASE,
        )
        if match:
            label = match.group(1).strip()
            raw_id = match.group(2).strip()
            obj_id = self._parse_id(raw_id)
            return AnchorCommand(object_label=label, object_id=obj_id)
        
        match = re.search(
            r"['\"]([^'\"]+)['\"].*?id[=:]\s*(\S+)",
            step, re.IGNORECASE,
        )
        if match:
            label = match.group(1).strip()
            raw_id = match.group(2).strip().rstrip(".,;)")
            obj_id = self._parse_id(raw_id)
            return AnchorCommand(object_label=label, object_id=obj_id)
        
        match = re.search(
            r"with\s+(\w+(?:\s+\w+)?)\s*\(id:\s*([^\)]+)\)",
            step, re.IGNORECASE,
        )
        if match:
            label = match.group(1).strip()
            raw_id = match.group(2).strip()
            obj_id = self._parse_id(raw_id)
            return AnchorCommand(object_label=label, object_id=obj_id)
        
        return None

    @staticmethod
    def _parse_id(raw_id: str) -> Union[int, str]:
        raw_id = raw_id.strip().rstrip(".,;)")
        try:
            return int(raw_id)
        except ValueError:
            return raw_id

    def _parse_AtomTraj_command(self, step: str) -> Optional[AtomTrajCommand]:
        """Parse a AtomTraj command."""
        pattern = r"['\"](\w+)['\"](?:,\s*angle\s*=\s*(-?\d+(?:\.\d+)?))?\s*\((object-level|transitional)\)"
        match = re.search(pattern, step, re.IGNORECASE)
        
        if match:
            movement_type = match.group(1).lower()
            arc_angle = float(match.group(2)) if match.group(2) else 0.0
            movement_category = match.group(3).lower()
            
            if movement_type not in self.MOVEMENT_CONFIGS:
                print(f"  [WARNING] Unknown movement '{movement_type}', defaulting")
                if movement_category == 'transitional':
                    movement_type = 'arc'
                else:
                    movement_type = 'static'
            
            return AtomTrajCommand(
                movement_type=movement_type,
                movement_category=movement_category,
                arc_angle=arc_angle,
            )
        
        return None

    # =========================================================================
    # Anchor Management
    # =========================================================================

    def get_anchor(self, object_id: Union[int, str], object_label: str = "") -> CameraAnchor:
        """Get anchor for an object, using cache if available."""
        if object_id in self._anchor_cache:
            return self._anchor_cache[object_id]
        
        prefs = self._lookup_viewing_prefs(object_id, object_label)
        
        if prefs:
            print(f"  Determining anchor for: {object_label} (id: {object_id}) "
                  f"[prefs: elevation={prefs.get('elevation', '?')}, "
                  f"distance={prefs.get('distance', '?')}]")
        else:
            print(f"  Determining anchor for: {object_label} (id: {object_id}) "
                  f"[no prefs — geometry fallback]")
        
        anchor = self.anchor_determinator.determine_anchor(
            object_id=object_id,
            object_label=object_label,
            viewing_prefs=prefs,
        )
        print(f"    Position: {anchor.position}, Look at: {anchor.look_at}, Score: {anchor.score:.3f}")
        
        self._anchor_cache[object_id] = anchor
        return anchor

    def _lookup_viewing_prefs(
        self,
        object_id: Union[int, str] = None,
        object_label: str = "",
    ) -> Optional[Dict]:
        """Look up viewing preferences by ID first, then by label (fuzzy match)."""
        if not self.viewing_preferences:
            return None
        
        if object_id is not None:
            obj_id_str = str(object_id)
            if obj_id_str in self.viewing_preferences:
                return self.viewing_preferences[obj_id_str]
        
        if not object_label:
            return None
        
        if object_label in self.viewing_preferences:
            return self.viewing_preferences[object_label]
        
        normalised = object_label.lower().replace('_', ' ').strip()
        for key, prefs in self.viewing_preferences.items():
            if key.lower().replace('_', ' ').strip() == normalised:
                return prefs
        
        for key, prefs in self.viewing_preferences.items():
            key_norm = key.lower().replace('_', ' ').strip()
            if normalised in key_norm or key_norm in normalised:
                return prefs
        
        return None

    # =========================================================================
    # Coordinate Conversion
    # =========================================================================

    def _zup_to_yup(self, point: np.ndarray) -> np.ndarray:
        return np.array([point[0], point[2], point[1]])
    
    def _yup_to_zup(self, point: np.ndarray) -> np.ndarray:
        return np.array([point[0], point[2], point[1]])

    def _anchor_to_pose(self, anchor: CameraAnchor) -> Dict[str, np.ndarray]:
        """Convert anchor to Y-up position + euler rotation."""
        position = anchor.position.copy()
        look_at = anchor.look_at.copy()
        
        position_yup = self._zup_to_yup(position)
        look_at_yup = self._zup_to_yup(look_at)
        
        forward = look_at_yup - position_yup
        forward_norm = np.linalg.norm(forward)
        if forward_norm < 1e-6:
            forward = np.array([0.0, 0.0, 1.0])
        else:
            forward = forward / forward_norm
        
        yaw = np.rad2deg(np.arctan2(forward[0], forward[2]))
        pitch = np.rad2deg(np.arcsin(-forward[1]))
        
        return {
            'position': position_yup,
            'position_zup': position,
            'rotation': np.array([pitch, yaw, 0.0]),
            'look_at_yup': look_at_yup,
        }

    def _convert_c2w_yup_to_zup(self, c2w_yup: np.ndarray) -> np.ndarray:
        """Convert c2w matrices from Y-up to Z-up."""
        S = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.float64)
        Rx_m90 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]], dtype=np.float64)

        t_zup = (S @ c2w_yup[:, :3, 3][..., None])[..., 0]
        R_zup = S @ c2w_yup[:, :3, :3] @ S.T @ Rx_m90

        c2w_zup = np.tile(np.eye(4, dtype=np.float64), (c2w_yup.shape[0], 1, 1))
        c2w_zup[:, :3, :3] = R_zup
        c2w_zup[:, :3, 3] = t_zup
        return c2w_zup

    # =========================================================================
    # Trajectory Creation
    # =========================================================================

    def _compute_anchor_pitch_offset(
        self, anchor: CameraAnchor,
        object_center_yup: np.ndarray,
        camera_pos_yup: np.ndarray,
    ) -> float:
        """Compute pitch offset to match anchor's original viewing angle."""
        offset_to_center = object_center_yup - camera_pos_yup
        h_dist = np.linalg.norm(offset_to_center[[0, 2]])
        orbit_pitch = np.rad2deg(np.arctan2(offset_to_center[1], h_dist)) if h_dist > 1e-6 else 0.0

        look_at_yup = self._zup_to_yup(anchor.look_at)
        offset_to_look = look_at_yup - camera_pos_yup
        h_dist_look = np.linalg.norm(offset_to_look[[0, 2]])
        anchor_pitch = np.rad2deg(np.arctan2(offset_to_look[1], h_dist_look)) if h_dist_look > 1e-6 else 0.0

        return anchor_pitch - orbit_pitch

    def _create_object_centric_trajectory(
        self,
        config: Dict,
        anchor: CameraAnchor,
        start_pose: Dict[str, np.ndarray],
        movement_type: str,
    ) -> ParametricTrajectoryBase:
        """Create orbit, pan, tilt, dolly, crane, zoom, or static trajectory."""
        TrajectoryClass = config['class']
        defaults = config.get('default_params', {}).copy()
        
        object_center_yup = self._zup_to_yup(anchor.look_at)
        camera_pos_yup = start_pose['position']
        
        offset = camera_pos_yup - object_center_yup
        h_radius = np.linalg.norm(offset[[0, 2]])
        height = offset[1]
        start_angle = np.rad2deg(np.arctan2(offset[2], offset[0]))
        
        if TrajectoryClass == CircularOrbit:
            angle_span = defaults.get('angle_span', 180)
            pitch_offset = self._compute_anchor_pitch_offset(anchor, object_center_yup, camera_pos_yup)
            pitch_offset += defaults.get('pitch_offset', 0)
            
            return CircularOrbit(
                object_center=tuple(object_center_yup),
                radius=max(h_radius, 0.5),
                height=height,
                start_angle=start_angle,
                end_angle=start_angle + angle_span,
                pitch_offset=pitch_offset,
                name=movement_type,
            )
        
        elif TrajectoryClass == StationaryPan:
            yaw_delta = defaults.get('yaw_delta', 0)
            base_yaw = start_pose['rotation'][1]
            base_pitch = start_pose['rotation'][0]
            
            return StationaryPan(
                camera_position=tuple(camera_pos_yup),
                start_yaw=base_yaw,
                end_yaw=base_yaw + yaw_delta,
                pitch=base_pitch,
                name=movement_type,
            )
        
        elif TrajectoryClass == StationaryTilt:
            pitch_delta = defaults.get('pitch_delta', 0)
            base_yaw = start_pose['rotation'][1]
            base_pitch = start_pose['rotation'][0]
            
            return StationaryTilt(
                camera_position=tuple(camera_pos_yup),
                start_pitch=base_pitch,
                end_pitch=base_pitch + pitch_delta,
                yaw=base_yaw,
                name=movement_type,
            )
        
        elif TrajectoryClass == DollyMove:
            dist_3d = np.linalg.norm(offset)
            start_radius = max(dist_3d, 0.5)

            radius_ratio = defaults.get('radius_ratio', 1.0)
            end_radius = start_radius * radius_ratio
            end_radius = np.clip(end_radius, 0.3, 20.0)

            approach_angle = start_angle

            pitch_offset = self._compute_anchor_pitch_offset(
                anchor, object_center_yup, camera_pos_yup
            )
            
            return DollyMove(
                object_center=tuple(object_center_yup),
                start_radius=start_radius,
                end_radius=end_radius,
                height=height,
                approach_angle=approach_angle,
                pitch_offset=pitch_offset,
                name=movement_type,
            )
        
        elif TrajectoryClass == ZoomLens:
            peak_multiplier = defaults.get('peak_multiplier', 2.0)
            base_yaw = start_pose['rotation'][1]
            base_pitch = start_pose['rotation'][0]
            
            return ZoomLens(
                camera_position=tuple(camera_pos_yup),
                peak_multiplier=peak_multiplier,
                yaw=base_yaw,
                pitch=base_pitch,
                name=movement_type,
            )
        
        elif TrajectoryClass == CraneShot:
            end_elevation = defaults.get('end_elevation', 85.0)
            
            pitch_offset = self._compute_anchor_pitch_offset(
                anchor, object_center_yup, camera_pos_yup
            )
            pitch_offset += defaults.get('pitch_offset', 0)
            
            start_elevation = np.rad2deg(np.arctan2(height, max(h_radius, 0.01)))
            orbit_radius = np.sqrt(h_radius**2 + height**2)
            
            return CraneShot(
                object_center=tuple(object_center_yup),
                radius=max(orbit_radius, 0.5),
                start_elevation=start_elevation,
                end_elevation=end_elevation,
                approach_angle=start_angle,
                pitch_offset=pitch_offset,
                name=movement_type,
            )
        
        # Fallback: static hold
        return StationaryPan(
            camera_position=tuple(camera_pos_yup),
            start_yaw=start_pose['rotation'][1],
            end_yaw=start_pose['rotation'][1],
            pitch=start_pose['rotation'][0],
            name=movement_type,
        )

    def _create_transitional_trajectory(
        self,
        config: Dict,
        start_pose: Dict[str, np.ndarray],
        end_pose: Dict[str, np.ndarray],
        arc_angle: float = 0.0,
    ) -> TransitionalArc:
        """Create an arc transition between two poses."""
        return TransitionalArc(
            start_position=tuple(start_pose['position']),
            start_rotation=tuple(start_pose['rotation']),
            end_position=tuple(end_pose['position']),
            end_rotation=tuple(end_pose['rotation']),
            arc_angle=arc_angle,
        )

    # =========================================================================
    # Execution
    # =========================================================================

    def execute_tools(
        self,
        cmd: AtomTrajCommand,
        start_anchor: CameraAnchor,
        end_anchor: Optional[CameraAnchor] = None,
        n_frames: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Execute a single trajectory command."""
        config = self.MOVEMENT_CONFIGS.get(cmd.movement_type, self.MOVEMENT_CONFIGS['arc'])
        
        if n_frames is None:
            n_frames = config.get('default_frames', 60)
        
        start_pose = self._anchor_to_pose(start_anchor)
        end_pose = self._anchor_to_pose(end_anchor) if end_anchor else None
        
        print(f"  Executing: {cmd.movement_type} ({cmd.movement_category}), {n_frames} frames")
        
        if cmd.movement_category == 'object-level':
            trajectory = self._create_object_centric_trajectory(
                config, start_anchor, start_pose, cmd.movement_type
            )
        else:
            if end_pose is None:
                end_pose = start_pose
            arc_angle = cmd.arc_angle if cmd.arc_angle != 0.0 else config['default_params'].get('arc_angle', 0.0)
            trajectory = self._create_transitional_trajectory(
                config, start_pose, end_pose, arc_angle
            )
        
        # Generate in Y-up, convert to Z-up
        result = trajectory.generate(n_frames, fps=int(self.fps))
        
        c2w_zup = self._convert_c2w_yup_to_zup(result['c2w'])
        result['c2w'] = c2w_zup
        result['positions'] = c2w_zup[:, :3, 3]
        result['rotations_matrix'] = c2w_zup[:, :3, :3]
        result['movement_type'] = cmd.movement_type
        result['movement_category'] = cmd.movement_category
        result['arc_angle'] = cmd.arc_angle
        result['start_anchor'] = start_anchor
        result['end_anchor'] = end_anchor
        result['coordinate_system'] = 'z-up'
        
        print(f"    Position range (Z-up): {result['positions'].min(axis=0).round(2)} to {result['positions'].max(axis=0).round(2)}")
        if 'focal_multiplier' in result:
            fm = result['focal_multiplier']
            print(f"    Focal multiplier range: [{fm.min():.2f}, {fm.max():.2f}]")
        
        return result

    def execute_plan(
        self,
        plan: TrajectoryPlan,
        base_name: str = "trajectory",
    ) -> TrajectoryResult:
        """Execute a full trajectory plan."""
        result = TrajectoryResult()
        traj_count = 0
        current_anchor: Optional[CameraAnchor] = None
        pending_anchors: List[CameraAnchor] = []
        
        print(f"\n=== Executing Plan: {len(plan.commands)} commands ===")

        output_dir = self.project_root / self.workspace / base_name
        if output_dir.exists():
            import shutil
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for i, cmd in enumerate(plan.commands):
            if isinstance(cmd, AnchorCommand):
                anchor = self.get_anchor(cmd.object_id, cmd.object_label)
                result.all_anchors.append(anchor)
                
                if current_anchor is None:
                    current_anchor = anchor
                else:
                    pending_anchors.append(anchor)
            
            elif isinstance(cmd, AtomTrajCommand):
                if current_anchor is None:
                    continue
                
                end_anchor = None
                if cmd.movement_category == 'transitional' and pending_anchors:
                    end_anchor = pending_anchors.pop(0)
                
                traj_count += 1
                try:
                    output = self.execute_tools(cmd, current_anchor, end_anchor)
                    result.trajectory_outputs.append(output)
                    output_path = output_dir / f"step_{traj_count:02d}_{cmd.movement_type}.npy"
                    np.save(output_path, output['c2w'])
                    result.trajectory_paths.append(output_path)
                except Exception as e:
                    print(f"  ERROR: {e}")
                    import traceback; traceback.print_exc()
                    output = None
                
                segment = TrajectorySegment(
                    start_anchor=current_anchor,
                    end_anchor=end_anchor,
                    movement_type=cmd.movement_type,
                    movement_category=cmd.movement_category,
                    arc_angle=cmd.arc_angle,
                    trajectory_output=output,
                )
                result.segments.append(segment)
                
                if end_anchor is not None:
                    current_anchor = end_anchor
        
        print(f"\n=== Done: {len(result.segments)} segments, {len(result.all_anchors)} anchors ===")
        return result

    def execute_from_llm_output(
        self,
        llm_output: Union[str, Dict],
        base_name: str = "trajectory",
    ) -> TrajectoryResult:
        """Parse LLM output and execute."""
        plan = self.parse_llm_output(llm_output)
        return self.execute_plan(plan, base_name)


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("Testing Trajectory Executor with SDF Collision Checking")
    print("=" * 60)
    
    executor = TrajectoryExecutor(
        project_root=".",
        bbox_path="data/InteriorGS/compressed/0001_839920/labels.json",
        mesh_path="data/InteriorGS/collision_mesh/0001_839920/839920_collision.usd",
        debug_candidates=True,
    )
    
    llm_output_legacy = {
        "observation": "Zoom into clock, then reveal bookshelf.",
        "reasoning": "zoom_in_out on clock, arc to bookshelf, move_out.",
        "object_sequence": ["clock", "bookshelf"],
        "viewing_preferences": {
            "clock": {"elevation": "low", "distance": "close"},
            "bookshelf": {"elevation": "low", "distance": "medium"},
        },
        "atomic_trajectories": (
            "1. Call Anchor Determinator with 'clock' (id: 33). "
            "2. Call AtomTraj with 'zoom_in_out' (object-level). "
            "3. Call Anchor Determinator with 'bookshelf' (id: 130). "
            "4. Call AtomTraj with 'arc', angle=30 (transitional). "
            "5. Call AtomTraj with 'move_out' (object-level). "
            "6. Call traj_compose. 7. Render video."
        ),
    }
    
    result = executor.execute_from_llm_output(llm_output_legacy, base_name="clock_bookshelf")
    
    print(f"\nSegments: {len(result.segments)}")
    for i, seg in enumerate(result.segments):
        print(f"  {i+1}. {seg.movement_type} ({seg.movement_category})")
        if seg.trajectory_output:
            print(f"     Frames: {seg.trajectory_output['n_frames']}")
            if 'focal_multiplier' in seg.trajectory_output:
                fm = seg.trajectory_output['focal_multiplier']
                print(f"     Focal multiplier: [{fm.min():.2f}, {fm.max():.2f}]")


if __name__ == "__main__":
    main()