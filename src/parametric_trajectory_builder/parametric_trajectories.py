"""
Parametric Camera Trajectories (Simplified)
============================================

Matches the simplified dialog vocabulary:

1. TRANSITIONAL: arc
   - Fixed: start_position, start_rotation, end_position, end_rotation (12 params)
   - Free: arc_angle (1 param) — controls how much the path curves between endpoints
   
2. OBJECT-CENTRIC: orbit_full, orbit_half, orbit_quarter, pan_left, pan_right,
                    move_in, move_out, crane, tilt_up, tilt_down, static,
                    zoom_in_out, zoom_out_in
   - Fixed: object_center (3 params) — what the camera looks at / where the camera sits
   - Free: radius, height, angles, etc.

NEW: zoom_in_out / zoom_out_in — stationary camera with focal-length ramp.
     The camera does NOT move; instead the focal length smoothly increases
     then returns (zoom_in_out) or decreases then returns (zoom_out_in).
     The result dict includes a 'focal_multiplier' array (per-frame).

UPDATED: move_in / move_out — "dolly on a ramp".  The camera translates
     along the 3D ray from its position through object_center (not just
     horizontal XY movement).  start_radius / end_radius are 3D Euclidean
     distances from object_center.  Values are respected literally — no
     swapping.
"""

import numpy as np
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Union
from abc import ABC, abstractmethod
import json


# =============================================================================
# 1. Parameter Specification
# =============================================================================

@dataclass
class ParameterSpec:
    """Specification for a single parameter"""
    name: str
    value: float
    min_bound: float = -np.inf
    max_bound: float = np.inf
    description: str = ""
    unit: str = ""
    is_fixed: bool = False


# =============================================================================
# 2. Base Class
# =============================================================================

class ParametricTrajectoryBase(ABC):
    """Base class for all parametric trajectories"""
    
    @abstractmethod
    def get_free_parameters(self) -> np.ndarray:
        pass
    
    @abstractmethod
    def set_free_parameters(self, params: np.ndarray) -> None:
        pass
    
    @abstractmethod
    def get_free_parameter_specs(self) -> List[ParameterSpec]:
        pass
    
    @abstractmethod
    def get_fixed_parameters(self) -> np.ndarray:
        pass
    
    @abstractmethod
    def set_fixed_parameters(self, params: np.ndarray) -> None:
        pass
    
    @abstractmethod
    def get_fixed_parameter_specs(self) -> List[ParameterSpec]:
        pass
    
    @abstractmethod
    def evaluate(self, t: Union[float, np.ndarray]) -> Dict[str, np.ndarray]:
        pass
    
    @abstractmethod
    def get_trajectory_type(self) -> str:
        pass
    
    def get_num_free_parameters(self) -> int:
        return len(self.get_free_parameters())
    
    def get_num_fixed_parameters(self) -> int:
        return len(self.get_fixed_parameters())
    
    def generate(self, n_frames: int, fps: int = 30) -> Dict[str, np.ndarray]:
        t = np.linspace(0, 1, n_frames)
        result = self.evaluate(t)
        
        positions = result['position']
        rotations_euler = result['rotation_euler']
        
        rotations_matrix = self._euler_to_rotation_matrix(rotations_euler)
        c2w = self._build_c2w_matrices(positions, rotations_matrix)
        
        output = {
            "positions": positions,
            "rotations_euler": rotations_euler,
            "rotations_matrix": rotations_matrix,
            "c2w": c2w,
            "timestamps": np.arange(n_frames) / fps,
            "n_frames": n_frames,
            "fps": fps,
            "trajectory_type": self.get_trajectory_type(),
            "fixed_parameters": self.get_fixed_parameters().copy(),
            "free_parameters": self.get_free_parameters().copy(),
        }
        
        # Propagate per-frame focal_multiplier if the trajectory produces one
        if 'focal_multiplier' in result:
            output['focal_multiplier'] = result['focal_multiplier']
        
        return output
    
    def _euler_to_rotation_matrix(self, euler_angles: np.ndarray) -> np.ndarray:
        if euler_angles.ndim == 1:
            euler_angles = euler_angles.reshape(1, 3)
        
        n = len(euler_angles)
        rad = np.deg2rad(euler_angles)
        pitch, yaw, roll = rad[:, 0], rad[:, 1], rad[:, 2]
        
        cos_p, sin_p = np.cos(pitch), np.sin(pitch)
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        cos_r, sin_r = np.cos(roll), np.sin(roll)
        
        R = np.zeros((n, 3, 3))
        R[:, 0, 0] = cos_y * cos_r
        R[:, 0, 1] = sin_p * sin_y * cos_r - cos_p * sin_r
        R[:, 0, 2] = cos_p * sin_y * cos_r + sin_p * sin_r
        R[:, 1, 0] = cos_y * sin_r
        R[:, 1, 1] = sin_p * sin_y * sin_r + cos_p * cos_r
        R[:, 1, 2] = cos_p * sin_y * sin_r - sin_p * cos_r
        R[:, 2, 0] = -sin_y
        R[:, 2, 1] = sin_p * cos_y
        R[:, 2, 2] = cos_p * cos_y
        
        return R
    
    def _build_c2w_matrices(self, positions: np.ndarray, rotations: np.ndarray) -> np.ndarray:
        if positions.ndim == 1:
            positions = positions.reshape(1, 3)
            rotations = rotations.reshape(1, 3, 3)
        
        n = len(positions)
        c2w = np.zeros((n, 4, 4))
        c2w[:, :3, :3] = rotations
        c2w[:, :3, 3] = positions
        c2w[:, 3, 3] = 1.0
        return c2w


# =============================================================================
# 3. TRANSITIONAL: Arc
# =============================================================================

class TransitionalArc(ParametricTrajectoryBase):
    """
    Arc trajectory connecting two fixed endpoints.
    
    Fixed (12): start_position, start_rotation, end_position, end_rotation
    Free  (1):  arc_angle — controls lateral offset of the path
    """
    
    def __init__(self,
                 start_position: Tuple[float, float, float] = (0, 0, 0),
                 start_rotation: Tuple[float, float, float] = (0, 0, 0),
                 end_position: Tuple[float, float, float] = (0, 0, 2),
                 end_rotation: Tuple[float, float, float] = (0, 0, 0),
                 arc_angle: float = 0.0):
        
        self.start_position = np.array(start_position, dtype=np.float64)
        self.start_rotation = np.array(start_rotation, dtype=np.float64)
        self.end_position = np.array(end_position, dtype=np.float64)
        self.end_rotation = np.array(end_rotation, dtype=np.float64)
        self.arc_angle = arc_angle
    
    def get_trajectory_type(self) -> str:
        return "transitional_arc"
    
    def get_fixed_parameters(self) -> np.ndarray:
        return np.concatenate([
            self.start_position, self.start_rotation,
            self.end_position, self.end_rotation
        ])
    
    def set_fixed_parameters(self, params: np.ndarray) -> None:
        self.start_position = params[0:3].copy()
        self.start_rotation = params[3:6].copy()
        self.end_position = params[6:9].copy()
        self.end_rotation = params[9:12].copy()
    
    def get_fixed_parameter_specs(self) -> List[ParameterSpec]:
        return [
            ParameterSpec("start_x", self.start_position[0], -50, 50, "Start X", "m", True),
            ParameterSpec("start_y", self.start_position[1], -50, 50, "Start Y", "m", True),
            ParameterSpec("start_z", self.start_position[2], -50, 50, "Start Z", "m", True),
            ParameterSpec("start_pitch", self.start_rotation[0], -90, 90, "Start pitch", "deg", True),
            ParameterSpec("start_yaw", self.start_rotation[1], -180, 180, "Start yaw", "deg", True),
            ParameterSpec("start_roll", self.start_rotation[2], -45, 45, "Start roll", "deg", True),
            ParameterSpec("end_x", self.end_position[0], -50, 50, "End X", "m", True),
            ParameterSpec("end_y", self.end_position[1], -50, 50, "End Y", "m", True),
            ParameterSpec("end_z", self.end_position[2], -50, 50, "End Z", "m", True),
            ParameterSpec("end_pitch", self.end_rotation[0], -90, 90, "End pitch", "deg", True),
            ParameterSpec("end_yaw", self.end_rotation[1], -180, 180, "End yaw", "deg", True),
            ParameterSpec("end_roll", self.end_rotation[2], -45, 45, "End roll", "deg", True),
        ]
    
    def get_free_parameters(self) -> np.ndarray:
        return np.array([self.arc_angle])
    
    def set_free_parameters(self, params: np.ndarray) -> None:
        self.arc_angle = params[0]
    
    def get_free_parameter_specs(self) -> List[ParameterSpec]:
        return [
            ParameterSpec("arc_angle", self.arc_angle, -89, 89,
                          "Lateral bulge angle (0=straight, ±45=offset equals half-chord)",
                          "deg", False),
        ]
    
    def evaluate(self, t: Union[float, np.ndarray]) -> Dict[str, np.ndarray]:
        t = np.atleast_1d(t)
        n = len(t)
        
        t_eased = np.where(t < 0.5, 2 * t * t, 1 - 0.5 * (2 - 2 * t) ** 2)
        
        position = ((1 - t_eased)[:, np.newaxis] * self.start_position +
                    t_eased[:, np.newaxis] * self.end_position)
        
        if abs(self.arc_angle) > 1e-6:
            chord_xz = self.end_position[[0, 2]] - self.start_position[[0, 2]]
            chord_len = np.linalg.norm(chord_xz)
            
            if chord_len > 1e-6:
                chord_dir = chord_xz / chord_len
                perp_dir = np.array([-chord_dir[1], chord_dir[0]])
                amplitude = np.tan(np.deg2rad(self.arc_angle)) * chord_len * 0.5
                bulge = amplitude * np.sin(np.pi * t_eased)
                position[:, 0] += bulge * perp_dir[0]
                position[:, 2] += bulge * perp_dir[1]
        
        rotation = ((1 - t_eased)[:, np.newaxis] * self.start_rotation +
                    t_eased[:, np.newaxis] * self.end_rotation)
        
        return {'position': position, 'rotation_euler': rotation}


# =============================================================================
# 4. OBJECT-CENTRIC TRAJECTORIES
# =============================================================================

class ObjectCentricTrajectory(ParametricTrajectoryBase):
    """
    Base class for object-centric trajectories.
    Fixed (3): object_center — what the camera looks at
    """
    
    def __init__(self, object_center: Tuple[float, float, float] = (0, 0, 0)):
        self.object_center = np.array(object_center, dtype=np.float64)
    
    def get_fixed_parameters(self) -> np.ndarray:
        return self.object_center.copy()
    
    def set_fixed_parameters(self, params: np.ndarray) -> None:
        self.object_center = params[0:3].copy()
    
    def get_fixed_parameter_specs(self) -> List[ParameterSpec]:
        return [
            ParameterSpec("center_x", self.object_center[0], -50, 50, "Object center X", "m", True),
            ParameterSpec("center_y", self.object_center[1], -50, 50, "Object center Y", "m", True),
            ParameterSpec("center_z", self.object_center[2], -50, 50, "Object center Z", "m", True),
        ]
    
    def _compute_look_at_rotation(self, position: np.ndarray,
                                   pitch_offset: float = 0.0,
                                   roll: float = 0.0) -> np.ndarray:
        if position.ndim == 1:
            position = position.reshape(1, 3)
        
        n = len(position)
        rotation = np.zeros((n, 3))
        
        dx = self.object_center[0] - position[:, 0]
        dy = self.object_center[1] - position[:, 1]
        dz = self.object_center[2] - position[:, 2]
        
        yaw = np.rad2deg(np.arctan2(dx, dz))
        horizontal_dist = np.sqrt(dx**2 + dz**2)
        pitch = -np.rad2deg(np.arctan2(dy, horizontal_dist))
        
        rotation[:, 0] = pitch + pitch_offset
        rotation[:, 1] = yaw
        rotation[:, 2] = roll
        
        return rotation


# -----------------------------------------------------------------------------
# 4.1 Circular Orbit
# -----------------------------------------------------------------------------

class CircularOrbit(ObjectCentricTrajectory):
    """
    Circular orbit around object center.
    
    Fixed (3): object_center
    Free  (5): radius, height, start_angle, end_angle, pitch_offset
    """
    
    def __init__(self, object_center=(0, 0, 0), radius=2.0, height=0.0,
                 start_angle=0.0, end_angle=360.0, pitch_offset=0.0,
                 name="orbit"):
        super().__init__(object_center)
        self.radius = radius
        self.height = height
        self.start_angle = start_angle
        self.end_angle = end_angle
        self.pitch_offset = pitch_offset
        self.name = name
    
    def get_trajectory_type(self) -> str:
        return f"object_centric_{self.name}"
    
    def get_free_parameters(self) -> np.ndarray:
        return np.array([self.radius, self.height, self.start_angle,
                         self.end_angle, self.pitch_offset])
    
    def set_free_parameters(self, params: np.ndarray) -> None:
        self.radius = params[0]
        self.height = params[1]
        self.start_angle = params[2]
        self.end_angle = params[3]
        self.pitch_offset = params[4]
    
    def get_free_parameter_specs(self) -> List[ParameterSpec]:
        return [
            ParameterSpec("radius", self.radius, 0.5, 20, "Orbit radius", "m", False),
            ParameterSpec("height", self.height, -10, 10, "Height above center", "m", False),
            ParameterSpec("start_angle", self.start_angle, -360, 360, "Start angle", "deg", False),
            ParameterSpec("end_angle", self.end_angle, -360, 720, "End angle", "deg", False),
            ParameterSpec("pitch_offset", self.pitch_offset, -60, 60, "Pitch offset", "deg", False),
        ]
    
    def evaluate(self, t: Union[float, np.ndarray]) -> Dict[str, np.ndarray]:
        t = np.atleast_1d(t)
        n = len(t)
        
        angles = np.deg2rad(self.start_angle + t * (self.end_angle - self.start_angle))
        
        position = np.zeros((n, 3))
        position[:, 0] = self.object_center[0] + self.radius * np.cos(angles)
        position[:, 2] = self.object_center[2] + self.radius * np.sin(angles)
        position[:, 1] = self.object_center[1] + self.height
        
        rotation = self._compute_look_at_rotation(position, self.pitch_offset)
        return {'position': position, 'rotation_euler': rotation}
    
    @classmethod
    def orbit_full(cls, object_center=(0, 0, 0), radius=2.0, height=0.0,
                   start_angle=0.0, pitch_offset=0.0):
        return cls(object_center, radius, height, start_angle,
                   start_angle + 360.0, pitch_offset, name="orbit_full")
    
    @classmethod
    def orbit_half(cls, object_center=(0, 0, 0), radius=2.0, height=0.0,
                   start_angle=0.0, pitch_offset=0.0):
        return cls(object_center, radius, height, start_angle,
                   start_angle + 180.0, pitch_offset, name="orbit_half")
    
    @classmethod
    def orbit_quarter(cls, object_center=(0, 0, 0), radius=2.0, height=0.0,
                      start_angle=0.0, pitch_offset=0.0):
        return cls(object_center, radius, height, start_angle,
                   start_angle + 90.0, pitch_offset, name="orbit_quarter")


# -----------------------------------------------------------------------------
# 4.2 Stationary Pan
# -----------------------------------------------------------------------------

class StationaryPan(ObjectCentricTrajectory):
    """
    Stationary camera that pans horizontally.
    
    Fixed (3): camera_position (stored as object_center)
    Free  (3): start_yaw, end_yaw, pitch
    """
    
    def __init__(self, camera_position=(0, 0, 0),
                 start_yaw=-45.0, end_yaw=45.0, pitch=0.0,
                 name="pan"):
        super().__init__(camera_position)
        self.start_yaw = start_yaw
        self.end_yaw = end_yaw
        self.pitch = pitch
        self.name = name
    
    @property
    def camera_position(self):
        return self.object_center
    
    def get_trajectory_type(self) -> str:
        return f"object_centric_{self.name}"
    
    def get_fixed_parameter_specs(self) -> List[ParameterSpec]:
        return [
            ParameterSpec("camera_x", self.object_center[0], -50, 50, "Camera X", "m", True),
            ParameterSpec("camera_y", self.object_center[1], -50, 50, "Camera Y", "m", True),
            ParameterSpec("camera_z", self.object_center[2], -50, 50, "Camera Z", "m", True),
        ]
    
    def get_free_parameters(self) -> np.ndarray:
        return np.array([self.start_yaw, self.end_yaw, self.pitch])
    
    def set_free_parameters(self, params: np.ndarray) -> None:
        self.start_yaw = params[0]
        self.end_yaw = params[1]
        self.pitch = params[2]
    
    def get_free_parameter_specs(self) -> List[ParameterSpec]:
        return [
            ParameterSpec("start_yaw", self.start_yaw, -180, 180, "Start yaw", "deg", False),
            ParameterSpec("end_yaw", self.end_yaw, -180, 180, "End yaw", "deg", False),
            ParameterSpec("pitch", self.pitch, -90, 90, "Pitch", "deg", False),
        ]
    
    def evaluate(self, t: Union[float, np.ndarray]) -> Dict[str, np.ndarray]:
        t = np.atleast_1d(t)
        n = len(t)
        
        t_eased = np.where(t < 0.5, 2 * t * t, 1 - 0.5 * (2 - 2 * t) ** 2)
        
        position = np.tile(self.camera_position, (n, 1))
        
        rotation = np.zeros((n, 3))
        rotation[:, 0] = self.pitch
        rotation[:, 1] = self.start_yaw + t_eased * (self.end_yaw - self.start_yaw)
        rotation[:, 2] = 0.0
        
        return {'position': position, 'rotation_euler': rotation}
    
    @classmethod
    def pan_left(cls, camera_position=(0, 0, 0), yaw_range=45.0, pitch=0.0):
        return cls(camera_position, start_yaw=yaw_range / 2,
                   end_yaw=-yaw_range / 2, pitch=pitch, name="pan_left")
    
    @classmethod
    def pan_right(cls, camera_position=(0, 0, 0), yaw_range=45.0, pitch=0.0):
        return cls(camera_position, start_yaw=-yaw_range / 2,
                   end_yaw=yaw_range / 2, pitch=pitch, name="pan_right")


# -----------------------------------------------------------------------------
# 4.3 Dolly Move  — UPDATED: "dolly on a ramp"
# -----------------------------------------------------------------------------

class DollyMove(ObjectCentricTrajectory):
    """
    Dolly move: camera translates toward or away from the object center
    along a 3D ray (not just horizontally).

    The camera moves along the line connecting its starting position to
    object_center.  The 3D direction is determined by approach_angle
    (horizontal direction, in the XZ plane) and height (Y offset from
    object_center).  Together they define a ray from object_center
    outward; the camera slides along this ray.

    start_radius and end_radius are 3D Euclidean distances from
    object_center measured along this ray.  Values are respected
    literally — no implicit swapping:
      - move_in:  set start_radius > end_radius  (far → near)
      - move_out: set start_radius < end_radius  (near → far)
    
    Fixed (3): object_center
    Free  (4): start_radius, end_radius, height, pitch_offset
    """
    
    def __init__(self, object_center=(0, 0, 0),
                 start_radius=3.0, end_radius=1.0, height=0.0,
                 approach_angle=0.0, pitch_offset=0.0,
                 name="move_in"):
        super().__init__(object_center)
        self.start_radius = start_radius
        self.end_radius = end_radius
        self.height = height
        self.approach_angle = approach_angle
        self.pitch_offset = pitch_offset
        self.name = name
    
    def get_trajectory_type(self) -> str:
        return f"object_centric_{self.name}"
    
    def get_free_parameters(self) -> np.ndarray:
        return np.array([self.start_radius, self.end_radius,
                         self.height, self.pitch_offset])
    
    def set_free_parameters(self, params: np.ndarray) -> None:
        self.start_radius = params[0]
        self.end_radius = params[1]
        self.height = params[2]
        self.pitch_offset = params[3]
    
    def get_free_parameter_specs(self) -> List[ParameterSpec]:
        return [
            ParameterSpec("start_radius", self.start_radius, 0.3, 20,
                          "Starting 3D distance from object center", "m", False),
            ParameterSpec("end_radius", self.end_radius, 0.3, 20,
                          "Ending 3D distance from object center", "m", False),
            ParameterSpec("height", self.height, -10, 10,
                          "Height above object center (defines ray elevation)", "m", False),
            ParameterSpec("pitch_offset", self.pitch_offset, -60, 60,
                          "Pitch offset from look-at", "deg", False),
        ]
    
    def evaluate(self, t: Union[float, np.ndarray]) -> Dict[str, np.ndarray]:
        t = np.atleast_1d(t)
        n = len(t)
        
        t_eased = np.where(t < 0.5, 2 * t * t, 1 - 0.5 * (2 - 2 * t) ** 2)
        
        # 3D distance from object_center interpolated over time
        dist = self.start_radius + t_eased * (self.end_radius - self.start_radius)
        
        # Build 3D unit direction: object_center → camera
        # Horizontal direction from approach_angle (in XZ plane, Y-up convention)
        angle_rad = np.deg2rad(self.approach_angle)
        dir_x = np.cos(angle_rad)   # X component
        dir_z = np.sin(angle_rad)   # Z component
        dir_y = 0.0                 # will be set from height
        
        # The ray direction incorporates the height offset:
        # at start_radius, the camera is at (center + start_radius * direction)
        # where direction includes the vertical component from height.
        # We define the ray as: center + dist * unit_direction_3d
        # where unit_direction_3d points from center toward
        # (center_x + h_dist * dir_x, center_y + height, center_z + h_dist * dir_z)
        #
        # To make start_radius the actual 3D Euclidean distance along this ray,
        # we normalize the direction vector.
        raw_dir = np.array([dir_x * self.start_radius,
                            self.height,
                            dir_z * self.start_radius], dtype=np.float64)
        ray_len = np.linalg.norm(raw_dir)
        if ray_len < 1e-8:
            # Degenerate: camera on top of center
            direction = np.array([dir_x, 0.0, dir_z])
        else:
            direction = raw_dir / ray_len
        
        # Camera positions along the 3D ray from object_center
        position = np.zeros((n, 3))
        position[:, 0] = self.object_center[0] + dist * direction[0]
        position[:, 1] = self.object_center[1] + dist * direction[1]
        position[:, 2] = self.object_center[2] + dist * direction[2]
        
        rotation = self._compute_look_at_rotation(position, self.pitch_offset)
        
        return {'position': position, 'rotation_euler': rotation}
    
    @classmethod
    def move_in(cls, object_center=(0, 0, 0), start_radius=3.0, end_radius=1.0,
                height=0.0, approach_angle=0.0, pitch_offset=0.0):
        """Camera approaches object: start_radius > end_radius."""
        return cls(object_center, start_radius, end_radius, height,
                   approach_angle, pitch_offset, name="move_in")
    
    @classmethod
    def move_out(cls, object_center=(0, 0, 0), start_radius=1.0, end_radius=3.0,
                 height=0.0, approach_angle=0.0, pitch_offset=0.0):
        """Camera recedes from object: start_radius < end_radius."""
        return cls(object_center, start_radius, end_radius, height,
                   approach_angle, pitch_offset, name="move_out")


# -----------------------------------------------------------------------------
# 4.4 Crane Shot
# -----------------------------------------------------------------------------

class CraneShot(ObjectCentricTrajectory):
    """
    Crane shot: VERTICAL orbit around the object.
    
    Fixed (3): object_center
    Free  (4): radius, start_elevation, end_elevation, pitch_offset
    """
    
    def __init__(self, object_center=(0, 0, 0),
                 radius=2.5, start_elevation=10.0, end_elevation=85.0,
                 approach_angle=0.0, pitch_offset=0.0,
                 name="crane"):
        super().__init__(object_center)
        self.radius = radius
        self.start_elevation = start_elevation
        self.end_elevation = end_elevation
        self.approach_angle = approach_angle
        self.pitch_offset = pitch_offset
        self.name = name
    
    def get_trajectory_type(self) -> str:
        return f"object_centric_{self.name}"
    
    def get_free_parameters(self) -> np.ndarray:
        return np.array([self.radius, self.start_elevation,
                         self.end_elevation, self.pitch_offset])
    
    def set_free_parameters(self, params: np.ndarray) -> None:
        self.radius = params[0]
        self.start_elevation = params[1]
        self.end_elevation = params[2]
        self.pitch_offset = params[3]
    
    def get_free_parameter_specs(self) -> List[ParameterSpec]:
        return [
            ParameterSpec("radius", self.radius, 0.5, 20,
                          "Orbit radius", "m", False),
            ParameterSpec("start_elevation", self.start_elevation, -10, 85,
                          "Start elevation angle (0=eye level)", "deg", False),
            ParameterSpec("end_elevation", self.end_elevation, 10, 90,
                          "End elevation angle (90=directly above)", "deg", False),
            ParameterSpec("pitch_offset", self.pitch_offset, -60, 60,
                          "Pitch offset at start", "deg", False),
        ]
    
    def evaluate(self, t: Union[float, np.ndarray]) -> Dict[str, np.ndarray]:
        t = np.atleast_1d(t)
        n = len(t)
        
        t_eased = np.where(t < 0.5, 2 * t * t, 1 - 0.5 * (2 - 2 * t) ** 2)
        
        elevation = self.start_elevation + t_eased * (self.end_elevation - self.start_elevation)
        elevation_rad = np.deg2rad(elevation)
        
        approach_rad = np.deg2rad(self.approach_angle)
        dir_x = np.cos(approach_rad)
        dir_z = np.sin(approach_rad)
        
        h_dist = self.radius * np.cos(elevation_rad)
        v_dist = self.radius * np.sin(elevation_rad)
        
        position = np.zeros((n, 3))
        position[:, 0] = self.object_center[0] + h_dist * dir_x
        position[:, 2] = self.object_center[2] + h_dist * dir_z
        position[:, 1] = self.object_center[1] + v_dist
        
        base_rotation = self._compute_look_at_rotation(position, pitch_offset=0.0)
        start_pitch = base_rotation[0, 0] + self.pitch_offset
        end_pitch = 90.0
        pitch_blend = start_pitch * (1 - t_eased) + end_pitch * t_eased
        
        rotation = np.zeros((n, 3))
        rotation[:, 0] = pitch_blend
        rotation[:, 1] = base_rotation[:, 1]
        rotation[:, 2] = 0.0
        
        return {'position': position, 'rotation_euler': rotation}


# -----------------------------------------------------------------------------
# 4.5 Stationary Tilt
# -----------------------------------------------------------------------------

class StationaryTilt(ObjectCentricTrajectory):
    """
    Stationary camera that tilts vertically (pitch change).
    
    Fixed (3): camera_position (stored as object_center)
    Free  (3): start_pitch, end_pitch, yaw
    """
    
    def __init__(self, camera_position=(0, 0, 0),
                 start_pitch=0.0, end_pitch=-30.0, yaw=0.0,
                 name="tilt"):
        super().__init__(camera_position)
        self.start_pitch = start_pitch
        self.end_pitch = end_pitch
        self.yaw = yaw
        self.name = name
    
    @property
    def camera_position(self):
        return self.object_center
    
    def get_trajectory_type(self) -> str:
        return f"object_centric_{self.name}"
    
    def get_fixed_parameter_specs(self) -> List[ParameterSpec]:
        return [
            ParameterSpec("camera_x", self.object_center[0], -50, 50,
                          "Camera X", "m", True),
            ParameterSpec("camera_y", self.object_center[1], -50, 50,
                          "Camera Y", "m", True),
            ParameterSpec("camera_z", self.object_center[2], -50, 50,
                          "Camera Z", "m", True),
        ]
    
    def get_free_parameters(self) -> np.ndarray:
        return np.array([self.start_pitch, self.end_pitch, self.yaw])
    
    def set_free_parameters(self, params: np.ndarray) -> None:
        self.start_pitch = params[0]
        self.end_pitch = params[1]
        self.yaw = params[2]
    
    def get_free_parameter_specs(self) -> List[ParameterSpec]:
        return [
            ParameterSpec("start_pitch", self.start_pitch, -90, 90,
                          "Start pitch", "deg", False),
            ParameterSpec("end_pitch", self.end_pitch, -90, 90,
                          "End pitch", "deg", False),
            ParameterSpec("yaw", self.yaw, -180, 180,
                          "Yaw (facing direction)", "deg", False),
        ]
    
    def evaluate(self, t: Union[float, np.ndarray]) -> Dict[str, np.ndarray]:
        t = np.atleast_1d(t)
        n = len(t)
        
        t_eased = np.where(t < 0.5, 2 * t * t, 1 - 0.5 * (2 - 2 * t) ** 2)
        
        position = np.tile(self.camera_position, (n, 1))
        
        rotation = np.zeros((n, 3))
        rotation[:, 0] = self.start_pitch + t_eased * (self.end_pitch - self.start_pitch)
        rotation[:, 1] = self.yaw
        rotation[:, 2] = 0.0
        
        return {'position': position, 'rotation_euler': rotation}
    
    @classmethod
    def tilt_up(cls, camera_position=(0, 0, 0), pitch_range=45.0, yaw=0.0):
        return cls(camera_position, start_pitch=pitch_range / 2,
                   end_pitch=-pitch_range / 2, yaw=yaw, name="tilt_up")
    
    @classmethod
    def tilt_down(cls, camera_position=(0, 0, 0), pitch_range=45.0, yaw=0.0):
        return cls(camera_position, start_pitch=-pitch_range / 2,
                   end_pitch=pitch_range / 2, yaw=yaw, name="tilt_down")


# -----------------------------------------------------------------------------
# 4.6 Zoom Lens — covers zoom_in_out, zoom_out_in
# -----------------------------------------------------------------------------

class ZoomLens(ObjectCentricTrajectory):
    """
    Optical zoom: stationary camera with focal-length change.

    The camera does NOT translate — only the focal length changes.
    This produces a genuine "zoom" (field-of-view narrows/widens) as
    opposed to a dolly move which physically moves the camera.

    Because the trajectory must be continuous with the segments before
    and after it, the focal length returns to its original value by the
    end of the segment:

        zoom_in_out:  1× → peak_multiplier → 1×   (narrow FOV then back)
        zoom_out_in:  1× → 1/peak_multiplier → 1×  (widen FOV then back)

    The per-frame focal multiplier is stored in the evaluate() result
    under the key 'focal_multiplier' (shape (N,)).  Downstream code
    (executor, combiner, renderer) should multiply the base focal
    lengths (fx, fy) by this array to produce the actual per-frame
    intrinsics.

    Fixed (3): camera_position (stored as object_center)
    Free  (3): peak_multiplier, yaw, pitch

    peak_multiplier controls how far the focal length departs from 1×:
        zoom_in_out default 2.0  → FOV halves at midpoint
        zoom_out_in default 0.5  → FOV doubles at midpoint
    """

    def __init__(self, camera_position=(0, 0, 0),
                 peak_multiplier=2.0, yaw=0.0, pitch=0.0,
                 name="zoom_in_out"):
        super().__init__(camera_position)
        self.peak_multiplier = peak_multiplier
        self.yaw = yaw
        self.pitch = pitch
        self.name = name

    @property
    def camera_position(self):
        return self.object_center

    def get_trajectory_type(self) -> str:
        return f"object_centric_{self.name}"

    def get_fixed_parameter_specs(self) -> List[ParameterSpec]:
        return [
            ParameterSpec("camera_x", self.object_center[0], -50, 50,
                          "Camera X", "m", True),
            ParameterSpec("camera_y", self.object_center[1], -50, 50,
                          "Camera Y", "m", True),
            ParameterSpec("camera_z", self.object_center[2], -50, 50,
                          "Camera Z", "m", True),
        ]

    def get_free_parameters(self) -> np.ndarray:
        return np.array([self.peak_multiplier, self.yaw, self.pitch])

    def set_free_parameters(self, params: np.ndarray) -> None:
        self.peak_multiplier = params[0]
        self.yaw = params[1]
        self.pitch = params[2]

    def get_free_parameter_specs(self) -> List[ParameterSpec]:
        return [
            ParameterSpec("peak_multiplier", self.peak_multiplier, 0.2, 5.0,
                          "Peak focal-length multiplier at midpoint", "×", False),
            ParameterSpec("yaw", self.yaw, -180, 180,
                          "Yaw (facing direction)", "deg", False),
            ParameterSpec("pitch", self.pitch, -90, 90,
                          "Pitch", "deg", False),
        ]

    def evaluate(self, t: Union[float, np.ndarray]) -> Dict[str, np.ndarray]:
        t = np.atleast_1d(t)
        n = len(t)

        t_eased = np.where(t < 0.5, 2 * t * t, 1 - 0.5 * (2 - 2 * t) ** 2)

        position = np.tile(self.camera_position, (n, 1))

        rotation = np.zeros((n, 3))
        rotation[:, 0] = self.pitch
        rotation[:, 1] = self.yaw
        rotation[:, 2] = 0.0

        focal_multiplier = 1.0 + (self.peak_multiplier - 1.0) * np.sin(np.pi * t_eased)

        return {
            'position': position,
            'rotation_euler': rotation,
            'focal_multiplier': focal_multiplier,
        }

    @classmethod
    def zoom_in_out(cls, camera_position=(0, 0, 0),
                    peak_multiplier=2.0, yaw=0.0, pitch=0.0):
        return cls(camera_position, peak_multiplier=peak_multiplier,
                   yaw=yaw, pitch=pitch, name="zoom_in_out")

    @classmethod
    def zoom_out_in(cls, camera_position=(0, 0, 0),
                    peak_multiplier=0.5, yaw=0.0, pitch=0.0):
        return cls(camera_position, peak_multiplier=peak_multiplier,
                   yaw=yaw, pitch=pitch, name="zoom_out_in")


# =============================================================================
# 5. Factory
# =============================================================================

def create_trajectory(name: str, **kwargs) -> ParametricTrajectoryBase:
    """
    Create a trajectory from a dialog movement name.
    
    Args:
        name: One of 'orbit_full', 'orbit_half', 'orbit_quarter',
              'pan_left', 'pan_right', 'move_in', 'move_out',
              'crane', 'tilt_up', 'tilt_down', 'zoom_in_out',
              'zoom_out_in', 'arc'
        **kwargs: Parameters forwarded to the constructor
    """
    factories = {
        "orbit_full": CircularOrbit.orbit_full,
        "orbit_half": CircularOrbit.orbit_half,
        "orbit_quarter": CircularOrbit.orbit_quarter,
        "pan_left": StationaryPan.pan_left,
        "pan_right": StationaryPan.pan_right,
        "move_in": DollyMove.move_in,
        "move_out": DollyMove.move_out,
        "crane": lambda **kw: CraneShot(**kw),
        "tilt_up": StationaryTilt.tilt_up,
        "tilt_down": StationaryTilt.tilt_down,
        "zoom_in_out": ZoomLens.zoom_in_out,
        "zoom_out_in": ZoomLens.zoom_out_in,
        "arc": lambda **kw: TransitionalArc(**kw),
    }
    
    if name not in factories:
        raise ValueError(
            f"Unknown trajectory '{name}'. "
            f"Valid names: {sorted(factories.keys())}"
        )
    
    return factories[name](**kwargs)


# =============================================================================
# 6. Optimization Helper
# =============================================================================

class TrajectoryOptimizer:
    """Helper for trajectory optimization"""
    
    def __init__(self, trajectory: ParametricTrajectoryBase):
        self.trajectory = trajectory
    
    def get_optimization_variables(self):
        specs = self.trajectory.get_free_parameter_specs()
        params = self.trajectory.get_free_parameters()
        lower = np.array([s.min_bound for s in specs])
        upper = np.array([s.max_bound for s in specs])
        return params, lower, upper
    
    def set_optimization_result(self, params: np.ndarray):
        self.trajectory.set_free_parameters(params)
    
    def print_summary(self):
        print(f"\n{'='*70}")
        print(f"Trajectory: {self.trajectory.get_trajectory_type()}")
        print(f"{'='*70}")
        
        print(f"\nFIXED ({self.trajectory.get_num_fixed_parameters()} params):")
        for s in self.trajectory.get_fixed_parameter_specs():
            print(f"  {s.name:<20} = {s.value:>10.4f} {s.unit}")
        
        print(f"\nFREE ({self.trajectory.get_num_free_parameters()} params):")
        print(f"  {'Name':<20} {'Value':>10} {'Min':>10} {'Max':>10}")
        for s in self.trajectory.get_free_parameter_specs():
            print(f"  {s.name:<20} {s.value:>10.4f} {s.min_bound:>10.2f} {s.max_bound:>10.2f}")


# =============================================================================
# 7. Visualization
# =============================================================================

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def visualize_trajectory(trajectory, n_frames=60, title=None, save_path=None):
    gen = trajectory.generate(n_frames)
    pos, rot = gen['positions'], gen['rotations_euler']
    rot_mat = gen['rotations_matrix']
    
    if title is None:
        title = trajectory.get_trajectory_type()
    
    has_focal = 'focal_multiplier' in gen
    n_plots = 5 if has_focal else 4
    
    fig = plt.figure(figsize=(16, 12 if has_focal else 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    rows = 2 if not has_focal else 3
    
    # 3D view
    ax1 = fig.add_subplot(rows, 2, 1, projection='3d')
    ax1.plot(pos[:, 0], pos[:, 2], pos[:, 1], 'b-', linewidth=2)
    ax1.scatter(*pos[0, [0, 2, 1]], c='green', s=150, marker='o', label='Start')
    ax1.scatter(*pos[-1, [0, 2, 1]], c='red', s=150, marker='s', label='End')
    
    if isinstance(trajectory, ObjectCentricTrajectory):
        c = trajectory.object_center
        ax1.scatter(c[0], c[2], c[1], c='orange', s=200, marker='*', label='Object')
    
    for i, idx in enumerate(np.linspace(0, n_frames - 1, min(8, n_frames), dtype=int)):
        _draw_frustum(ax1, pos[idx], rot_mat[idx], 0.12, 0.3 + 0.6 * i / 7)
    
    ax1.set_xlabel('X'); ax1.set_ylabel('Z'); ax1.set_zlabel('Y')
    ax1.legend(fontsize=8)
    
    pts = (np.vstack([pos, trajectory.object_center])
           if isinstance(trajectory, ObjectCentricTrajectory) else pos)
    r = max(pts.max(0) - pts.min(0)) * 0.6 + 0.5
    mid = pts.mean(0)
    ax1.set_xlim(mid[0] - r, mid[0] + r)
    ax1.set_ylim(mid[2] - r, mid[2] + r)
    ax1.set_zlim(mid[1] - r, mid[1] + r)
    
    # Info panel
    ax2 = fig.add_subplot(rows, 2, 2)
    ax2.axis('off')
    info = f"FIXED ({trajectory.get_num_fixed_parameters()}):\n"
    for s in trajectory.get_fixed_parameter_specs():
        info += f"  {s.name}: {s.value:.3f}\n"
    info += f"\nFREE ({trajectory.get_num_free_parameters()}):\n"
    for s in trajectory.get_free_parameter_specs():
        info += f"  {s.name}: {s.value:.3f}\n"
    ax2.text(0.05, 0.95, info, transform=ax2.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    # Position / rotation plots
    frames = np.arange(n_frames)
    ax3 = fig.add_subplot(rows, 2, 3)
    ax3.plot(frames, pos[:, 0], 'r-', label='X')
    ax3.plot(frames, pos[:, 1], 'g-', label='Y')
    ax3.plot(frames, pos[:, 2], 'b-', label='Z')
    ax3.set_xlabel('Frame'); ax3.set_ylabel('Position')
    ax3.legend(); ax3.grid(True, alpha=0.3)
    
    ax4 = fig.add_subplot(rows, 2, 4)
    ax4.plot(frames, rot[:, 0], 'r-', label='Pitch')
    ax4.plot(frames, rot[:, 1], 'g-', label='Yaw')
    ax4.plot(frames, rot[:, 2], 'b-', label='Roll')
    ax4.set_xlabel('Frame'); ax4.set_ylabel('Angle (deg)')
    ax4.legend(); ax4.grid(True, alpha=0.3)
    
    if has_focal:
        ax5 = fig.add_subplot(rows, 2, 5)
        fm = gen['focal_multiplier']
        ax5.plot(frames, fm, 'm-', linewidth=2, label='Focal multiplier')
        ax5.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        ax5.set_xlabel('Frame'); ax5.set_ylabel('Focal multiplier (×)')
        ax5.set_title('Focal Length')
        ax5.legend(); ax5.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    return fig


def _draw_frustum(ax, pos, R, scale=0.15, alpha=0.6):
    frustum = np.array([
        [0, 0, 0], [-0.6, -0.4, 1], [0.6, -0.4, 1],
        [0.6, 0.4, 1], [-0.6, 0.4, 1]
    ]) * scale
    pts = (R @ frustum.T).T + pos
    for e in [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]:
        ax.plot3D([pts[e[0], 0], pts[e[1], 0]],
                  [pts[e[0], 2], pts[e[1], 2]],
                  [pts[e[0], 1], pts[e[1], 1]],
                  color='purple', alpha=alpha, linewidth=0.8)


# =============================================================================
# 8. Main Demo
# =============================================================================

if __name__ == "__main__":
    import os

    output_dir = "outputs/tool_set_movie_parametric"
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("SIMPLIFIED PARAMETRIC TRAJECTORIES")
    print("=" * 70)
    print("""
    Vocabulary:
      Object-level:  orbit_full, orbit_half, orbit_quarter, pan_left, pan_right,
                     move_in, move_out, crane, tilt_up, tilt_down,
                     zoom_in_out, zoom_out_in
      Transitional:  arc (angle parameter)
    """)
    
    # --- Object-Centric ---
    print("\n--- OBJECT-CENTRIC ---")
    
    full = CircularOrbit.orbit_full(object_center=(0, 0.5, 0), radius=3.0, height=1.0)
    TrajectoryOptimizer(full).print_summary()
    
    half = CircularOrbit.orbit_half(object_center=(0, 0.5, 0), radius=2.5, height=0.5)
    TrajectoryOptimizer(half).print_summary()
    
    quarter = CircularOrbit.orbit_quarter(object_center=(0, 0.5, 0), radius=2.0)
    TrajectoryOptimizer(quarter).print_summary()
    
    pan_l = StationaryPan.pan_left(camera_position=(0, 1.5, 2), yaw_range=60)
    TrajectoryOptimizer(pan_l).print_summary()
    
    pan_r = StationaryPan.pan_right(camera_position=(0, 1.5, 2), yaw_range=60)
    TrajectoryOptimizer(pan_r).print_summary()
    
    dolly_in = DollyMove.move_in(
        object_center=(0, 0.5, 0), start_radius=3.0, end_radius=1.0,
        height=0.5, approach_angle=45.0
    )
    TrajectoryOptimizer(dolly_in).print_summary()
    
    dolly_out = DollyMove.move_out(
        object_center=(0, 0.5, 0), start_radius=1.0, end_radius=4.0,
        height=0.5, approach_angle=45.0
    )
    TrajectoryOptimizer(dolly_out).print_summary()
    
    crane = CraneShot(
        object_center=(0, 0.5, 0), radius=2.5,
        start_elevation=10.0, end_elevation=85.0,
        approach_angle=0.0,
    )
    TrajectoryOptimizer(crane).print_summary()
    
    tilt_u = StationaryTilt.tilt_up(camera_position=(0, 1.5, 2), pitch_range=60, yaw=0)
    TrajectoryOptimizer(tilt_u).print_summary()
    
    tilt_d = StationaryTilt.tilt_down(camera_position=(0, 1.5, 2), pitch_range=60, yaw=0)
    TrajectoryOptimizer(tilt_d).print_summary()
    
    zoom_io = ZoomLens.zoom_in_out(camera_position=(0, 1.5, 2), peak_multiplier=2.0, yaw=0, pitch=-10)
    TrajectoryOptimizer(zoom_io).print_summary()
    
    zoom_oi = ZoomLens.zoom_out_in(camera_position=(0, 1.5, 2), peak_multiplier=0.5, yaw=0, pitch=-10)
    TrajectoryOptimizer(zoom_oi).print_summary()
    
    # --- Transitional ---
    print("\n--- TRANSITIONAL ---")
    
    arc_straight = TransitionalArc(
        start_position=(-2, 1, 0), start_rotation=(-10, 30, 0),
        end_position=(2, 1, 0), end_rotation=(-10, -30, 0),
        arc_angle=0
    )
    TrajectoryOptimizer(arc_straight).print_summary()
    
    arc_curved = TransitionalArc(
        start_position=(-2, 1, 0), start_rotation=(-10, 30, 0),
        end_position=(2, 1, 0), end_rotation=(-10, -30, 0),
        arc_angle=45
    )
    TrajectoryOptimizer(arc_curved).print_summary()
    
    # --- Factory demo ---
    print("\n--- FACTORY ---")
    t1 = create_trajectory("orbit_half", object_center=(1, 0, 1), radius=2.0)
    t2 = create_trajectory("pan_right", camera_position=(0, 1, 0))
    t3 = create_trajectory("arc", start_position=(0, 0, 0), end_position=(3, 0, 3), arc_angle=30)
    t4 = create_trajectory("move_in", object_center=(1, 0, 1), start_radius=4.0, end_radius=1.5)
    t5 = create_trajectory("move_out", object_center=(1, 0, 1), start_radius=1.0, end_radius=5.0)
    t6 = create_trajectory("crane", object_center=(1, 0, 1), radius=2.0)
    t7 = create_trajectory("tilt_up", camera_position=(0, 1, 2), pitch_range=50)
    t8 = create_trajectory("tilt_down", camera_position=(0, 1, 2), pitch_range=50)
    t9 = create_trajectory("zoom_in_out", camera_position=(0, 1, 2), peak_multiplier=2.5)
    t10 = create_trajectory("zoom_out_in", camera_position=(0, 1, 2), peak_multiplier=0.4)
    for t in [t1, t2, t3, t4, t5, t6, t7, t8, t9, t10]:
        TrajectoryOptimizer(t).print_summary()
    
    # --- Visualizations ---
    print("\n\nGenerating visualizations...")
    visualize_trajectory(full, 120, "orbit_full (360°)", f'{output_dir}/orbit_full.png')
    visualize_trajectory(half, 90, "orbit_half (180°)", f'{output_dir}/orbit_half.png')
    visualize_trajectory(quarter, 60, "orbit_quarter (90°)", f'{output_dir}/orbit_quarter.png')
    visualize_trajectory(pan_l, 60, "pan_left", f'{output_dir}/pan_left.png')
    visualize_trajectory(pan_r, 60, "pan_right", f'{output_dir}/pan_right.png')
    visualize_trajectory(dolly_in, 60, "move_in (dolly in: 3.0 → 1.0)", f'{output_dir}/move_in.png')
    visualize_trajectory(dolly_out, 60, "move_out (dolly out: 1.0 → 4.0)", f'{output_dir}/move_out.png')
    visualize_trajectory(crane, 120, "crane (orbit + rise to top-down)", f'{output_dir}/crane.png')
    visualize_trajectory(tilt_u, 60, "tilt_up", f'{output_dir}/tilt_up.png')
    visualize_trajectory(tilt_d, 60, "tilt_down", f'{output_dir}/tilt_down.png')
    visualize_trajectory(zoom_io, 120, "zoom_in_out (1× → 2× → 1×)", f'{output_dir}/zoom_in_out.png')
    visualize_trajectory(zoom_oi, 120, "zoom_out_in (1× → 0.5× → 1×)", f'{output_dir}/zoom_out_in.png')
    visualize_trajectory(arc_straight, 60, "arc (angle=0, straight)", f'{output_dir}/arc_straight.png')
    visualize_trajectory(arc_curved, 60, "arc (angle=45)", f'{output_dir}/arc_curved.png')
    
    # --- Summary ---
    print("\n" + "=" * 70)
    print("PARAMETER SUMMARY")
    print("=" * 70)
    print(f"\n{'Trajectory':<25} {'Fixed':<8} {'Free':<8}")
    print("-" * 45)
    for name, traj in [
        ("arc", arc_curved),
        ("orbit_full", full),
        ("orbit_half", half),
        ("orbit_quarter", quarter),
        ("pan_left", pan_l),
        ("pan_right", pan_r),
        ("move_in", dolly_in),
        ("move_out", dolly_out),
        ("crane", crane),
        ("tilt_up", tilt_u),
        ("tilt_down", tilt_d),
        ("zoom_in_out", zoom_io),
        ("zoom_out_in", zoom_oi),
    ]:
        f = traj.get_num_fixed_parameters()
        p = traj.get_num_free_parameters()
        print(f"  {name:<23} {f:<8} {p:<8}")
    
    print("\nDone!")