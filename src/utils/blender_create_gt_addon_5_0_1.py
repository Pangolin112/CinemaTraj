"""
Blender Addon: Camera Trajectory Creator v5.3-b52
==================================================

Adapted for Blender 5.0+ (layered/slotted Action API).

Key API changes from 3.x/4.x → 5.0:
  - action.fcurves is removed; F-Curves live in channelbags.
  - Use action.fcurve_ensure_for_datablock(datablock, data_path, index=...)
    to get/create F-Curves (auto-creates layer, strip, slot, channelbag).
  - Use bpy_extras.anim_utils.action_get_channelbag_for_slot() to iterate.
  - bl_info still works as legacy addon format.

v5.3 features:
  - CAMERA PARAMETERS exposed in panel for ALL movement types:
    Base Focal Length (mm), Sensor Width, computed FOV display.
    Camera lens is updated live as you change focal/sensor.
  - Per-object INITIAL VIEWPOINT bug fix: changing IV sliders now
    auto-applies to trajectory params so the preview updates immediately
    (previously only the first change took effect).
  - New "Auto-Apply IV" toggle: when ON (default), IV slider changes
    feed directly into trajectory params.  When OFF, you must click
    "Apply to Trajectory" manually.
  - STATIC movement type: camera holds position, looking at anchor.

v5.3-b50-fix:
  - FIX: cam_pos in _get_preview_data now uses approach_angle and elevation
    (spherical coords) so that pan, tilt, zoom movements correctly respond
    to IV approach angle and elevation changes.

v5.3-b51-fix:
  - FIX: Orbit (full/half/quarter) now respects start_elevation (IV elevation)
    by decomposing radius into horizontal and vertical components.
  - FIX: Orbit with force_start_pos (continuing from arc) now derives
    elevation and effective radius from the 3D position of the previous
    segment's last frame, preventing vertical jumps.

v5.3-b52-fix:
  - NEW: Orbit end_angle property — orbits now use explicit start_angle
    and end_angle instead of fixed 90°/180°/360° spans, giving full
    control over the sweep.  The orbit type presets (quarter/half/full)
    auto-set end_angle = start_angle + 90/180/360 as defaults, but the
    user can override end_angle freely.
  - Panel exposes End Angle for all orbit types.

v5.2 features (retained):
  - Per-object ANCHOR POINTS with OFFSET / ABSOLUTE modes.
  - Per-object INITIAL VIEWPOINT stored as custom properties.
  - "Apply Initial Viewpoint" button.
  - Anchor & viewpoint data stored on the objects themselves.

v5.1 fixes (retained):
  - move_in vs move_out direction enforcement.
  - LINEAR keyframe interpolation for exact curve matching.
  - "Exact KF" toggle for pixel-perfect paths.

v5 features (retained):
  - Continuation from previous segment via force_start_pos.
  - Select and edit any committed segment; subsequent segments cascade.
"""

bl_info = {
    "name": "Camera Trajectory Creator",
    "author": "TrajScene Tools",
    "version": (5, 3, 2),
    "blender": (5, 0, 1),
    "location": "View3D > Sidebar > CamTraj",
    "description": "Parametric camera trajectories with per-object anchors, initial viewpoints & camera FOV control",
    "category": "Animation",
}

import bpy
import gpu
import math
import numpy as np
from gpu_extras.batch import batch_for_shader
from mathutils import Vector, Euler, Matrix, Quaternion
from bpy.props import (
    FloatProperty, IntProperty, EnumProperty,
    PointerProperty, StringProperty, BoolProperty,
    FloatVectorProperty,
)
from bpy.types import Panel, Operator, PropertyGroup


# ============================================================================
# Blender version detection
# ============================================================================

BL_VERSION = bpy.app.version  # e.g. (5, 0, 1)
USE_LAYERED_ACTIONS = BL_VERSION >= (5, 0, 0)


# ============================================================================
# Core math: look-at rotation for Blender camera
# ============================================================================

def look_at_blender(cam_pos, target_pos):
    """
    Compute 3x3 rotation for Blender camera at cam_pos looking at target_pos.
    Convention: local -Z = forward, +Y = up, +X = right.
    """
    cam_pos = np.asarray(cam_pos, dtype=np.float64)
    target_pos = np.asarray(target_pos, dtype=np.float64)

    forward = target_pos - cam_pos
    d = np.linalg.norm(forward)
    if d < 1e-8:
        return np.eye(3)
    forward /= d

    cam_z = -forward
    world_up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(forward, world_up)) > 0.9999:
        world_up = np.array([0.0, 1.0, 0.0])

    cam_x = np.cross(world_up, cam_z)
    n = np.linalg.norm(cam_x)
    if n < 1e-10:
        cam_x = np.array([1.0, 0.0, 0.0])
    else:
        cam_x /= n

    cam_y = np.cross(cam_z, cam_x)
    n = np.linalg.norm(cam_y)
    if n < 1e-10:
        cam_y = np.array([0.0, 0.0, 1.0])
    else:
        cam_y /= n

    R = np.column_stack([cam_x, cam_y, cam_z])
    if np.linalg.det(R) < 0:
        cam_x = -cam_x
        R = np.column_stack([cam_x, cam_y, cam_z])
    return R


# ============================================================================
# Easing
# ============================================================================

def ease_in_out(t):
    return np.where(t < 0.5, 2 * t * t, 1 - 0.5 * (2 - 2 * t) ** 2)


# ============================================================================
# Approach angle helpers
# ============================================================================

def _approach_to_xy(angle_deg):
    """Convert approach angle (0°=+Y, CCW) to unit direction in XY."""
    a = np.radians(angle_deg) + np.pi / 2
    return np.cos(a), np.sin(a)


# ============================================================================
# FOV helpers
# ============================================================================

def focal_to_hfov(focal_mm, sensor_width_mm):
    """Compute horizontal FOV in degrees from focal length and sensor width."""
    if focal_mm < 1e-6:
        return 180.0
    return math.degrees(2.0 * math.atan(sensor_width_mm / (2.0 * focal_mm)))


def hfov_to_focal(hfov_deg, sensor_width_mm):
    """Compute focal length in mm from horizontal FOV and sensor width."""
    hfov_rad = math.radians(max(0.1, min(179.9, hfov_deg)))
    return sensor_width_mm / (2.0 * math.tan(hfov_rad / 2.0))


# ============================================================================
# Per-object anchor & initial viewpoint helpers
# ============================================================================

def get_bbox_center_world(obj):
    """
    Return the world-space bounding box center for an object.

    Works for meshes, curves, and any object type that has bound_box.
    Falls back to obj.location if bound_box is unavailable or degenerate.

    Returns: np.array(3,) world-space bbox center.
    """
    if obj is None:
        return np.array([0.0, 0.0, 0.0])

    try:
        bb = obj.bound_box
        if bb and len(bb) == 8:
            corners_local = np.array([list(corner) for corner in bb],
                                     dtype=np.float64)
            bb_min = corners_local.min(axis=0)
            bb_max = corners_local.max(axis=0)
            center_local = (bb_min + bb_max) * 0.5

            mat_world = np.array(obj.matrix_world, dtype=np.float64)
            center_local_h = np.array([*center_local, 1.0])
            center_world = (mat_world @ center_local_h)[:3]
            return center_world
    except (AttributeError, TypeError):
        pass

    return np.array(obj.location, dtype=np.float64)


def get_object_anchor(obj):
    """
    Return the effective look-at (anchor) position for a target object.

    Uses custom properties stored on the object:
      - "camtraj_anchor_mode": 'ORIGIN' (default), 'OFFSET', or 'ABSOLUTE'
      - "camtraj_anchor_offset": [x, y, z] offset from bbox center
      - "camtraj_anchor_absolute": [x, y, z] world-space anchor position

    For ORIGIN mode, uses the mesh bounding box center (world-space).

    Returns: np.array(3,) world-space anchor position.
    """
    if obj is None:
        return np.array([0.0, 0.0, 0.0])

    mode = obj.get("camtraj_anchor_mode", 'ORIGIN')
    bbox_center = get_bbox_center_world(obj)

    if mode == 'OFFSET':
        offset = obj.get("camtraj_anchor_offset", [0.0, 0.0, 0.0])
        return bbox_center + np.array(offset, dtype=np.float64)
    elif mode == 'ABSOLUTE':
        abs_pos = obj.get("camtraj_anchor_absolute", bbox_center.tolist())
        return np.array(abs_pos, dtype=np.float64)
    else:  # 'ORIGIN' → bbox center
        return bbox_center


def get_initial_viewpoint(obj):
    """
    Return the initial viewpoint parameters stored on a target object.

    Returns: dict with keys approach_angle, distance, elevation, height_offset
    """
    if obj is None:
        return {
            'approach_angle': 0.0,
            'distance': 2.0,
            'elevation': 15.0,
            'height_offset': 0.5,
        }
    return {
        'approach_angle': obj.get("camtraj_iv_approach_angle", 0.0),
        'distance': obj.get("camtraj_iv_distance", 2.0),
        'elevation': obj.get("camtraj_iv_elevation", 15.0),
        'height_offset': obj.get("camtraj_iv_height_offset", 0.5),
    }


def set_object_anchor(obj, mode, offset=None, absolute=None):
    """Store anchor settings as custom properties on the object."""
    obj["camtraj_anchor_mode"] = mode
    if offset is not None:
        obj["camtraj_anchor_offset"] = list(offset)
    if absolute is not None:
        obj["camtraj_anchor_absolute"] = list(absolute)


def set_initial_viewpoint(obj, approach_angle, distance, elevation, height_offset):
    """Store initial viewpoint settings as custom properties on the object."""
    obj["camtraj_iv_approach_angle"] = approach_angle
    obj["camtraj_iv_distance"] = distance
    obj["camtraj_iv_elevation"] = elevation
    obj["camtraj_iv_height_offset"] = height_offset


# ============================================================================
# Trajectory generation — all in Blender Z-up coords
# ============================================================================

def generate_trajectory(move_type, n_frames, params):
    """
    Generate trajectory in Blender world coordinates (Z-up).

    Returns:
        positions:  (N, 3) float64
        rot_mats:   (N, 3, 3) float64
        focals:     (N,) float64 or None
    """
    t = np.linspace(0, 1, n_frames)
    te = ease_in_out(t)

    center = np.array(params.get('center', [0, 0, 0]), dtype=np.float64)
    anchor = np.array(params.get('anchor', center), dtype=np.float64)
    cam_pos = np.array(params.get('camera_position', [0, 0, 0]), dtype=np.float64)
    force_start = params.get('force_start_pos', None)
    focals = None

    # ------------------------------------------------------------------
    # Static — camera holds position, looking at anchor
    # ------------------------------------------------------------------
    if move_type == 'static':
        if force_start is not None:
            cam_pos = np.asarray(force_start, dtype=np.float64)
        else:
            dx, dy = _approach_to_xy(params.get('approach_angle', 0.0))
            radius = params.get('radius', 2.0)
            height = params.get('height', 0.5)
            elev_rad = np.radians(params.get('start_elevation', 0.0))
            h_dist = radius * np.cos(elev_rad)
            v_dist = radius * np.sin(elev_rad)
            cam_pos = anchor + np.array([dx * h_dist, dy * h_dist,
                                         v_dist + height],
                                        dtype=np.float64)

        positions = np.tile(cam_pos, (n_frames, 1))
        R_fixed = look_at_blender(cam_pos, anchor)
        rot_mats = np.tile(R_fixed, (n_frames, 1, 1))

    # ------------------------------------------------------------------
    # Orbits
    # ------------------------------------------------------------------
    elif move_type in ('orbit_full', 'orbit_half', 'orbit_quarter'):
        radius = params['radius']
        height = params['height']
        sa_rad = np.radians(params['start_angle']) + np.pi / 2

        # Incorporate start_elevation into orbit plane.
        start_elev_rad = np.radians(params.get('start_elevation', 0.0))
        h_radius = radius * np.cos(start_elev_rad)
        v_offset = radius * np.sin(start_elev_rad)

        orbit_center = center.copy()

        # FIX (v5.3-b52): Use explicit end_angle instead of fixed spans.
        # The orbit type presets provide default end_angle values
        # (start+90, start+180, start+360) but the user can override.
        ea_rad = np.radians(params.get('end_angle', params['start_angle'] + 90)) + np.pi / 2

        if force_start is not None:
            fs = np.asarray(force_start, dtype=np.float64)
            dx = fs[0] - orbit_center[0]
            dy = fs[1] - orbit_center[1]
            sa_rad = np.arctan2(dy, dx)
            # Recompute ea_rad relative to the new sa_rad so the sweep
            # span is preserved even when the start angle is overridden.
            original_span = ea_rad - (np.radians(params['start_angle']) + np.pi / 2)
            ea_rad = sa_rad + original_span

            # Derive elevation from the forced start position so
            # continuing from an arc at a different height works.
            dz = fs[2] - orbit_center[2]
            h_dist_fs = np.sqrt(dx**2 + dy**2)
            fs_radius_3d = np.sqrt(dx**2 + dy**2 + dz**2)
            if fs_radius_3d > 1e-6:
                start_elev_rad = np.arctan2(dz, max(h_dist_fs, 1e-6))
                h_radius = fs_radius_3d * np.cos(start_elev_rad)
                v_offset = fs_radius_3d * np.sin(start_elev_rad)

        # Sweep from sa_rad to ea_rad
        angles = sa_rad + t * (ea_rad - sa_rad)
        positions = np.column_stack([
            orbit_center[0] + h_radius * np.cos(angles),
            orbit_center[1] + h_radius * np.sin(angles),
            np.full(n_frames, orbit_center[2] + v_offset + height),
        ])

        look_target = anchor.copy()
        total_height = v_offset + height
        if abs(total_height) > 0.01:
            look_target[2] = anchor[2] + total_height * 0.3

        rot_mats = np.array([look_at_blender(positions[i], look_target)
                             for i in range(n_frames)])

        pitch_off = np.radians(params.get('pitch_offset', 0))
        if abs(pitch_off) > 1e-6:
            for i in range(n_frames):
                cam_x = rot_mats[i][:, 0]
                K = np.array([[0, -cam_x[2], cam_x[1]],
                              [cam_x[2], 0, -cam_x[0]],
                              [-cam_x[1], cam_x[0], 0]])
                R_p = (np.eye(3) + np.sin(pitch_off) * K
                       + (1 - np.cos(pitch_off)) * K @ K)
                rot_mats[i] = R_p @ rot_mats[i]

    # ------------------------------------------------------------------
    # Pan
    # ------------------------------------------------------------------
    elif move_type in ('pan_left', 'pan_right'):
        yr = np.radians(params['yaw_range'])
        pitch_offset = np.radians(params.get('pitch', 0))
        s_yaw, e_yaw = ((-yr / 2, yr / 2) if move_type == 'pan_left'
                 else (yr / 2, -yr / 2))

        if force_start is not None:
            cam_pos = np.asarray(force_start, dtype=np.float64)
            s_yaw, e_yaw = ((0, yr) if move_type == 'pan_left'
                             else (0, -yr))
        else:
            s_yaw, e_yaw = ((-yr / 2, yr / 2) if move_type == 'pan_left'
                             else (yr / 2, -yr / 2))

        ref_dir = anchor - cam_pos
        ref_dist_xy = np.linalg.norm(ref_dir[:2])
        ref_dist_3d = np.linalg.norm(ref_dir)
        base_yaw = (np.arctan2(ref_dir[1], ref_dir[0])
                    if ref_dist_xy > 1e-6 else 0.0)
        base_pitch = (np.arctan2(ref_dir[2], ref_dist_xy)
                      if ref_dist_3d > 1e-6 else 0.0)
        base_pitch += pitch_offset

        positions = np.tile(cam_pos, (n_frames, 1))
        rot_mats = np.zeros((n_frames, 3, 3))
        for i in range(n_frames):
            yaw = base_yaw + s_yaw + te[i] * (e_yaw - s_yaw)
            look_dir = np.array([np.cos(yaw) * np.cos(base_pitch),
                                  np.sin(yaw) * np.cos(base_pitch),
                                  np.sin(base_pitch)])
            rot_mats[i] = look_at_blender(cam_pos, cam_pos + look_dir * 10.0)

    # ------------------------------------------------------------------
    # Dolly (move_in / move_out)
    # ------------------------------------------------------------------
    elif move_type in ('move_in', 'move_out'):
        sd = params['start_radius']
        ed = params['end_radius']
        height = params['height']

        if force_start is not None:
            fs = np.asarray(force_start, dtype=np.float64)
            ray = fs - anchor
            ray_len = np.linalg.norm(ray)
            if ray_len < 1e-8:
                dx, dy = _approach_to_xy(params['approach_angle'])
                ray = np.array([dx, dy, 0.0])
                ray_len = 1.0
            direction = ray / ray_len
        else:
            dx, dy = _approach_to_xy(params['approach_angle'])
            ray = np.array([dx * sd, dy * sd, height], dtype=np.float64)
            ray_len = np.linalg.norm(ray)
            if ray_len < 1e-8:
                ray = np.array([dx, dy, 0.0])
                ray_len = 1.0
            direction = ray / ray_len

        dist = sd + t * (ed - sd)
        positions = anchor[np.newaxis, :] + dist[:, np.newaxis] * direction[np.newaxis, :]

        look_target = anchor.copy()
        avg_z = np.mean(positions[:, 2]) - anchor[2]
        if abs(avg_z) > 0.01:
            look_target[2] = anchor[2] + avg_z * 0.3

        rot_mats = np.array([look_at_blender(positions[i], look_target)
                             for i in range(n_frames)])

    # ------------------------------------------------------------------
    # Crane
    # ------------------------------------------------------------------
    elif move_type == 'crane':
        radius = params['radius']
        se = np.radians(params['start_elevation'])
        ee = np.radians(params['end_elevation'])
        dx, dy = _approach_to_xy(params['approach_angle'])

        if force_start is not None:
            fs = np.asarray(force_start, dtype=np.float64)
            off_x = fs[0] - center[0]
            off_y = fs[1] - center[1]
            off_z = fs[2] - center[2]
            h_dist = np.sqrt(off_x**2 + off_y**2)
            radius = np.sqrt(off_x**2 + off_y**2 + off_z**2)
            se = np.arctan2(off_z, max(h_dist, 1e-6))
            if h_dist > 1e-6:
                dx, dy = off_x / h_dist, off_y / h_dist

        elev = se + te * (ee - se)
        h_dist = radius * np.cos(elev)
        v_dist = radius * np.sin(elev)

        positions = np.column_stack([
            center[0] + h_dist * dx,
            center[1] + h_dist * dy,
            center[2] + v_dist,
        ])
        rot_mats = np.array([look_at_blender(positions[i], anchor)
                             for i in range(n_frames)])

    # ------------------------------------------------------------------
    # Tilt
    # ------------------------------------------------------------------
    elif move_type in ('tilt_up', 'tilt_down'):
        pr = np.radians(params['pitch_range'])
        sp, ep = ((-pr / 2, pr / 2) if move_type == 'tilt_up'
                  else (pr / 2, -pr / 2))

        if force_start is not None:
            cam_pos = np.asarray(force_start, dtype=np.float64)
            sp, ep = ((0, pr) if move_type == 'tilt_up'
                       else (0, -pr))
        else:
            sp, ep = ((-pr / 2, pr / 2) if move_type == 'tilt_up'
                       else (pr / 2, -pr / 2))

        ref_dir = anchor - cam_pos
        ref_dist_xy = np.linalg.norm(ref_dir[:2])
        ref_dist_3d = np.linalg.norm(ref_dir)
        base_yaw = (np.arctan2(ref_dir[1], ref_dir[0])
                    if ref_dist_xy > 1e-6
                    else np.radians(params.get('yaw', 0)))
        base_pitch = (np.arctan2(ref_dir[2], ref_dist_xy)
                      if ref_dist_3d > 1e-6 else 0.0)

        positions = np.tile(cam_pos, (n_frames, 1))
        rot_mats = np.zeros((n_frames, 3, 3))
        for i in range(n_frames):
            pitch = base_pitch + sp + te[i] * (ep - sp)
            look_dir = np.array([np.cos(base_yaw) * np.cos(pitch),
                                  np.sin(base_yaw) * np.cos(pitch),
                                  np.sin(pitch)])
            rot_mats[i] = look_at_blender(cam_pos, cam_pos + look_dir * 10.0)

    # ------------------------------------------------------------------
    # Zoom
    # ------------------------------------------------------------------
    elif move_type in ('zoom_in_out', 'zoom_out_in'):
        if force_start is not None:
            cam_pos = np.asarray(force_start, dtype=np.float64)

        peak = params['peak_multiplier']
        bf = params['base_focal']
        positions = np.tile(cam_pos, (n_frames, 1))
        R_fixed = look_at_blender(cam_pos, anchor)
        rot_mats = np.tile(R_fixed, (n_frames, 1, 1))
        focals = bf * (1.0 + (peak - 1.0) * np.sin(np.pi * te))

    # ------------------------------------------------------------------
    # Arc
    # ------------------------------------------------------------------
    elif move_type == 'arc':
        sp = np.array(params['start_position'], dtype=np.float64)
        ep = np.array(params['end_position'], dtype=np.float64)
        sr = np.array(params.get('start_rotation_mat', np.eye(3)),
                      dtype=np.float64)
        er = np.array(params.get('end_rotation_mat', np.eye(3)),
                      dtype=np.float64)
        arc_a = np.radians(params.get('arc_angle', 0))

        if force_start is not None:
            sp = np.asarray(force_start, dtype=np.float64)

        positions = (sp[np.newaxis, :]
                     + te[:, np.newaxis] * (ep - sp)[np.newaxis, :])

        if abs(arc_a) > 1e-6:
            chord_xy = ep[:2] - sp[:2]
            cl = np.linalg.norm(chord_xy)
            if cl > 1e-6:
                cd = chord_xy / cl
                perp = np.array([-cd[1], cd[0]])
                amp = np.tan(arc_a) * cl * 0.5
                bulge = amp * np.sin(np.pi * te)
                positions[:, 0] += bulge * perp[0]
                positions[:, 1] += bulge * perp[1]

        from mathutils import Quaternion as MQuat
        q_start = Matrix(sr.tolist()).to_3x3().to_quaternion()
        q_end = Matrix(er.tolist()).to_3x3().to_quaternion()
        if q_start.dot(q_end) < 0:
            q_end = -q_end

        rot_mats = np.zeros((n_frames, 3, 3))
        for i in range(n_frames):
            q = q_start.slerp(q_end, te[i])
            rot_mats[i] = np.array(q.to_matrix())

    else:
        raise ValueError(f"Unknown movement: {move_type}")

    return positions, rot_mats, focals


# ============================================================================
# Committed segments storage
# ============================================================================

_committed_segments = []


# ============================================================================
# Params builder
# ============================================================================

def _build_params_from_props(props, center, anchor, cam_pos):
    """Build trajectory params dict from property group values."""
    return {
        'center': center.tolist(),
        'anchor': anchor.tolist(),
        'camera_position': cam_pos.tolist(),
        'radius': props.radius,
        'height': props.height,
        'start_angle': props.start_angle,
        'end_angle': props.end_angle,
        'pitch_offset': props.pitch_offset,
        'approach_angle': props.approach_angle,
        'start_radius': props.start_radius,
        'end_radius': props.end_radius,
        'start_elevation': props.start_elevation,
        'end_elevation': props.end_elevation,
        'yaw_range': props.yaw_range,
        'pitch_range': props.pitch_range,
        'yaw': props.yaw_fixed,
        'pitch': props.pitch_fixed,
        'peak_multiplier': props.peak_multiplier,
        'base_focal': props.base_focal,
        'arc_angle': props.arc_angle,
    }


def _get_preview_data(context, force_start_pos=None):
    """Build trajectory params from current panel settings."""
    props = context.scene.camtraj
    target = props.target_object

    center = np.array(target.location) if target else np.array([0, 0, 0])
    anchor = get_object_anchor(target)

    dx, dy = _approach_to_xy(props.approach_angle)
    elev_rad = np.radians(props.start_elevation)
    h_dist = props.radius * np.cos(elev_rad)
    v_dist = props.radius * np.sin(elev_rad)

    cam_pos = np.array([
        anchor[0] + dx * h_dist,
        anchor[1] + dy * h_dist,
        anchor[2] + v_dist + props.height,
    ])

    params = _build_params_from_props(props, center, anchor, cam_pos)

    fsp = force_start_pos
    if fsp is None and props.continue_from_previous and len(_committed_segments) > 0:
        edit_idx = props.edit_segment_index
        if edit_idx >= 0:
            if edit_idx > 0:
                fsp = _committed_segments[edit_idx - 1]['positions'][-1].copy()
        else:
            fsp = _committed_segments[-1]['positions'][-1].copy()

    if fsp is not None:
        params['force_start_pos'] = (fsp.tolist() if hasattr(fsp, 'tolist')
                                     else list(fsp))

    move = props.move_type
    if move == 'arc':
        if fsp is not None:
            sp = np.asarray(fsp, dtype=np.float64)
        elif props.continue_from_previous and len(_committed_segments) > 0:
            sp = _committed_segments[-1]['positions'][-1].copy()
        else:
            sp = cam_pos.copy()

        sr_mat = np.eye(3)
        prev_idx = (props.edit_segment_index - 1
                    if props.edit_segment_index >= 0
                    else len(_committed_segments) - 1)
        if 0 <= prev_idx < len(_committed_segments):
            sr_mat = _committed_segments[prev_idx]['rot_mats'][-1].copy()

        if target is not None:
            dx_ep, dy_ep = _approach_to_xy(props.approach_angle)
            ep_elev_rad = np.radians(props.start_elevation)
            ep_h_dist = props.radius * np.cos(ep_elev_rad)
            ep_v_dist = props.radius * np.sin(ep_elev_rad)
            ep = anchor + np.array([dx_ep * ep_h_dist,
                                    dy_ep * ep_h_dist,
                                    ep_v_dist + props.height],
                                   dtype=np.float64)
            er_mat = look_at_blender(ep, anchor)
        else:
            ep = sp + np.array([3, 0, 0])
            er_mat = np.eye(3)
            ep[2] = sp[2] + props.height
        params['start_position'] = sp.tolist()
        params['end_position'] = ep.tolist()
        params['start_rotation_mat'] = sr_mat
        params['end_rotation_mat'] = er_mat

    return params


# ============================================================================
# Cascade: regenerate segments from edit_index onward
# ============================================================================

def _cascade_segments_from(start_idx):
    """Re-generate segments from start_idx onward, chaining start positions."""
    for i in range(start_idx, len(_committed_segments)):
        seg = _committed_segments[i]
        params = seg['params'].copy()

        if i > 0:
            prev_end = _committed_segments[i - 1]['positions'][-1].copy()
            params['force_start_pos'] = prev_end.tolist()
        elif 'force_start_pos' in params:
            del params['force_start_pos']

        positions, rot_mats, focals = generate_trajectory(
            seg['move_type'], seg['n_frames'], params
        )

        seg['positions'] = positions
        seg['rot_mats'] = rot_mats
        seg['focals'] = focals
        seg['params'] = params


# ============================================================================
# Visualization
# ============================================================================

_draw_handler = None
TRAJ_COLLECTION = "CamTraj_Viz"

SEGMENT_COLORS = [
    (0.15, 0.5, 1.0),
    (0.15, 0.85, 0.25),
    (1.0, 0.5, 0.05),
    (0.85, 0.15, 0.75),
    (1.0, 0.9, 0.1),
    (0.05, 0.85, 0.85),
]


def _get_or_create_collection(name):
    if name in bpy.data.collections:
        return bpy.data.collections[name]
    col = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(col)
    return col


def _get_or_create_material(name, color, alpha=1.0):
    if name in bpy.data.materials:
        mat = bpy.data.materials[name]
        mat.diffuse_color = (*color, alpha)
        if mat.use_nodes and mat.node_tree:
            for node in mat.node_tree.nodes:
                if node.type == 'EMISSION':
                    node.inputs[0].default_value = (*color, 1.0)
        return mat
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (*color, alpha)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    emission = nodes.new('ShaderNodeEmission')
    emission.inputs[0].default_value = (*color, 1.0)
    emission.inputs[1].default_value = 3.0
    output = nodes.new('ShaderNodeOutputMaterial')
    links.new(emission.outputs[0], output.inputs[0])
    return mat


def _create_curve_from_points(name, positions, color, bevel_depth=0.015,
                               collection=None):
    n = len(positions)
    if n < 2:
        return None
    curve_data = bpy.data.curves.new(name, 'CURVE')
    curve_data.dimensions = '3D'
    curve_data.bevel_depth = bevel_depth
    curve_data.bevel_resolution = 2
    curve_data.fill_mode = 'FULL'
    spline = curve_data.splines.new('POLY')
    spline.points.add(n - 1)
    for i in range(n):
        spline.points[i].co = (*positions[i], 1.0)
    curve_obj = bpy.data.objects.new(name, curve_data)
    mat_name = f"TrajMat_{name}"
    mat = _get_or_create_material(mat_name, color)
    curve_obj.data.materials.append(mat)
    col = collection or _get_or_create_collection(TRAJ_COLLECTION)
    col.objects.link(curve_obj)
    curve_obj.hide_select = True
    return curve_obj


def _create_endpoint_markers(name_prefix, start_pos, end_pos, collection=None):
    col = collection or _get_or_create_collection(TRAJ_COLLECTION)
    for label, pos, color in [("Start", start_pos, (0, 1, 0)),
                                ("End", end_pos, (1, 0, 0))]:
        obj_name = f"{name_prefix}_{label}"
        old = bpy.data.objects.get(obj_name)
        if old:
            bpy.data.objects.remove(old, do_unlink=True)
        empty = bpy.data.objects.new(obj_name, None)
        empty.empty_display_type = 'SPHERE'
        empty.empty_display_size = 0.025
        empty.location = Vector(pos)
        empty.hide_select = True
        empty.color = (*color, 1.0)
        empty.show_in_front = True
        col.objects.link(empty)


def _create_frustum_empties(name_prefix, positions, rot_mats, n_frustums=6,
                             color=(1, 1, 1), scale=0.03, collection=None):
    col = collection or _get_or_create_collection(TRAJ_COLLECTION)
    n = len(positions)
    step = max(1, n // n_frustums)
    indices = list(range(0, n, step))
    if indices[-1] != n - 1:
        indices.append(n - 1)
    for idx, fi in enumerate(indices):
        obj_name = f"{name_prefix}_Cam{idx}"
        old = bpy.data.objects.get(obj_name)
        if old:
            bpy.data.objects.remove(old, do_unlink=True)
        empty = bpy.data.objects.new(obj_name, None)
        empty.empty_display_type = 'SINGLE_ARROW'
        empty.empty_display_size = scale
        empty.location = Vector(positions[fi])
        R = rot_mats[fi]
        U, _, Vt = np.linalg.svd(R)
        R_clean = U @ Vt
        if np.linalg.det(R_clean) < 0:
            U[:, -1] *= -1
            R_clean = U @ Vt
        mat = Matrix(R_clean.tolist()).to_3x3()
        q_cam = mat.to_quaternion()
        q_flip = Quaternion((0, 1, 0, 0))
        q_final = q_cam @ q_flip
        empty.rotation_mode = 'QUATERNION'
        empty.rotation_quaternion = q_final
        empty.hide_select = True
        empty.color = (*color, 1.0)
        empty.show_in_front = True
        col.objects.link(empty)


def _create_anchor_marker(anchor_pos, collection=None):
    """Create/update a diamond marker showing the current anchor point."""
    col = collection or _get_or_create_collection(TRAJ_COLLECTION)
    obj_name = "TrajAnchor"
    old = bpy.data.objects.get(obj_name)
    if old:
        bpy.data.objects.remove(old, do_unlink=True)
    marker = bpy.data.objects.new(obj_name, None)
    marker.empty_display_type = 'PLAIN_AXES'
    marker.empty_display_size = 0.08
    marker.location = Vector(anchor_pos)
    marker.hide_select = True
    marker.color = (1.0, 0.2, 0.2, 1.0)
    marker.show_in_front = True
    col.objects.link(marker)
    return marker


def _remove_objects_with_prefix(prefix):
    to_remove = [obj for obj in bpy.data.objects if obj.name.startswith(prefix)]
    for obj in to_remove:
        bpy.data.objects.remove(obj, do_unlink=True)


def _remove_preview_objects():
    _remove_objects_with_prefix("_TrajPreview")


def _remove_all_viz_objects():
    _remove_objects_with_prefix("_TrajPreview")
    _remove_objects_with_prefix("TrajPath_")
    _remove_objects_with_prefix("TrajTarget")
    _remove_objects_with_prefix("TrajAnchor")
    col = bpy.data.collections.get(TRAJ_COLLECTION)
    if col and len(col.objects) == 0:
        bpy.data.collections.remove(col)


def _rebuild_all_committed_curves():
    """Rebuild all committed segment curves."""
    _remove_objects_with_prefix("TrajPath_")
    for i, seg in enumerate(_committed_segments):
        _create_committed_curve(i, seg['positions'], seg['rot_mats'],
                                 seg['name'], seg['color_index'])


def _update_camera_lens(context):
    """Update TrajCamera lens from panel properties (live feedback)."""
    props = context.scene.camtraj
    cam_obj = bpy.data.objects.get("TrajCamera")
    if cam_obj and cam_obj.data:
        cam_obj.data.lens = props.base_focal
        cam_obj.data.sensor_width = props.sensor_width


def _update_preview_curve(context):
    props = context.scene.camtraj
    _remove_preview_objects()

    _update_camera_lens(context)

    if not props.show_preview:
        return
    try:
        params = _get_preview_data(context)
        positions, rot_mats, _ = generate_trajectory(props.move_type, 80, params)
    except:
        return
    if len(positions) < 2:
        return

    col = _get_or_create_collection(TRAJ_COLLECTION)

    if props.edit_segment_index >= 0:
        ci = props.edit_segment_index
        preview_color = SEGMENT_COLORS[ci % len(SEGMENT_COLORS)]
    else:
        preview_color = (1.0, 1.0, 1.0)

    _create_curve_from_points("_TrajPreview_Curve", positions, preview_color,
                               bevel_depth=0.02, collection=col)
    _create_endpoint_markers("_TrajPreview", positions[0], positions[-1], col)
    _create_frustum_empties("_TrajPreview", positions, rot_mats, n_frustums=6,
                             color=preview_color, scale=0.04, collection=col)

    target = props.target_object
    if target is not None:
        old = bpy.data.objects.get("TrajTarget")
        if old:
            bpy.data.objects.remove(old, do_unlink=True)
        marker = bpy.data.objects.new("TrajTarget", None)
        marker.empty_display_type = 'PLAIN_AXES'
        marker.empty_display_size = 0.06
        marker.location = target.location.copy()
        marker.hide_select = True
        marker.color = (1.0, 0.8, 0.0, 1.0)
        marker.show_in_front = True
        col.objects.link(marker)

        anchor = get_object_anchor(target)
        _create_anchor_marker(anchor, col)


def _create_committed_curve(seg_index, positions, rot_mats, name, color_idx):
    col = _get_or_create_collection(TRAJ_COLLECTION)
    color = SEGMENT_COLORS[color_idx % len(SEGMENT_COLORS)]
    prefix = f"TrajPath_{seg_index}"
    _create_curve_from_points(f"{prefix}_Curve", positions, color,
                               bevel_depth=0.012, collection=col)
    _create_endpoint_markers(prefix, positions[0], positions[-1], col)
    _create_frustum_empties(prefix, positions, rot_mats, n_frustums=6,
                             color=color, scale=0.08, collection=col)


def _draw_callback():
    pass


def register_draw_handler():
    global _draw_handler
    if _draw_handler is None:
        _draw_handler = bpy.types.SpaceView3D.draw_handler_add(
            _draw_callback, (), 'WINDOW', 'POST_VIEW')


def unregister_draw_handler():
    global _draw_handler
    if _draw_handler is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_draw_handler, 'WINDOW')
        _draw_handler = None


# ============================================================================
# F-curve helpers — Blender 5.0+ layered action API
# ============================================================================

def _get_channelbag_for_datablock(action, datablock):
    from bpy_extras import anim_utils
    action_slot = datablock.animation_data.action_slot
    return anim_utils.action_get_channelbag_for_slot(action, action_slot)


def _ensure_channelbag_for_datablock(action, datablock):
    from bpy_extras import anim_utils
    action_slot = datablock.animation_data.action_slot
    return anim_utils.action_ensure_channelbag_for_slot(action, action_slot)


def ensure_fcurve(action, datablock, data_path, index):
    return action.fcurve_ensure_for_datablock(
        datablock, data_path, index=index
    )


def batch_insert_kf(action, datablock, data_path, index, frames, values,
                    interpolation='LINEAR'):
    fc = ensure_fcurve(action, datablock, data_path, index)
    existing = len(fc.keyframe_points)
    fc.keyframe_points.add(len(frames))
    for i, (f, v) in enumerate(zip(frames, values)):
        kf = fc.keyframe_points[existing + i]
        kf.co = (float(f), float(v))
        kf.interpolation = interpolation
        if interpolation == 'LINEAR':
            kf.handle_left_type = 'VECTOR'
            kf.handle_right_type = 'VECTOR'
        else:
            kf.handle_left_type = 'AUTO_CLAMPED'
            kf.handle_right_type = 'AUTO_CLAMPED'
    fc.update()


def tag_redraw(context):
    for area in context.screen.areas:
        if area.type == 'VIEW_3D':
            area.tag_redraw()


# ============================================================================
# Keyframe rebuild from all committed segments
# ============================================================================

def _rebuild_all_keyframes(context):
    props = context.scene.camtraj

    cam_obj = bpy.data.objects.get("TrajCamera")
    if cam_obj is None:
        cam_data = bpy.data.cameras.new("TrajCameraData")
        cam_data.lens = props.base_focal
        cam_data.sensor_width = props.sensor_width
        cam_obj = bpy.data.objects.new("TrajCamera", cam_data)
        context.scene.collection.objects.link(cam_obj)

    cam_data = cam_obj.data
    cam_data.lens = props.base_focal
    cam_data.sensor_width = props.sensor_width
    try:
        cam_data.display_size = 0.1
    except AttributeError:
        try:
            cam_data.draw_size = 0.1
        except AttributeError:
            pass
    cam_data.clip_start = 0.01
    cam_data.clip_end = 50.0
    cam_obj.rotation_mode = 'QUATERNION'
    context.scene.camera = cam_obj

    if cam_obj.animation_data:
        cam_obj.animation_data_clear()
    if cam_data.animation_data:
        cam_data.animation_data_clear()

    cam_obj.animation_data_create()
    action = bpy.data.actions.new("TrajCameraAction")
    cam_obj.animation_data.action = action

    if not cam_obj.animation_data.action_slot:
        if action.slots:
            cam_obj.animation_data.action_slot = action.slots[0]
        else:
            slot = action.slots.new(id_type='OBJECT', name=cam_obj.name)
            cam_obj.animation_data.action_slot = slot

    has_focals = any(seg['focals'] is not None for seg in _committed_segments)
    data_action = None
    if has_focals:
        cam_data.animation_data_create()
        data_action = bpy.data.actions.new("TrajCameraDataAction")
        cam_data.animation_data.action = data_action

        if not cam_data.animation_data.action_slot:
            if data_action.slots:
                cam_data.animation_data.action_slot = data_action.slots[0]
            else:
                slot = data_action.slots.new(id_type='CAMERA', name=cam_data.name)
                cam_data.animation_data.action_slot = slot

    if props.exact_keyframes:
        step = 1
    else:
        step = max(1, props.keyframe_step)

    interp = 'LINEAR'

    for seg in _committed_segments:
        positions = seg['positions']
        rot_mats = seg['rot_mats']
        focals = seg['focals']
        n_frames = seg['n_frames']
        f_start = seg['frame_start']

        quats = np.zeros((n_frames, 4))
        for i in range(n_frames):
            R = rot_mats[i]
            U, _, Vt = np.linalg.svd(R)
            R_clean = U @ Vt
            if np.linalg.det(R_clean) < 0:
                U[:, -1] *= -1
                R_clean = U @ Vt
            mat = Matrix(R_clean.tolist()).to_3x3()
            q = mat.to_quaternion()
            quats[i] = [q.w, q.x, q.y, q.z]

        for i in range(1, n_frames):
            if np.dot(quats[i], quats[i - 1]) < 0:
                quats[i] = -quats[i]

        indices = list(range(0, n_frames, step))
        if indices[-1] != n_frames - 1:
            indices.append(n_frames - 1)
        indices = np.array(indices)
        frame_numbers = (f_start + indices).astype(np.float64)

        for axis in range(3):
            batch_insert_kf(action, cam_obj, "location", axis,
                           frame_numbers, positions[indices, axis], interp)

        for qi in range(4):
            batch_insert_kf(action, cam_obj, "rotation_quaternion", qi,
                           frame_numbers, quats[indices, qi], interp)

        if focals is not None and data_action is not None:
            batch_insert_kf(data_action, cam_data, "lens", 0,
                           frame_numbers, focals[indices], interp)


# ============================================================================
# Properties
# ============================================================================

MOVE_ITEMS = [
    ('static',         "Static (Hold)",          "Camera holds position, looking at anchor"),
    ('orbit_full',     "Orbit Full (360°)",      ""),
    ('orbit_half',     "Orbit Half (180°)",       ""),
    ('orbit_quarter',  "Orbit Quarter (90°)",     ""),
    ('pan_left',       "Pan Left",                ""),
    ('pan_right',      "Pan Right",               ""),
    ('move_in',        "Move In (Dolly)",         ""),
    ('move_out',       "Move Out (Dolly)",        ""),
    ('crane',          "Crane",                   ""),
    ('tilt_up',        "Tilt Up",                 ""),
    ('tilt_down',      "Tilt Down",               ""),
    ('zoom_in_out',    "Zoom In→Out",             ""),
    ('zoom_out_in',    "Zoom Out→In",             ""),
    ('arc',            "Arc (Transitional)",       ""),
]

ANCHOR_MODE_ITEMS = [
    ('ORIGIN',   "BBox Center",     "Look at the object's bounding box center"),
    ('OFFSET',   "Offset",          "Look at bbox center + custom XYZ offset"),
    ('ABSOLUTE', "Absolute",        "Look at an absolute world position"),
]


def _on_update(self, context):
    try:
        _update_preview_curve(context)
    except:
        pass
    tag_redraw(context)


def _on_camera_update(self, context):
    _update_camera_lens(context)
    _on_update(self, context)


def _on_anchor_update(self, context):
    props = context.scene.camtraj
    target = props.target_object
    if target is None:
        return
    set_object_anchor(
        target,
        mode=props.anchor_mode,
        offset=list(props.anchor_offset),
        absolute=list(props.anchor_absolute),
    )
    _on_update(self, context)


def _on_iv_update(self, context):
    props = context.scene.camtraj
    target = props.target_object
    if target is None:
        return

    set_initial_viewpoint(
        target,
        approach_angle=props.iv_approach_angle,
        distance=props.iv_distance,
        elevation=props.iv_elevation,
        height_offset=props.iv_height_offset,
    )

    if props.auto_apply_iv:
        props["approach_angle"] = props.iv_approach_angle
        props["radius"] = props.iv_distance
        props["start_angle"] = props.iv_approach_angle
        props["height"] = props.iv_height_offset
        props["start_elevation"] = props.iv_elevation
        props["start_radius"] = props.iv_distance

    _on_update(self, context)


def _on_move_type_update(self, context):
    """When movement type changes, auto-set end_angle for orbit presets."""
    props = context.scene.camtraj
    m = props.move_type
    if m == 'orbit_quarter':
        props["end_angle"] = props.start_angle + 90.0
    elif m == 'orbit_half':
        props["end_angle"] = props.start_angle + 180.0
    elif m == 'orbit_full':
        props["end_angle"] = props.start_angle + 360.0
    _on_update(self, context)


def _on_target_update(self, context):
    props = context.scene.camtraj
    target = props.target_object
    if target is not None:
        mode = target.get("camtraj_anchor_mode", 'ORIGIN')
        if mode not in ('ORIGIN', 'OFFSET', 'ABSOLUTE'):
            mode = 'ORIGIN'
        props["anchor_mode"] = ANCHOR_MODE_ITEMS_INDEX.get(mode, 0)
        offset = target.get("camtraj_anchor_offset", [0.0, 0.0, 0.0])
        props["anchor_offset"] = offset
        abs_pos = target.get("camtraj_anchor_absolute",
                             list(target.location))
        props["anchor_absolute"] = abs_pos

        iv = get_initial_viewpoint(target)
        props["iv_approach_angle"] = iv['approach_angle']
        props["iv_distance"] = iv['distance']
        props["iv_elevation"] = iv['elevation']
        props["iv_height_offset"] = iv['height_offset']

    _on_update(self, context)


ANCHOR_MODE_ITEMS_INDEX = {'ORIGIN': 0, 'OFFSET': 1, 'ABSOLUTE': 2}


class CamTrajProperties(PropertyGroup):
    move_type: EnumProperty(name="Movement", items=MOVE_ITEMS,
                            default='orbit_quarter', update=_on_move_type_update)

    frame_start: IntProperty(name="Start Frame", default=1, min=1)
    frame_end: IntProperty(name="End Frame", default=60, min=2)

    radius: FloatProperty(name="Radius", default=2.0, min=0.1, max=50.0,
                          step=10, update=_on_update)
    height: FloatProperty(name="Height (Z offset)", default=0.5,
                          min=-20.0, max=20.0, step=10, update=_on_update)
    start_angle: FloatProperty(name="Start Angle (°)", default=0.0,
                                min=-360, max=360, step=100, update=_on_update)
    end_angle: FloatProperty(
        name="End Angle (°)",
        default=90.0,
        min=-720, max=720, step=100,
        description="End angle for orbit sweep (degrees). "
                    "The camera sweeps from Start Angle to End Angle",
        update=_on_update,
    )
    pitch_offset: FloatProperty(name="Pitch Offset (°)", default=0.0,
                                 min=-60, max=60, step=100, update=_on_update)
    approach_angle: FloatProperty(name="Approach Angle (°)", default=0.0,
                                   min=-360, max=360, step=100, update=_on_update)

    start_radius: FloatProperty(name="Start Distance", default=3.0,
                                 min=0.1, max=50.0, step=10, update=_on_update)
    end_radius: FloatProperty(name="End Distance", default=1.0,
                               min=0.1, max=50.0, step=10, update=_on_update)

    start_elevation: FloatProperty(name="Start Elevation (°)", default=10.0,
                                    min=-10, max=85, step=100, update=_on_update)
    end_elevation: FloatProperty(name="End Elevation (°)", default=70.0,
                                  min=10, max=90, step=100, update=_on_update)

    yaw_range: FloatProperty(name="Yaw Range (°)", default=45.0,
                              min=5, max=180, step=100, update=_on_update)
    pitch_range: FloatProperty(name="Pitch Range (°)", default=45.0,
                                min=5, max=120, step=100, update=_on_update)
    yaw_fixed: FloatProperty(name="Yaw (°)", default=0.0,
                              min=-180, max=180, step=100, update=_on_update)
    pitch_fixed: FloatProperty(name="Pitch (°)", default=0.0,
                                min=-90, max=90, step=100, update=_on_update)

    peak_multiplier: FloatProperty(name="Peak Multiplier", default=2.0,
                                    min=0.2, max=5.0, step=10, update=_on_update)

    base_focal: FloatProperty(
        name="Focal Length (mm)",
        default=50.0,
        min=1.0, max=500.0,
        step=100,
        description="Camera focal length in millimetres",
        update=_on_camera_update,
    )
    sensor_width: FloatProperty(
        name="Sensor Width (mm)",
        default=100.0,
        min=1.0, max=100.0,
        step=10,
        description="Camera sensor width in millimetres (36 mm = full-frame)",
        update=_on_camera_update,
    )

    arc_angle: FloatProperty(name="Arc Angle (°)", default=30.0,
                              min=-89, max=89, step=100, update=_on_update)

    target_object: PointerProperty(name="Target Object", type=bpy.types.Object,
                                    update=_on_target_update)
    continue_from_previous: BoolProperty(name="Continue From Previous",
                                          default=True)
    segment_count: IntProperty(name="Segments Added", default=0)
    keyframe_step: IntProperty(name="KF Every N Frames", default=5,
                                min=1, max=30)
    exact_keyframes: BoolProperty(
        name="Exact KF (every frame)",
        default=False,
        description="Keyframe every single frame for exact path matching",
    )

    show_preview: BoolProperty(name="Show Live Preview", default=True,
                                update=_on_update)
    show_committed: BoolProperty(name="Show Committed", default=True,
                                  update=_on_update)

    edit_segment_index: IntProperty(name="Edit Segment Index", default=-1)

    anchor_mode: EnumProperty(
        name="Anchor Mode",
        items=ANCHOR_MODE_ITEMS,
        default='ORIGIN',
        description="How to determine the camera look-at point for this object",
        update=_on_anchor_update,
    )
    anchor_offset: FloatVectorProperty(
        name="Anchor Offset",
        size=3,
        default=(0.0, 0.0, 0.0),
        subtype='TRANSLATION',
        description="XYZ offset from object origin for the look-at point",
        update=_on_anchor_update,
    )
    anchor_absolute: FloatVectorProperty(
        name="Anchor Position",
        size=3,
        default=(0.0, 0.0, 0.0),
        subtype='XYZ',
        description="Absolute world-space look-at position",
        update=_on_anchor_update,
    )

    auto_apply_iv: BoolProperty(
        name="Auto-Apply IV",
        default=True,
        description="Automatically push IV slider changes into trajectory parameters",
    )
    iv_approach_angle: FloatProperty(
        name="IV Approach (°)",
        default=0.0,
        min=-360, max=360, step=100,
        description="Initial camera approach angle around the object",
        update=_on_iv_update,
    )
    iv_distance: FloatProperty(
        name="IV Distance",
        default=1.0,
        min=0.1, max=50.0, step=10,
        description="Initial camera distance from the object",
        update=_on_iv_update,
    )
    iv_elevation: FloatProperty(
        name="IV Elevation (°)",
        default=15.0,
        min=-10, max=85, step=100,
        description="Initial camera elevation angle",
        update=_on_iv_update,
    )
    iv_height_offset: FloatProperty(
        name="IV Height Offset",
        default=0.5,
        min=-20.0, max=20.0, step=10,
        description="Initial camera height offset from object",
        update=_on_iv_update,
    )


# ============================================================================
# Operators
# ============================================================================

class CAMTRAJ_OT_add_segment(Operator):
    bl_idname = "camtraj.add_segment"
    bl_label = "Commit Segment"
    bl_description = "Insert keyframes for current trajectory segment"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.camtraj
        move = props.move_type
        f_start = props.frame_start
        f_end = props.frame_end
        n_frames = max(f_end - f_start + 1, 2)

        try:
            params = _get_preview_data(context)
            positions, rot_mats, focals = generate_trajectory(move, n_frames,
                                                               params)
        except Exception as e:
            self.report({'ERROR'}, str(e))
            import traceback; traceback.print_exc()
            return {'CANCELLED'}

        edit_idx = props.edit_segment_index

        seg_data = {
            'positions': positions.copy(),
            'rot_mats': rot_mats.copy(),
            'focals': focals.copy() if focals is not None else None,
            'move_type': move,
            'params': params.copy(),
            'n_frames': n_frames,
            'frame_start': f_start,
            'frame_end': f_end,
            'name': '',
            'color_index': 0,
            'target_object_name': (props.target_object.name
                                   if props.target_object else None),
            'target_origin': (list(props.target_object.location)
                              if props.target_object else None),
            'anchor_position': (get_object_anchor(props.target_object).tolist()
                                if props.target_object else None),
            'anchor_mode': (props.target_object.get("camtraj_anchor_mode", 'ORIGIN')
                            if props.target_object else None),
        }

        if edit_idx >= 0 and edit_idx < len(_committed_segments):
            seg_data['name'] = f"{edit_idx}: {move}"
            seg_data['color_index'] = edit_idx
            _committed_segments[edit_idx] = seg_data

            if edit_idx + 1 < len(_committed_segments):
                _cascade_segments_from(edit_idx + 1)

            props.edit_segment_index = -1

            _rebuild_all_keyframes(context)
            _rebuild_all_committed_curves()
            _remove_preview_objects()

            if _committed_segments:
                last = _committed_segments[-1]
                props.frame_start = last['frame_end'] + 1
                props.frame_end = last['frame_end'] + n_frames

            tag_redraw(context)
            n_cascaded = len(_committed_segments) - edit_idx - 1
            self.report({'INFO'},
                        f"Updated segment {edit_idx} '{move}'"
                        + (f" + cascaded {n_cascaded}" if n_cascaded else ""))
            return {'FINISHED'}

        seg_data['name'] = f"{props.segment_count}: {move}"
        seg_data['color_index'] = props.segment_count
        _committed_segments.append(seg_data)

        _create_committed_curve(props.segment_count, positions, rot_mats,
                                 move, props.segment_count)
        _remove_preview_objects()

        _rebuild_all_keyframes(context)

        props.segment_count += 1
        props.frame_start = f_end + 1
        props.frame_end = f_end + n_frames
        context.scene.frame_end = max(context.scene.frame_end, f_end)

        tag_redraw(context)
        self.report({'INFO'}, f"Committed '{move}': frames {f_start}–{f_end}")
        return {'FINISHED'}


class CAMTRAJ_OT_edit_segment(Operator):
    """Select a committed segment for editing."""
    bl_idname = "camtraj.edit_segment"
    bl_label = "Edit Segment"
    bl_options = {'REGISTER', 'UNDO'}

    segment_index: IntProperty(name="Segment Index", default=0)

    def execute(self, context):
        props = context.scene.camtraj
        idx = self.segment_index

        if idx < 0 or idx >= len(_committed_segments):
            self.report({'ERROR'}, f"Invalid segment index: {idx}")
            return {'CANCELLED'}

        seg = _committed_segments[idx]
        props.edit_segment_index = idx

        props.move_type = seg['move_type']
        props.frame_start = seg['frame_start']
        props.frame_end = seg['frame_end']

        p = seg['params']
        props.radius = p.get('radius', 2.0)
        props.height = p.get('height', 0.5)
        props.start_angle = p.get('start_angle', 0.0)
        props.end_angle = p.get('end_angle', p.get('start_angle', 0.0) + 90.0)
        props.pitch_offset = p.get('pitch_offset', 0.0)
        props.approach_angle = p.get('approach_angle', 0.0)
        props.start_radius = p.get('start_radius', 3.0)
        props.end_radius = p.get('end_radius', 1.0)
        props.start_elevation = p.get('start_elevation', 10.0)
        props.end_elevation = p.get('end_elevation', 70.0)
        props.yaw_range = p.get('yaw_range', 45.0)
        props.pitch_range = p.get('pitch_range', 45.0)
        props.yaw_fixed = p.get('yaw', 0.0)
        props.pitch_fixed = p.get('pitch', 0.0)
        props.peak_multiplier = p.get('peak_multiplier', 2.0)
        props.base_focal = p.get('base_focal', 50.0)
        props.arc_angle = p.get('arc_angle', 30.0)

        _update_preview_curve(context)
        tag_redraw(context)
        self.report({'INFO'}, f"Editing segment {idx}: {seg['move_type']}")
        return {'FINISHED'}


class CAMTRAJ_OT_cancel_edit(Operator):
    """Cancel editing and return to append mode."""
    bl_idname = "camtraj.cancel_edit"
    bl_label = "Cancel Edit"

    def execute(self, context):
        props = context.scene.camtraj
        props.edit_segment_index = -1
        if _committed_segments:
            last = _committed_segments[-1]
            props.frame_start = last['frame_end'] + 1
            props.frame_end = last['frame_end'] + 60
        _update_preview_curve(context)
        tag_redraw(context)
        self.report({'INFO'}, "Edit cancelled")
        return {'FINISHED'}


class CAMTRAJ_OT_undo_last(Operator):
    bl_idname = "camtraj.undo_last"
    bl_label = "Undo Last"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        if _committed_segments:
            removed = _committed_segments.pop()
            props = context.scene.camtraj
            seg_idx = props.segment_count - 1
            _remove_objects_with_prefix(f"TrajPath_{seg_idx}")
            props.segment_count = max(0, props.segment_count - 1)
            if _committed_segments:
                last = _committed_segments[-1]
                props.frame_start = last['frame_end'] + 1
                props.frame_end = last['frame_end'] + 60
            else:
                props.frame_start = 1
                props.frame_end = 60
            _rebuild_all_keyframes(context)
            tag_redraw(context)
            self.report({'INFO'}, f"Removed: {removed['name']}")
        return {'FINISHED'}


class CAMTRAJ_OT_scrub(Operator):
    bl_idname = "camtraj.scrub"
    bl_label = "Scrub Preview"

    _timer = None
    _playing = False

    def modal(self, context, event):
        if event.type == 'TIMER' and CAMTRAJ_OT_scrub._playing:
            props = context.scene.camtraj
            f = context.scene.frame_current + 1
            if f > props.frame_end:
                f = props.frame_start
            context.scene.frame_current = f
            tag_redraw(context)
        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            self.cancel(context)
            return {'CANCELLED'}
        return {'PASS_THROUGH'}

    def execute(self, context):
        if CAMTRAJ_OT_scrub._playing:
            CAMTRAJ_OT_scrub._playing = False
            return {'FINISHED'}
        CAMTRAJ_OT_scrub._playing = True
        context.scene.frame_current = context.scene.camtraj.frame_start
        wm = context.window_manager
        CAMTRAJ_OT_scrub._timer = wm.event_timer_add(
            1.0 / context.scene.render.fps, window=context.window)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def cancel(self, context):
        CAMTRAJ_OT_scrub._playing = False
        if CAMTRAJ_OT_scrub._timer:
            context.window_manager.event_timer_remove(CAMTRAJ_OT_scrub._timer)
            CAMTRAJ_OT_scrub._timer = None


class CAMTRAJ_OT_set_target(Operator):
    bl_idname = "camtraj.set_target"
    bl_label = "Pick Active"

    def execute(self, context):
        if context.active_object:
            context.scene.camtraj.target_object = context.active_object
            tag_redraw(context)
            self.report({'INFO'}, f"Target: {context.active_object.name}")
        return {'FINISHED'}


class CAMTRAJ_OT_look_through(Operator):
    bl_idname = "camtraj.look_through"
    bl_label = "Look Through"

    def execute(self, context):
        cam = bpy.data.objects.get("TrajCamera")
        if cam:
            context.scene.camera = cam
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    area.spaces[0].region_3d.view_perspective = 'CAMERA'
                    break
        return {'FINISHED'}


class CAMTRAJ_OT_apply_initial_viewpoint(Operator):
    """Load the target object's stored initial viewpoint into trajectory params."""
    bl_idname = "camtraj.apply_initial_viewpoint"
    bl_label = "Apply Initial Viewpoint"
    bl_description = ("Load this object's preferred initial camera placement "
                      "into the trajectory parameters")
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.camtraj
        target = props.target_object
        if target is None:
            self.report({'WARNING'}, "No target object selected")
            return {'CANCELLED'}

        iv = get_initial_viewpoint(target)

        props.approach_angle = iv['approach_angle']
        props.radius = iv['distance']
        props.start_angle = iv['approach_angle']
        props.height = iv['height_offset']
        props.start_elevation = iv['elevation']
        props.start_radius = iv['distance']

        _update_preview_curve(context)
        tag_redraw(context)
        self.report({'INFO'},
                    f"Applied initial viewpoint from '{target.name}': "
                    f"dist={iv['distance']:.1f}, "
                    f"approach={iv['approach_angle']:.0f}°, "
                    f"elev={iv['elevation']:.0f}°, "
                    f"h_off={iv['height_offset']:.1f}")
        return {'FINISHED'}


class CAMTRAJ_OT_set_anchor_from_cursor(Operator):
    """Set the anchor absolute position from the 3D cursor."""
    bl_idname = "camtraj.set_anchor_from_cursor"
    bl_label = "Anchor from 3D Cursor"
    bl_description = "Set the anchor (look-at) position to the 3D cursor location"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.camtraj
        target = props.target_object
        if target is None:
            self.report({'WARNING'}, "No target object selected")
            return {'CANCELLED'}

        cursor_loc = context.scene.cursor.location
        props.anchor_mode = 'ABSOLUTE'
        props.anchor_absolute = (cursor_loc.x, cursor_loc.y, cursor_loc.z)

        self.report({'INFO'},
                    f"Anchor set to cursor: "
                    f"({cursor_loc.x:.2f}, {cursor_loc.y:.2f}, {cursor_loc.z:.2f})")
        return {'FINISHED'}


class CAMTRAJ_OT_reset(Operator):
    bl_idname = "camtraj.reset"
    bl_label = "Reset All"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        global _committed_segments
        cam_obj = bpy.data.objects.get("TrajCamera")
        if cam_obj:
            if cam_obj.animation_data:
                cam_obj.animation_data_clear()
            if cam_obj.data and cam_obj.data.animation_data:
                cam_obj.data.animation_data_clear()
        _committed_segments = []
        _remove_all_viz_objects()
        props = context.scene.camtraj
        props.segment_count = 0
        props.frame_start = 1
        props.frame_end = 60
        props.edit_segment_index = -1
        tag_redraw(context)
        self.report({'INFO'}, "Reset all")
        return {'FINISHED'}


class CAMTRAJ_OT_export(Operator):
    bl_idname = "camtraj.export"
    bl_label = "Export JSON"

    filepath: StringProperty(subtype='FILE_PATH', default="//trajectory.json")

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        import json as _json
        from bpy_extras import anim_utils

        props = context.scene.camtraj

        cam_obj = bpy.data.objects.get("TrajCamera")
        if not cam_obj:
            self.report({'ERROR'}, "No TrajCamera")
            return {'CANCELLED'}

        obj_fcurves = []
        if cam_obj.animation_data and cam_obj.animation_data.action:
            action = cam_obj.animation_data.action
            action_slot = cam_obj.animation_data.action_slot
            if action_slot:
                cb = anim_utils.action_get_channelbag_for_slot(action, action_slot)
                if cb:
                    obj_fcurves = list(cb.fcurves)

        data_fcurves = []
        cam_data = cam_obj.data
        if cam_data and cam_data.animation_data and cam_data.animation_data.action:
            data_action = cam_data.animation_data.action
            data_slot = cam_data.animation_data.action_slot
            if data_slot:
                cb = anim_utils.action_get_channelbag_for_slot(data_action, data_slot)
                if cb:
                    data_fcurves = list(cb.fcurves)

        frames = []
        for f in range(context.scene.frame_start,
                       context.scene.frame_end + 1):
            loc = [0.0] * 3
            quat = [1.0, 0.0, 0.0, 0.0]
            focal = props.base_focal

            for fc in obj_fcurves:
                if fc.data_path == "location":
                    loc[fc.array_index] = fc.evaluate(f)
                elif fc.data_path == "rotation_quaternion":
                    quat[fc.array_index] = fc.evaluate(f)

            for fc in data_fcurves:
                if fc.data_path == "lens":
                    focal = fc.evaluate(f)

            q = Quaternion((quat[0], quat[1], quat[2], quat[3]))
            mat = q.to_matrix().to_4x4()
            mat.translation = Vector(loc)
            c2w = [[mat[r][c] for c in range(4)] for r in range(4)]

            frames.append({
                'frame': f,
                'position': loc,
                'quaternion_wxyz': quat,
                'focal_length_mm': focal,
                'c2w': c2w,
            })

        segments_info = []
        for seg in _committed_segments:
            seg_info = {
                'name': seg['name'],
                'move_type': seg['move_type'],
                'frame_start': seg['frame_start'],
                'frame_end': seg['frame_end'],
                'n_frames': seg['n_frames'],
                'target_object': seg.get('target_object_name'),
                'target_origin': seg.get('target_origin'),
                'anchor_position': seg.get('anchor_position'),
                'anchor_mode': seg.get('anchor_mode'),
            }
            segments_info.append(seg_info)

        data = {
            'camera_name': 'TrajCamera',
            'coordinate_system': 'blender_zup',
            'n_frames': len(frames),
            'fps': context.scene.render.fps,
            'focal_length_mm': props.base_focal,
            'sensor_width_mm': props.sensor_width,
            'hfov_deg': focal_to_hfov(props.base_focal, props.sensor_width),
            'segments': segments_info,
            'frames': frames,
        }

        path = bpy.path.abspath(self.filepath)
        with open(path, 'w') as fp:
            _json.dump(data, fp, indent=2)
        self.report({'INFO'}, f"Exported {len(frames)} frames → {path}")
        return {'FINISHED'}


# ============================================================================
# Panel
# ============================================================================

class CAMTRAJ_PT_main(Panel):
    bl_label = "Camera Trajectory"
    bl_idname = "CAMTRAJ_PT_main"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'CamTraj'

    def draw(self, context):
        layout = self.layout
        props = context.scene.camtraj
        editing = props.edit_segment_index >= 0

        if editing:
            box = layout.box()
            box.alert = True
            row = box.row()
            row.label(text=f"EDITING Segment {props.edit_segment_index}",
                      icon='GREASEPENCIL')
            row.operator("camtraj.cancel_edit", text="Cancel", icon='X')

        # Target
        box = layout.box()
        box.label(text="Target Object", icon='OBJECT_DATA')
        row = box.row(align=True)
        row.prop(props, "target_object", text="")
        row.operator("camtraj.set_target", text="", icon='EYEDROPPER')
        if props.target_object:
            loc = props.target_object.location
            box.label(text=f"  Origin: ({loc.x:.2f}, {loc.y:.2f}, {loc.z:.2f})")

        # Anchor
        if props.target_object:
            sub = box.box()
            sub.label(text="Anchor (Look-At Point)", icon='CURSOR')
            sub.prop(props, "anchor_mode", text="Mode")
            if props.anchor_mode == 'OFFSET':
                sub.prop(props, "anchor_offset", text="Offset")
                anchor = get_object_anchor(props.target_object)
                sub.label(text=f"  Effective: ({anchor[0]:.2f}, "
                               f"{anchor[1]:.2f}, {anchor[2]:.2f})")
            elif props.anchor_mode == 'ABSOLUTE':
                sub.prop(props, "anchor_absolute", text="Position")
                sub.operator("camtraj.set_anchor_from_cursor",
                             icon='PIVOT_CURSOR')
            else:
                anchor = get_object_anchor(props.target_object)
                sub.label(text=f"  = BBox Center ({anchor[0]:.2f}, "
                               f"{anchor[1]:.2f}, {anchor[2]:.2f})")

        # Initial Viewpoint
        if props.target_object:
            sub = box.box()
            sub.label(text="Initial Viewpoint", icon='CAMERA_DATA')
            sub.prop(props, "auto_apply_iv",
                     text="Auto-Apply to Trajectory",
                     icon='LINKED' if props.auto_apply_iv else 'UNLINKED')
            sub.prop(props, "iv_approach_angle")
            sub.prop(props, "iv_distance")
            sub.prop(props, "iv_elevation")
            sub.prop(props, "iv_height_offset")
            if not props.auto_apply_iv:
                sub.operator("camtraj.apply_initial_viewpoint",
                             text="Apply to Trajectory", icon='IMPORT')

        # Movement
        box = layout.box()
        box.label(text="Movement", icon='ANIM')
        box.prop(props, "move_type", text="")

        # Camera params
        box = layout.box()
        box.label(text="Camera", icon='CAMERA_DATA')
        box.prop(props, "base_focal")
        box.prop(props, "sensor_width")
        hfov = focal_to_hfov(props.base_focal, props.sensor_width)
        box.label(text=f"  H-FOV: {hfov:.1f}")

        # Movement-specific params
        box = layout.box()
        box.label(text="Parameters", icon='PREFERENCES')
        m = props.move_type

        has_prev = (props.continue_from_previous
                    and len(_committed_segments) > 0)

        if m == 'static':
            if not has_prev:
                box.prop(props, "radius", text="Distance")
                box.prop(props, "height")
                box.prop(props, "approach_angle")
                box.prop(props, "start_elevation", text="Elevation")
            else:
                box.label(text="  (Position from prev endpoint)")
            box.label(text="  Camera holds still, looking at anchor")
        elif m in ('orbit_full', 'orbit_half', 'orbit_quarter'):
            box.prop(props, "radius")
            box.prop(props, "height")
            if not has_prev:
                box.prop(props, "start_angle")
            else:
                box.label(text="  (Start angle auto from prev endpoint)")
            box.prop(props, "end_angle")
            sweep = props.end_angle - props.start_angle
            box.label(text=f"  Sweep: {sweep:.1f}°")
            box.prop(props, "start_elevation", text="Elevation")
            box.prop(props, "pitch_offset")
        elif m in ('pan_left', 'pan_right'):
            box.prop(props, "yaw_range")
            box.prop(props, "pitch_fixed", text="Pitch")
        elif m in ('move_in', 'move_out'):
            box.prop(props, "start_radius")
            box.prop(props, "end_radius")
            box.prop(props, "height")
            if not has_prev:
                box.prop(props, "approach_angle")
            else:
                box.label(text="  (Approach auto from prev endpoint)")
        elif m == 'crane':
            box.prop(props, "radius")
            box.prop(props, "start_elevation")
            box.prop(props, "end_elevation")
            if not has_prev:
                box.prop(props, "approach_angle")
            else:
                box.label(text="  (Approach auto from prev endpoint)")
        elif m in ('tilt_up', 'tilt_down'):
            box.prop(props, "pitch_range")
            box.prop(props, "yaw_fixed", text="Yaw")
        elif m in ('zoom_in_out', 'zoom_out_in'):
            box.prop(props, "peak_multiplier")
        elif m == 'arc':
            box.prop(props, "arc_angle")
            box.prop(props, "height")

        # Frames
        box = layout.box()
        box.label(text="Frames", icon='TIME')
        row = box.row(align=True)
        row.prop(props, "frame_start")
        row.prop(props, "frame_end")
        box.prop(props, "keyframe_step")
        box.prop(props, "exact_keyframes")
        if not editing:
            box.prop(props, "continue_from_previous")

        # Display
        box = layout.box()
        box.label(text="Display", icon='HIDE_OFF')
        row = box.row(align=True)
        row.prop(props, "show_preview", toggle=True, text="Live Preview")
        row.prop(props, "show_committed", toggle=True, text="History")

        # Actions
        box = layout.box()
        box.label(text="Actions", icon='PLAY')
        row = box.row(align=True)
        row.scale_y = 1.6
        if editing:
            row.operator("camtraj.add_segment", text="Apply Edit",
                         icon='CHECKMARK')
        else:
            row.operator("camtraj.add_segment", icon='KEYFRAME')
        row = box.row(align=True)
        if not editing:
            row.operator("camtraj.undo_last", text="Undo Last",
                         icon='LOOP_BACK')
        row.operator("camtraj.reset", text="Reset All", icon='TRASH')
        row = box.row(align=True)
        playing = CAMTRAJ_OT_scrub._playing
        row.operator("camtraj.scrub",
                     text="Stop" if playing else "Scrub",
                     icon='PAUSE' if playing else 'PLAY')
        row.operator("camtraj.look_through", text="", icon='CAMERA_DATA')

        layout.separator()
        layout.operator("camtraj.export", text="Export JSON", icon='EXPORT')

        # Segments list
        if _committed_segments:
            layout.separator()
            box = layout.box()
            box.label(text=f"Segments: {len(_committed_segments)}",
                      icon='INFO')
            for i, seg in enumerate(_committed_segments):
                row = box.row(align=True)
                ci = seg['color_index']
                if editing and i == props.edit_segment_index:
                    row.alert = True

                row.label(
                    text=f"  [{ci}] "
                         f"{seg['name']}  "
                         f"[{seg['frame_start']}-{seg['frame_end']}]")

                op = row.operator("camtraj.edit_segment", text="",
                                  icon='GREASEPENCIL')
                op.segment_index = i


# ============================================================================
# Registration
# ============================================================================

classes = (
    CamTrajProperties,
    CAMTRAJ_OT_add_segment,
    CAMTRAJ_OT_edit_segment,
    CAMTRAJ_OT_cancel_edit,
    CAMTRAJ_OT_undo_last,
    CAMTRAJ_OT_scrub,
    CAMTRAJ_OT_set_target,
    CAMTRAJ_OT_look_through,
    CAMTRAJ_OT_apply_initial_viewpoint,
    CAMTRAJ_OT_set_anchor_from_cursor,
    CAMTRAJ_OT_reset,
    CAMTRAJ_OT_export,
    CAMTRAJ_PT_main,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.camtraj = PointerProperty(type=CamTrajProperties)
    register_draw_handler()


def unregister():
    unregister_draw_handler()
    del bpy.types.Scene.camtraj
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    try:
        unregister()
    except:
        pass
    try:
        _remove_all_viz_objects()
    except:
        pass
    _committed_segments.clear()
    register()