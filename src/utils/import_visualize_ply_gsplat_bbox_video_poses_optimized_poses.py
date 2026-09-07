import numpy as np
import torch
import viser
import viser.transforms as tf
from plyfile import PlyData
from gsplat import rasterization
import json
import trimesh
import time

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
from src.anchor_selector.anchor_selector import CameraAnchor
from src.trajectory_optimizer.trajectory_optimizer import CameraTrajectory, CameraPose, CameraIntrinsics


# ─── Global rotation (identity — no transform needed) ─────────────────────────
ROT_X_90 = np.array([
    [1,  0,  0],
    [0,  1,  0],
    [0,  0,  1],
], dtype=np.float64)

ROT_X_90_T = torch.tensor(ROT_X_90, dtype=torch.float32)


def rotate_positions(positions: np.ndarray) -> np.ndarray:
    """Rotate (N,3) positions by the global rotation."""
    return (ROT_X_90 @ positions.T).T


def rotate_rotation_matrix(R: np.ndarray) -> np.ndarray:
    """Rotate a (3,3) rotation matrix: R' = ROT @ R."""
    return ROT_X_90 @ R


def load_3dgs_ply(path, device="cuda"):
    """Load 3D Gaussian Splatting PLY file."""
    plydata = PlyData.read(path)
    vertex = plydata['vertex']
    
    xyz = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=1)
    
    SH_C0 = 0.28209479177387814
    colors = np.stack([vertex['f_dc_0'], vertex['f_dc_1'], vertex['f_dc_2']], axis=1)
    colors = 0.5 + SH_C0 * colors
    colors = np.clip(colors, 0, 1)
    
    opacities = 1 / (1 + np.exp(-vertex['opacity']))
    scales = np.exp(np.stack([vertex['scale_0'], vertex['scale_1'], vertex['scale_2']], axis=1))
    
    rots = np.stack([vertex['rot_0'], vertex['rot_1'], vertex['rot_2'], vertex['rot_3']], axis=1)
    rots = rots / np.linalg.norm(rots, axis=1, keepdims=True)
    
    # ── Apply global rotation ──
    xyz = rotate_positions(xyz)
    
    # Rotate quaternions: convert to matrix, rotate, convert back
    from scipy.spatial.transform import Rotation as SciRotation
    rot_matrices = SciRotation.from_quat(rots[:, [1, 2, 3, 0]]).as_matrix()  # wxyz -> xyzw for scipy
    rot_matrices = np.array([ROT_X_90 @ R for R in rot_matrices])
    rots_scipy = SciRotation.from_matrix(rot_matrices)
    rots_xyzw = rots_scipy.as_quat()  # xyzw
    rots = rots_xyzw[:, [3, 0, 1, 2]]  # back to wxyz
    rots = rots / np.linalg.norm(rots, axis=1, keepdims=True)
    
    return {
        'means': torch.tensor(xyz, dtype=torch.float32, device=device),
        'colors': torch.tensor(colors, dtype=torch.float32, device=device),
        'opacities': torch.tensor(opacities, dtype=torch.float32, device=device),
        'scales': torch.tensor(scales, dtype=torch.float32, device=device),
        'quats': torch.tensor(rots, dtype=torch.float32, device=device),
    }


def load_bounding_boxes(json_path):
    """
    Load bounding boxes from JSON file.
    
    Supports both:
    - New scene graph format: dict with "objects" key, OBB arrays
    - Legacy labels.json format: list of dicts with "bounding_box" key
    
    Returns:
        (objects_list, format_type)
        - objects_list: normalized list of dicts with 'id', 'label', 'corners'
        - format_type: 'scene_graph' or 'legacy'
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    if isinstance(data, dict) and "objects" in data:
        # New scene graph format
        from scipy.spatial.transform import Rotation
        
        objects_dict = data["objects"]
        rooms = data.get("rooms", {})
        
        # Build reverse mapping: object_id → room_name
        obj_to_room = {}
        for room_name, obj_ids in rooms.items():
            for obj_id in obj_ids:
                obj_to_room[obj_id] = room_name
        
        objects_list = []
        for obj_id, obj_data in objects_dict.items():
            obb = obj_data["obb"]
            center = np.array(obb[0:3], dtype=np.float64)
            size = np.array(obb[3:6], dtype=np.float64)
            qxyzw = np.array(obb[6:10], dtype=np.float64)
            
            # Compute 8 corners from OBB
            R = Rotation.from_quat(qxyzw).as_matrix()
            half = size / 2.0
            signs = np.array([
                [-1, -1, -1], [-1, -1,  1], [-1,  1, -1], [-1,  1,  1],
                [ 1, -1, -1], [ 1, -1,  1], [ 1,  1, -1], [ 1,  1,  1],
            ], dtype=np.float64)
            local_corners = signs * half
            corners = (R @ local_corners.T).T + center
            
            # ── Apply global rotation ──
            center = rotate_positions(center.reshape(1, 3)).squeeze()
            corners = rotate_positions(corners)
            
            label = obj_id.rsplit("_", 1)[0]
            
            objects_list.append({
                'id': obj_id,
                'label': label,
                'corners': corners,
                'center': center,
                'size': size,
                'against_wall': obj_data.get('against_wall', False),
                'attached_to_ceiling': obj_data.get('attached_to_ceiling', False),
                'room': obj_to_room.get(obj_id),
            })
        
        return objects_list, 'scene_graph'
    
    else:
        # Legacy labels.json format — rotate corners
        for item in data:
            if 'bounding_box' in item and isinstance(item['bounding_box'], list):
                corners = np.array([[p['x'], p['y'], p['z']] for p in item['bounding_box']])
                corners = rotate_positions(corners)
                item['bounding_box'] = [{'x': c[0], 'y': c[1], 'z': c[2]} for c in corners]
        return data, 'legacy'


def load_trajectory_json(json_path):
    """Load trajectory from GenDoP-format JSON."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    intrinsics = CameraIntrinsics(
        width=data.get('w', 512.0),
        height=data.get('h', 512.0),
        fx=data.get('fl_x', 256.0),
        fy=data.get('fl_y', 256.0),
        cx=data.get('cx', 256.0),
        cy=data.get('cy', 256.0),
    )
    
    poses = []
    positions = []
    rotations = []
    
    for i, frame in enumerate(data.get('frames', [])):
        transform = np.array(frame['transform_matrix'])
        rotation = transform[:3, :3]
        position = transform[:3, 3]
        
        # ── Apply global rotation ──
        position = (ROT_X_90 @ position)
        rotation = ROT_X_90 @ rotation
        
        poses.append(CameraPose(
            position=position.copy(),
            rotation=rotation.copy(),
            timestamp=frame.get('timestamp', i / 30.0),
        ))
        positions.append(position)
        rotations.append(rotation)
    
    return CameraTrajectory(
        poses=poses,
        timestamps=np.array([p.timestamp for p in poses]),
        positions=np.array(positions),
        rotations=np.array(rotations),
        intrinsics=intrinsics,
    )


def get_label_color(label):
    """Assign consistent bright, saturated colors to different label types."""
    color_map = {
        'door': (255, 0, 0),           # red
        'door_frame': (255, 50, 50),    # bright red
        'window': (0, 100, 255),        # bright blue
        'wardrobe': (0, 255, 0),        # green
        'curtain': (255, 165, 0),       # orange
        'cabinet': (0, 200, 0),         # green (darker)
        'table': (255, 255, 0),         # yellow
        'chair': (0, 255, 255),         # cyan
        'sofa': (180, 0, 255),          # purple
        'tv': (255, 0, 200),            # magenta/pink
        'refrigerator': (0, 255, 128),  # spring green
        'kitchen_counter': (200, 200, 0), # dark yellow
        'sink': (0, 180, 255),          # sky blue
        'plant': (0, 220, 0),           # bright green
        'lamp': (255, 220, 0),          # golden yellow
        'picture': (255, 0, 255),       # magenta
        'painting': (255, 0, 255),      # magenta
        'book': (255, 140, 0),          # dark orange
        'bottle': (80, 80, 255),        # medium blue
        'bag': (255, 0, 128),           # rose
        'box': (200, 200, 0),           # olive-yellow
        'speaker': (0, 200, 200),       # teal
        'trash_can': (180, 180, 180),   # light gray
        'towel': (240, 240, 240),       # near-white
        'footwear': (255, 160, 50),     # bright amber
        'obstacle': (200, 200, 200),    # light gray
        'wall': (150, 150, 180),        # blue-gray
    }
    return color_map.get(label, (255, 255, 255))


# ─── OBB visualization settings ──────────────────────────────────────────────
OBB_COLOR_OVERRIDE = None  # None → use per-label colors from get_label_color
OBB_LINE_WIDTH = 4.5       # default was 2.0


def add_bounding_box_from_corners(server, name, corners, color):
    """
    Add a wireframe bounding box from 8 corner points.
    
    Works with both:
    - New format corners: (8, 3) numpy array from obb_to_corners()
    - Legacy format corners: list of 8 dicts with 'x', 'y', 'z'
    
    Returns (handle, center).
    """
    if isinstance(corners, list) and len(corners) > 0 and isinstance(corners[0], dict):
        corners = np.array([[p['x'], p['y'], p['z']] for p in corners])
    else:
        corners = np.array(corners, dtype=np.float64)
    
    edges = [
        (0, 2), (2, 6), (6, 4), (4, 0),
        (1, 3), (3, 7), (7, 5), (5, 1),
        (0, 1), (2, 3), (4, 5), (6, 7),
    ]
    
    points = np.array([[corners[start], corners[end]] for start, end in edges])

    # Use override color if set, otherwise use the passed-in color
    display_color = OBB_COLOR_OVERRIDE if OBB_COLOR_OVERRIDE is not None else color
    color_array = np.array(display_color, dtype=np.uint8)
    
    handle = server.scene.add_line_segments(
        name,
        points=points,
        colors=color_array,
        line_width=OBB_LINE_WIDTH,
    )
    
    center = corners.mean(axis=0)
    return handle, center


# Keep old name as alias
def add_bounding_box(server, name, bbox_points, color):
    """Legacy wrapper — bbox_points is list of 8 dicts with x,y,z."""
    return add_bounding_box_from_corners(server, name, bbox_points, color)


def render_gaussians(gaussians, viewmat, K, width, height):
    """Render Gaussians using gsplat."""
    renders, alphas, meta = rasterization(
        means=gaussians['means'],
        quats=gaussians['quats'],
        scales=gaussians['scales'],
        opacities=gaussians['opacities'],
        colors=gaussians['colors'],
        viewmats=viewmat[None],
        Ks=K[None],
        width=width,
        height=height,
        packed=False,
        render_mode="RGB",
    )
    return renders[0].clamp(0, 1)


def load_mesh_auto(mesh_path: str) -> trimesh.Trimesh:
    """
    Load mesh from PLY/OBJ/STL (via trimesh) or USD.
    Auto-detects format by file extension.
    """
    mesh_path = Path(mesh_path)
    ext = mesh_path.suffix.lower()
    
    if ext in ('.usd', '.usda', '.usdc'):
        mesh = _load_usd_mesh(mesh_path)
    else:
        mesh = trimesh.load(str(mesh_path), force='mesh')
        print(f"Loaded mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
    
    # ── Apply global rotation ──
    mesh.vertices = rotate_positions(np.array(mesh.vertices, dtype=np.float64))
    
    return mesh


def _load_usd_mesh(mesh_path) -> trimesh.Trimesh:
    """Load USD mesh with correct transform handling."""
    from pxr import Usd, UsdGeom
    
    mesh_path = Path(mesh_path)
    stage = Usd.Stage.Open(str(mesh_path))
    
    meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
    print(f"USD meters per unit: {meters_per_unit}")
    
    time_code = Usd.TimeCode.Default()
    xform_cache = UsdGeom.XformCache(time_code)
    
    all_meshes = []
    
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        
        mesh_prim = UsdGeom.Mesh(prim)
        points = mesh_prim.GetPointsAttr().Get(time_code)
        if points is None or len(points) == 0:
            continue
        
        points = np.array(points, dtype=np.float64)
        
        world_xform = xform_cache.GetLocalToWorldTransform(prim)
        world_matrix = np.array(world_xform).reshape(4, 4).T
        
        points_h = np.hstack([points, np.ones((len(points), 1))])
        points_world = (world_matrix @ points_h.T).T[:, :3]
        
        face_counts = mesh_prim.GetFaceVertexCountsAttr().Get(time_code)
        face_indices = mesh_prim.GetFaceVertexIndicesAttr().Get(time_code)
        
        if face_counts is None or face_indices is None:
            continue
        
        triangles = []
        idx = 0
        for count in face_counts:
            if count == 3:
                triangles.append([face_indices[idx], face_indices[idx+1], face_indices[idx+2]])
            elif count == 4:
                triangles.append([face_indices[idx], face_indices[idx+1], face_indices[idx+2]])
                triangles.append([face_indices[idx], face_indices[idx+2], face_indices[idx+3]])
            elif count > 4:
                for i in range(1, count - 1):
                    triangles.append([face_indices[idx], face_indices[idx+i], face_indices[idx+i+1]])
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
    print(f"Final mesh: {len(combined.vertices)} verts, {len(combined.faces)} faces")
    
    return combined


def add_mesh_to_viser(server, name, mesh, color=(200, 200, 200)):
    """Add a trimesh to viser scene. Returns handle."""
    vertices = np.array(mesh.vertices, dtype=np.float32)
    faces = np.array(mesh.faces, dtype=np.uint32)
    
    handle = server.scene.add_mesh_simple(
        name,
        vertices=vertices,
        faces=faces,
        color=color,
        wireframe=False,
        opacity=0.5,
    )
    return handle


def visualize_anchor(server, anchor: CameraAnchor, name: str = "/anchor"):
    """Visualize camera anchor in viser. Returns list of handles."""
    handles = []
    
    h = server.scene.add_icosphere(
        f"{name}/position",
        radius=0.1,
        position=anchor.position,
        color=(0, 255, 0),
    )
    handles.append(h)
    
    h = server.scene.add_line_segments(
        f"{name}/lookat_line",
        points=np.array([[anchor.position, anchor.look_at]]),
        colors=np.array([0, 255, 255], dtype=np.uint8),
        line_width=2.0,
    )
    handles.append(h)
    
    h = server.scene.add_label(
        f"{name}/label",
        text=f"{anchor.object_label} (score: {anchor.score:.2f})",
        position=anchor.position + np.array([0, 0, 0.2]),
    )
    handles.append(h)
    
    return handles


def add_trajectory_to_viser(
    server, 
    trajectory: CameraTrajectory, 
    name: str = "/trajectory",
    color_scheme: str = "rainbow",
    base_color: tuple = None,
    line_width: float = 3.0,
    show_frustums: bool = True,
    collision_mask: np.ndarray = None,
):
    """
    Add trajectory visualization to viser scene.
    
    Returns:
        all_handles: list of scene handles
        frustum_indices: indices of frustum visualizations
    """
    if len(trajectory) == 0:
        return [], []
    
    positions = trajectory.positions
    rotations = trajectory.rotations
    n_poses = len(positions)
    
    path_segments = []
    path_colors = []
    
    import matplotlib.pyplot as plt
    
    if color_scheme == "rainbow":
        colors = plt.cm.rainbow(np.linspace(0, 1, n_poses))[:, :3]
    elif color_scheme == "solid" and base_color is not None:
        colors = np.tile(np.array(base_color) / 255.0, (n_poses, 1))
    elif color_scheme == "collision" and collision_mask is not None:
        colors = np.zeros((n_poses, 3))
        colors[~collision_mask] = [0, 1, 0]
        colors[collision_mask] = [1, 0, 0]
    else:
        colors = plt.cm.rainbow(np.linspace(0, 1, n_poses))[:, :3]
    
    all_handles = []
    
    for i in range(n_poses - 1):
        path_segments.append([positions[i], positions[i + 1]])
        color = ((colors[i] + colors[i + 1]) / 2 * 255).astype(np.uint8)
        path_colors.append(color)

    if len(path_segments) > 0:
        path_colors_array = np.array(path_colors, dtype=np.uint8)
        path_colors_expanded = np.stack([path_colors_array, path_colors_array], axis=1)
        
        h = server.scene.add_line_segments(
            f"{name}/path",
            points=np.array(path_segments),
            colors=path_colors_expanded,
            line_width=line_width,
        )
        all_handles.append(h)
    
    frustum_handles = []
    
    if show_frustums:
        n_frustums = min(20, n_poses)
        frustum_indices = np.linspace(0, n_poses - 1, n_frustums, dtype=int)
        
        frustum_scale = 0.15
        
        for idx in frustum_indices:
            pos = positions[idx]
            rot = rotations[idx]
            
            near = frustum_scale
            fov = 0.8
            aspect = 1.5
            
            h_val = near * np.tan(fov / 2)
            w = h_val * aspect
            
            corners_cam = np.array([
                [0, 0, 0],
                [-w, -h_val, near],
                [w, -h_val, near],
                [w, h_val, near],
                [-w, h_val, near],
            ])
            
            corners_world = (rot @ corners_cam.T).T + pos
            
            frustum_edges = [
                (0, 1), (0, 2), (0, 3), (0, 4),
                (1, 2), (2, 3), (3, 4), (4, 1),
            ]
            
            frustum_segments = np.array([[corners_world[s], corners_world[e]] for s, e in frustum_edges])
            
            color = (colors[idx] * 255).astype(np.uint8)
            
            fh = server.scene.add_line_segments(
                f"{name}/frustum_{idx}",
                points=frustum_segments,
                colors=color,
                line_width=1.5,
            )
            frustum_handles.append(fh)
    else:
        frustum_indices = []
    
    # No START/END spheres or labels — just path + frustums
    
    all_handles.extend(frustum_handles)
    
    return all_handles, frustum_indices


def add_trajectory_comparison(
    server,
    original_trajectory: CameraTrajectory,
    optimized_trajectory: CameraTrajectory,
    mesh: trimesh.Trimesh = None,
    collision_margin: float = 0.3,
):
    """Add both original and optimized trajectories for comparison."""
    handles = {
        'original': [],
        'optimized': [],
        'collision_markers': [],
        'diff': [],
    }
    
    original_collisions = None
    optimized_collisions = None
    
    if mesh is not None:
        proximity = trimesh.proximity.ProximityQuery(mesh)
        
        orig_distances = proximity.signed_distance(original_trajectory.positions)
        original_collisions = orig_distances < collision_margin
        n_orig_colliding = np.sum(original_collisions)
        print(f"Original trajectory: {n_orig_colliding}/{len(original_trajectory)} colliding frames")
        
        opt_distances = proximity.signed_distance(optimized_trajectory.positions)
        optimized_collisions = opt_distances < collision_margin
        n_opt_colliding = np.sum(optimized_collisions)
        print(f"Optimized trajectory: {n_opt_colliding}/{len(optimized_trajectory)} colliding frames")
    
    orig_handles, orig_frustum_indices = add_trajectory_to_viser(
        server, original_trajectory, name="/trajectory_original",
        color_scheme="solid", base_color=(255, 100, 50),
        line_width=2.5, show_frustums=True, collision_mask=original_collisions,
    )
    handles['original'] = orig_handles
    
    opt_handles, opt_frustum_indices = add_trajectory_to_viser(
        server, optimized_trajectory, name="/trajectory_optimized",
        color_scheme="solid", base_color=(50, 150, 255),
        line_width=3.5, show_frustums=True, collision_mask=optimized_collisions,
    )
    handles['optimized'] = opt_handles
    
    if original_collisions is not None:
        collision_indices = np.where(original_collisions)[0]
        for idx in collision_indices[::5]:
            pos = original_trajectory.positions[idx]
            marker_name = f"/collision_markers/original_{idx}"
            h = server.scene.add_icosphere(
                marker_name, radius=0.05, position=pos, color=(255, 0, 0),
            )
            handles['collision_markers'].append(h)
    
    if len(original_trajectory) == len(optimized_trajectory):
        diff_segments = []
        diff_colors = []
        
        for i in range(0, len(original_trajectory), 10):
            orig_pos = original_trajectory.positions[i]
            opt_pos = optimized_trajectory.positions[i]
            diff = np.linalg.norm(opt_pos - orig_pos)
            
            if diff > 0.01:
                diff_segments.append([orig_pos, opt_pos])
                if diff < 0.1:
                    diff_colors.append([100, 255, 100])
                elif diff < 0.3:
                    diff_colors.append([255, 255, 100])
                else:
                    diff_colors.append([255, 100, 100])
        
        if len(diff_segments) > 0:
            diff_colors_array = np.array(diff_colors, dtype=np.uint8)
            diff_colors_expanded = np.stack([diff_colors_array, diff_colors_array], axis=1)
            
            h = server.scene.add_line_segments(
                "/trajectory_diff/connections",
                points=np.array(diff_segments),
                colors=diff_colors_expanded,
                line_width=1.0,
            )
            handles['diff'] = [h]
    
    return handles, original_collisions, optimized_collisions


def add_coordinate_axes(server, origin=None, scale=1.0, name="/axes"):
    """Add XYZ coordinate axes to the viser scene."""
    if origin is None:
        origin = np.array([0.0, 0.0, 0.0])
    else:
        origin = np.array(origin)
    
    handles = []
    
    x_end = origin + np.array([scale, 0, 0])
    y_end = origin + np.array([0, scale, 0])
    z_end = origin + np.array([0, 0, scale])
    
    h = server.scene.add_line_segments(
        f"{name}/x_axis", points=np.array([[origin, x_end]]),
        colors=np.array([255, 0, 0], dtype=np.uint8), line_width=3.0,
    )
    handles.append(h)
    h = server.scene.add_label(f"{name}/x_label", text="X", position=x_end + np.array([0.1 * scale, 0, 0]))
    handles.append(h)
    
    h = server.scene.add_line_segments(
        f"{name}/y_axis", points=np.array([[origin, y_end]]),
        colors=np.array([0, 255, 0], dtype=np.uint8), line_width=3.0,
    )
    handles.append(h)
    h = server.scene.add_label(f"{name}/y_label", text="Y", position=y_end + np.array([0, 0.1 * scale, 0]))
    handles.append(h)
    
    h = server.scene.add_line_segments(
        f"{name}/z_axis", points=np.array([[origin, z_end]]),
        colors=np.array([0, 0, 255], dtype=np.uint8), line_width=3.0,
    )
    handles.append(h)
    h = server.scene.add_label(f"{name}/z_label", text="Z", position=z_end + np.array([0, 0, 0.1 * scale]))
    handles.append(h)
    
    h = server.scene.add_icosphere(f"{name}/origin", radius=0.05 * scale, position=origin, color=(255, 255, 255))
    handles.append(h)
    
    return handles


def load_anchors_json(json_path):
    """Load anchors from JSON file. Handles both string and int object IDs."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    anchors = []
    for item in data.get('anchors', []):
        obj_id = item['object_id']
        if obj_id is not None:
            try:
                obj_id = int(obj_id)
            except (ValueError, TypeError):
                pass  # Keep as string
        
        # ── Apply global rotation to anchor positions ──
        position = ROT_X_90 @ np.array(item['position'])
        look_at = ROT_X_90 @ np.array(item['look_at'])
        up = ROT_X_90 @ np.array(item['up'])
        
        anchor = CameraAnchor(
            position=position,
            look_at=look_at,
            up=up,
            score=item['score'],
            object_id=obj_id,
            object_label=item['object_label'],
        )
        if 'transform_matrix' in item:
            T = np.array(item['transform_matrix'])
            T[:3, 3] = ROT_X_90 @ T[:3, 3]
            T[:3, :3] = ROT_X_90 @ T[:3, :3]
            anchor.transform_matrix = T
        anchors.append(anchor)
    
    return anchors


def add_anchor_frustum(server, name: str, position: np.ndarray, rotation: np.ndarray, 
                        color=(0, 255, 0), frustum_scale=0.2):
    """Add a camera frustum visualization for an anchor."""
    near = frustum_scale
    fov = 0.8
    aspect = 1.5
    
    h_val = near * np.tan(fov / 2)
    w = h_val * aspect
    
    corners_cam = np.array([
        [0, 0, 0], [-w, -h_val, near], [w, -h_val, near],
        [w, h_val, near], [-w, h_val, near],
    ])
    
    corners_world = (rotation @ corners_cam.T).T + position
    
    frustum_edges = [
        (0, 1), (0, 2), (0, 3), (0, 4),
        (1, 2), (2, 3), (3, 4), (4, 1),
    ]
    
    frustum_segments = np.array([[corners_world[s], corners_world[e]] for s, e in frustum_edges])
    color_array = np.array(color, dtype=np.uint8)
    
    handles = []
    
    h = server.scene.add_line_segments(
        f"{name}/frustum", points=frustum_segments,
        colors=color_array, line_width=2.5,
    )
    handles.append(h)
    
    forward_end = position + rotation[:, 2] * frustum_scale * 2
    h = server.scene.add_line_segments(
        f"{name}/forward", points=np.array([[position, forward_end]]),
        colors=np.array([255, 255, 0], dtype=np.uint8), line_width=3.0,
    )
    handles.append(h)
    
    return handles


def add_anchors_to_viser(server, anchors: list, name: str = "/anchors"):
    """Add all anchor visualizations to viser scene."""
    from src.trajectory_optimizer.trajectory_optimizer import TrajectoryCombiner
    
    all_handles = []
    combiner = TrajectoryCombiner()
    
    colors = [
        (255, 100, 100), (100, 255, 100), (100, 100, 255),
        (255, 255, 100), (255, 100, 255), (100, 255, 255),
        (255, 180, 100), (180, 100, 255),
    ]
    
    for i, anchor in enumerate(anchors):
        color = colors[i % len(colors)]
        anchor_name = f"{name}/anchor_{i}_{anchor.object_label}"
        
        pose = combiner.anchor_to_pose(anchor)
        
        frustum_handles = add_anchor_frustum(
            server, anchor_name, pose.position, pose.rotation,
            color=color, frustum_scale=0.25,
        )
        for fh in frustum_handles:
            all_handles.append((fh, "frustum"))
        
        h = server.scene.add_icosphere(
            f"{anchor_name}/position", radius=0.08,
            position=anchor.position, color=color,
        )
        all_handles.append((h, "position"))
        
        h = server.scene.add_line_segments(
            f"{anchor_name}/lookat_line",
            points=np.array([[anchor.position, anchor.look_at]]),
            colors=np.array([200, 200, 200], dtype=np.uint8), line_width=1.5,
        )
        all_handles.append((h, "lookat"))
        
        h = server.scene.add_icosphere(
            f"{anchor_name}/lookat_target", radius=0.05,
            position=anchor.look_at, color=(255, 255, 255),
        )
        all_handles.append((h, "lookat"))
        
        h = server.scene.add_label(
            f"{anchor_name}/label",
            text=f"{anchor.object_label} (id:{anchor.object_id}, score:{anchor.score:.2f})",
            position=anchor.position + np.array([0, 0, 0.2]),
        )
        all_handles.append((h, "label"))
    
    return all_handles


# ─── Label font size via HTML styling ─────────────────────────────────────────
# Viser's add_label doesn't support font_size directly, so we inject HTML markup.
# If your viser version doesn't render HTML in labels, this gracefully degrades
# to plain text (the tags just show as literal text — update viser if so).



def visualize_assets(ply_path, bbox_json_path, mesh_path=None, trajectory_path=None, 
         optimized_trajectory_path=None, anchor_path=None, port=8080):
    """
    Main visualization function.
    
    Args:
        ply_path: Path to 3DGS PLY file
        bbox_json_path: Path to scene graph JSON or legacy labels.json
        mesh_path: Path to collision mesh (PLY, OBJ, USD, etc.) — auto-detected
        trajectory_path: Path to trajectory JSON
        optimized_trajectory_path: Path to optimized trajectory JSON
        anchor_path: Path to anchors JSON
        port: Viser server port
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load Gaussians
    print(f"Loading {ply_path}...")
    gaussians = load_3dgs_ply(ply_path, device)
    print(f"Loaded {gaussians['means'].shape[0]} Gaussians")
    
    # Load bounding boxes (auto-detects format)
    print(f"Loading {bbox_json_path}...")
    objects, bbox_format = load_bounding_boxes(bbox_json_path)
    print(f"Loaded {len(objects)} objects (format: {bbox_format})")
    
    # Load mesh if provided (auto-detects PLY/USD)
    mesh = None
    if mesh_path and Path(mesh_path).exists():
        print(f"Loading {mesh_path}...")
        mesh = load_mesh_auto(mesh_path)
        print(f"Loaded mesh with {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    # Load trajectories
    trajectory = None
    if trajectory_path and Path(trajectory_path).exists():
        print(f"Loading original trajectory from {trajectory_path}...")
        trajectory = load_trajectory_json(trajectory_path)
        print(f"Loaded original trajectory with {len(trajectory)} poses")
    
    optimized_trajectory = None
    if optimized_trajectory_path and Path(optimized_trajectory_path).exists():
        print(f"Loading optimized trajectory from {optimized_trajectory_path}...")
        optimized_trajectory = load_trajectory_json(optimized_trajectory_path)
        print(f"Loaded optimized trajectory with {len(optimized_trajectory)} poses")
    
    # Load anchors
    anchors = None
    if anchor_path and Path(anchor_path).exists():
        print(f"Loading anchors from {anchor_path}...")
        anchors = load_anchors_json(anchor_path)
        print(f"Loaded {len(anchors)} anchors")
    
    # Compute scene center and scale
    scene_center = gaussians['means'].mean(dim=0).cpu().numpy()
    scene_scale = (gaussians['means'] - gaussians['means'].mean(dim=0)).norm(dim=1).quantile(0.95).item()
    
    server = viser.ViserServer(host="0.0.0.0", port=port)
    print(f"Viser server running at http://localhost:{port}")
    
    # ========== 3DGS Rendering Toggle ==========
    show_3dgs = [True]
    
    with server.gui.add_folder("3D Gaussian Splatting"):
        show_3dgs_checkbox = server.gui.add_checkbox("Show 3DGS", initial_value=True)
    
    @show_3dgs_checkbox.on_update
    def update_3dgs_visibility(_):
        show_3dgs[0] = show_3dgs_checkbox.value
        if not show_3dgs[0]:
            for c in server.get_clients().values():
                try:
                    c.scene.set_background_image(
                        np.zeros((2, 2, 3), dtype=np.uint8),
                        format="jpeg", jpeg_quality=1,
                    )
                except Exception:
                    pass
    
    # Add mesh to scene
    mesh_handle = None
    if mesh:
        with server.gui.add_folder("Mesh"):
            show_mesh = server.gui.add_checkbox("Show Mesh", initial_value=False)
        
        mesh_handle = add_mesh_to_viser(server, "/mesh", mesh)
        mesh_handle.visible = False
        
        @show_mesh.on_update
        def update_mesh_visibility(_):
            mesh_handle.visible = show_mesh.value
    
    # GUI controls
    with server.gui.add_folder("Render Settings"):
        resolution_slider = server.gui.add_slider("Resolution", min=256, max=1920, step=64, initial_value=1024)
        scale_slider = server.gui.add_slider("Gaussian Scale", min=0.1, max=3.0, step=0.1, initial_value=1.0)
    
    with server.gui.add_folder("Bounding Boxes"):
        show_boxes = server.gui.add_checkbox("Show Boxes", initial_value=True)
        show_labels = server.gui.add_checkbox("Show Labels", initial_value=True)
    
    # Add bounding boxes and labels
    bbox_handles = []
    label_handles = []
    
    skipped = 0
    
    if bbox_format == 'scene_graph':
        for item in objects:
            obj_id = item['id']
            label = item['label']
            corners = item['corners']
            color = get_label_color(label)
            
            box_name = f"/boxes/box_{obj_id}"
            box_handle, center = add_bounding_box_from_corners(server, box_name, corners, color)
            bbox_handles.append(box_handle)
            
            label_name = f"/labels/label_{obj_id}"
            label_pos = center + np.array([0, 0, 0.3])
            room_str = f" [{item.get('room', '')}]" if item.get('room') else ""
            wall_str = " [wall]" if item.get('against_wall') else ""
            ceil_str = " [ceil]" if item.get('attached_to_ceiling') else ""
            label_text = f"{label} ({obj_id}){room_str}{wall_str}{ceil_str}"
            lh = server.scene.add_label(
                label_name,
                text=label_text,
                position=label_pos,
            )
            label_handles.append(lh)
    else:
        for item in objects:
            if 'bounding_box' not in item:
                skipped += 1
                continue
            
            ins_id = item['ins_id']
            label = item['label']
            bbox_points = item['bounding_box']
            color = get_label_color(label)
            
            box_name = f"/boxes/box_{ins_id}"
            box_handle, center = add_bounding_box(server, box_name, bbox_points, color)
            bbox_handles.append(box_handle)
            
            label_name = f"/labels/label_{ins_id}"
            label_pos = center + np.array([0, 0, 0.3])
            label_text = f"{label} ({ins_id})"
            lh = server.scene.add_label(
                label_name,
                text=label_text,
                position=label_pos,
            )
            label_handles.append(lh)
    
    print(f"Added {len(bbox_handles)} boxes" + (f", skipped {skipped} items without bounding boxes" if skipped else ""))
    
    @show_boxes.on_update
    def update_box_visibility(_):
        for h in bbox_handles:
            h.visible = show_boxes.value
    
    @show_labels.on_update
    def update_label_visibility(_):
        for h in label_handles:
            h.visible = show_labels.value
    
    # ========== TRAJECTORY COMPARISON VISUALIZATION ==========
    trajectory_handles = {'original': [], 'optimized': [], 'collision_markers': [], 'diff': []}
    current_pose_idx = [0]
    active_trajectory = [None]
    original_collisions = None
    optimized_collisions = None
    
    # Playback sphere handle — shared across comparison / single trajectory paths
    current_pose_handle = [None]
    show_playback_sphere = [False]  # hidden by default
    
    has_comparison = trajectory is not None and optimized_trajectory is not None
    
    if has_comparison:
        print("\n=== Adding trajectory comparison visualization ===")
        trajectory_handles, original_collisions, optimized_collisions = add_trajectory_comparison(
            server, trajectory, optimized_trajectory, mesh=mesh, collision_margin=0.3,
        )
        
        with server.gui.add_folder("Trajectory Comparison"):
            show_original = server.gui.add_checkbox("Show Original (Orange)", initial_value=True)
            show_optimized = server.gui.add_checkbox("Show Optimized (Blue)", initial_value=True)
            show_collision_markers = server.gui.add_checkbox("Show Collision Markers", initial_value=True)
            show_diff_lines = server.gui.add_checkbox("Show Difference Lines", initial_value=True)
            
            n_orig_collisions = np.sum(original_collisions) if original_collisions is not None else 0
            n_opt_collisions = np.sum(optimized_collisions) if optimized_collisions is not None else 0
            server.gui.add_markdown(
                f"**Collision Stats:**\n"
                f"- Original: {n_orig_collisions}/{len(trajectory)} frames\n"
                f"- Optimized: {n_opt_collisions}/{len(optimized_trajectory)} frames"
            )
            
            trajectory_selector = server.gui.add_dropdown(
                "Active Trajectory", options=["Original", "Optimized"],
                initial_value="Optimized",
            )
            active_trajectory[0] = optimized_trajectory
        
        @show_original.on_update
        def update_original_visibility(_):
            for h in trajectory_handles.get('original', []):
                try: h.visible = show_original.value
                except Exception: pass
            # Hide playback sphere when both trajectories are hidden
            any_visible = show_original.value or show_optimized.value
            show_playback_sphere[0] = any_visible
            if current_pose_handle[0] is not None:
                current_pose_handle[0].visible = any_visible
        
        @show_optimized.on_update
        def update_optimized_visibility(_):
            for h in trajectory_handles.get('optimized', []):
                try: h.visible = show_optimized.value
                except Exception: pass
            any_visible = show_original.value or show_optimized.value
            show_playback_sphere[0] = any_visible
            if current_pose_handle[0] is not None:
                current_pose_handle[0].visible = any_visible
        
        @show_collision_markers.on_update
        def update_collision_visibility(_):
            for h in trajectory_handles.get('collision_markers', []):
                try: h.visible = show_collision_markers.value
                except Exception: pass
        
        @show_diff_lines.on_update
        def update_diff_visibility(_):
            for h in trajectory_handles.get('diff', []):
                try: h.visible = show_diff_lines.value
                except Exception: pass
        
        @trajectory_selector.on_update
        def update_active_trajectory(_):
            if trajectory_selector.value == "Original":
                active_trajectory[0] = trajectory
            else:
                active_trajectory[0] = optimized_trajectory
    
    elif trajectory is not None:
        active_trajectory[0] = trajectory
        orig_handles, _ = add_trajectory_to_viser(
            server, trajectory, "/trajectory",
            color_scheme="rainbow", line_width=3.0,
        )
        trajectory_handles['original'] = orig_handles
        
        with server.gui.add_folder("Trajectory"):
            show_trajectory = server.gui.add_checkbox("Show Trajectory", initial_value=True)
        
        @show_trajectory.on_update
        def update_trajectory_visibility(_):
            for h in trajectory_handles.get('original', []):
                try: h.visible = show_trajectory.value
                except Exception: pass
            show_playback_sphere[0] = show_trajectory.value
            if current_pose_handle[0] is not None:
                current_pose_handle[0].visible = show_trajectory.value
    
    # Playback controls
    is_playing = [False]
    
    if active_trajectory[0] is not None:
        with server.gui.add_folder("Playback Controls"):
            pose_slider = server.gui.add_slider(
                "Current Pose", min=0, max=len(active_trajectory[0]) - 1,
                step=1, initial_value=0,
            )
            play_button = server.gui.add_button("Play")
            stop_button = server.gui.add_button("Stop")
            goto_pose_button = server.gui.add_button("Go to Pose View")
            playback_speed = server.gui.add_slider("Playback Speed", min=0.1, max=5.0, step=0.1, initial_value=1.0)
        
        # Create playback sphere — hidden by default
        current_pose_handle[0] = server.scene.add_icosphere(
            "/playback/current_pose", radius=0.12,
            position=active_trajectory[0].positions[0], color=(255, 255, 0),
        )
        current_pose_handle[0].visible = False
        
        @pose_slider.on_update
        def update_current_pose(_):
            traj = active_trajectory[0]
            if traj is None: return
            idx = int(min(pose_slider.value, len(traj) - 1))
            current_pose_idx[0] = idx
            pos = traj.positions[idx]
            if current_pose_handle[0] is not None:
                current_pose_handle[0].remove()
            current_pose_handle[0] = server.scene.add_icosphere(
                "/playback/current_pose", radius=0.12, position=pos, color=(255, 255, 0),
            )
            current_pose_handle[0].visible = show_playback_sphere[0]
        
        @play_button.on_click
        def on_play(_):
            is_playing[0] = True
            show_playback_sphere[0] = True
            if current_pose_handle[0] is not None:
                current_pose_handle[0].visible = True
        
        @stop_button.on_click
        def on_stop(_):
            is_playing[0] = False
        
        @goto_pose_button.on_click
        def on_goto_pose(_):
            traj = active_trajectory[0]
            if traj is None: return
            idx = int(pose_slider.value)
            pos = traj.positions[idx]
            rot = traj.rotations[idx]
            forward = rot[:, 2]
            look_at = pos + forward * 2.0
            for c in server.get_clients().values():
                c.camera.position = pos
                c.camera.look_at = look_at

    # Add coordinate axes
    with server.gui.add_folder("Coordinate Axes"):
        show_axes = server.gui.add_checkbox("Show Axes", initial_value=True)
        axes_scale = server.gui.add_slider("Axes Scale", min=0.1, max=10.0, step=0.1, initial_value=5.0)
        axes_at_origin = server.gui.add_checkbox("At World Origin", initial_value=True)

    axes_origin = np.array([0.0, 0.0, 0.0])
    axes_handles = add_coordinate_axes(server, origin=axes_origin, scale=1.0, name="/axes")

    @show_axes.on_update
    def update_axes_visibility(_):
        for h in axes_handles: h.visible = show_axes.value

    @axes_scale.on_update
    def update_axes_scale(_):
        for h in axes_handles:
            try: h.remove()
            except Exception: pass
        origin = np.array([0.0, 0.0, 0.0]) if axes_at_origin.value else scene_center
        axes_handles.clear()
        axes_handles.extend(add_coordinate_axes(server, origin=origin, scale=axes_scale.value, name="/axes"))

    @axes_at_origin.on_update
    def update_axes_origin(_):
        for h in axes_handles:
            try: h.remove()
            except Exception: pass
        origin = np.array([0.0, 0.0, 0.0]) if axes_at_origin.value else scene_center
        axes_handles.clear()
        axes_handles.extend(add_coordinate_axes(server, origin=origin, scale=axes_scale.value, name="/axes"))
    
    # Add anchors visualization
    anchor_handles = []
    if anchors and len(anchors) > 0:
        with server.gui.add_folder("Anchors"):
            show_anchors = server.gui.add_checkbox("Show Anchors", initial_value=True)
            show_anchor_frustums = server.gui.add_checkbox("Show Frustums", initial_value=True)
            show_anchor_labels = server.gui.add_checkbox("Show Labels", initial_value=True)
            
            anchor_options = [f"{i}: {a.object_label} (id:{a.object_id})" for i, a in enumerate(anchors)]
            anchor_selector = server.gui.add_dropdown("Select Anchor", options=anchor_options)
            goto_anchor_button = server.gui.add_button("Go to Anchor")
        
        anchor_handles = add_anchors_to_viser(server, anchors, "/anchors")
        
        @show_anchors.on_update
        def update_anchor_visibility(_):
            for h, tag in anchor_handles:
                try: h.visible = show_anchors.value
                except Exception: pass
        
        @show_anchor_frustums.on_update
        def update_anchor_frustum_visibility(_):
            for h, tag in anchor_handles:
                if tag == "frustum":
                    try: h.visible = show_anchor_frustums.value
                    except Exception: pass
        
        @show_anchor_labels.on_update
        def update_anchor_label_visibility(_):
            for h, tag in anchor_handles:
                if tag == "label":
                    try: h.visible = show_anchor_labels.value
                    except Exception: pass
        
        @goto_anchor_button.on_click
        def on_goto_anchor(_):
            selected = anchor_selector.value
            if selected:
                idx = int(selected.split(":")[0])
                anchor = anchors[idx]
                from src.trajectory_optimizer.trajectory_optimizer import TrajectoryCombiner
                combiner = TrajectoryCombiner()
                pose = combiner.anchor_to_pose(anchor)
                forward = pose.rotation[:, 2]
                look_at = pose.position + forward * 2.0
                for c in server.get_clients().values():
                    c.camera.position = pose.position
                    c.camera.look_at = look_at

    # ========== 3DGS Rendering Callback ==========
    @server.on_client_connect
    def on_client_connect(client: viser.ClientHandle):
        client.camera.position = scene_center + np.array([0, 0, scene_scale * 2])
        client.camera.look_at = scene_center
        client.camera.up_direction = (0, 0, 1)  # Z-up
        
        @client.camera.on_update
        def on_camera_update(camera: viser.CameraHandle):
            if not show_3dgs[0]: return
            
            width = resolution_slider.value
            height = int(width * 9 / 16)
            fov = camera.fov
            
            fy = height / (2 * np.tan(fov / 2))
            fx = fy
            K = torch.tensor([
                [fx, 0, width / 2],
                [0, fy, height / 2],
                [0, 0, 1]
            ], dtype=torch.float32, device=device)
            
            c2w = np.eye(4)
            c2w[:3, :3] = tf.SO3(camera.wxyz).as_matrix()
            c2w[:3, 3] = camera.position
            w2c = np.linalg.inv(c2w)
            viewmat = torch.tensor(w2c, dtype=torch.float32, device=device)
            
            scaled_gaussians = gaussians.copy()
            scaled_gaussians['scales'] = gaussians['scales'] * scale_slider.value
            
            with torch.no_grad():
                image = render_gaussians(scaled_gaussians, viewmat, K, width, height)
            
            image_np = (image.cpu().numpy() * 255).astype(np.uint8)
            client.scene.set_background_image(image_np, format="jpeg", jpeg_quality=90)

    # ========== Main loop with playback ==========
    try:
        last_time = time.time()
        while True:
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time
            
            traj = active_trajectory[0]
            if traj is not None and len(traj) > 0 and is_playing[0]:
                speed = playback_speed.value
                frames_to_advance = dt * 30.0 * speed
                
                new_idx = current_pose_idx[0] + frames_to_advance
                if new_idx >= len(traj) - 1:
                    new_idx = 0
                
                current_pose_idx[0] = new_idx
                idx = int(new_idx)
                pose_slider.value = idx
                
                pos = traj.positions[idx]
                if current_pose_handle[0] is not None:
                    current_pose_handle[0].remove()
                current_pose_handle[0] = server.scene.add_icosphere(
                    "/playback/current_pose", radius=0.12,
                    position=pos, color=(255, 255, 0),
                )
                current_pose_handle[0].visible = show_playback_sphere[0]
            
            time.sleep(0.033)
            
    except KeyboardInterrupt:
        print("Shutting down...")


if __name__ == "__main__":
    # ========== ScanNet++ scene graph format ==========
    scene_id = "09c1414f1b"
    ply_path = f"data/ScanNetpp/scenes/{scene_id}/dslr/ply/point_cloud_30000.ply"
    bbox_json_path = f"data/ScanNetpp/scenes/{scene_id}/dslr/sg/{scene_id}-simple.json"
    mesh_path = f"outputs/scannetpp/{scene_id}/obb_mesh.ply"  # PLY mesh from build_obb_mesh.py
    trajectory_path = f"outputs/scannetpp/{scene_id}/combined_trajectory.json"
    optimized_trajectory_path = None
    anchors_path = f"outputs/scannetpp/{scene_id}/anchors.json"
    port = 8080
    
    visualize_assets(
        ply_path, 
        bbox_json_path, 
        mesh_path, 
        trajectory_path, 
        optimized_trajectory_path,
        anchors_path, 
        port,
    )