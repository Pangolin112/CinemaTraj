"""
Utility Functions
==================
Prompt parsing, trajectory I/O, visualization, and coordinate transforms.
"""

import re
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from scene_reconstruction import qvec2rotmat, rotmat2qvec


# ---------------------------------------------------------------------------
# Prompt Parsing
# ---------------------------------------------------------------------------

def parse_prompt_segments(prompt: str) -> List[str]:
    """
    Parse a compound camera instruction prompt into individual segments.
    
    The prompt is split on common delimiters used in cinematographic instructions:
    commas, 'then', 'and then', 'next', 'followed by', semicolons.
    
    Each segment is stripped and filtered for non-empty strings.
    
    Examples:
        "zoom in on the mug, pan to the bear" 
            -> ["zoom in on the mug", "pan to the bear"]
        
        "Overview of the table, zoom in the mug, pan to the cookies"
            -> ["Overview of the table", "zoom in the mug", "pan to the cookies"]
            
        "Close up of the lego bulldozer; then pan to the textbook"
            -> ["Close up of the lego bulldozer", "pan to the textbook"]
    """
    # Replace multi-word delimiters first
    text = prompt
    for delimiter in ["and then", "then", "next", "followed by", "and finally", "finally"]:
        text = re.sub(rf'\b{delimiter}\b', ',', text, flags=re.IGNORECASE)

    # Split on comma, semicolon, or period (but not within quotes)
    segments = re.split(r'[,;.]', text)

    # Clean up each segment
    segments = [seg.strip() for seg in segments if seg.strip()]

    # Filter out segments that are too short to be meaningful
    segments = [seg for seg in segments if len(seg.split()) >= 2]

    return segments


def parse_cinematographic_intent(segment: str) -> Dict:
    """
    Parse a single instruction segment to extract cinematographic intent.
    
    Identifies:
        - shot_type: close up, wide shot, high angle, low angle, etc.
        - movement: zoom in, pull out, pan, tilt, orbit, focus, etc.
        - target_object: the referenced object
    
    Returns:
        Dict with keys 'shot_type', 'movement', 'target_object'
    """
    segment_lower = segment.lower().strip()

    # Shot type patterns
    shot_types = {
        "close up": r"close\s*up",
        "wide shot": r"wide\s*(shot|angle|view)",
        "high angle": r"high\s*angle",
        "low angle": r"low\s*angle",
        "overview": r"overview|bird.?s?\s*eye",
        "medium shot": r"medium\s*(shot|view)",
    }
    detected_shot = None
    for shot_name, pattern in shot_types.items():
        if re.search(pattern, segment_lower):
            detected_shot = shot_name
            break

    # Movement patterns
    movements = {
        "zoom in": r"zoom\s*in",
        "zoom out": r"zoom\s*out|pull\s*out|pull\s*back",
        "pan": r"\bpan\b",
        "tilt": r"\btilt\b",
        "orbit": r"\borbit\b",
        "dolly": r"\bdolly\b",
        "focus": r"\bfocus\b",
        "push in": r"push\s*in|pull\s*in|move\s*(closer|in)",
        "show": r"\bshow\b|\bdisplay\b",
    }
    detected_movement = None
    for move_name, pattern in movements.items():
        if re.search(pattern, segment_lower):
            detected_movement = move_name
            break

    # Extract target object (everything after movement/shot keywords and prepositions)
    # Remove shot type and movement phrases
    target = segment_lower
    for pattern in list(shot_types.values()) + list(movements.values()):
        target = re.sub(pattern, "", target)

    # Remove common prepositions and articles
    for word in ["of", "the", "a", "an", "to", "on", "at", "from", "around", "into", "in"]:
        target = re.sub(rf"\b{word}\b", "", target)

    target = target.strip()
    target = re.sub(r'\s+', ' ', target)  # Collapse whitespace

    return {
        "shot_type": detected_shot,
        "movement": detected_movement,
        "target_object": target if target else None,
        "raw_segment": segment,
    }


# ---------------------------------------------------------------------------
# Trajectory I/O
# ---------------------------------------------------------------------------

def save_trajectory(trajectory: Dict, path: Path):
    """Save trajectory to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "positions": trajectory["positions"].tolist(),
        "rotations": trajectory["rotations"].tolist(),
        "keyframe_indices": trajectory.get("keyframe_indices", []),
        "num_frames": len(trajectory["positions"]),
        "num_corrections": trajectory.get("num_corrections", 0),
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_trajectory(path: Path) -> Dict:
    """Load trajectory from JSON file."""
    with open(path, "r") as f:
        data = json.load(f)

    return {
        "positions": np.array(data["positions"]),
        "rotations": np.array(data["rotations"]),
        "keyframe_indices": data.get("keyframe_indices", []),
        "num_corrections": data.get("num_corrections", 0),
    }


# ---------------------------------------------------------------------------
# COLMAP Data Loading Helpers
# ---------------------------------------------------------------------------

def load_colmap_cameras(model_dir: Path) -> dict:
    """Convenience function to load COLMAP cameras."""
    from scene_reconstruction import read_cameras_binary, read_cameras_text

    bin_path = Path(model_dir) / "cameras.bin"
    txt_path = Path(model_dir) / "cameras.txt"

    if bin_path.exists():
        return read_cameras_binary(str(bin_path))
    elif txt_path.exists():
        return read_cameras_text(str(txt_path))
    else:
        raise FileNotFoundError(f"No cameras file in {model_dir}")


def load_colmap_images(model_dir: Path) -> dict:
    """Convenience function to load COLMAP images."""
    from scene_reconstruction import read_images_binary, read_images_text

    bin_path = Path(model_dir) / "images.bin"
    txt_path = Path(model_dir) / "images.txt"

    if bin_path.exists():
        return read_images_binary(str(bin_path))
    elif txt_path.exists():
        return read_images_text(str(txt_path))
    else:
        raise FileNotFoundError(f"No images file in {model_dir}")


def get_camera_intrinsics_matrix(camera_info) -> np.ndarray:
    """
    Convert COLMAP camera params to 3x3 intrinsics matrix K.
    
    Supports SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL, RADIAL, OPENCV models.
    """
    model = camera_info.model
    params = camera_info.params

    if model == "SIMPLE_PINHOLE":
        f, cx, cy = params[0], params[1], params[2]
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
    elif model == "PINHOLE":
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    elif model in ("SIMPLE_RADIAL", "RADIAL"):
        f, cx, cy = params[0], params[1], params[2]
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
    elif model == "OPENCV":
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    else:
        # Default: assume first param is focal length
        f = params[0]
        cx = camera_info.width / 2.0
        cy = camera_info.height / 2.0
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])

    return K


# ---------------------------------------------------------------------------
# Coordinate Transforms
# ---------------------------------------------------------------------------

def camera_to_world(R: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert COLMAP's camera-to-world representation.
    
    COLMAP stores:
        R: rotation (world-to-camera)
        t: translation (world-to-camera)
    
    Camera center in world: C = -R^T @ t
    Camera-to-world rotation: R_c2w = R^T
    """
    R_c2w = R.T
    C = -R.T @ t
    return R_c2w, C


def world_to_camera(R_c2w: np.ndarray, C: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert camera-to-world to COLMAP's world-to-camera convention."""
    R_w2c = R_c2w.T
    t = -R_w2c @ C
    return R_w2c, t


def compute_look_at_rotation(
    camera_pos: np.ndarray,
    target_pos: np.ndarray,
    up: np.ndarray = np.array([0, 1, 0]),
) -> np.ndarray:
    """
    Compute rotation matrix for a camera looking at a target point.
    
    Args:
        camera_pos: (3,) camera position.
        target_pos: (3,) target point to look at.
        up: (3,) world up vector.
        
    Returns:
        (3, 3) camera-to-world rotation matrix.
    """
    forward = target_pos - camera_pos
    forward = forward / (np.linalg.norm(forward) + 1e-8)

    right = np.cross(forward, up)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-6:
        # Camera looking straight up/down, choose arbitrary right
        up = np.array([0, 0, 1])
        right = np.cross(forward, up)
        right_norm = np.linalg.norm(right)
    right = right / (right_norm + 1e-8)

    up_corrected = np.cross(right, forward)
    up_corrected = up_corrected / (np.linalg.norm(up_corrected) + 1e-8)

    # R = [right | up | -forward] (columns)
    R = np.stack([right, up_corrected, -forward], axis=-1)
    return R


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

# def visualize_trajectory_3d(
#     trajectory: Dict,
#     keyframe_positions: Optional[np.ndarray] = None,
#     points3d: Optional[dict] = None,
#     save_path: Optional[Path] = None,
# ):
#     """
#     Create a 3D visualization of the camera trajectory using plotly.
    
#     Generates an interactive HTML file showing:
#         - Camera trajectory path (line)
#         - Keyframe positions (highlighted markers)
#         - Sparse point cloud (if available)
#         - Camera frustum indicators at keyframes
#     """
#     try:
#         import plotly.graph_objects as go
#     except ImportError:
#         print("    Warning: plotly not available. Install with: pip install plotly")
#         _visualize_trajectory_matplotlib(trajectory, keyframe_positions, points3d, save_path)
#         return

#     positions = trajectory["positions"]
#     fig = go.Figure()

#     # Point cloud
#     if points3d and len(points3d) > 0:
#         pts = np.array([p.xyz for p in points3d.values()])
#         colors = np.array([p.rgb for p in points3d.values()])

#         # Subsample if too many points
#         if len(pts) > 10000:
#             indices = np.random.choice(len(pts), 10000, replace=False)
#             pts = pts[indices]
#             colors = colors[indices]

#         color_strings = [f"rgb({r},{g},{b})" for r, g, b in colors]

#         fig.add_trace(go.Scatter3d(
#             x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
#             mode="markers",
#             marker=dict(size=1, color=color_strings, opacity=0.3),
#             name="Point Cloud",
#         ))

#     # Trajectory path
#     fig.add_trace(go.Scatter3d(
#         x=positions[:, 0], y=positions[:, 1], z=positions[:, 2],
#         mode="lines",
#         line=dict(color="red", width=4),
#         name="Camera Trajectory",
#     ))

#     # Keyframe markers
#     if keyframe_positions is not None:
#         fig.add_trace(go.Scatter3d(
#             x=keyframe_positions[:, 0],
#             y=keyframe_positions[:, 1],
#             z=keyframe_positions[:, 2],
#             mode="markers",
#             marker=dict(size=8, color="blue", symbol="diamond"),
#             name="Keyframes",
#         ))

#     # Start and end markers
#     fig.add_trace(go.Scatter3d(
#         x=[positions[0, 0]], y=[positions[0, 1]], z=[positions[0, 2]],
#         mode="markers",
#         marker=dict(size=10, color="green", symbol="circle"),
#         name="Start",
#     ))
#     fig.add_trace(go.Scatter3d(
#         x=[positions[-1, 0]], y=[positions[-1, 1]], z=[positions[-1, 2]],
#         mode="markers",
#         marker=dict(size=10, color="orange", symbol="square"),
#         name="End",
#     ))

#     fig.update_layout(
#         title="Camera Trajectory Visualization",
#         scene=dict(
#             xaxis_title="X",
#             yaxis_title="Y",
#             zaxis_title="Z",
#             aspectmode="data",
#         ),
#         width=1200,
#         height=800,
#     )

#     if save_path:
#         fig.write_html(str(save_path))
#     else:
#         fig.show()


def _visualize_trajectory_matplotlib(
    trajectory: Dict,
    keyframe_positions: Optional[np.ndarray] = None,
    points3d: Optional[dict] = None,
    save_path: Optional[Path] = None,
):
    """Fallback matplotlib visualization."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("    Warning: matplotlib not available for visualization.")
        return

    positions = trajectory["positions"]
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Point cloud
    if points3d and len(points3d) > 0:
        pts = np.array([p.xyz for p in points3d.values()])
        colors = np.array([p.rgb for p in points3d.values()]) / 255.0
        if len(pts) > 5000:
            idx = np.random.choice(len(pts), 5000, replace=False)
            pts, colors = pts[idx], colors[idx]
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=colors, s=0.5, alpha=0.3)

    # Trajectory
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], "r-", linewidth=2, label="Trajectory")

    # Keyframes
    if keyframe_positions is not None:
        ax.scatter(
            keyframe_positions[:, 0], keyframe_positions[:, 1], keyframe_positions[:, 2],
            c="blue", s=80, marker="D", label="Keyframes",
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.title("Camera Trajectory Visualization")

    if save_path:
        plt.savefig(str(save_path).replace(".html", ".png"), dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def compute_trajectory_stats(trajectory: Dict) -> Dict:
    """Compute statistics about a trajectory."""
    positions = trajectory["positions"]
    N = len(positions)

    # Step sizes
    steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    total_length = np.sum(steps)
    avg_speed = np.mean(steps)
    speed_std = np.std(steps)

    # Acceleration (2nd derivative)
    if N > 2:
        accel = np.diff(positions, n=2, axis=0)
        avg_accel = np.mean(np.linalg.norm(accel, axis=1))
    else:
        avg_accel = 0.0

    return {
        "num_frames": N,
        "total_path_length": float(total_length),
        "avg_step_size": float(avg_speed),
        "step_size_std": float(speed_std),
        "avg_acceleration": float(avg_accel),
        "num_keyframes": len(trajectory.get("keyframe_indices", [])),
    }


"""
Replace the visualize_trajectory_3d function and add _add_camera_frustums 
in your utils.py. Also update the call in main.py to pass frustum params.

Drop-in replacement - same function signature plus optional new args.
"""

# ---- Add this to utils.py (replace existing visualize_trajectory_3d) ----

def visualize_trajectory_3d(
    trajectory: Dict,
    keyframe_positions: Optional[np.ndarray] = None,
    points3d: Optional[dict] = None,
    save_path: Optional[Path] = None,
    frustum_scale: float = 0.15,
    frustum_every: int = 5,
):
    """
    Create a 3D visualization of the camera trajectory using plotly.

    Shows camera frustums to verify orientation. Each frustum has:
      - Blue wireframe: image plane + edges from camera center (shows look direction)
      - Red line: up indicator on top of image plane
      - The frustum "opens" toward where the camera is looking

    Args:
        trajectory: Dict with 'positions' (N,3) and 'rotations' (N,3,3).
            rotations are camera-to-world (c2w) matrices.
        frustum_scale: Size of frustum visualization.
        frustum_every: Draw a frustum every N frames.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("    Warning: plotly not available. Install with: pip install plotly")
        return

    positions = trajectory["positions"]
    rotations = trajectory.get("rotations", None)
    fig = go.Figure()

    # Point cloud
    if points3d and len(points3d) > 0:
        pts = np.array([p.xyz for p in points3d.values()])
        colors = np.array([p.rgb for p in points3d.values()])

        if len(pts) > 10000:
            indices = np.random.choice(len(pts), 10000, replace=False)
            pts = pts[indices]
            colors = colors[indices]

        color_strings = [f"rgb({r},{g},{b})" for r, g, b in colors]

        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode="markers",
            marker=dict(size=1, color=color_strings, opacity=0.3),
            name="Point Cloud",
        ))

    # Trajectory path
    fig.add_trace(go.Scatter3d(
        x=positions[:, 0], y=positions[:, 1], z=positions[:, 2],
        mode="lines",
        line=dict(color="red", width=4),
        name="Camera Trajectory",
    ))

    # Camera frustums
    if rotations is not None and len(rotations) == len(positions):
        _add_camera_frustums(
            fig, positions, rotations,
            frustum_scale=frustum_scale,
            every=frustum_every,
        )

    # Keyframe markers
    if keyframe_positions is not None:
        fig.add_trace(go.Scatter3d(
            x=keyframe_positions[:, 0],
            y=keyframe_positions[:, 1],
            z=keyframe_positions[:, 2],
            mode="markers",
            marker=dict(size=8, color="blue", symbol="diamond"),
            name="Keyframes",
        ))

    # Start and end markers
    fig.add_trace(go.Scatter3d(
        x=[positions[0, 0]], y=[positions[0, 1]], z=[positions[0, 2]],
        mode="markers+text",
        marker=dict(size=10, color="green", symbol="circle"),
        text=["START"],
        textposition="top center",
        name="Start",
    ))
    fig.add_trace(go.Scatter3d(
        x=[positions[-1, 0]], y=[positions[-1, 1]], z=[positions[-1, 2]],
        mode="markers+text",
        marker=dict(size=10, color="orange", symbol="square"),
        text=["END"],
        textposition="top center",
        name="End",
    ))

    fig.update_layout(
        title="Camera Trajectory Visualization (frustums show look direction)",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
        ),
        width=1200,
        height=800,
    )

    if save_path:
        fig.write_html(str(save_path))
        print(f"    Saved trajectory visualization to {save_path}")
    else:
        fig.show()


def _add_camera_frustums(
    fig,
    positions: np.ndarray,
    rotations: np.ndarray,
    frustum_scale: float = 0.15,
    aspect_ratio: float = 16.0 / 9.0,
    every: int = 5,
):
    """
    Add camera frustum wireframes to a plotly figure.

    Assumes rotations are c2w matrices in OpenCV convention:
        col 0 = right  (X)
        col 1 = down   (Y)
        col 2 = forward (Z)  <-- camera looks along +Z

    The frustum "opens" in the +Z direction of the camera,
    so if cameras appear to look backward, the rotation convention
    is likely OpenGL (where forward = -Z) or w2c instead of c2w.
    """
    import plotly.graph_objects as go

    half_w = frustum_scale * 0.5 * aspect_ratio
    half_h = frustum_scale * 0.5
    d = frustum_scale  # depth of frustum

    # Image plane corners in camera-local coords (OpenCV: Z = forward)
    corners_local = np.array([
        [-half_w, -half_h, d],  # top-left
        [ half_w, -half_h, d],  # top-right
        [ half_w,  half_h, d],  # bottom-right
        [-half_w,  half_h, d],  # bottom-left
    ])

    all_frustum_x = []
    all_frustum_y = []
    all_frustum_z = []

    all_up_x = []
    all_up_y = []
    all_up_z = []

    for i in range(0, len(positions), every):
        pos = positions[i]
        R = rotations[i]  # (3, 3) c2w

        # Transform corners to world
        corners_world = (R @ corners_local.T).T + pos  # (4, 3)

        # Lines from camera center to each corner
        for c in range(4):
            all_frustum_x.extend([pos[0], corners_world[c, 0], None])
            all_frustum_y.extend([pos[1], corners_world[c, 1], None])
            all_frustum_z.extend([pos[2], corners_world[c, 2], None])

        # Image plane rectangle
        for c in range(4):
            cn = (c + 1) % 4
            all_frustum_x.extend([corners_world[c, 0], corners_world[cn, 0], None])
            all_frustum_y.extend([corners_world[c, 1], corners_world[cn, 1], None])
            all_frustum_z.extend([corners_world[c, 2], corners_world[cn, 2], None])

        # Up indicator: triangle on top edge of image plane
        # In OpenCV, camera Y points down, so "up" in camera = -Y
        top_center = (corners_world[0] + corners_world[1]) / 2.0
        up_tip = pos + R @ np.array([0, -half_h * 1.5, d])
        all_up_x.extend([top_center[0], up_tip[0], None])
        all_up_y.extend([top_center[1], up_tip[1], None])
        all_up_z.extend([top_center[2], up_tip[2], None])

    fig.add_trace(go.Scatter3d(
        x=all_frustum_x, y=all_frustum_y, z=all_frustum_z,
        mode="lines",
        line=dict(color="rgba(0, 100, 255, 0.6)", width=2),
        name="Frustums (look direction)",
        hoverinfo="skip",
    ))

    fig.add_trace(go.Scatter3d(
        x=all_up_x, y=all_up_y, z=all_up_z,
        mode="lines",
        line=dict(color="rgba(255, 50, 50, 0.8)", width=3),
        name="Up indicator",
        hoverinfo="skip",
    ))
