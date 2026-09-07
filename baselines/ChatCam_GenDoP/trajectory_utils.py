"""
Trajectory Utilities
=====================
I/O, visualization, and format conversion for camera trajectories.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Optional


def save_trajectory_json(trajectory: dict, path: str):
    """Save trajectory in a standard JSON format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    c2ws = trajectory.get("c2ws")
    intrinsics = trajectory.get("intrinsics")

    if c2ws is None:
        data = {"error": "No trajectory generated"}
    else:
        data = {
            "num_frames": len(c2ws),
            "intrinsics": intrinsics,
            "frames": [],
        }
        for i, c2w in enumerate(c2ws):
            data["frames"].append({
                "frame_id": i,
                "transform_matrix": c2w.tolist(),
            })

    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"    Saved trajectory ({len(c2ws) if c2ws is not None else 0} frames) to {path}")


def load_trajectory_json(path: str) -> dict:
    """Load trajectory from JSON."""
    with open(path) as f:
        data = json.load(f)

    c2ws = np.array([frame["transform_matrix"] for frame in data["frames"]], dtype=np.float32)
    return {
        "c2ws": c2ws,
        "intrinsics": data.get("intrinsics"),
        "num_frames": len(c2ws),
    }


def c2ws_to_nerfstudio_path(
    c2ws: np.ndarray,
    intrinsics: dict,
    output_path: str,
    fps: int = 24,
):
    """
    Export trajectory as Nerfstudio camera path JSON for ns-render.
    
    Usage:
        ns-render camera-path --load-config <config.yml> \
            --camera-path-filename <output_path> \
            --output-path rendered.mp4
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    w = intrinsics.get("w", 512)
    h = intrinsics.get("h", 512)
    fx = intrinsics.get("fx", 500)

    camera_path = []
    for c2w in c2ws:
        camera_path.append({
            "camera_to_world": c2w.flatten().tolist(),
            "fov": float(2 * np.degrees(np.arctan(w / (2 * fx)))),
            "aspect": w / h,
        })

    data = {
        "camera_type": "perspective",
        "render_height": h,
        "render_width": w,
        "camera_path": camera_path,
        "fps": fps,
        "seconds": len(c2ws) / fps,
        "is_cycle": False,
        "smoothness_value": 0.0,
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"    Exported Nerfstudio camera path to {output_path}")


def visualize_c2ws(
    c2ws: np.ndarray,
    save_path: str,
    title: str = "",
):
    """
    Visualize camera trajectory from c2w matrices.
    Draws 3 orthogonal views (front, top, side) combined into one image.
    
    Adapted from GenDoP's draw_json function.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("    Visualization requires matplotlib")
        return

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    positions = c2ws[:, :3, 3]
    num_frames = len(c2ws)

    # Color gradient: rainbow from start to end
    colors = plt.cm.rainbow(np.linspace(0, 1, num_frames))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Front view (XY plane)
    ax = axes[0]
    ax.scatter(positions[:, 0], positions[:, 1], c=colors, s=3)
    ax.plot(positions[:, 0], positions[:, 1], 'k-', alpha=0.3, linewidth=0.5)
    ax.scatter(positions[0, 0], positions[0, 1], c='green', s=60, marker='o', zorder=5, label='Start')
    ax.scatter(positions[-1, 0], positions[-1, 1], c='red', s=60, marker='s', zorder=5, label='End')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Front (XY)")
    ax.set_aspect("equal")
    ax.legend(fontsize=8)

    # Top view (XZ plane)
    ax = axes[1]
    ax.scatter(positions[:, 0], positions[:, 2], c=colors, s=3)
    ax.plot(positions[:, 0], positions[:, 2], 'k-', alpha=0.3, linewidth=0.5)
    ax.scatter(positions[0, 0], positions[0, 2], c='green', s=60, marker='o', zorder=5)
    ax.scatter(positions[-1, 0], positions[-1, 2], c='red', s=60, marker='s', zorder=5)
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_title("Top (XZ)")
    ax.set_aspect("equal")

    # Side view (YZ plane)
    ax = axes[2]
    ax.scatter(positions[:, 2], positions[:, 1], c=colors, s=3)
    ax.plot(positions[:, 2], positions[:, 1], 'k-', alpha=0.3, linewidth=0.5)
    ax.scatter(positions[0, 2], positions[0, 1], c='green', s=60, marker='o', zorder=5)
    ax.scatter(positions[-1, 2], positions[-1, 1], c='red', s=60, marker='s', zorder=5)
    ax.set_xlabel("Z")
    ax.set_ylabel("Y")
    ax.set_title("Side (ZY)")
    ax.set_aspect("equal")

    # Add camera direction arrows at key frames
    for ax_idx, (dim1, dim2) in enumerate([(0, 1), (0, 2), (2, 1)]):
        ax = axes[ax_idx]
        step = max(1, num_frames // 10)
        for i in range(0, num_frames, step):
            # Camera forward direction (negative Z in camera frame)
            forward = c2ws[i, :3, 2]  # Z column
            pos = positions[i]
            scale = 0.1 * np.max(np.abs(positions))
            ax.annotate(
                "", xy=(pos[dim1] - forward[dim1] * scale, pos[dim2] - forward[dim2] * scale),
                xytext=(pos[dim1], pos[dim2]),
                arrowprops=dict(arrowstyle="->", color=colors[i], lw=1.0),
            )

    fig.suptitle(title, fontsize=11, y=0.98)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Trajectory visualization saved to {save_path}")


def compute_trajectory_metrics(c2ws: np.ndarray) -> dict:
    """Compute basic trajectory metrics."""
    positions = c2ws[:, :3, 3]
    N = len(positions)

    # Path length
    steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    total_length = float(np.sum(steps))
    avg_speed = float(np.mean(steps))
    speed_std = float(np.std(steps))

    # Acceleration
    if N > 2:
        accel = np.diff(positions, n=2, axis=0)
        avg_accel = float(np.mean(np.linalg.norm(accel, axis=1)))
    else:
        avg_accel = 0.0

    # Rotation smoothness
    from scipy.spatial.transform import Rotation
    rots = Rotation.from_matrix(c2ws[:, :3, :3])
    angle_diffs = []
    for i in range(1, N):
        rel = rots[i - 1].inv() * rots[i]
        angle_diffs.append(float(rel.magnitude()))
    avg_rot_change = float(np.mean(angle_diffs)) if angle_diffs else 0.0

    return {
        "num_frames": N,
        "total_path_length": total_length,
        "avg_speed": avg_speed,
        "speed_std": speed_std,
        "avg_acceleration": avg_accel,
        "avg_rotation_change_rad": avg_rot_change,
    }
