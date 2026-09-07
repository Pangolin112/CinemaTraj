from __future__ import annotations

"""
ChatCam: Empowering Camera Control through Conversational AI
==============================================================
Implementation based on Liu et al. (2024), HKUST & Dartmouth College.

Pipeline:
    1. LLM Agent (GPT-4) parses user instructions into observation/reasoning/plan
    2. CineGPT (replaced by GenDoP) generates atomic camera trajectories from text
    3. Anchor Determinator places trajectories in 3D scenes via CLIP + refinement
    4. Trajectory Composition combines sub-trajectories through anchor points
    5. 3D Visualization with point cloud and camera frustums
    6. (Optional) Video rendering via gsplat (3D Gaussian Splatting)

Usage:
    python chatcam.py --scene_dir /path/to/scene --instruction "Pan from the sofa to the window"
    python chatcam.py --scene_dir /path/to/scene --interactive  # Multi-turn conversation
    python chatcam.py --scene_dir /path/to/scene --instruction "..." --render --gsplat_model_path /path/to/model.ply
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import argparse
import json
import re
import numpy as np
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

from llm_agent import ChatCamAgent
from anchor_determinator import AnchorDeterminator
from cinegpt_wrapper import CineGPTWrapper
from trajectory_composer import TrajectoryComposer
from trajectory_utils import (
    save_trajectory_json,
    visualize_c2ws,
    c2ws_to_nerfstudio_path,
)


# ======================================================================
# Point cloud loading (for visualization and collision reference)
# ======================================================================

def load_point_cloud(scene_dir: Path, point_cloud_path: str = None) -> np.ndarray | None:
    """
    Load a scene point cloud for visualization.

    Primary source: COLMAP ``points3D.bin`` / ``points3D.txt`` in the
    ``sparse/0`` folder (the actual reconstructed 3D points).

    Fallback: explicit ``point_cloud_path`` or common .ply locations.
    """
    # --- 1. Try COLMAP sparse points (preferred) ---
    colmap_dir = scene_dir / "sparse" / "0"
    pts = _load_colmap_points3d(colmap_dir)
    if pts is not None:
        return pts

    # --- 2. Explicit path ---
    if point_cloud_path:
        pts = _load_ply_points(Path(point_cloud_path))
        if pts is not None:
            return pts

    # --- 3. Other common locations ---
    for candidate in [
        scene_dir / "point_cloud.ply",
        scene_dir.parent / "mesh" / "mesh_aligned_0.05.ply",
        scene_dir.parent / "scans" / "mesh_aligned_0.05.ply",
    ]:
        pts = _load_ply_points(candidate)
        if pts is not None:
            return pts

    return None


def _load_colmap_points3d(colmap_dir: Path) -> np.ndarray | None:
    """
    Read COLMAP ``points3D.bin`` or ``points3D.txt`` and return an (N, 3)
    float32 array of 3D point positions.
    """
    bin_path = colmap_dir / "points3D.bin"
    txt_path = colmap_dir / "points3D.txt"

    if bin_path.exists():
        try:
            from scene_reconstruction import read_points3D_binary
            points3d = read_points3D_binary(str(bin_path))
            pts = np.array([p.xyz for p in points3d.values()], dtype=np.float32)
            if len(pts) > 500_000:
                pts = pts[np.random.choice(len(pts), 500_000, replace=False)]
            print(f"  Loaded COLMAP point cloud from {bin_path}: {len(pts)} points")
            return pts
        except Exception as e:
            print(f"  Could not parse {bin_path}: {e}")

    if txt_path.exists():
        try:
            from scene_reconstruction import read_points3D_text
            points3d = read_points3D_text(str(txt_path))
            pts = np.array([p.xyz for p in points3d.values()], dtype=np.float32)
            if len(pts) > 500_000:
                pts = pts[np.random.choice(len(pts), 500_000, replace=False)]
            print(f"  Loaded COLMAP point cloud from {txt_path}: {len(pts)} points")
            return pts
        except Exception:
            # Fallback: manual parsing of points3D.txt
            try:
                pts = _parse_points3d_txt(txt_path)
                if pts is not None and len(pts) > 0:
                    print(f"  Loaded COLMAP point cloud from {txt_path}: {len(pts)} points")
                    return pts
            except Exception as e:
                print(f"  Could not parse {txt_path}: {e}")

    return None


def _parse_points3d_txt(txt_path: Path) -> np.ndarray | None:
    """
    Manually parse COLMAP ``points3D.txt``.

    Format per line (ignoring comment lines starting with #):
        POINT3D_ID  X  Y  Z  R  G  B  ERROR  TRACK[]
    """
    points = []
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 4:
                try:
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    points.append([x, y, z])
                except ValueError:
                    continue
    if not points:
        return None
    pts = np.array(points, dtype=np.float32)
    if len(pts) > 500_000:
        pts = pts[np.random.choice(len(pts), 500_000, replace=False)]
    return pts


def _load_ply_points(ply_path: Path) -> np.ndarray | None:
    """Load 3D points from a .ply file (mesh or point cloud)."""
    if not ply_path.exists():
        return None
    try:
        import trimesh
        obj = trimesh.load(str(ply_path), process=False)
        if hasattr(obj, "vertices"):
            pts = np.array(obj.vertices, dtype=np.float32)
            if len(pts) > 500_000:
                pts = pts[np.random.choice(len(pts), 500_000, replace=False)]
            print(f"  Loaded point cloud from {ply_path}: {len(pts)} points")
            return pts
    except ImportError:
        try:
            from plyfile import PlyData
            ply = PlyData.read(str(ply_path))
            v = ply["vertex"]
            pts = np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float32)
            if len(pts) > 500_000:
                pts = pts[np.random.choice(len(pts), 500_000, replace=False)]
            print(f"  Loaded point cloud from {ply_path}: {len(pts)} points")
            return pts
        except Exception as e:
            print(f"  Could not load {ply_path}: {e}")
    except Exception as e:
        print(f"  Could not load {ply_path}: {e}")
    return None


# ======================================================================
# 3D Visualization with camera frustums (plotly, matplotlib fallback)
# ======================================================================

def visualize_trajectory_3d(
    c2ws: np.ndarray,
    save_path: Path,
    title: str = "",
    point_cloud: np.ndarray | None = None,
    frustum_scale: float = 0.15,
    frustum_every: int = 5,
):
    """
    Create a 3D interactive visualization of the camera trajectory.

    Args:
        c2ws:           (N, 4, 4) camera-to-world matrices.
        save_path:      Path for the output HTML file.
        title:          Title / prompt string shown on the plot.
        point_cloud:    (P, 3) scene point cloud (optional).
        frustum_scale:  Size of each camera frustum wireframe.
        frustum_every:  Draw a frustum every N frames.
    """
    positions = c2ws[:, :3, 3]
    rotations = c2ws[:, :3, :3]

    try:
        import plotly.graph_objects as go
    except ImportError:
        print("    plotly not available – falling back to matplotlib")
        _visualize_matplotlib(positions, save_path, title, point_cloud)
        return

    fig = go.Figure()

    # --- point cloud ---
    if point_cloud is not None:
        pts = point_cloud
        if len(pts) > 30_000:
            pts = pts[np.random.choice(len(pts), 30_000, replace=False)]
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode="markers",
            marker=dict(size=1, color="gray", opacity=0.25),
            name="Point Cloud",
        ))

    # --- trajectory line ---
    fig.add_trace(go.Scatter3d(
        x=positions[:, 0], y=positions[:, 1], z=positions[:, 2],
        mode="lines",
        line=dict(color="red", width=4),
        name="Camera Trajectory",
    ))

    # --- camera frustums ---
    _add_camera_frustums(fig, positions, rotations, frustum_scale, frustum_every)

    # --- start / end markers ---
    fig.add_trace(go.Scatter3d(
        x=[positions[0, 0]], y=[positions[0, 1]], z=[positions[0, 2]],
        mode="markers+text", text=["START"], textposition="top center",
        marker=dict(size=10, color="green", symbol="circle"),
        name="Start",
    ))
    fig.add_trace(go.Scatter3d(
        x=[positions[-1, 0]], y=[positions[-1, 1]], z=[positions[-1, 2]],
        mode="markers+text", text=["END"], textposition="top center",
        marker=dict(size=10, color="orange", symbol="square"),
        name="End",
    ))

    fig.update_layout(
        title=f"ChatCam Trajectory – {title}" if title else "ChatCam Trajectory",
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
                   aspectmode="data"),
        width=1200, height=800,
    )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(save_path))
    print(f"    Saved 3D visualization to {save_path}")


def _add_camera_frustums(
    fig,
    positions: np.ndarray,
    rotations: np.ndarray,
    frustum_scale: float = 0.15,
    every: int = 5,
    aspect_ratio: float = 16 / 9,
):
    """Add camera frustum wireframes to a plotly figure."""
    import plotly.graph_objects as go

    half_w = frustum_scale * 0.5 * aspect_ratio
    half_h = frustum_scale * 0.5
    d = frustum_scale

    # Image-plane corners in camera-local coords (OpenCV: +Z = forward)
    corners_local = np.array([
        [-half_w, -half_h, d],
        [ half_w, -half_h, d],
        [ half_w,  half_h, d],
        [-half_w,  half_h, d],
    ])

    fx, fy, fz = [], [], []
    ux, uy, uz = [], [], []

    for i in range(0, len(positions), every):
        pos = positions[i]
        R = rotations[i]
        corners_w = (R @ corners_local.T).T + pos

        # Edges from center to corners
        for c in range(4):
            fx.extend([pos[0], corners_w[c, 0], None])
            fy.extend([pos[1], corners_w[c, 1], None])
            fz.extend([pos[2], corners_w[c, 2], None])

        # Image-plane rectangle
        for c in range(4):
            cn = (c + 1) % 4
            fx.extend([corners_w[c, 0], corners_w[cn, 0], None])
            fy.extend([corners_w[c, 1], corners_w[cn, 1], None])
            fz.extend([corners_w[c, 2], corners_w[cn, 2], None])

        # Up indicator (camera -Y in OpenCV convention)
        top_center = (corners_w[0] + corners_w[1]) / 2
        up_tip = pos + R @ np.array([0, -half_h * 1.5, d])
        ux.extend([top_center[0], up_tip[0], None])
        uy.extend([top_center[1], up_tip[1], None])
        uz.extend([top_center[2], up_tip[2], None])

    fig.add_trace(go.Scatter3d(
        x=fx, y=fy, z=fz, mode="lines",
        line=dict(color="rgba(0,100,255,0.6)", width=2),
        name="Frustums", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter3d(
        x=ux, y=uy, z=uz, mode="lines",
        line=dict(color="rgba(255,50,50,0.8)", width=3),
        name="Up indicator", hoverinfo="skip",
    ))


def _visualize_matplotlib(
    positions: np.ndarray,
    save_path: Path,
    title: str = "",
    point_cloud: np.ndarray | None = None,
):
    """Fallback matplotlib 3D scatter plot."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("    matplotlib not available for visualization.")
        return

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    if point_cloud is not None:
        pts = point_cloud
        if len(pts) > 5000:
            pts = pts[np.random.choice(len(pts), 5000, replace=False)]
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=0.5, alpha=0.2, c="gray")
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], "r-", lw=2)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    plt.title(title or "Camera Trajectory")

    png_path = str(save_path).replace(".html", ".png")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved visualization to {png_path}")


# ======================================================================
# gsplat Renderer (renders trajectory frames → video)
# ======================================================================

class GsplatTrajectoryRenderer:
    """
    Renders a camera trajectory into a video using a 3DGS model via gsplat.
    """

    def __init__(self, model_path: str, device: str = "cuda",
                 image_width: int = 1920, image_height: int = 1080):
        self.model_path = Path(model_path)
        self.device = device
        self.W = image_width
        self.H = image_height
        self.splat_data = None
        self._load_model()

    # ---- model loading --------------------------------------------------

    def _load_model(self):
        if not self.model_path.exists():
            print(f"    Warning: model not found at {self.model_path}")
            return
        try:
            import torch as _torch
            if self.model_path.suffix == ".ply":
                self._load_ply()
            elif self.model_path.suffix in (".pt", ".ckpt"):
                self._load_checkpoint()
            else:
                print(f"    Unknown model format: {self.model_path.suffix}")
        except Exception as e:
            print(f"    Could not load model: {e}")

    def _load_ply(self):
        import torch as _torch
        try:
            from plyfile import PlyData
            ply = PlyData.read(str(self.model_path))
            v = ply["vertex"]
            means = np.stack([v["x"], v["y"], v["z"]], axis=-1)
            self.splat_data = {
                "means": _torch.tensor(means, dtype=_torch.float32, device=self.device),
            }
            # Optional attributes
            if "opacity" in v.data.dtype.names:
                self.splat_data["opacities"] = _torch.tensor(
                    np.array(v["opacity"]), dtype=_torch.float32, device=self.device)
            scale_names = sorted(n for n in v.data.dtype.names if n.startswith("scale_"))
            if len(scale_names) >= 3:
                self.splat_data["scales"] = _torch.tensor(
                    np.stack([v[s] for s in scale_names[:3]], axis=-1),
                    dtype=_torch.float32, device=self.device)
            rot_names = sorted(n for n in v.data.dtype.names if n.startswith("rot_"))
            if len(rot_names) >= 4:
                self.splat_data["quats"] = _torch.tensor(
                    np.stack([v[r] for r in rot_names[:4]], axis=-1),
                    dtype=_torch.float32, device=self.device)
            sh_dc = sorted(n for n in v.data.dtype.names if n.startswith("f_dc_"))
            if sh_dc:
                self.splat_data["sh_dc"] = _torch.tensor(
                    np.stack([v[s] for s in sh_dc], axis=-1),
                    dtype=_torch.float32, device=self.device)
            has_full = all(k in self.splat_data for k in ("opacities", "scales", "quats"))
            print(f"    Loaded {len(means)} Gaussians ({'full' if has_full else 'basic'} mode)")
        except ImportError:
            import trimesh
            mesh = trimesh.load(str(self.model_path), process=False)
            self.splat_data = {
                "means": _torch.tensor(np.array(mesh.vertices),
                                       dtype=_torch.float32, device=self.device),
            }
            print(f"    Loaded {len(mesh.vertices)} points (basic mode)")

    def _load_checkpoint(self):
        import torch as _torch
        ckpt = _torch.load(str(self.model_path), map_location=self.device, weights_only=False)
        if isinstance(ckpt, dict) and "splats" in ckpt:
            self.splat_data = {
                k: v.to(self.device) if isinstance(v, _torch.Tensor) else v
                for k, v in ckpt["splats"].items()
            }
            n = len(next(iter(self.splat_data.values())))
            print(f"    Loaded gsplat checkpoint: {n} Gaussians")

    # ---- rendering -------------------------------------------------------

    def render_frame(self, c2w: np.ndarray, intrinsics: dict | None = None) -> np.ndarray | None:
        """Render one frame given a 4×4 c2w matrix."""
        if self.splat_data is None:
            return None
        try:
            import torch as _torch
            from gsplat import rasterization

            fx = intrinsics.get("fx", 500.0) if intrinsics else 500.0
            fy = intrinsics.get("fy", 500.0) if intrinsics else 500.0
            cx = intrinsics.get("cx", self.W / 2) if intrinsics else self.W / 2
            cy = intrinsics.get("cy", self.H / 2) if intrinsics else self.H / 2

            Ks = _torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
                               dtype=_torch.float32, device=self.device).unsqueeze(0)
            w2c = np.linalg.inv(c2w.astype(np.float64)).astype(np.float32)
            viewmats = _torch.tensor(w2c, dtype=_torch.float32, device=self.device).unsqueeze(0)

            means = self.splat_data["means"]
            N = len(means)
            quats = self.splat_data.get(
                "quats", _torch.tensor([[1, 0, 0, 0]], device=self.device).expand(N, -1))
            scales = self.splat_data.get(
                "scales", _torch.full((N, 3), -5.0, device=self.device))
            opacities = self.splat_data.get(
                "opacities", _torch.ones(N, device=self.device))

            if "sh_dc" in self.splat_data:
                colors = _torch.sigmoid(self.splat_data["sh_dc"][:, :3].reshape(N, 3))
            elif "colors" in self.splat_data:
                colors = self.splat_data["colors"]
            else:
                colors = _torch.ones(N, 3, device=self.device) * 0.5

            renders, _, _ = rasterization(
                means=means, quats=quats,
                scales=_torch.exp(scales),
                opacities=_torch.sigmoid(opacities),
                colors=colors, viewmats=viewmats, Ks=Ks,
                width=self.W, height=self.H, packed=False,
            )
            rgb = renders[0].clamp(0, 1).detach().cpu().numpy()
            return (rgb * 255).astype(np.uint8)
        except Exception as e:
            print(f"    Render error: {e}")
            return None

    def render_trajectory(
        self, c2ws: np.ndarray, intrinsics: dict | None = None,
        output_path: Path | str = None, fps: int = 24,
    ) -> np.ndarray | None:
        """Render all frames and optionally save as MP4."""
        N = len(c2ws)
        print(f"    Rendering {N} frames...")
        frames = []
        for i in range(N):
            if (i + 1) % 30 == 0 or i == 0:
                print(f"      Frame {i+1}/{N}")
            frame = self.render_frame(c2ws[i], intrinsics)
            if frame is not None:
                frames.append(frame)
            else:
                frames.append(np.zeros((self.H, self.W, 3), dtype=np.uint8))
        frames = np.array(frames)

        if output_path and len(frames) > 0:
            self._save_video(frames, Path(output_path), fps)
        return frames

    # @staticmethod
    # def _save_video(frames: np.ndarray, output_path: Path, fps: int = 24):
    #     output_path.parent.mkdir(parents=True, exist_ok=True)
    #     try:
    #         import cv2
    #         H, W = frames.shape[1], frames.shape[2]
    #         writer = cv2.VideoWriter(
    #             str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    #         for f in frames:
    #             writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    #         writer.release()
    #         print(f"    Video saved: {output_path}")
    #     except ImportError:
    #         try:
    #             import imageio
    #             w = imageio.get_writer(str(output_path), fps=fps, codec="libx264")
    #             for f in frames:
    #                 w.append_data(f)
    #             w.close()
    #             print(f"    Video saved: {output_path}")
    #         except ImportError:
    #             print("    Neither cv2 nor imageio available for video saving.")

    @staticmethod
    def _save_video(frames: np.ndarray, output_path: Path, fps: int = 24):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import cv2
            H, W = frames.shape[1], frames.shape[2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (W, H))
            for f in frames:
                writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            writer.release()

            # Re-encode to H.264 for browser compatibility
            import subprocess, shutil
            h264_path = str(output_path) + ".h264.mp4"
            ret = subprocess.run([
                "ffmpeg", "-y", "-i", str(output_path),
                "-c:v", "libx264", "-crf", "23",
                "-c:a", "aac", "-movflags", "+faststart",
                h264_path
            ], capture_output=True)
            if ret.returncode == 0:
                shutil.move(h264_path, str(output_path))
                print(f"    Video saved: {output_path} (H.264)")
            else:
                print(f"    Video saved: {output_path} (mp4v)")
                print(f"      H.264 re-encode failed. Run manually:")
                print(f"      ffmpeg -y -i \"{output_path}\" -c:v libx264 -crf 23 -movflags +faststart \"{output_path}\"")
        except ImportError:
            try:
                import imageio
                w = imageio.get_writer(
                    str(output_path), fps=fps, codec="libx264",
                    output_params=["-pix_fmt", "yuv420p"],
                )
                for f in frames:
                    w.append_data(f)
                w.close()
                print(f"    Video saved: {output_path} (H.264 via imageio)")
            except ImportError:
                print("    Neither cv2 nor imageio available for video saving.")


# Noise words stripped from descriptions during anchor–trajectory matching.
# These are camera-motion verbs, prepositions, and other function words that
# carry no object-identity information.
_MATCH_NOISE_WORDS = {
    "pan", "tilt", "zoom", "dolly", "orbit", "move", "rotate",
    "turn", "pull", "push", "sweep", "glide", "track", "crane",
    "forward", "backward", "left", "right", "up", "down",
    "upwards", "downwards", "in", "out", "around", "through",
    "to", "from", "the", "a", "an", "of", "on", "at", "with",
    "and", "then", "into", "while", "increasing", "decreasing",
    "focal", "length", "u-turn", "s-shaped", "slowly", "quickly",
    "close", "make", "shot", "start", "starting", "begin",
    "maintain", "size", "frame",
}


class ChatCam:
    """
    Main ChatCam system that orchestrates camera trajectory generation
    through conversational AI.
    
    Components:
        - LLM Agent: Parses instructions, plans tool calls
        - CineGPT (GenDoP): Text-conditioned trajectory generation
        - Anchor Determinator: CLIP-based anchor selection + refinement
        - Trajectory Composer: Combines sub-trajectories with anchors
    """

    def __init__(
        self,
        scene_dir: str,
        cinegpt_config: dict = None,
        openai_api_key: str = None,
        llm_model: str = "gpt-4o",
        clip_model: str = "ViT-B/32",
        device: str = "cuda",
        point_cloud_path: str = None,
        gsplat_model_path: str = None,
        render: bool = False,
        refine_anchors: bool = False,
        gendop_convention: str = "opengl",
        fps: int = 24,
    ):
        self.scene_dir = Path(scene_dir)
        self.device = device if (torch is not None and torch.cuda.is_available()) else "cpu"
        self.render_enabled = render
        self.fps = fps

        # GenDoP → COLMAP camera convention conversion.
        #
        # GenDoP camera axes:  X=right, Y=up,   Z=backward (looks along -Z)
        # COLMAP/OpenCV axes:  X=right, Y=down, Z=forward  (looks along +Z)
        #
        # Only the ROTATION part of c2w needs conversion (flip Y and Z
        # columns).  The TRANSLATION (camera position in world space) is
        # convention-independent and must not be modified.
        #
        # We store a 3×3 matrix that is RIGHT-multiplied onto the rotation:
        #     colmap_R = gendop_R @ conv_rot      (flip cols 1 and 2)
        #     colmap_t = gendop_t                  (unchanged)
        self._gendop_convert = (gendop_convention != "none")
        print(f"  GenDoP convention conversion: {'enabled' if self._gendop_convert else 'disabled'}")

        print("=" * 60)
        print("ChatCam: Empowering Camera Control through Conversational AI")
        print("=" * 60)

        # Initialize LLM Agent
        print("\n[Init] LLM Agent...")
        self.agent = ChatCamAgent(
            api_key=openai_api_key or os.environ.get("OPENAI_API_KEY"),
            model=llm_model,
        )

        # Initialize CineGPT (GenDoP wrapper)
        print("[Init] CineGPT (GenDoP)...")
        self.cinegpt = CineGPTWrapper(
            config=cinegpt_config,
            device=self.device,
        )

        # Initialize Anchor Determinator
        print("[Init] Anchor Determinator...")
        self.anchor_det = AnchorDeterminator(
            scene_dir=scene_dir,
            clip_model=clip_model,
            device=self.device,
        )

        # Initialize Trajectory Composer
        self.composer = TrajectoryComposer()

        # Load scene intrinsics from COLMAP cameras
        print("[Init] Scene intrinsics...")
        self.scene_intrinsics = self._load_scene_intrinsics()
        if self.scene_intrinsics:
            print(f"    fx={self.scene_intrinsics['fx']:.1f} "
                  f"fy={self.scene_intrinsics['fy']:.1f} "
                  f"{self.scene_intrinsics['w']}x{self.scene_intrinsics['h']}")
        else:
            print("    Warning: Could not load scene intrinsics, using defaults")

        # Load scene point cloud for visualization
        print("[Init] Scene geometry...")
        self.point_cloud = load_point_cloud(self.scene_dir, point_cloud_path)

        # Initialize gsplat renderer (lazy — only if rendering requested)
        self.renderer = None
        self.gsplat_model_path = gsplat_model_path
        if render and gsplat_model_path:
            print("[Init] gsplat Renderer...")
            intr = self.scene_intrinsics or {}
            self.renderer = GsplatTrajectoryRenderer(
                model_path=gsplat_model_path,
                device=self.device,
                image_width=intr.get("w", 1920),
                image_height=intr.get("h", 1080),
            )

        # Connect differentiable render function for anchor refinement
        if refine_anchors and self.renderer is not None and self.renderer.splat_data is not None:
            print("[Init] Connecting gsplat to Anchor Refinement...")
            render_fn = self._make_differentiable_render_fn()
            self.anchor_det.set_render_function(render_fn)
        elif refine_anchors and self.renderer is None:
            print("[Init] Warning: --refine_anchors requires --render + --gsplat_model_path")

        # Conversation history for multi-turn
        self.conversation_history = []

    def _load_scene_intrinsics(self) -> dict | None:
        """
        Load camera intrinsics from COLMAP cameras in sparse/0.
        Returns the intrinsics of the first camera (most scenes use a single camera).
        """
        sparse_dir = self.scene_dir / "sparse" / "0"
        if not sparse_dir.exists():
            return None

        try:
            from scene_reconstruction import read_cameras_binary, read_cameras_text

            cameras_bin = sparse_dir / "cameras.bin"
            cameras_txt = sparse_dir / "cameras.txt"

            if cameras_bin.exists():
                cameras = read_cameras_binary(str(cameras_bin))
            elif cameras_txt.exists():
                cameras = read_cameras_text(str(cameras_txt))
            else:
                return None

            if not cameras:
                return None

            cam = list(cameras.values())[0]
            params = cam.params
            model = cam.model

            if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
                return {
                    "fx": float(params[0]), "fy": float(params[0]),
                    "cx": float(params[1]), "cy": float(params[2]),
                    "w": int(cam.width), "h": int(cam.height),
                }
            elif model in ("PINHOLE", "OPENCV"):
                return {
                    "fx": float(params[0]), "fy": float(params[1]),
                    "cx": float(params[2]), "cy": float(params[3]),
                    "w": int(cam.width), "h": int(cam.height),
                }
            else:
                return {
                    "fx": float(params[0]), "fy": float(params[0]),
                    "cx": cam.width / 2.0, "cy": cam.height / 2.0,
                    "w": int(cam.width), "h": int(cam.height),
                }

        except Exception as e:
            print(f"    Could not load COLMAP cameras: {e}")
            return None

    def _make_differentiable_render_fn(self):
        """
        Create a differentiable render function for anchor refinement.

        Returns a callable ``render_fn(c2w_tensor, intrinsics)`` that:
          - Takes a *torch.Tensor* c2w (4×4, with grad) and intrinsics dict
          - Renders via gsplat rasterization (differentiable)
          - Returns a torch.Tensor (H, W, 3) in [0, 1] with gradients

        The gradient chain is:
            camera params → c2w → w2c (viewmat) → gsplat rasterization → pixels
        """
        renderer = self.renderer
        device = self.device
        scene_intr = self.scene_intrinsics

        def render_fn(c2w_tensor, intrinsics=None):
            """Differentiable gsplat render at a single camera pose."""
            try:
                import torch as _t
                from gsplat import rasterization

                splat = renderer.splat_data
                if splat is None:
                    return None

                intr = intrinsics or scene_intr or {}
                W = intr.get("w", renderer.W)
                H = intr.get("h", renderer.H)
                fx = intr.get("fx", 500.0)
                fy = intr.get("fy", 500.0)
                cx = intr.get("cx", W / 2.0)
                cy = intr.get("cy", H / 2.0)

                Ks = _t.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
                               dtype=_t.float32, device=device).unsqueeze(0)

                # Differentiable w2c from c2w.
                # torch.linalg.inv is differentiable for 4×4 matrices.
                c2w_f = c2w_tensor.float()
                w2c = _t.linalg.inv(c2w_f).unsqueeze(0)  # (1, 4, 4)

                means = splat["means"]
                N = len(means)

                quats = splat.get(
                    "quats",
                    _t.tensor([[1, 0, 0, 0]], device=device).expand(N, -1).float(),
                )
                scales = splat.get(
                    "scales",
                    _t.full((N, 3), -5.0, device=device),
                )
                opacities = splat.get(
                    "opacities",
                    _t.ones(N, device=device),
                )

                if "sh_dc" in splat:
                    colors = _t.sigmoid(splat["sh_dc"][:, :3].reshape(N, 3))
                elif "colors" in splat:
                    colors = splat["colors"]
                else:
                    colors = _t.ones(N, 3, device=device) * 0.5

                renders, _, _ = rasterization(
                    means=means,
                    quats=quats,
                    scales=_t.exp(scales),
                    opacities=_t.sigmoid(opacities),
                    colors=colors,
                    viewmats=w2c,
                    Ks=Ks,
                    width=W,
                    height=H,
                    packed=False,
                )

                # (1, H, W, 3) → (H, W, 3), clamped to [0, 1]
                return renders[0].clamp(0, 1)

            except Exception as e:
                print(f"    [Differentiable Render] Error: {e}")
                return None

        return render_fn

    def process_instruction(self, instruction: str, output_dir: str = "./output") -> dict:
        """
        Process a single camera operation instruction end-to-end.
        
        Args:
            instruction: Natural language camera instruction.
            output_dir: Directory to save outputs.
            
        Returns:
            Dict with trajectory, rendered frames info, and agent plan.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: LLM Agent parses instruction
        print(f"\n{'='*60}")
        print(f"User: {instruction}")
        print(f"{'='*60}")

        plan = self.agent.parse_instruction(
            instruction,
            conversation_history=self.conversation_history,
        )
        print(f"\n[Agent] Observation: {plan['observation']}")
        print(f"[Agent] Reasoning: {plan['reasoning']}")
        print(f"[Agent] Plan:")
        for step in plan["plan_steps"]:
            print(f"  {step}")

        # Step 2: Execute the plan - call tools as specified
        tool_results = self._execute_plan(plan)

        # ── Save anchor poses for eval.py coverage metric ──
        anchor_results = [r for r in tool_results if r["type"] == "anchor"]
        if anchor_results:
            anchors_out = {"anchors": [], "method": "chatcam_clip"}
            for i, ar in enumerate(anchor_results):
                ad = ar["data"]  # dict from AnchorDeterminator.determine_anchor()
                c2w = ad.get("c2w")
                entry = {
                    "anchor_idx": i,
                    "text": ar.get("description", ad.get("text_prompt", "")),
                    "similarity": ad.get("similarity", -1.0),
                    "image_name": ad.get("image_name", ""),
                }
                if c2w is not None:
                    c2w_np = np.array(c2w)
                    entry["c2w"] = c2w_np.tolist()
                    if c2w_np.shape == (4, 4):
                        entry["position"] = c2w_np[:3, 3].tolist()
                if ad.get("intrinsics"):
                    entry["intrinsics"] = ad["intrinsics"]
                anchors_out["anchors"].append(entry)
 
            anchors_file = output_dir / "anchors.json"
            with open(anchors_file, "w") as f:
                json.dump(anchors_out, f, indent=2, default=str)
            print(f"  Saved {len(anchors_out['anchors'])} anchor poses -> {anchors_file}")

        # Step 3: Compose final trajectory (pass plan for interleaving)
        final_trajectory = self._compose_trajectory(tool_results, plan)

        # Step 4: Save outputs
        traj_path = output_dir / "trajectory.json"
        save_trajectory_json(final_trajectory, traj_path)

        c2ws = final_trajectory.get("c2ws")

        # Use scene intrinsics from COLMAP (not GenDoP's internal intrinsics)
        intrinsics = self.scene_intrinsics or final_trajectory.get("intrinsics")

        # 2D overview (existing)
        vis_path_2d = output_dir / "trajectory_vis.png"
        if c2ws is not None:
            visualize_c2ws(c2ws, vis_path_2d, instruction)

        # 3D interactive visualization with point cloud and frustums
        vis_path_3d = output_dir / "trajectory_3d.html"
        if c2ws is not None:
            print("\n[Visualization]")
            visualize_trajectory_3d(
                c2ws=np.array(c2ws, dtype=np.float64),
                save_path=vis_path_3d,
                title=instruction,
                point_cloud=self.point_cloud,
            )

        # Export camera path for external renderers
        ns_path = output_dir / "camera_path.json"
        if c2ws is not None and intrinsics:
            c2ws_to_nerfstudio_path(
                c2ws,
                intrinsics,
                ns_path,
            )

        # Optional: render video via gsplat
        if c2ws is not None and self.render_enabled and self.renderer is not None:
            print("\n[Rendering]")
            video_path = output_dir / "rendered_video.mp4"
            self.renderer.render_trajectory(
                c2ws=np.array(c2ws, dtype=np.float32),
                intrinsics=intrinsics,
                output_path=video_path,
                fps=self.fps,
            )

        # Update conversation history
        self.conversation_history.append({
            "role": "user",
            "content": instruction,
        })
        self.conversation_history.append({
            "role": "assistant",
            "content": json.dumps(plan, indent=2),
        })

        return {
            "plan": plan,
            "tool_results": tool_results,
            "trajectory": final_trajectory,
            "output_dir": str(output_dir),
        }

    def _execute_plan(self, plan: dict) -> list:
        """
        Execute the agent's plan by calling CineGPT and Anchor Determinator.
        
        Parses the plan steps and dispatches to appropriate tools.
        """
        results = []

        for step in plan.get("tool_calls", []):
            tool_name = step["tool"]
            args = step["args"]

            if tool_name == "infer_cinegpt":
                print(f"\n  [CineGPT] Generating trajectory: \"{args['traj_description']}\"")
                traj = self.cinegpt.generate(
                    text=args["traj_description"],
                    image_path=args.get("image_path"),
                    depth_path=args.get("depth_path"),
                )

                # Convert GenDoP c2ws to match COLMAP/OpenCV camera convention.
                #
                # GenDoP: X=right, Y=up, Z=backward  (OpenGL-like)
                # COLMAP: X=right, Y=down, Z=forward  (OpenCV)
                #
                # Conversion: negate Y and Z columns of the 3×3 rotation.
                # Translation (world-space camera position) is unchanged.
                if traj.get("c2ws") is not None and self._gendop_convert:
                    conv_rot = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
                    converted = []
                    for c2w in traj["c2ws"]:
                        c2w_new = c2w.copy()
                        c2w_new[:3, :3] = c2w[:3, :3] @ conv_rot  # flip rotation cols only
                        # c2w_new[:3, 3] unchanged (world-space position)
                        converted.append(c2w_new)
                    traj["c2ws"] = np.array(converted)

                results.append({
                    "type": "trajectory",
                    "description": args["traj_description"],
                    "data": traj,
                })

            elif tool_name == "get_anchor":
                print(f"\n  [Anchor] Finding anchor: \"{args['anchor_description']}\"")
                anchor = self.anchor_det.determine_anchor(
                    text_prompt=args["anchor_description"],
                    refine=args.get("refine", True),
                )
                results.append({
                    "type": "anchor",
                    "description": args["anchor_description"],
                    "data": anchor,
                })

            elif tool_name == "traj_compose":
                # This is handled in _compose_trajectory
                results.append({
                    "type": "compose_instruction",
                    "data": args.get("compose", []),
                })

        return results

    # ==================================================================
    # Trajectory composition with interleaved element reconstruction
    # ==================================================================

    def _compose_trajectory(self, tool_results: list, plan: dict = None) -> dict:
        """
        Compose final trajectory from tool results (anchors + trajectories).

        Reconstructs the interleaved anchor/trajectory sequence by:
          1. Parsing the numbered plan steps to recover the ordered list of
             anchor and CineGPT calls.
          2. For each anchor, matching its description against trajectory
             descriptions to determine which trajectory it is adjacent to.
          3. Building an ``elements`` list in the correct interleaved order
             and passing it to ``TrajectoryComposer.compose()``.
        """
        # Collect generated data, preserving duplicates for trajectories.
        # tool_results are in execution order; anchor_map uses description
        # as key (anchors are always unique), but traj_list preserves order
        # since CineGPT may be called with identical descriptions
        # (e.g. "pan right" twice).
        anchor_map = {}          # description -> data dict (has 'c2w')
        traj_list = []           # list of (description, data) in execution order

        for result in tool_results:
            if result["type"] == "anchor":
                anchor_map[result["description"]] = result["data"]
            elif result["type"] == "trajectory":
                traj_list.append((result["description"], result["data"]))

        if not anchor_map and not traj_list:
            print("  [Warning] No trajectories or anchors generated")
            return {"c2ws": None, "intrinsics": None, "num_frames": 0}

        # If we only have anchors (no trajectories), pass them directly
        # to the composer which will interpolate between them.
        if not traj_list:
            anchor_elements = [anchor_map[d] for d in anchor_map.values()
                               if isinstance(d, dict)] if False else \
                              list(anchor_map.values())
            # Build a static trajectory or interpolation
            # (the composer expects at least one trajectory between anchors)
            print("  [Warning] No trajectories generated; returning anchor poses only")
            anchor_c2ws = np.array([a["c2w"] for a in anchor_map.values()])
            intrinsics = None
            for a in anchor_map.values():
                if "intrinsics" in a:
                    intrinsics = a["intrinsics"]
                    break
            return {
                "c2ws": anchor_c2ws,
                "intrinsics": intrinsics,
                "num_frames": len(anchor_c2ws),
            }

        # Parse plan text into ordered calls
        ordered_calls = self._parse_plan_to_ordered_calls(plan)

        anchor_descs = [c["description"] for c in ordered_calls
                        if c["tool"] == "get_anchor"]
        traj_descs = [c["description"] for c in ordered_calls
                      if c["tool"] == "infer_cinegpt"]

        # Build traj_map that maps (description, occurrence_index) -> data
        # so that duplicate descriptions like "pan right" get separate entries.
        # We match plan-order traj_descs to execution-order traj_list by
        # consuming each traj_list entry in order for each matching description.
        traj_queue = {}  # description -> list of data dicts (FIFO)
        for desc, data in traj_list:
            traj_queue.setdefault(desc, []).append(data)

        # traj_map_indexed: index into traj_descs -> data
        traj_map_indexed = {}
        for t, tdesc in enumerate(traj_descs):
            if tdesc in traj_queue and traj_queue[tdesc]:
                traj_map_indexed[t] = traj_queue[tdesc].pop(0)

        # Build the interleaved elements list
        elements = self._build_interleaved_elements(
            anchor_descs, traj_descs, anchor_map, traj_map_indexed,
        )

        if not elements:
            print("  [Warning] Could not reconstruct interleaved sequence")
            # Fallback: concatenate trajectories in plan order
            all_trajs = [traj_map_indexed[t] for t in sorted(traj_map_indexed)]
            if all_trajs:
                return self.composer.compose(all_trajs)
            return {"c2ws": None, "intrinsics": None, "num_frames": 0}

        # Log the reconstructed sequence
        print("\n  [Compose] Reconstructed element sequence:")
        for i, elem in enumerate(elements):
            if "c2w" in elem:
                print(f"    [{i}] ANCHOR")
            elif "c2ws" in elem:
                print(f"    [{i}] TRAJ ({len(elem['c2ws'])} frames)")

        composed = self.composer.compose(elements)

        num = composed.get("num_frames", 0)
        print(f"    Composed trajectory: {num} frames")
        return composed

    # ------------------------------------------------------------------

    @staticmethod
    def _parse_plan_to_ordered_calls(plan: dict) -> list:
        """
        Parse the numbered plan text and return tool calls **in the order
        they appear**, preserving interleaving information.

        Each returned dict: ``{"tool": "get_anchor"|"infer_cinegpt", "description": str}``
        """
        ordered = []
        if plan is None:
            return ordered

        plan_steps = plan.get("plan_steps", [])

        for step in plan_steps:
            m = re.search(
                r"(?:Call|call)\s+Anchor\s+Determinator\s+with\s+['\"](.+?)['\"]",
                step, re.IGNORECASE,
            )
            if m:
                ordered.append({"tool": "get_anchor", "description": m.group(1)})
                continue

            m = re.search(
                r"(?:Call|call)\s+CineGPT\s+with\s+['\"](.+?)['\"]",
                step, re.IGNORECASE,
            )
            if m:
                ordered.append({"tool": "infer_cinegpt", "description": m.group(1)})
                continue

        return ordered

    # ------------------------------------------------------------------

    @staticmethod
    def _anchor_match_score(anchor_desc: str, traj_desc: str) -> float:
        """
        Compute a matching score between an anchor description and a
        trajectory description based on noun-token overlap.

        Returns 0 if there is no meaningful overlap.
        """
        adesc_lower = anchor_desc.lower()
        traj_lower = traj_desc.lower()

        anchor_tokens = set(re.findall(r"[a-z]+", adesc_lower)) - _MATCH_NOISE_WORDS
        traj_tokens = set(re.findall(r"[a-z]+", traj_lower)) - _MATCH_NOISE_WORDS

        if not anchor_tokens:
            return 0.0

        overlap = anchor_tokens & traj_tokens
        if not overlap:
            return 0.0

        # Base: fraction of anchor tokens found in trajectory
        score = len(overlap) / len(anchor_tokens)

        # Bonus for per-token substring presence
        for tok in anchor_tokens:
            if tok in traj_lower:
                score += 0.1

        return score

    # ------------------------------------------------------------------

    def _build_interleaved_elements(
        self,
        anchor_descs: list,
        traj_descs: list,
        anchor_map: dict,
        traj_map: dict,
    ) -> list:
        """
        Reconstruct the interleaved ``[anchor, traj, anchor, traj, ...]``
        sequence from the plan's anchor and trajectory descriptions.

        **Algorithm**

        The LLM agent lists anchors in *path order* (the order they appear
        along the camera trajectory) and trajectories in *execution order*.
        A trajectory whose description mentions an anchor's object name
        (e.g. "move to the colorful bulldozer") *ends* at that anchor.

        1. For every anchor, find the trajectory whose description best
           matches it.  That trajectory is the one that *arrives at* the
           anchor, so the anchor is emitted right after it.
        2. Unmatched anchors (no trajectory mentions their object) are
           placed by interpolation:
           - Before the first matched anchor → position −1 (before all
             trajectories; these are pure starting waypoints).
           - After the last matched anchor → same position as the last
             matched anchor.
           - Between two matched anchors → linearly interpolated.
        3. Walk through trajectory indices 0 … M−1, emitting each
           trajectory and any anchors that are positioned at that index.

        **No-adjacency guarantee**: At most one anchor may occupy each
        "slot" (position −1 or after trajectory *t*).  When more anchors
        than available slots exist, lower-priority anchors are dropped to
        prevent adjacent anchors in the output.  The first anchor, the
        last anchor, and textually-matched anchors are kept preferentially.
        """
        K = len(anchor_descs)
        M = len(traj_descs)

        if K == 0:
            return [traj_map[t] for t in range(M) if t in traj_map]
        if M == 0:
            return [anchor_map[d] for d in anchor_descs if d in anchor_map]

        # --- Step 1: for each anchor, find its best-matching trajectory ---
        # anchor_pos[a] = trajectory index after which anchor a is emitted.
        #   -1  → before the first trajectory
        #   None → not yet assigned
        anchor_pos = [None] * K

        anchor_best_traj = [None] * K
        anchor_best_score = [0.0] * K
        for a, adesc in enumerate(anchor_descs):
            for t, tdesc in enumerate(traj_descs):
                sc = self._anchor_match_score(adesc, tdesc)
                if sc > anchor_best_score[a]:
                    anchor_best_score[a] = sc
                    anchor_best_traj[a] = t

        # Resolve conflicts: if multiple anchors match the same trajectory,
        # the one with the higher score wins.
        traj_claimed_by = {}   # traj_idx -> (anchor_idx, score)
        for a in range(K):
            if anchor_best_score[a] < 0.3:
                continue
            t = anchor_best_traj[a]
            if t not in traj_claimed_by or anchor_best_score[a] > traj_claimed_by[t][1]:
                if t in traj_claimed_by:
                    prev_a = traj_claimed_by[t][0]
                    anchor_best_traj[prev_a] = None
                    anchor_pos[prev_a] = None
                traj_claimed_by[t] = (a, anchor_best_score[a])
                anchor_pos[a] = t
            else:
                anchor_best_traj[a] = None

        # --- Step 2: fill in unmatched anchors ---
        # Available slots: -1, 0, 1, ..., M-1 (total M+1 slots).
        # K anchors need K distinct slots.  When K <= M+1, this is always
        # possible without dropping any anchor.
        posts = [(a, anchor_pos[a]) for a in range(K) if anchor_pos[a] is not None]

        if not posts:
            # No textual matches at all → distribute K anchors evenly.
            # First anchor is the starting waypoint (slot -1).
            # Remaining K-1 anchors spread across slots 0 .. M-1.
            anchor_pos[0] = -1
            if K > 1:
                rest_slots = np.linspace(0, M - 1, K - 1)
                for i, a in enumerate(range(1, K)):
                    anchor_pos[a] = int(round(rest_slots[i]))
        else:
            # Use matched anchors as fixed "posts" and fill unmatched
            # anchors into the gaps between them.

            # Before first post: spread into slots from -1 to post_pos-1
            first_post_a, first_post_pos = posts[0]
            unmatched_before = [a for a in range(0, first_post_a)
                                if anchor_pos[a] is None]
            if unmatched_before:
                avail_start = -1
                avail_end = first_post_pos - 1  # one slot before the first post
                if avail_end < avail_start:
                    avail_end = avail_start
                n = len(unmatched_before)
                if n == 1:
                    positions = [avail_start]
                else:
                    positions = np.linspace(avail_start, avail_end, n)
                for i, a in enumerate(unmatched_before):
                    anchor_pos[a] = int(round(positions[i]))

            # After last post: spread into slots from post_pos+1 to M-1
            last_post_a, last_post_pos = posts[-1]
            unmatched_after = [a for a in range(last_post_a + 1, K)
                               if anchor_pos[a] is None]
            if unmatched_after:
                avail_start = last_post_pos + 1
                avail_end = M - 1
                if avail_start > avail_end:
                    avail_start = avail_end
                n = len(unmatched_after)
                if n == 1:
                    positions = [avail_end]
                else:
                    positions = np.linspace(avail_start, avail_end, n)
                for i, a in enumerate(unmatched_after):
                    anchor_pos[a] = int(round(positions[i]))

            # Between consecutive posts: spread into the gap
            for pi in range(len(posts) - 1):
                a_start, pos_start = posts[pi]
                a_end, pos_end = posts[pi + 1]
                unmatched = [a for a in range(a_start + 1, a_end)
                             if anchor_pos[a] is None]
                if not unmatched:
                    continue
                # Available slots: pos_start+1 ... pos_end-1
                gap_start = pos_start + 1
                gap_end = pos_end - 1
                n = len(unmatched)
                if gap_end < gap_start:
                    # No room — squeeze next to post_start
                    for a in unmatched:
                        anchor_pos[a] = pos_start
                else:
                    if n == 1:
                        positions = [(gap_start + gap_end) / 2]
                    else:
                        positions = np.linspace(gap_start, gap_end, n)
                    for i, a in enumerate(unmatched):
                        anchor_pos[a] = int(round(positions[i]))

        # Ensure monotonicity (anchors are in path order)
        for a in range(1, K):
            if anchor_pos[a] < anchor_pos[a - 1]:
                anchor_pos[a] = anchor_pos[a - 1]

        # --- Step 2b: enforce at most one anchor per slot ---
        # Available slots are: -1, 0, 1, ..., M-1  (total M+1 slots).
        # If multiple anchors map to the same slot, keep only the one
        # with highest priority.  Priority (descending):
        #   1. Textually matched anchors
        #   2. First anchor (a == 0) and last anchor (a == K-1)
        #   3. Others (by anchor index, prefer earlier)
        #
        # Dropped anchors are logged but do not appear in the output.
        slot_occupant = {}   # slot -> anchor_idx (winner so far)
        dropped = set()

        def _anchor_priority(a_idx):
            """Higher = more important to keep."""
            prio = 0
            if anchor_pos[a_idx] is not None and a_idx in {
                c[0] for c in traj_claimed_by.values()
            }:
                prio += 100          # textually matched
            if a_idx == 0:
                prio += 50           # first anchor (starting point)
            if a_idx == K - 1:
                prio += 50           # last anchor (ending point)
            prio -= a_idx * 0.01     # tie-break: earlier wins
            return prio

        for a in range(K):
            slot = anchor_pos[a]
            if slot in slot_occupant:
                existing = slot_occupant[slot]
                if _anchor_priority(a) > _anchor_priority(existing):
                    dropped.add(existing)
                    slot_occupant[slot] = a
                else:
                    dropped.add(a)
            else:
                slot_occupant[slot] = a

        if dropped:
            dropped_names = [anchor_descs[a] for a in sorted(dropped)]
            print(f"    [Compose] Warning: dropped {len(dropped)} anchor(s) "
                  f"to avoid adjacency: {dropped_names}")

        # --- Step 3: assemble output ---
        elements = []

        # Anchors at position −1 (before the first trajectory)
        for a in range(K):
            if anchor_pos[a] == -1 and a not in dropped:
                adata = anchor_map.get(anchor_descs[a])
                if adata is not None:
                    elements.append(adata)

        for t in range(M):
            tdata = traj_map.get(t)
            if tdata is not None:
                elements.append(tdata)

            # Anchors positioned after this trajectory
            for a in range(K):
                if anchor_pos[a] == t and a not in dropped:
                    adata = anchor_map.get(anchor_descs[a])
                    if adata is not None:
                        elements.append(adata)

        return elements

    # ==================================================================

    def interactive_session(self, output_dir: str = "./output"):
        """Run an interactive multi-turn conversation session."""
        print("\n" + "=" * 60)
        print("ChatCam Interactive Session")
        print("Type 'quit' to exit, 'reset' to clear history")
        print("=" * 60)

        turn = 0
        while True:
            try:
                instruction = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if instruction.lower() in ("quit", "exit", "q"):
                break
            if instruction.lower() == "reset":
                self.conversation_history = []
                print("  [System] Conversation history cleared.")
                continue
            if not instruction:
                continue

            turn += 1
            turn_dir = os.path.join(output_dir, f"turn_{turn:03d}")

            result = self.process_instruction(instruction, turn_dir)
            print(f"\n  [ChatCam] Trajectory saved to {turn_dir}/")


def main():
    parser = argparse.ArgumentParser(description="ChatCam: Camera Control through Conversational AI")

    scene_name = "0cf2e9402d" # [09c1414f1b, 0f25f24a4f, 0eba3981c9, 1cefb55d50, 0cf2e9402d, 0d2ee665be, 5f99900f09, 21d970d8de, 6115eddb86, 0a7cc12c0e, 00a231a370, 0b031f3119, 0dce89ab21, 00dd871005, 0e75f3c4d9, 0f0191b10b, 1b75758486, 1bb93d185e, 1c7a683c92, 4e0b8cbd33]
    data_dir = "data/ScanNetpp/scenes"
    parser.add_argument("--scene_dir", type=str, default=f"{data_dir}/{scene_name}/dslr",
                        help="Path to 3D scene directory (images + COLMAP)")
    parser.add_argument("--instruction", type=str, default="Close up of the refrigerator, pan to the sink, zoom in to wash machine, pan to the range hood.", #"Move right of the sink, orbit left to the computer screen, move forward to the board, zoom in to the red sofa.", #"Close up of the refrigerator, pan to the sink, zoom in to TV, pan to the sofa.", #"Close up of the refrigerator, pan to the sink, zoom in to wash machine, pan to the range hood.",
                        help="Single camera instruction to process")
    parser.add_argument("--interactive", action="store_true",
                        help="Run interactive multi-turn conversation")
    parser.add_argument("--output_dir", type=str, default=f"outputs/baselines/ChatCam_GenDoP/{scene_name}",
                        help="Output directory")

    # Model configs
    parser.add_argument("--llm_model", type=str, default="gpt-4.1",
                        help="LLM model for agent (gpt-4, gpt-3.5-turbo)")
    parser.add_argument("--clip_model", type=str, default="ViT-B/32",
                        help="CLIP model for anchor determination")
    parser.add_argument("--cinegpt_resume", type=str, default="third_party/GenDoP/checkpoints/text_directorial.safetensors",
                        help="Path to GenDoP/CineGPT checkpoint")
    parser.add_argument("--cinegpt_cond_mode", type=str, default="text",
                        choices=["text", "image", "image+text", "depth+image+text"],
                        help="CineGPT conditioning mode")

    # Scene geometry & rendering
    parser.add_argument("--point_cloud_path", type=str, default=None,
                        help="Path to .ply file (point cloud or mesh) for visualization")
    parser.add_argument("--gsplat_model_path", type=str, default=f"{data_dir}/{scene_name}/dslr/ply/point_cloud.ply",
                        help="Path to trained 3DGS model (.ply or .pt) for rendering")
    parser.add_argument("--render", action="store_false",
                        help="Render video from trajectory using gsplat")
    parser.add_argument("--refine_anchors", action="store_true",
                        help="Refine anchor poses via differentiable rendering + CLIP (Eq. 5-6). "
                             "Requires --render and --gsplat_model_path.")
    parser.add_argument("--no_gendop_convert", action="store_true",
                        help="Disable GenDoP→COLMAP camera convention conversion (for debugging)")
    parser.add_argument("--fps", type=int, default=24,
                        help="Output video frame rate")

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--openai_api_key", type=str,
                        default=os.environ.get("OPENAI_API_KEY"))

    args = parser.parse_args()

    cinegpt_config = {
        "resume": args.cinegpt_resume,
        "cond_mode": args.cinegpt_cond_mode,
    }

    chatcam = ChatCam(
        scene_dir=args.scene_dir,
        cinegpt_config=cinegpt_config,
        llm_model=args.llm_model,
        openai_api_key=args.openai_api_key,
        clip_model=args.clip_model,
        device=args.device,
        point_cloud_path=args.point_cloud_path,
        gsplat_model_path=args.gsplat_model_path,
        render=args.render,
        refine_anchors=args.refine_anchors,
        gendop_convention="none" if args.no_gendop_convert else "opengl",
        fps=args.fps,
    )

    if args.interactive:
        chatcam.interactive_session(args.output_dir)
    elif args.instruction:
        chatcam.process_instruction(args.instruction, args.output_dir)
    else:
        print("Please provide --instruction or --interactive flag.")


if __name__ == "__main__":
    main()