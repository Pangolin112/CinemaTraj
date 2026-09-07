"""
Trajectory Rendering Module
=============================
Renders novel views along a camera trajectory using a trained 3DGS model (gsplat).
Produces a video output by rendering each frame and compositing into a video.

No nerfstudio dependency - uses gsplat directly for Gaussian Splatting rendering.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple


class GsplatRenderer:
    """
    Renders a camera trajectory into a video using a trained 3DGS model via gsplat.

    Loads a .ply checkpoint (3DGS point cloud with Gaussian attributes)
    and renders novel views at each camera pose along the trajectory.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        image_width: int = 1920,
        image_height: int = 1080,
    ):
        """
        Args:
            model_path: Path to trained 3DGS model. Can be:
                - A .ply file (exported Gaussian splat)
                - A .pt/.ckpt checkpoint from gsplat training
                - A directory containing point_cloud.ply or model checkpoint
            device: CUDA device.
            image_width: Render width.
            image_height: Render height.
        """
        self.model_path = Path(model_path)
        self.device = device
        self.image_width = image_width
        self.image_height = image_height
        self.splat_data = None
        self._load_model()

    def _load_model(self):
        """Load 3DGS model from checkpoint or .ply file."""
        try:
            import torch

            model_path = self.model_path

            # Find the actual model file
            if model_path.is_dir():
                # Search for checkpoint or ply
                candidates = [
                    model_path / "point_cloud.ply",
                    model_path / "model.pt",
                    model_path / "ckpt" / "model.pt",
                ]
                # Also search for gsplat checkpoints
                for p in model_path.rglob("*.pt"):
                    candidates.append(p)
                for p in model_path.rglob("*.ply"):
                    candidates.append(p)

                for c in candidates:
                    if c.exists():
                        model_path = c
                        break

            if not model_path.exists():
                print(f"    Warning: Model not found at {self.model_path}")
                return

            if model_path.suffix == ".ply":
                self._load_ply(model_path)
            elif model_path.suffix in (".pt", ".ckpt"):
                self._load_checkpoint(model_path)
            else:
                print(f"    Warning: Unknown model format: {model_path.suffix}")

        except ImportError as e:
            print(f"    Warning: Missing dependency for loading model: {e}")
        except Exception as e:
            print(f"    Warning: Could not load model: {e}")

    def _load_ply(self, ply_path: Path):
        """Load Gaussian splat from .ply file."""
        import torch

        try:
            from plyfile import PlyData
        except ImportError:
            import trimesh
            mesh = trimesh.load(str(ply_path), process=False)
            # Basic point cloud - no Gaussian attributes
            self.splat_data = {
                "means": torch.tensor(np.array(mesh.vertices), dtype=torch.float32, device=self.device),
            }
            print(f"    Loaded point cloud from {ply_path}: {len(mesh.vertices)} points (basic mode)")
            return

        ply_data = PlyData.read(str(ply_path))
        vertex = ply_data["vertex"]

        # Extract Gaussian parameters
        means = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=-1)

        self.splat_data = {
            "means": torch.tensor(means, dtype=torch.float32, device=self.device),
        }

        # Try to load full Gaussian attributes (scales, rotations, SH coefficients, opacities)
        try:
            # Opacities
            if "opacity" in vertex.data.dtype.names:
                opacities = np.array(vertex["opacity"])
                self.splat_data["opacities"] = torch.tensor(opacities, dtype=torch.float32, device=self.device)

            # Scales (log scale)
            scale_names = [n for n in vertex.data.dtype.names if n.startswith("scale_")]
            if len(scale_names) >= 3:
                scales = np.stack([vertex[s] for s in sorted(scale_names)[:3]], axis=-1)
                self.splat_data["scales"] = torch.tensor(scales, dtype=torch.float32, device=self.device)

            # Rotations (quaternion)
            rot_names = [n for n in vertex.data.dtype.names if n.startswith("rot_")]
            if len(rot_names) >= 4:
                quats = np.stack([vertex[r] for r in sorted(rot_names)[:4]], axis=-1)
                self.splat_data["quats"] = torch.tensor(quats, dtype=torch.float32, device=self.device)

            # Spherical harmonics
            sh_names = sorted([n for n in vertex.data.dtype.names if n.startswith("f_rest_") or n.startswith("f_dc_")])
            if sh_names:
                sh_dc = [n for n in vertex.data.dtype.names if n.startswith("f_dc_")]
                sh_rest = [n for n in vertex.data.dtype.names if n.startswith("f_rest_")]
                sh_dc_data = np.stack([vertex[s] for s in sorted(sh_dc)], axis=-1) if sh_dc else None
                sh_rest_data = np.stack([vertex[s] for s in sorted(sh_rest)], axis=-1) if sh_rest else None

                if sh_dc_data is not None:
                    self.splat_data["sh_dc"] = torch.tensor(sh_dc_data, dtype=torch.float32, device=self.device)
                if sh_rest_data is not None:
                    self.splat_data["sh_rest"] = torch.tensor(sh_rest_data, dtype=torch.float32, device=self.device)

            n_gaussians = len(means)
            has_full = all(k in self.splat_data for k in ["opacities", "scales", "quats"])
            mode = "full Gaussian" if has_full else "basic"
            print(f"    Loaded {n_gaussians} Gaussians from {ply_path} ({mode} mode)")

        except Exception as e:
            print(f"    Loaded {len(means)} points (some attributes missing: {e})")

    def _load_checkpoint(self, ckpt_path: Path):
        """Load gsplat checkpoint (.pt file)."""
        import torch

        ckpt = torch.load(str(ckpt_path), map_location=self.device, weights_only=False)

        # gsplat checkpoints typically store splats as a dict
        if isinstance(ckpt, dict):
            if "splats" in ckpt:
                splats = ckpt["splats"]
                self.splat_data = {}
                for k, v in splats.items():
                    if isinstance(v, torch.Tensor):
                        self.splat_data[k] = v.to(self.device)
                    else:
                        self.splat_data[k] = v
                n = len(next(iter(self.splat_data.values())))
                print(f"    Loaded gsplat checkpoint: {n} Gaussians from {ckpt_path}")
            elif "means" in ckpt:
                self.splat_data = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in ckpt.items()}
                print(f"    Loaded checkpoint from {ckpt_path}")
            else:
                print(f"    Warning: Unrecognized checkpoint format in {ckpt_path}")
                print(f"    Keys: {list(ckpt.keys())[:10]}")
        else:
            print(f"    Warning: Unexpected checkpoint type: {type(ckpt)}")

    def render_frame(
        self,
        position: np.ndarray,
        rotation: np.ndarray,
        intrinsics: Optional[Dict] = None,
    ) -> Optional[np.ndarray]:
        """
        Render a single frame at the given camera pose using gsplat rasterization.

        Args:
            position: (3,) camera position in world coordinates.
            rotation: (3, 3) camera rotation matrix (camera-to-world).
            intrinsics: Optional dict with 'fx', 'fy', 'cx', 'cy'.

        Returns:
            RGB image as (H, W, 3) uint8 numpy array, or None if rendering fails.
        """
        if self.splat_data is None:
            return None

        try:
            import torch
            from gsplat import rasterization

            W, H = self.image_width, self.image_height

            # Camera intrinsics
            if intrinsics:
                fx = intrinsics.get("fx", 500.0)
                fy = intrinsics.get("fy", 500.0)
                cx = intrinsics.get("cx", W / 2.0)
                cy = intrinsics.get("cy", H / 2.0)
            else:
                fx = fy = 500.0
                cx, cy = W / 2.0, H / 2.0

            Ks = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
                              dtype=torch.float32, device=self.device).unsqueeze(0)

            # World-to-camera: invert c2w
            c2w = np.eye(4, dtype=np.float32)
            c2w[:3, :3] = rotation
            c2w[:3, 3] = position
            w2c = np.linalg.inv(c2w)
            viewmats = torch.tensor(w2c, dtype=torch.float32, device=self.device).unsqueeze(0)

            means = self.splat_data["means"]
            N = len(means)

            # Get or create default Gaussian parameters
            quats = self.splat_data.get("quats", torch.tensor([[1, 0, 0, 0]], dtype=torch.float32, device=self.device).expand(N, -1))
            scales = self.splat_data.get("scales", torch.full((N, 3), -5.0, dtype=torch.float32, device=self.device))
            opacities = self.splat_data.get("opacities", torch.ones(N, dtype=torch.float32, device=self.device))

            # Colors from SH or default
            if "sh_dc" in self.splat_data:
                # Use DC component as base color
                colors = self.splat_data["sh_dc"].reshape(N, -1)
                if colors.shape[1] == 3:
                    colors = torch.sigmoid(colors)
                else:
                    colors = torch.sigmoid(colors[:, :3])
            elif "colors" in self.splat_data:
                colors = self.splat_data["colors"]
            else:
                colors = torch.ones(N, 3, dtype=torch.float32, device=self.device) * 0.5

            # Rasterize
            renders, alphas, info = rasterization(
                means=means,
                quats=quats,
                scales=torch.exp(scales),
                opacities=torch.sigmoid(opacities),
                colors=colors,
                viewmats=viewmats,
                Ks=Ks,
                width=W,
                height=H,
                packed=False,
            )

            # Extract RGB image
            rgb = renders[0].clamp(0, 1).detach().cpu().numpy()
            rgb = (rgb * 255).astype(np.uint8)
            return rgb

        except ImportError:
            print("    gsplat not installed. Install with: pip install gsplat")
            return None
        except Exception as e:
            print(f"    Render error: {e}")
            return None

    def render_trajectory(
        self,
        trajectory: Dict,
        intrinsics: Optional[Dict] = None,
        output_path: Optional[Path] = None,
        fps: int = 24,
    ) -> Optional[np.ndarray]:
        """
        Render all frames along a trajectory and save as video.

        Args:
            trajectory: Dict with 'positions' (N, 3) and 'rotations' (N, 3, 3).
            intrinsics: Camera intrinsics dict.
            output_path: Path to save the output video (mp4).
            fps: Frames per second for the output video.

        Returns:
            frames: (N, H, W, 3) uint8 array of rendered frames, or None.
        """
        positions = trajectory["positions"]
        rotations = trajectory["rotations"]
        N = len(positions)

        frames = []
        print(f"    Rendering {N} frames...")

        for i in range(N):
            if (i + 1) % 10 == 0 or i == 0:
                print(f"      Frame {i+1}/{N}")

            frame = self.render_frame(positions[i], rotations[i], intrinsics)
            if frame is not None:
                frames.append(frame)
            else:
                placeholder = np.zeros(
                    (self.image_height, self.image_width, 3), dtype=np.uint8
                )
                frames.append(placeholder)

        frames = np.array(frames)

        if output_path and len(frames) > 0:
            self._save_video(frames, output_path, fps)

        return frames

    # def _save_video(self, frames: np.ndarray, output_path: Path, fps: int = 24):
    #     """Save frames as an MP4 video."""
    #     output_path = Path(output_path)
    #     output_path.parent.mkdir(parents=True, exist_ok=True)

    #     try:
    #         import cv2
    #         H, W = frames.shape[1], frames.shape[2]
    #         fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    #         writer = cv2.VideoWriter(str(output_path), fourcc, fps, (W, H))
    #         for frame in frames:
    #             bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    #             writer.write(bgr)
    #         writer.release()
    #         print(f"    Video saved: {output_path}")
    #     except ImportError:
    #         try:
    #             import imageio
    #             writer = imageio.get_writer(str(output_path), fps=fps, codec="libx264")
    #             for frame in frames:
    #                 writer.append_data(frame)
    #             writer.close()
    #             print(f"    Video saved: {output_path}")
    #         except ImportError:
    #             print("    Warning: Neither cv2 nor imageio available for video saving.")
    #             print("    Install with: pip install opencv-python imageio")

    def _save_video(self, frames: np.ndarray, output_path: Path, fps: int = 24):
        """Save frames as an MP4 video, re-encoded to H.264 for browser compatibility."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            import cv2
            H, W = frames.shape[1], frames.shape[2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (W, H))
            for frame in frames:
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                writer.write(bgr)
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
                writer = imageio.get_writer(
                    str(output_path), fps=fps, codec="libx264",
                    output_params=["-pix_fmt", "yuv420p"],
                )
                for frame in frames:
                    writer.append_data(frame)
                writer.close()
                print(f"    Video saved: {output_path} (H.264 via imageio)")
            except ImportError:
                print("    Warning: Neither cv2 nor imageio available for video saving.")


class CameraPathExporter:
    """
    Export camera trajectories in common JSON formats for rendering with
    external tools (gsplat viewer, viser, custom renderers).
    """

    @staticmethod
    def export_gsplat_path(
        trajectory: Dict,
        intrinsics: Dict,
        output_path: Path,
        image_width: int = 1920,
        image_height: int = 1080,
        fps: int = 24,
    ):
        """
        Export trajectory as a JSON camera path compatible with gsplat/viser viewers.

        The exported file contains a list of camera poses (world-to-camera)
        and intrinsics that can be loaded by gsplat's rendering scripts.
        """
        positions = trajectory["positions"]
        rotations = trajectory["rotations"]
        N = len(positions)

        fx = intrinsics.get("fx", 500.0)
        fy = intrinsics.get("fy", 500.0)
        cx = intrinsics.get("cx", image_width / 2.0)
        cy = intrinsics.get("cy", image_height / 2.0)

        camera_path = []
        for i in range(N):
            c2w = np.eye(4)
            c2w[:3, :3] = rotations[i]
            c2w[:3, 3] = positions[i]

            camera_path.append({
                "camera_to_world": c2w.flatten().tolist(),
                "fov": float(2 * np.degrees(np.arctan(image_width / (2 * fx)))),
                "aspect": image_width / image_height,
            })

        data = {
            "camera_type": "perspective",
            "render_height": image_height,
            "render_width": image_width,
            "camera_path": camera_path,
            "fps": fps,
            "seconds": N / fps,
        }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"    Exported camera path ({N} frames): {output_path}")

    @staticmethod
    def export_colmap_format(
        trajectory: Dict,
        intrinsics: Dict,
        output_path: Path,
    ):
        """Export trajectory as COLMAP-style images.txt for compatibility."""
        from scipy.spatial.transform import Rotation

        positions = trajectory["positions"]
        rotations = trajectory["rotations"]
        N = len(positions)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            f.write("# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n")
            f.write("# POINTS2D[] as (X Y POINT3D_ID)\n")
            for i in range(N):
                R = rotations[i]
                t = positions[i]
                # COLMAP convention: w2c
                R_w2c = R.T
                t_w2c = -R.T @ t
                qvec = Rotation.from_matrix(R_w2c).as_quat()  # (x, y, z, w)
                qvec = np.array([qvec[3], qvec[0], qvec[1], qvec[2]])  # (w, x, y, z)
                f.write(f"{i+1} {qvec[0]:.8f} {qvec[1]:.8f} {qvec[2]:.8f} {qvec[3]:.8f} "
                        f"{t_w2c[0]:.8f} {t_w2c[1]:.8f} {t_w2c[2]:.8f} 1 frame_{i:05d}.jpg\n")
                f.write("\n")

        print(f"    Exported COLMAP images.txt ({N} frames): {output_path}")