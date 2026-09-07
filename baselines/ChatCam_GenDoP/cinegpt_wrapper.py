"""
CineGPT Wrapper (GenDoP Integration)
======================================
Wraps the GenDoP model as CineGPT for text-conditioned camera trajectory generation.

GenDoP uses a GPT-based autoregressive model with:
    - VQ-VAE trajectory tokenizer for discretizing camera trajectories
    - Cross-modal transformer decoder for text -> trajectory token generation
    - Camera parameterization: translation, rotation (S²×S²), focal length, velocity

Reference: Section 3.1 of Liu et al. (2024) + GenDoP codebase
"""

import numpy as np
import json
import os
import time
from pathlib import Path
from typing import Dict, Optional, List

try:
    import torch
except ImportError:
    torch = None


class CineGPTWrapper:
    """
    Wrapper around GenDoP for text-conditioned camera trajectory generation.
    
    CineGPT translates natural language trajectory descriptions into
    camera trajectories (sequences of c2w matrices + intrinsics).
    
    Supports multiple conditioning modes:
        - text: Text description only
        - image+text: Image + text description
        - depth+image+text: Depth + image + text
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        device: str = "cuda",
    ):
        self.device = device if (torch is not None and torch.cuda.is_available()) else "cpu"
        self.model = None
        self.opt = None

        if config and config.get("resume"):
            self._load_model(config)
        else:
            print("    CineGPT: No checkpoint provided, using analytical trajectory generation")

    def _load_model(self, config: dict):
        try:
            import sys
            gendop_path = os.path.join(os.path.dirname(__file__), '..', '..', 'third_party', 'GenDoP')
            gendop_path = os.path.abspath(gendop_path)
            if gendop_path not in sys.path:
                sys.path.insert(0, gendop_path)

            import tyro
            from core.options import AllConfigs
            from core.models import LMM
            from core.utils import monkey_patch_transformers
            from safetensors.torch import load_file

            # Use tyro to properly instantiate AllConfigs (same as eval.py)
            cond_mode = config.get("cond_mode", "text")
            resume = config.get("resume")
            self.opt = tyro.cli(AllConfigs, args=[
                "ArAE",
                "--resume", resume,
                "--cond_mode", cond_mode,
            ])

            if self.opt.cond_mode == "text":
                self.opt.num_cond_tokens = 77
            elif self.opt.cond_mode == "depth+image+text":
                self.opt.num_cond_tokens = 591

            monkey_patch_transformers()

            self.model = LMM(self.opt)

            if resume.endswith("safetensors"):
                ckpt = load_file(resume, device="cpu")
            else:
                ckpt = torch.load(resume, map_location="cpu")

            self.model.load_state_dict(ckpt, strict=False)
            self.model = self.model.half().eval().to(self.device)
            print(f"    CineGPT: Loaded GenDoP from {resume}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"    CineGPT: Could not load GenDoP model: {e}")
            self.model = None

    def generate(
        self,
        text: str,
        image_path: Optional[str] = None,
        depth_path: Optional[str] = None,
    ) -> dict:
        """
        Generate a camera trajectory from a text description.
        
        Args:
            text: Natural language trajectory description
                  (e.g., "pan forward slowly", "orbit left", "dolly zoom")
            image_path: Optional conditioning image path
            depth_path: Optional depth map path
            
        Returns:
            Dict with:
                - c2ws: (N, 4, 4) camera-to-world matrices
                - intrinsics: Dict with fx, fy, cx, cy, w, h
                - num_frames: Number of frames
                - description: Input text
        """
        if self.model is not None:
            return self._generate_with_model(text, image_path, depth_path)
        else:
            return self._generate_analytical(text)

    def _generate_with_model(
        self,
        text: str,
        image_path: Optional[str] = None,
        depth_path: Optional[str] = None,
    ) -> dict:
        """Generate trajectory using GenDoP model."""
        import cv2
        from third_party.GenDoP.core.utils import (
            token_to_camera, sample_from_dense_cameras
        )

        opt = self.opt

        # Prepare conditioning inputs
        if opt.cond_mode == "text":
            conds = [text]
        elif opt.cond_mode == "image+text" and image_path:
            rgb = self._load_image(image_path, opt.target_height, opt.target_width)
            rgb_batch = rgb.unsqueeze(0).to(self.device)
            conds = [[text], rgb_batch]
        elif opt.cond_mode == "depth+image+text" and image_path and depth_path:
            rgb = self._load_image(image_path, opt.target_height, opt.target_width)
            depth = self._load_depth(depth_path, opt.target_height, opt.target_width)
            rgb_batch = rgb.unsqueeze(0).to(self.device)
            depth_batch = depth.unsqueeze(0).to(self.device)
            conds = [[text], rgb_batch, depth_batch]
        else:
            conds = [text]

        # Generate tokens
        t0 = time.time()
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                tokens = self.model.generate(
                    conds,
                    max_new_tokens=opt.test_max_seq_length,
                    clean=True,
                )
        t1 = time.time()
        print(f"    CineGPT generation time: {t1-t0:.2f}s")

        # Decode tokens to camera trajectory
        token = tokens[0]
        if token[:-1].shape[0] != opt.pose_length * 10:
            # Fallback: default token sequence
            token = torch.tensor([256, 128, 128, 128, 128, 128, 128, 36, 64, 60]) / 256 * opt.discrete_bins
            token = token.repeat(opt.pose_length)
            coords = token.reshape(-1, 10)
        else:
            coords = token[:-1].reshape(-1, 10)

        coords = torch.tensor(coords, dtype=torch.float32)
        discrete_bins = opt.discrete_bins

        # Split into trajectory and intrinsics tokens
        coords_traj = coords[:, :7]
        coords_instri = coords[:, 7:]
        coords_scale = coords_instri[:, -1]

        # De-quantize
        temp_traj = coords_traj / (0.5 * discrete_bins) - 1
        temp_instri = coords_instri / (discrete_bins / 10)
        scale = torch.exp(coords_scale / discrete_bins * 4 - 2)

        camera_tokens = torch.cat([temp_traj, temp_instri], dim=1)
        camera_tokens = camera_tokens.unsqueeze(0)
        camera_pose = token_to_camera(camera_tokens, 512, 512)

        # Extract c2w matrices
        c2ws = camera_pose[:, :, :12].cpu().numpy()
        scale_value = scale[0].cpu().numpy()
        c2ws = c2ws.reshape((-1, 3, 4))
        c2ws[:, :3, 3] *= scale_value

        # Add homogeneous row
        row = np.array([0, 0, 0, 1], dtype=np.float32)
        c2ws = np.array([np.vstack((m, row)) for m in c2ws])

        # Densify trajectory via interpolation (to 120 frames)
        traj_tensor = camera_pose[:, :, :12]
        camera_list = []
        for i in range(120):
            t = torch.full((1, 1), fill_value=i / 120)
            camera = sample_from_dense_cameras(traj_tensor, t)
            camera_list.append(camera[0])
        camera_tensor = torch.cat(camera_list, dim=0)
        dense_c2ws = camera_tensor.cpu().numpy().reshape(-1, 3, 4)
        dense_c2ws = np.array([np.vstack((m, row)) for m in dense_c2ws])

        # Extract intrinsics from first frame
        f_x, f_y, c_x, c_y, w, h = camera_pose[0][0][-6:].tolist()

        return {
            "c2ws": dense_c2ws,  # (120, 4, 4)
            "c2ws_keyframes": c2ws,  # Keyframe poses
            "intrinsics": {
                "fx": f_x, "fy": f_y,
                "cx": c_x, "cy": c_y,
                "w": int(w), "h": int(h),
            },
            "num_frames": len(dense_c2ws),
            "description": text,
        }

    def _generate_analytical(self, text: str) -> dict:
        """
        Analytical trajectory generation fallback when GenDoP is not available.
        
        Generates basic trajectories from keyword parsing:
            - pan left/right/forward/backward
            - zoom in/out
            - orbit left/right
            - tilt up/down
            - dolly zoom
            - look around / rotate
            - S-shaped path
            - U-turn
        """
        text_lower = text.lower().strip()
        num_frames = 120
        t = np.linspace(0, 1, num_frames)

        # Start from identity
        c2ws = np.zeros((num_frames, 4, 4), dtype=np.float32)
        for i in range(num_frames):
            c2ws[i] = np.eye(4)

        # Parse trajectory type and generate
        if "pan" in text_lower and "left" in text_lower:
            for i, ti in enumerate(t):
                c2ws[i][:3, 3] = [-ti * 2.0, 0, 0]

        elif "pan" in text_lower and "right" in text_lower:
            for i, ti in enumerate(t):
                c2ws[i][:3, 3] = [ti * 2.0, 0, 0]

        elif "pan" in text_lower and ("forward" in text_lower or "straight" in text_lower):
            for i, ti in enumerate(t):
                c2ws[i][:3, 3] = [0, 0, -ti * 2.0]

        elif "move forward" in text_lower or "push forward" in text_lower:
            for i, ti in enumerate(t):
                c2ws[i][:3, 3] = [0, 0, -ti * 3.0]

        elif "pull back" in text_lower or "move backward" in text_lower:
            for i, ti in enumerate(t):
                c2ws[i][:3, 3] = [0, 0, ti * 3.0]

        elif "zoom in" in text_lower:
            for i, ti in enumerate(t):
                c2ws[i][:3, 3] = [0, 0, -ti * 2.0]

        elif "zoom out" in text_lower:
            for i, ti in enumerate(t):
                c2ws[i][:3, 3] = [0, 0, ti * 2.0]

        elif "orbit" in text_lower or "look around" in text_lower:
            radius = 1.5
            for i, ti in enumerate(t):
                angle = ti * 2 * np.pi
                c2ws[i][:3, 3] = [radius * np.cos(angle), 0, radius * np.sin(angle)]
                # Point toward center with proper right-handed frame
                forward = -c2ws[i][:3, 3].copy()
                forward = forward / (np.linalg.norm(forward) + 1e-8)
                up = np.array([0, 1, 0])
                right = np.cross(up, forward)
                right_norm = np.linalg.norm(right)
                if right_norm < 1e-6:
                    right = np.array([1, 0, 0])
                else:
                    right = right / right_norm
                up = np.cross(forward, right)
                up = up / (np.linalg.norm(up) + 1e-8)
                c2ws[i][:3, :3] = np.stack([right, up, forward], axis=-1)

        elif "tilt" in text_lower and "up" in text_lower:
            for i, ti in enumerate(t):
                angle = ti * np.pi / 6  # 30 degrees
                R = np.array([
                    [1, 0, 0],
                    [0, np.cos(angle), -np.sin(angle)],
                    [0, np.sin(angle), np.cos(angle)],
                ])
                c2ws[i][:3, :3] = R

        elif "tilt" in text_lower and "down" in text_lower:
            for i, ti in enumerate(t):
                angle = -ti * np.pi / 6
                R = np.array([
                    [1, 0, 0],
                    [0, np.cos(angle), -np.sin(angle)],
                    [0, np.sin(angle), np.cos(angle)],
                ])
                c2ws[i][:3, :3] = R

        elif "dolly zoom" in text_lower:
            # Move forward while keeping subject same size (change focal length)
            for i, ti in enumerate(t):
                c2ws[i][:3, 3] = [0, 0, -ti * 3.0]

        elif "u-turn" in text_lower or "u turn" in text_lower:
            for i, ti in enumerate(t):
                if ti < 0.5:
                    c2ws[i][:3, 3] = [0, 0, -ti * 4.0]
                else:
                    angle = (ti - 0.5) * np.pi
                    c2ws[i][:3, 3] = [np.sin(angle) * 1.0, 0, -2.0 + (ti - 0.5) * 4.0]
                    R_y = np.array([
                        [np.cos(angle), 0, np.sin(angle)],
                        [0, 1, 0],
                        [-np.sin(angle), 0, np.cos(angle)],
                    ])
                    c2ws[i][:3, :3] = R_y

        elif "s-shaped" in text_lower or "s shaped" in text_lower:
            for i, ti in enumerate(t):
                x = np.sin(ti * 2 * np.pi) * 0.8
                z = -ti * 4.0
                c2ws[i][:3, 3] = [x, 0, z]

        elif "ascend" in text_lower or "crane up" in text_lower:
            for i, ti in enumerate(t):
                c2ws[i][:3, 3] = [0, ti * 2.0, 0]

        elif "descend" in text_lower or "crane down" in text_lower:
            for i, ti in enumerate(t):
                c2ws[i][:3, 3] = [0, -ti * 2.0, 0]

        elif "roll" in text_lower:
            for i, ti in enumerate(t):
                angle = ti * 2 * np.pi
                R = np.array([
                    [np.cos(angle), -np.sin(angle), 0],
                    [np.sin(angle), np.cos(angle), 0],
                    [0, 0, 1],
                ])
                c2ws[i][:3, :3] = R

        elif "turn left" in text_lower:
            for i, ti in enumerate(t):
                angle = ti * np.pi / 3
                R_y = np.array([
                    [np.cos(angle), 0, np.sin(angle)],
                    [0, 1, 0],
                    [-np.sin(angle), 0, np.cos(angle)],
                ])
                c2ws[i][:3, :3] = R_y

        elif "turn right" in text_lower:
            for i, ti in enumerate(t):
                angle = -ti * np.pi / 3
                R_y = np.array([
                    [np.cos(angle), 0, np.sin(angle)],
                    [0, 1, 0],
                    [-np.sin(angle), 0, np.cos(angle)],
                ])
                c2ws[i][:3, :3] = R_y

        else:
            # Default: gentle forward movement
            for i, ti in enumerate(t):
                c2ws[i][:3, 3] = [0, 0, -ti * 1.5]

        # Default intrinsics
        intrinsics = {
            "fx": 500.0, "fy": 500.0,
            "cx": 256.0, "cy": 256.0,
            "w": 512, "h": 512,
        }

        # Handle dolly zoom: varying focal length
        if "dolly zoom" in text_lower:
            # Focal length decreases as camera moves forward
            intrinsics["fx_sequence"] = [500.0 * (1.0 - 0.5 * ti) for ti in t]
            intrinsics["fy_sequence"] = [500.0 * (1.0 - 0.5 * ti) for ti in t]

        return {
            "c2ws": c2ws,
            "c2ws_keyframes": c2ws[::30],  # Every 30th frame
            "intrinsics": intrinsics,
            "num_frames": num_frames,
            "description": text,
        }

    def _load_image(self, path: str, target_h: int = 512, target_w: int = 512):
        """Load and preprocess image for GenDoP conditioning."""
        import cv2
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0
        image = image[..., [2, 1, 0]]  # BGR to RGB
        tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()

        h, w = tensor.shape[1], tensor.shape[2]
        if h > target_h:
            start = (h - target_h) // 2
            tensor = tensor[:, start:start + target_h, :]
        if w > target_w:
            start = (w - target_w) // 2
            tensor = tensor[:, :, start:start + target_w]

        if tensor.shape[1] < target_h or tensor.shape[2] < target_w:
            padded = torch.zeros((3, target_h, target_w), dtype=torch.float32)
            top = (target_h - tensor.shape[1]) // 2
            left = (target_w - tensor.shape[2]) // 2
            padded[:, top:top + tensor.shape[1], left:left + tensor.shape[2]] = tensor
            tensor = padded

        return tensor

    def _load_depth(self, path: str, target_h: int = 512, target_w: int = 512):
        """Load and preprocess depth map for GenDoP conditioning."""
        depth = np.load(path).astype(np.float32)
        tensor = torch.from_numpy(depth).unsqueeze(0).float()

        h, w = tensor.shape[1], tensor.shape[2]
        if h > target_h:
            start = (h - target_h) // 2
            tensor = tensor[:, start:start + target_h, :]
        if w > target_w:
            start = (w - target_w) // 2
            tensor = tensor[:, :, start:start + target_w]

        if tensor.shape[1] < target_h or tensor.shape[2] < target_w:
            padded = torch.zeros((1, target_h, target_w), dtype=torch.float32)
            top = (target_h - tensor.shape[1]) // 2
            left = (target_w - tensor.shape[2]) // 2
            padded[:, top:top + tensor.shape[1], left:left + tensor.shape[2]] = tensor
            tensor = padded

        return tensor
