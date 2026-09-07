"""
Anchor Determinator Module
============================
Identifies relevant objects in a 3D scene to serve as anchor points
for camera trajectory placement.

Two-stage process:
    1. Initial Anchor Selector: CLIP-based best-matching image selection (Eq. 4)
    2. Anchor Refinement: Gradient descent on CLIP similarity with rendered views (Eq. 5-6)

Reference: Section 3.2 of Liu et al. (2024)
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image
from glob import glob

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None
    F = None

try:
    import clip
except ImportError:
    clip = None


class AnchorDeterminator:
    """
    Determines anchor camera poses for placing trajectories in 3D scenes.
    
    Given a text description (e.g., "close-up of the Sydney Opera House"),
    finds the camera pose that renders the best matching view.
    """

    def __init__(
        self,
        scene_dir: str,
        clip_model: str = "ViT-B/32",
        device: str = "cuda",
    ):
        self.scene_dir = Path(scene_dir)
        self.device = device if (torch is not None and torch.cuda.is_available()) else "cpu"

        # Load CLIP model
        if clip is not None:
            self.clip_model, self.clip_preprocess = clip.load(clip_model, device=self.device)
            self.clip_model.eval()
            print(f"    Loaded CLIP {clip_model} for anchor determination")
        else:
            self.clip_model = None
            print("    Warning: CLIP not available. Install with: pip install git+https://github.com/openai/CLIP.git")

        # Load scene data (images + camera poses)
        self.images_info = self._load_scene_images()
        self.image_features = None  # Cached CLIP features

        # Optional: 3DGS/NeRF model for anchor refinement
        self.render_fn = None

    def _load_scene_images(self) -> List[dict]:
        """Load training images and their camera poses from the scene directory."""
        images_info = []

        # Find images
        image_dir = self.scene_dir / "images"
        if not image_dir.exists():
            image_dir = self.scene_dir  # Images might be in root

        image_paths = sorted(
            glob(str(image_dir / "*.jpg")) +
            glob(str(image_dir / "*.png")) +
            glob(str(image_dir / "*.JPG")) +
            glob(str(image_dir / "*.PNG"))
        )

        # Try to load camera poses from transforms.json (Nerfstudio format)
        transforms_path = self.scene_dir / "transforms.json"
        if transforms_path.exists():
            import json
            with open(transforms_path) as f:
                transforms = json.load(f)

            for frame in transforms.get("frames", []):
                file_path = frame.get("file_path", "")
                # Resolve relative path
                img_path = self.scene_dir / file_path
                if not img_path.exists():
                    img_path = self.scene_dir / "images" / Path(file_path).name
                if not img_path.exists():
                    continue

                c2w = np.array(frame["transform_matrix"], dtype=np.float32)
                images_info.append({
                    "path": str(img_path),
                    "name": Path(file_path).name,
                    "c2w": c2w,
                    "intrinsics": {
                        "fx": transforms.get("fl_x", 500),
                        "fy": transforms.get("fl_y", 500),
                        "cx": transforms.get("cx", 256),
                        "cy": transforms.get("cy", 256),
                        "w": transforms.get("w", 512),
                        "h": transforms.get("h", 512),
                    },
                })
        else:
            # Try COLMAP format
            images_info = self._load_colmap_poses(image_paths)

        if not images_info and image_paths:
            # No poses found, just load images with identity poses
            print(f"    Warning: No camera poses found, using images without poses")
            for p in image_paths:
                images_info.append({
                    "path": p,
                    "name": Path(p).name,
                    "c2w": np.eye(4, dtype=np.float32),
                    "intrinsics": None,
                })

        print(f"    Loaded {len(images_info)} scene images")
        return images_info

    def _load_colmap_poses(self, image_paths: List[str]) -> List[dict]:
        """Load COLMAP poses AND camera intrinsics for the scene."""
        sparse_dir = self.scene_dir / "sparse" / "0"
        if not sparse_dir.exists():
            sparse_dir = self.scene_dir / "colmap" / "sparse" / "0"
        if not sparse_dir.exists():
            return []

        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "trajectory_generation"))
            from scene_reconstruction import (
                read_images_binary, read_images_text,
                read_cameras_binary, read_cameras_text,
                qvec2rotmat,
            )

            # --- Load images ---
            images_bin = sparse_dir / "images.bin"
            images_txt = sparse_dir / "images.txt"
            if images_bin.exists():
                colmap_images = read_images_binary(str(images_bin))
            elif images_txt.exists():
                colmap_images = read_images_text(str(images_txt))
            else:
                return []

            # --- Load cameras (for intrinsics) ---
            colmap_cameras = {}
            cameras_bin = sparse_dir / "cameras.bin"
            cameras_txt = sparse_dir / "cameras.txt"
            try:
                if cameras_bin.exists():
                    colmap_cameras = read_cameras_binary(str(cameras_bin))
                elif cameras_txt.exists():
                    colmap_cameras = read_cameras_text(str(cameras_txt))
            except Exception as e:
                print(f"    Warning: Could not load COLMAP cameras: {e}")

            # Build name-to-path mapping
            name_to_path = {Path(p).name: p for p in image_paths}

            images_info = []
            for img_info in colmap_images.values():
                if img_info.name not in name_to_path:
                    continue

                R = qvec2rotmat(img_info.qvec)
                t = img_info.tvec

                # COLMAP stores world-to-camera, convert to camera-to-world
                c2w = np.eye(4, dtype=np.float32)
                c2w[:3, :3] = R.T
                c2w[:3, 3] = -R.T @ t

                # Extract intrinsics from the corresponding COLMAP camera
                intrinsics = None
                cam_id = img_info.camera_id
                if cam_id in colmap_cameras:
                    cam = colmap_cameras[cam_id]
                    params = cam.params
                    model = cam.model
                    if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
                        intrinsics = {
                            "fx": float(params[0]),
                            "fy": float(params[0]),
                            "cx": float(params[1]),
                            "cy": float(params[2]),
                            "w": int(cam.width),
                            "h": int(cam.height),
                        }
                    elif model in ("PINHOLE", "OPENCV"):
                        intrinsics = {
                            "fx": float(params[0]),
                            "fy": float(params[1]),
                            "cx": float(params[2]),
                            "cy": float(params[3]),
                            "w": int(cam.width),
                            "h": int(cam.height),
                        }
                    else:
                        intrinsics = {
                            "fx": float(params[0]),
                            "fy": float(params[0]),
                            "cx": cam.width / 2.0,
                            "cy": cam.height / 2.0,
                            "w": int(cam.width),
                            "h": int(cam.height),
                        }

                images_info.append({
                    "path": name_to_path[img_info.name],
                    "name": img_info.name,
                    "c2w": c2w,
                    "intrinsics": intrinsics,
                })

            return images_info

        except Exception as e:
            print(f"    Could not load COLMAP poses: {e}")
            return []

    @torch.no_grad()
    def _encode_images(self) -> torch.Tensor:
        """Encode all scene images into CLIP feature space. Cache the result."""
        if self.image_features is not None:
            return self.image_features

        if self.clip_model is None:
            return None

        all_features = []
        batch_size = 32

        for i in range(0, len(self.images_info), batch_size):
            batch = self.images_info[i:i + batch_size]
            images = []
            for info in batch:
                img = Image.open(info["path"]).convert("RGB")
                img_tensor = self.clip_preprocess(img)
                images.append(img_tensor)

            image_input = torch.stack(images).to(self.device)
            features = self.clip_model.encode_image(image_input)
            features = F.normalize(features, dim=-1)
            all_features.append(features.cpu())

        self.image_features = torch.cat(all_features, dim=0)
        return self.image_features

    @torch.no_grad()
    def _encode_text(self, text: str) -> torch.Tensor:
        """Encode text prompt into CLIP feature space."""
        if self.clip_model is None:
            return None
        text_tokens = clip.tokenize([text], truncate=True).to(self.device)
        features = self.clip_model.encode_text(text_tokens)
        features = F.normalize(features, dim=-1)
        return features.cpu()

    def select_initial_anchor(self, text_prompt: str) -> Tuple[int, float, dict]:
        """
        Initial Anchor Selector (Eq. 4 from the paper):
        
            i_anchor = argmax_i  f_image(I_i) · f_text(T) / (||f_image(I_i)|| · ||f_text(T)||)
        
        Selects the input image with highest CLIP cosine similarity to the text.
        """
        image_features = self._encode_images()
        text_features = self._encode_text(text_prompt)

        if image_features is None or text_features is None:
            return 0, 0.0, self.images_info[0] if self.images_info else {}

        similarities = (text_features @ image_features.T).squeeze(0).numpy()
        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])

        return best_idx, best_score, self.images_info[best_idx]

    def refine_anchor(
        self,
        initial_c2w: np.ndarray,
        text_prompt: str,
        intrinsics: Optional[dict] = None,
        lr: float = 0.002,
        num_steps: int = 200,
    ) -> np.ndarray:
        """
        Anchor Refinement (Eq. 5-6 from the paper):

            min_c  L_anchor(c) = - f_image(R(c)) · f_text(T)
                                   / (||f_image(R(c))|| · ||f_text(T)||)

            c_{t+1} = c_t - η ∇_c L_anchor(c_t)

        The rendering function R(c) must be differentiable w.r.t. the camera
        parameters c.  We use gsplat rasterization which supports this natively.

        The gradient flows:  camera params → c2w → viewmat → gsplat rasterization
        → rendered image → CLIP preprocessing → CLIP visual encoder → cosine loss.

        CLIP's visual encoder is frozen but still produces gradients w.r.t. its
        input pixels, which propagate back through the rendering.

        Args:
            initial_c2w: (4, 4) initial camera-to-world matrix from CLIP selection.
            text_prompt: Target text description.
            intrinsics: Camera intrinsics dict with fx, fy, cx, cy, w, h.
            lr: Learning rate for Adam optimizer.
            num_steps: Maximum optimization steps.

        Returns:
            Refined (4, 4) camera-to-world matrix (numpy).
        """
        if self.render_fn is None:
            print("    [Anchor Refinement] No render function available, skipping")
            return initial_c2w

        if torch is None:
            return initial_c2w

        # Pre-compute text features (frozen, no grad needed)
        with torch.no_grad():
            text_tokens = clip.tokenize([text_prompt], truncate=True).to(self.device)
            text_features = self.clip_model.encode_text(text_tokens)
            text_features = F.normalize(text_features.float(), dim=-1)  # (1, D) float32

        # Parameterize camera pose for optimization using 6D rotation
        # (Zhou et al., "On the Continuity of Rotation Representations", CVPR 2019)
        translation = torch.tensor(
            initial_c2w[:3, 3].copy(), dtype=torch.float32,
            device=self.device, requires_grad=True,
        )
        # 6D rotation: first two columns of the rotation matrix, flattened
        rot_6d = torch.tensor(
            initial_c2w[:3, :2].T.reshape(-1).copy(), dtype=torch.float32,
            device=self.device, requires_grad=True,
        )

        optimizer = torch.optim.Adam([translation, rot_6d], lr=lr)

        best_loss = float("inf")
        best_c2w = initial_c2w.copy()
        initial_score = None

        for step in range(num_steps):
            optimizer.zero_grad()

            # Reconstruct c2w from optimizable params
            c2w_tensor = self._params_to_c2w(translation, rot_6d)

            # Render the scene at the current camera pose (differentiable)
            rendered = self.render_fn(c2w_tensor, intrinsics)
            if rendered is None:
                print(f"    [Anchor Refinement] Render failed at step {step}, stopping")
                break

            # rendered should be a torch.Tensor (H, W, 3) in [0, 1]
            # If it's a numpy array, convert (but we lose gradients)
            if isinstance(rendered, np.ndarray):
                rendered = torch.tensor(rendered, dtype=torch.float32, device=self.device)
            if rendered.dim() == 3 and rendered.shape[-1] == 3:
                rendered = rendered.permute(2, 0, 1)  # (3, H, W)

            # Resize to CLIP's expected input (224x224) using differentiable interpolation
            rendered_resized = F.interpolate(
                rendered.unsqueeze(0), size=(224, 224),
                mode="bilinear", align_corners=False,
            )

            # Normalize with CLIP's expected mean/std
            clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                                     device=self.device).view(1, 3, 1, 1)
            clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                                    device=self.device).view(1, 3, 1, 1)
            rendered_normalized = (rendered_resized - clip_mean) / clip_std

            # Forward through CLIP visual encoder (frozen weights, but grad flows through input)
            image_features = self.clip_model.encode_image(rendered_normalized.half())
            image_features = image_features.float()
            image_features = F.normalize(image_features, dim=-1)

            # Negative cosine similarity loss (Eq. 5)
            loss = -(image_features @ text_features.T).squeeze()

            loss.backward()
            optimizer.step()

            current_loss = loss.item()
            if initial_score is None:
                initial_score = -current_loss

            if current_loss < best_loss:
                best_loss = current_loss
                best_c2w = c2w_tensor.detach().cpu().numpy().astype(np.float32)

            if (step + 1) % 50 == 0:
                print(f"    [Anchor Refinement] Step {step+1}/{num_steps}, "
                      f"similarity={-current_loss:.4f} (init={initial_score:.4f})")

            # Early stopping: if similarity is very high
            if -current_loss > 0.35:
                print(f"    [Anchor Refinement] Converged at step {step+1} "
                      f"(similarity={-current_loss:.4f})")
                break

        final_score = -best_loss
        print(f"    [Anchor Refinement] {initial_score:.4f} → {final_score:.4f}")
        return best_c2w

    def determine_anchor(
        self,
        text_prompt: str,
        refine: bool = True,
    ) -> dict:
        """
        Full anchor determination pipeline:
            1. Select initial anchor via CLIP similarity
            2. Optionally refine via gradient descent
        
        Returns:
            Dict with 'c2w', 'intrinsics', 'image_path', 'similarity'
        """
        idx, score, info = self.select_initial_anchor(text_prompt)
        print(f"    Initial anchor: {info.get('name', 'unknown')} (similarity: {score:.4f})")

        c2w = info.get("c2w", np.eye(4, dtype=np.float32))
        intrinsics = info.get("intrinsics")

        if refine and self.render_fn is not None:
            c2w = self.refine_anchor(c2w, text_prompt, intrinsics)

        return {
            "c2w": c2w,
            "intrinsics": intrinsics,
            "image_path": info.get("path"),
            "image_name": info.get("name"),
            "similarity": score,
            "text_prompt": text_prompt,
        }

    def set_render_function(self, render_fn):
        """Set the differentiable rendering function for anchor refinement."""
        self.render_fn = render_fn

    def _params_to_c2w(self, translation, rotation_6d):
        """Convert optimizable parameters to a 4x4 c2w matrix."""
        r6d = rotation_6d.reshape(2, 3)
        a1 = F.normalize(r6d[0], dim=0)
        a2 = r6d[1] - (a1 @ r6d[1]) * a1
        a2 = F.normalize(a2, dim=0)
        a3 = torch.linalg.cross(a1, a2)
        R = torch.stack([a1, a2, a3], dim=-1)
        c2w = torch.eye(4, device=self.device)
        c2w[:3, :3] = R
        c2w[:3, 3] = translation
        return c2w

    def _encode_rendered_image(self, image) -> torch.Tensor:
        """Encode a rendered image (PIL or tensor) into CLIP feature space."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray((image * 255).astype(np.uint8) if image.max() <= 1 else image)
        elif torch.is_tensor(image):
            img_np = image.detach().cpu().numpy()
            if img_np.shape[0] == 3:
                img_np = img_np.transpose(1, 2, 0)
            image = Image.fromarray((img_np * 255).astype(np.uint8) if img_np.max() <= 1 else img_np.astype(np.uint8))

        img_tensor = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        features = self.clip_model.encode_image(img_tensor)
        return F.normalize(features, dim=-1)