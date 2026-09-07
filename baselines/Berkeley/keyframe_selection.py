"""
CLIP-based Keyframe Selection Module
=====================================
Uses CLIP to match natural language camera instructions to training images,
selecting the best-aligned keyframes for trajectory generation.

Reference: Section 3.3 of Wu (2025)
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image

try:
    import clip
except ImportError:
    print("Warning: 'clip' package not found. Install with: pip install git+https://github.com/openai/CLIP.git")

from scene_reconstruction import qvec2rotmat


class CLIPKeyframeSelector:
    """
    Selects keyframes from a set of training images using CLIP similarity
    between camera instruction text and image features.
    
    For each segment of the camera instruction prompt, the selector finds
    the training image whose CLIP embedding has the highest cosine similarity
    with the text embedding of that segment.
    """

    def __init__(
        self,
        model_name: str = "ViT-B/32",
        device: str = "cuda",
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"  Loading CLIP model: {model_name} on {self.device}")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()

    @torch.no_grad()
    def encode_images(self, image_paths: List[Path]) -> torch.Tensor:
        """Encode a batch of images into CLIP feature space.
        
        Args:
            image_paths: List of paths to images.
            
        Returns:
            Normalized image features tensor of shape (N, D).
        """
        if not image_paths:
            raise ValueError(
                "No images to encode. Check that image_paths is non-empty "
                "and that the images directory contains files matching the "
                "expected extensions (jpg, JPG, png, PNG)."
            )

        all_features = []
        batch_size = 64

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            images = []
            for p in batch_paths:
                img = Image.open(p).convert("RGB")
                img_tensor = self.preprocess(img)
                images.append(img_tensor)

            image_input = torch.stack(images).to(self.device)
            features = self.model.encode_image(image_input)
            features = F.normalize(features, dim=-1)
            all_features.append(features.cpu())

        return torch.cat(all_features, dim=0)

    @torch.no_grad()
    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        """Encode text prompts into CLIP feature space.
        
        Args:
            texts: List of text strings.
            
        Returns:
            Normalized text features tensor of shape (N, D).
        """
        text_tokens = clip.tokenize(texts, truncate=True).to(self.device)
        features = self.model.encode_text(text_tokens)
        features = F.normalize(features, dim=-1)
        return features.cpu()

    def compute_similarity(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> np.ndarray:
        """Compute cosine similarity between image and text features.
        
        Equation 3.1 from the thesis:
            similarity(I, T) = CLIP(I) · CLIP(T) / (||CLIP(I)|| * ||CLIP(T)||)
        
        Since features are already L2-normalized, this is simply the dot product.
        
        Args:
            image_features: (N_images, D) normalized image features
            text_features: (N_texts, D) normalized text features
            
        Returns:
            Similarity matrix of shape (N_texts, N_images)
        """
        # Both are already normalized, so dot product = cosine similarity
        similarity = (text_features @ image_features.T).numpy()
        return similarity

    def select_keyframes(
        self,
        image_paths: List[Path],
        camera_poses: dict,
        prompt_segments: List[str],
        top_k: int = 1,
    ) -> List[Dict]:
        """
        Select keyframes by matching prompt segments to training images via CLIP.
        
        For each prompt segment, selects the top_k images with highest cosine
        similarity. Keyframes are constrained to the training image set to ensure
        they correspond to well-reconstructed regions of the scene.
        
        Args:
            image_paths: Paths to training images.
            camera_poses: Dict of COLMAP image info (id -> ImageInfo).
            prompt_segments: List of parsed instruction segments.
            top_k: Number of keyframes to select per segment.
            
        Returns:
            List of keyframe dicts with keys:
                - segment: the text instruction
                - image_name: filename of selected image
                - image_path: full path
                - similarity: cosine similarity score
                - position: camera position in world coordinates (3,)
                - rotation: rotation matrix (3, 3)
                - intrinsics: camera intrinsics if available
        """
        # Build mapping from image name to pose
        # COLMAP may store names with subdirectory prefixes
        # (e.g., "undistorted_images/DSC_0001.JPG")
        name_to_pose = {}
        for img_info in camera_poses.values():
            R_w2c = qvec2rotmat(img_info.qvec)
            t = img_info.tvec
            C = -R_w2c.T @ t  # World-space camera center
            R_c2w = R_w2c.T
            pose_data = {
                "rotation": R_c2w,
                "position": C,
                "camera_id": img_info.camera_id,
            }
            # Store under both full COLMAP name and just the filename
            name_to_pose[img_info.name] = pose_data
            basename = Path(img_info.name).name
            if basename != img_info.name:
                name_to_pose[basename] = pose_data

        # Filter image_paths to only those with known poses
        # Try matching by basename since COLMAP names may include subdir prefixes
        valid_paths = []
        valid_names = []
        for p in image_paths:
            basename = p.name
            if basename in name_to_pose:
                valid_paths.append(p)
                valid_names.append(basename)
            else:
                # Try matching with subdirectory prefix from COLMAP
                for colmap_name in camera_poses.values():
                    colmap_basename = Path(colmap_name.name).name
                    if colmap_basename == basename:
                        valid_paths.append(p)
                        valid_names.append(colmap_basename)
                        break

        if not valid_paths and image_paths:
            print(f"  Warning: No name matches found between {len(image_paths)} images "
                  f"and {len(name_to_pose)} COLMAP poses")
            sample_img = image_paths[0].name if image_paths else "N/A"
            sample_colmap = list(camera_poses.values())[0].name if camera_poses else "N/A"
            print(f"    Image sample: {sample_img}")
            print(f"    COLMAP sample: {sample_colmap}")
            # Use all images anyway
            valid_paths = image_paths
            valid_names = [p.name for p in image_paths]

        print(f"  Encoding {len(valid_paths)} images...")
        image_features = self.encode_images(valid_paths)

        print(f"  Encoding {len(prompt_segments)} text segments...")
        text_features = self.encode_texts(prompt_segments)

        # Compute similarity matrix: (num_segments, num_images)
        similarity = self.compute_similarity(image_features, text_features)

        # Select keyframes
        keyframes = []
        used_indices = set()  # Optionally prevent duplicate keyframes

        for seg_idx, segment in enumerate(prompt_segments):
            scores = similarity[seg_idx]

            # Sort by similarity, descending
            sorted_indices = np.argsort(scores)[::-1]

            selected = 0
            for idx in sorted_indices:
                if selected >= top_k:
                    break
                # Optionally skip already-used images to avoid duplicates
                # (commented out to allow re-use if needed)
                # if idx in used_indices:
                #     continue

                name = valid_names[idx]
                pose = name_to_pose.get(name, None)

                kf = {
                    "segment": segment,
                    "image_name": name,
                    "image_path": str(valid_paths[idx]),
                    "similarity": float(scores[idx]),
                    "position": pose["position"] if pose else np.zeros(3),
                    "rotation": pose["rotation"] if pose else np.eye(3),
                }
                keyframes.append(kf)
                used_indices.add(idx)
                selected += 1

        return keyframes

    @torch.no_grad()
    def compute_all_similarities(
        self,
        image_paths: List[Path],
        prompt: str,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Compute similarity scores between all images and a single prompt.
        Useful for visualization / debugging.
        
        Returns:
            scores: (N_images,) array of similarity scores
            names: list of image filenames
        """
        image_features = self.encode_images(image_paths)
        text_features = self.encode_texts([prompt])
        similarity = self.compute_similarity(image_features, text_features)
        scores = similarity[0]
        names = [p.name for p in image_paths]
        return scores, names


class KeyframeVisualizer:
    """Utility for visualizing keyframe selection results."""

    @staticmethod
    def create_keyframe_grid(
        keyframes: List[Dict],
        max_cols: int = 4,
        thumb_size: Tuple[int, int] = (320, 240),
    ) -> Optional[Image.Image]:
        """Create a grid image showing selected keyframes with their prompts."""
        try:
            from PIL import ImageDraw, ImageFont
        except ImportError:
            return None

        n = len(keyframes)
        if n == 0:
            return None

        cols = min(n, max_cols)
        rows = (n + cols - 1) // cols
        w, h = thumb_size
        padding = 40  # Space for text below each image

        grid = Image.new("RGB", (cols * w, rows * (h + padding)), "white")
        draw = ImageDraw.Draw(grid)

        for i, kf in enumerate(keyframes):
            row, col = divmod(i, cols)
            x, y = col * w, row * (h + padding)

            # Load and resize image
            img = Image.open(kf["image_path"]).convert("RGB")
            img = img.resize(thumb_size, Image.LANCZOS)
            grid.paste(img, (x, y))

            # Draw text label
            label = kf["segment"][:40]
            score = f"sim={kf['similarity']:.3f}"
            draw.text((x + 5, y + h + 2), label, fill="black")
            draw.text((x + 5, y + h + 18), score, fill="gray")

        return grid