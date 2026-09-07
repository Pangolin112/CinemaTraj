"""
Scene Reconstruction Module
============================
Wraps COLMAP for Structure-from-Motion reconstruction.
Produces camera poses, sparse point clouds, and prepares data for NeRF training.
"""

import subprocess
import os
import struct
import collections
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional


# ---------------------------------------------------------------------------
# COLMAP binary file readers (adapted from COLMAP scripts)
# ---------------------------------------------------------------------------

CameraModel = collections.namedtuple("CameraModel", ["model_id", "model_name", "num_params"])
CAMERA_MODELS = {
    0: CameraModel(0, "SIMPLE_PINHOLE", 3),
    1: CameraModel(1, "PINHOLE", 4),
    2: CameraModel(2, "SIMPLE_RADIAL", 4),
    3: CameraModel(3, "RADIAL", 5),
    4: CameraModel(4, "OPENCV", 8),
    5: CameraModel(5, "OPENCV_FISHEYE", 8),
    6: CameraModel(6, "FULL_OPENCV", 12),
    7: CameraModel(7, "FOV", 5),
    8: CameraModel(8, "SIMPLE_RADIAL_FISHEYE", 4),
    9: CameraModel(9, "RADIAL_FISHEYE", 5),
    10: CameraModel(10, "THIN_PRISM_FISHEYE", 12),
}

ImageInfo = collections.namedtuple("ImageInfo", [
    "id", "qvec", "tvec", "camera_id", "name", "xys", "point3d_ids"
])

CameraInfo = collections.namedtuple("CameraInfo", [
    "id", "model", "width", "height", "params"
])

Point3DInfo = collections.namedtuple("Point3DInfo", [
    "id", "xyz", "rgb", "error", "image_ids", "point2d_idxs"
])


def qvec2rotmat(qvec):
    """Convert quaternion (w, x, y, z) to 3x3 rotation matrix."""
    w, x, y, z = qvec
    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y],
    ])
    return R


def rotmat2qvec(R):
    """Convert 3x3 rotation matrix to quaternion (w, x, y, z)."""
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
    ]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


# ---------------------------------------------------------------------------
# Binary readers
# ---------------------------------------------------------------------------

def read_cameras_binary(path):
    """Read COLMAP cameras.bin file."""
    cameras = {}
    with open(path, "rb") as f:
        num_cameras = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_cameras):
            camera_id = struct.unpack("<I", f.read(4))[0]
            model_id = struct.unpack("<i", f.read(4))[0]
            width = struct.unpack("<Q", f.read(8))[0]
            height = struct.unpack("<Q", f.read(8))[0]
            num_params = CAMERA_MODELS[model_id].num_params
            params = np.array(struct.unpack(f"<{num_params}d", f.read(8 * num_params)))
            cameras[camera_id] = CameraInfo(
                id=camera_id,
                model=CAMERA_MODELS[model_id].model_name,
                width=width, height=height, params=params,
            )
    return cameras


def read_images_binary(path):
    """Read COLMAP images.bin file."""
    images = {}
    with open(path, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_images):
            image_id = struct.unpack("<I", f.read(4))[0]
            qvec = np.array(struct.unpack("<4d", f.read(32)))
            tvec = np.array(struct.unpack("<3d", f.read(24)))
            camera_id = struct.unpack("<I", f.read(4))[0]
            name = b""
            while True:
                ch = f.read(1)
                if ch == b"\x00":
                    break
                name += ch
            name = name.decode("utf-8")
            num_points2d = struct.unpack("<Q", f.read(8))[0]
            xys = np.array(struct.unpack(f"<{num_points2d * 2}d", f.read(16 * num_points2d)))
            xys = xys.reshape(-1, 2) if num_points2d > 0 else np.zeros((0, 2))
            point3d_ids = np.array(struct.unpack(f"<{num_points2d}q", f.read(8 * num_points2d)))
            images[image_id] = ImageInfo(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=name,
                xys=xys, point3d_ids=point3d_ids,
            )
    return images


def read_points3d_binary(path):
    """Read COLMAP points3D.bin file."""
    points3d = {}
    with open(path, "rb") as f:
        num_points = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_points):
            point_id = struct.unpack("<Q", f.read(8))[0]
            xyz = np.array(struct.unpack("<3d", f.read(24)))
            rgb = np.array(struct.unpack("<3B", f.read(3)))
            error = struct.unpack("<d", f.read(8))[0]
            track_length = struct.unpack("<Q", f.read(8))[0]
            image_ids = []
            point2d_idxs = []
            for _ in range(track_length):
                img_id = struct.unpack("<I", f.read(4))[0]
                p2d_idx = struct.unpack("<I", f.read(4))[0]
                image_ids.append(img_id)
                point2d_idxs.append(p2d_idx)
            points3d[point_id] = Point3DInfo(
                id=point_id, xyz=xyz, rgb=rgb, error=error,
                image_ids=np.array(image_ids),
                point2d_idxs=np.array(point2d_idxs),
            )
    return points3d


# ---------------------------------------------------------------------------
# Text readers
# ---------------------------------------------------------------------------

def read_cameras_text(path):
    """Read COLMAP cameras.txt file."""
    cameras = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            camera_id = int(parts[0])
            model_name = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = np.array([float(p) for p in parts[4:]])
            cameras[camera_id] = CameraInfo(
                id=camera_id, model=model_name,
                width=width, height=height, params=params,
            )
    return cameras


def read_images_text(path):
    """Read COLMAP images.txt file."""
    images = {}
    with open(path, "r") as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    # images.txt has pairs of lines: header + points
    for i in range(0, len(lines), 2):
        parts = lines[i].split()
        image_id = int(parts[0])
        qvec = np.array([float(parts[j]) for j in range(1, 5)])
        tvec = np.array([float(parts[j]) for j in range(5, 8)])
        camera_id = int(parts[8])
        name = parts[9]

        # Parse 2D points (next line)
        if i + 1 < len(lines):
            pts_parts = lines[i + 1].split()
            num_points = len(pts_parts) // 3
            xys = np.zeros((num_points, 2))
            point3d_ids = np.zeros(num_points, dtype=np.int64)
            for j in range(num_points):
                xys[j, 0] = float(pts_parts[3 * j])
                xys[j, 1] = float(pts_parts[3 * j + 1])
                point3d_ids[j] = int(pts_parts[3 * j + 2])
        else:
            xys = np.zeros((0, 2))
            point3d_ids = np.zeros(0, dtype=np.int64)

        images[image_id] = ImageInfo(
            id=image_id, qvec=qvec, tvec=tvec,
            camera_id=camera_id, name=name,
            xys=xys, point3d_ids=point3d_ids,
        )
    return images


def read_points3d_text(path):
    """Read COLMAP points3D.txt file.

    Format per line (after comment lines starting with #):
        POINT3D_ID X Y Z R G B ERROR TRACK[] as (IMAGE_ID POINT2D_IDX) pairs
    """
    points3d = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            point_id = int(parts[0])
            xyz = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
            rgb = np.array([int(parts[4]), int(parts[5]), int(parts[6])], dtype=np.uint8)
            error = float(parts[7])
            # Track: pairs of (IMAGE_ID, POINT2D_IDX)
            image_ids = []
            point2d_idxs = []
            for j in range(8, len(parts) - 1, 2):
                image_ids.append(int(parts[j]))
                point2d_idxs.append(int(parts[j + 1]))
            points3d[point_id] = Point3DInfo(
                id=point_id, xyz=xyz, rgb=rgb, error=error,
                image_ids=np.array(image_ids, dtype=np.int64),
                point2d_idxs=np.array(point2d_idxs, dtype=np.int64),
            )
    return points3d


class SceneReconstructor:
    """
    Wraps COLMAP to perform Structure-from-Motion on a set of input images.
    Produces camera poses, intrinsics, and a sparse 3D point cloud.
    """

    def __init__(self, data_dir, output_dir, colmap_executable="colmap"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.colmap_executable = colmap_executable

    def reconstruct(self):
        """Run full COLMAP reconstruction pipeline."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        database_path = self.output_dir / "database.db"
        sparse_dir = self.output_dir / "sparse"

        # Feature extraction
        subprocess.run([
            self.colmap_executable, "feature_extractor",
            "--database_path", str(database_path),
            "--image_path", str(self.data_dir),
            "--ImageReader.single_camera", "1",
        ], check=True)

        # Feature matching
        subprocess.run([
            self.colmap_executable, "exhaustive_matcher",
            "--database_path", str(database_path),
        ], check=True)

        # Sparse reconstruction
        sparse_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            self.colmap_executable, "mapper",
            "--database_path", str(database_path),
            "--image_path", str(self.data_dir),
            "--output_path", str(sparse_dir),
        ], check=True)

        # Load results from the first reconstruction (sparse/0)
        model_dir = sparse_dir / "0"
        return self._load_model(model_dir)

    def load_existing(self, model_dir):
        """Load an existing COLMAP model from a directory."""
        return self._load_model(Path(model_dir))

    def _load_model(self, model_dir):
        """Load cameras, images, and points3D from a COLMAP model directory."""
        model_dir = Path(model_dir)

        # Try binary format first, then text
        cameras_bin = model_dir / "cameras.bin"
        cameras_txt = model_dir / "cameras.txt"
        images_bin = model_dir / "images.bin"
        images_txt = model_dir / "images.txt"
        points_bin = model_dir / "points3D.bin"
        points_txt = model_dir / "points3D.txt"

        if cameras_bin.exists():
            cameras = read_cameras_binary(str(cameras_bin))
        elif cameras_txt.exists():
            cameras = read_cameras_text(str(cameras_txt))
        else:
            raise FileNotFoundError(f"No cameras file found in {model_dir}")

        if images_bin.exists():
            images = read_images_binary(str(images_bin))
        elif images_txt.exists():
            images = read_images_text(str(images_txt))
        else:
            raise FileNotFoundError(f"No images file found in {model_dir}")

        if points_bin.exists():
            points3d = read_points3d_binary(str(points_bin))
        elif points_txt.exists():
            points3d = read_points3d_text(str(points_txt))
        else:
            points3d = {}

        return cameras, images, points3d

    def get_camera_positions(self, images):
        """Extract world-space camera positions from COLMAP images.

        COLMAP stores camera-to-world as: R, t where the world position is
        C = -R^T @ t
        """
        positions = []
        for img_info in images.values():
            R = qvec2rotmat(img_info.qvec)
            t = img_info.tvec
            C = -R.T @ t
            positions.append(C)
        return np.array(positions)

    def get_point_cloud(self, points3d):
        """Extract point cloud positions and colors."""
        if not points3d:
            return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8)
        positions = np.array([p.xyz for p in points3d.values()])
        colors = np.array([p.rgb for p in points3d.values()])
        return positions, colors