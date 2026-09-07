"""
Watertight Occupied-Space Mesh for SDF Computation (v5)
========================================================

Goal: produce a single watertight mesh from OBBs such that SDF computation
gives correct signs — negative INSIDE objects/walls, positive in free space.

Previous approach (v1–v4) extracted the void/free-space surface via:
  voxelize OBBs → flood-fill void → marching cubes on void boundary
This failed because the void is one connected region and the marching-cubes
surface wraps the outer boundary (walls), losing interior object detail after
smoothing.

New approach (v5): run marching cubes directly on the OCCUPANCY field.
  1. Voxelize each OBB independently (per-box containment test).
  2. Run marching cubes on the occupied voxels (level=0.5).
  3. The result is a watertight mesh where each object is a solid blob.
  4. SDF of this mesh: negative inside objects, positive in free space.
     → For collision avoidance, check SDF > 0 (camera in free space).

Optionally, if you need SDF with the OPPOSITE convention (negative = free
space, positive = inside objects), just flip normals with --invert.

Usage:
    python create_free_space.py -i obb_mesh.ply -o watertight_obbs.ply
    python create_free_space.py -i obb_mesh.ply -o watertight_obbs.ply --json scene.json
    python create_free_space.py -i obb_mesh.ply -o watertight_obbs.ply --invert
"""

import argparse
import json
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import trimesh
from scipy import ndimage
from scipy.spatial.transform import Rotation
from skimage import measure


# =========================================================================
# Adaptive voxel size
# =========================================================================

def compute_adaptive_voxel_size(
    mesh: trimesh.Trimesh,
    base_resolution: int,
    min_voxels_per_thin_dim: int = 3,
    max_grid_dim: int = 512,
) -> float:
    bounds = mesh.bounds
    extent = bounds[1] - bounds[0]

    voxel_from_res = extent.max() / base_resolution

    edges = mesh.edges_unique
    edge_vecs = mesh.vertices[edges[:, 1]] - mesh.vertices[edges[:, 0]]
    edge_lengths = np.linalg.norm(edge_vecs, axis=1)
    thin_dim = np.percentile(edge_lengths[edge_lengths > 1e-6], 5)
    voxel_from_thin = thin_dim / min_voxels_per_thin_dim

    padded_extent = extent * 1.10
    voxel_floor = padded_extent.max() / max_grid_dim

    desired = min(voxel_from_res, voxel_from_thin)
    chosen = max(desired, voxel_floor)

    grid_approx = np.ceil(padded_extent / chosen).astype(int)
    print(f"  Voxel from resolution ({base_resolution}): {voxel_from_res:.5f}")
    print(f"  Voxel from thin structures (thin={thin_dim:.4f}m, "
          f"min_vox={min_voxels_per_thin_dim}): {voxel_from_thin:.5f}")
    print(f"  Voxel floor (max_grid_dim={max_grid_dim}): {voxel_floor:.5f}")
    print(f"  -> Chosen: {chosen:.5f}  "
          f"(grid ~ {grid_approx[0]}x{grid_approx[1]}x{grid_approx[2]})")
    return chosen


# =========================================================================
# Per-box voxelization
# =========================================================================

def voxelize_boxes_from_mesh(
    mesh: trimesh.Trimesh,
    min_bound: np.ndarray,
    grid_shape: np.ndarray,
    voxel_size: float,
) -> np.ndarray:
    """Decompose PLY into connected components, voxelize each as a box."""
    occupancy = np.zeros(grid_shape, dtype=bool)
    boxes = mesh.split(only_watertight=False)
    nx, ny, nz = int(grid_shape[0]), int(grid_shape[1]), int(grid_shape[2])

    print(f"  Decomposed into {len(boxes)} boxes")
    t0 = time.time()

    for bi, box_mesh in enumerate(boxes):
        obb = box_mesh.bounding_box_oriented
        T = obb.primitive.transform
        center = T[:3, 3]
        R = T[:3, :3]
        R_inv = R.T
        half_extents = obb.primitive.extents / 2.0

        # AABB in world space
        box_bounds = box_mesh.bounds
        ilo = np.maximum(
            np.floor((box_bounds[0] - min_bound) / voxel_size - 1).astype(int), 0
        )
        ihi = np.minimum(
            np.ceil((box_bounds[1] - min_bound) / voxel_size + 1).astype(int),
            grid_shape
        )

        xs = min_bound[0] + (np.arange(ilo[0], ihi[0]) + 0.5) * voxel_size
        ys = min_bound[1] + (np.arange(ilo[1], ihi[1]) + 0.5) * voxel_size
        zs = min_bound[2] + (np.arange(ilo[2], ihi[2]) + 0.5) * voxel_size

        if len(xs) == 0 or len(ys) == 0 or len(zs) == 0:
            continue

        gx, gy, gz = np.meshgrid(xs, ys, zs, indexing='ij')
        points = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])

        local = (R_inv @ (points - center).T).T
        tol = voxel_size * 0.01
        inside = (
            (np.abs(local[:, 0]) <= half_extents[0] + tol) &
            (np.abs(local[:, 1]) <= half_extents[1] + tol) &
            (np.abs(local[:, 2]) <= half_extents[2] + tol)
        )

        if not inside.any():
            continue

        vi = ((points[inside] - min_bound) / voxel_size).astype(int)
        vi[:, 0] = np.clip(vi[:, 0], 0, nx - 1)
        vi[:, 1] = np.clip(vi[:, 1], 0, ny - 1)
        vi[:, 2] = np.clip(vi[:, 2], 0, nz - 1)
        occupancy[vi[:, 0], vi[:, 1], vi[:, 2]] = True

        if (bi + 1) % 10 == 0 or bi == len(boxes) - 1:
            print(f"    Box {bi+1}/{len(boxes)}", end="\r")

    print()
    elapsed = time.time() - t0
    occ = occupancy.sum()
    print(f"  Per-box voxelization done in {elapsed:.1f}s — "
          f"{occ:,} occupied ({100*occ/occupancy.size:.1f}%)")
    return occupancy


def voxelize_boxes_from_json(
    json_path: str,
    min_bound: np.ndarray,
    grid_shape: np.ndarray,
    voxel_size: float,
) -> np.ndarray:
    """Voxelize directly from JSON OBB parameters (most accurate)."""
    with open(json_path, "r") as f:
        scene_json = json.load(f)

    objects = scene_json["objects"]
    occupancy = np.zeros(grid_shape, dtype=bool)
    nx, ny, nz = int(grid_shape[0]), int(grid_shape[1]), int(grid_shape[2])

    print(f"  Voxelizing from JSON: {len(objects)} objects")
    t0 = time.time()

    for oi, (obj_id, obj_data) in enumerate(objects.items()):
        obb = obj_data["obb"]
        cx, cy, cz, sx, sy, sz, qx, qy, qz, qw = obb

        center = np.array([cx, cy, cz])
        half_extents = np.array([sx, sy, sz]) / 2.0
        R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        R_inv = R.T

        # AABB of OBB in world space
        corners_local = np.array([
            [s0 * half_extents[0], s1 * half_extents[1], s2 * half_extents[2]]
            for s0 in [-1, 1] for s1 in [-1, 1] for s2 in [-1, 1]
        ])
        corners_world = (R @ corners_local.T).T + center
        box_min = corners_world.min(axis=0)
        box_max = corners_world.max(axis=0)

        ilo = np.maximum(
            np.floor((box_min - min_bound) / voxel_size - 1).astype(int), 0
        )
        ihi = np.minimum(
            np.ceil((box_max - min_bound) / voxel_size + 1).astype(int),
            grid_shape
        )

        xs = min_bound[0] + (np.arange(ilo[0], ihi[0]) + 0.5) * voxel_size
        ys = min_bound[1] + (np.arange(ilo[1], ihi[1]) + 0.5) * voxel_size
        zs = min_bound[2] + (np.arange(ilo[2], ihi[2]) + 0.5) * voxel_size

        if len(xs) == 0 or len(ys) == 0 or len(zs) == 0:
            continue

        gx, gy, gz = np.meshgrid(xs, ys, zs, indexing='ij')
        points = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])

        local = (R_inv @ (points - center).T).T
        tol = voxel_size * 0.01
        inside = (
            (np.abs(local[:, 0]) <= half_extents[0] + tol) &
            (np.abs(local[:, 1]) <= half_extents[1] + tol) &
            (np.abs(local[:, 2]) <= half_extents[2] + tol)
        )

        if not inside.any():
            continue

        vi = ((points[inside] - min_bound) / voxel_size).astype(int)
        vi[:, 0] = np.clip(vi[:, 0], 0, nx - 1)
        vi[:, 1] = np.clip(vi[:, 1], 0, ny - 1)
        vi[:, 2] = np.clip(vi[:, 2], 0, nz - 1)
        occupancy[vi[:, 0], vi[:, 1], vi[:, 2]] = True

    elapsed = time.time() - t0
    occ = occupancy.sum()
    print(f"  JSON voxelization done in {elapsed:.1f}s — "
          f"{occ:,} occupied ({100*occ/occupancy.size:.1f}%)")
    return occupancy


def voxelize_surface(mesh, min_bound, grid_shape, voxel_size):
    """Surface sampling supplement."""
    n_samples = max(100_000, len(mesh.faces) * 50)
    points, _ = trimesh.sample.sample_surface(mesh, n_samples)
    points = np.vstack([points, mesh.vertices])

    vi = ((points - min_bound) / voxel_size).astype(int)
    for ax in range(3):
        vi[:, ax] = np.clip(vi[:, ax], 0, int(grid_shape[ax]) - 1)

    occ = np.zeros(grid_shape, dtype=bool)
    occ[vi[:, 0], vi[:, 1], vi[:, 2]] = True
    return occ


# =========================================================================
# Build occupancy grid
# =========================================================================

def create_occupancy_grid(
    mesh: trimesh.Trimesh,
    voxel_size: float,
    padding: float = 0.05,
    closing_iterations: int = 1,
    json_path: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    bounds = mesh.bounds
    extent = bounds[1] - bounds[0]
    pad = extent * padding

    min_bound = bounds[0] - pad
    max_bound = bounds[1] + pad
    padded_extent = max_bound - min_bound
    grid_shape = np.ceil(padded_extent / voxel_size).astype(int)

    print(f"  Grid shape: {tuple(grid_shape)}, voxel size: {voxel_size:.5f}")

    if json_path:
        print(f"  Using JSON: {json_path}")
        occupancy = voxelize_boxes_from_json(
            json_path, min_bound, grid_shape, voxel_size
        )
    else:
        occupancy = voxelize_boxes_from_mesh(
            mesh, min_bound, grid_shape, voxel_size
        )

    # Surface supplement
    print("  Adding surface voxelization...")
    surface_occ = voxelize_surface(mesh, min_bound, grid_shape, voxel_size)
    added = (surface_occ & ~occupancy).sum()
    occupancy |= surface_occ
    print(f"  Surface pass added {added:,} voxels")

    # Closing to seal thin-wall gaps
    if closing_iterations > 0:
        struct = ndimage.generate_binary_structure(3, 1)
        before = occupancy.sum()
        occupancy = ndimage.binary_closing(
            occupancy, struct, iterations=closing_iterations
        )
        print(f"  Closing ({closing_iterations}x): "
              f"{before:,} -> {occupancy.sum():,}")

    return occupancy, min_bound, voxel_size


# =========================================================================
# Marching cubes on occupancy → watertight mesh
# =========================================================================

def extract_occupied_surface(
    occupancy: np.ndarray,
    min_bound: np.ndarray,
    voxel_size: float,
    smooth_iterations: int = 1,
    invert_normals: bool = False,
) -> trimesh.Trimesh:
    """
    Run marching cubes on the occupancy field directly.

    The resulting mesh wraps every occupied region (each OBB becomes a
    solid blob in the mesh).  Normals point OUTWARD from occupied space
    (into free space) by default.

    For SDF:
      - Default normals (outward): SDF < 0 inside objects, SDF > 0 in free space
      - Inverted normals (--invert): SDF > 0 inside objects, SDF < 0 in free space

    Choose based on your SDF library's convention.
    """
    # Pad with zeros so marching cubes closes the surface at grid boundaries
    padded = np.pad(occupancy.astype(float), pad_width=1,
                    mode='constant', constant_values=0.0)

    vertices, faces, _, _ = measure.marching_cubes(
        padded,
        level=0.5,
        spacing=(voxel_size, voxel_size, voxel_size),
    )

    # Shift back: padding added 1 voxel on each side
    vertices += (min_bound - np.array([voxel_size, voxel_size, voxel_size]))

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # Ensure normals point outward from occupied space.
    # Sample a point we KNOW is outside (a grid corner) and check that
    # the nearest face normal points toward it.
    outside_point = min_bound - np.array([1, 1, 1])  # definitely outside
    closest_face = mesh.nearest.on_surface([outside_point])[2][0]
    face_normal = mesh.face_normals[closest_face]
    face_center = mesh.triangles_center[closest_face]
    to_outside = outside_point - face_center
    if np.dot(face_normal, to_outside) < 0:
        mesh.invert()
        print("  Flipped normals to point outward from objects")

    if invert_normals:
        mesh.invert()
        print("  Inverted normals (--invert): normals now point INTO objects")

    print(f"  Mesh: {len(mesh.vertices):,} verts, {len(mesh.faces):,} faces")
    print(f"  Watertight: {mesh.is_watertight}")

    if smooth_iterations > 0:
        trimesh.smoothing.filter_laplacian(mesh, iterations=smooth_iterations)
        print(f"  Smoothed ({smooth_iterations} iterations)")

    return mesh


def cleanup_mesh(mesh: trimesh.Trimesh, min_component_fraction: float = 0.005):
    """Remove tiny disconnected fragments."""
    components = mesh.split(only_watertight=False)
    if len(components) <= 1:
        return mesh

    total_faces = len(mesh.faces)
    threshold = total_faces * min_component_fraction
    kept = [c for c in components if len(c.faces) >= threshold]
    removed = len(components) - len(kept)

    if removed > 0:
        print(f"  Removed {removed} small components")

    if not kept:
        return mesh
    return trimesh.util.concatenate(kept) if len(kept) > 1 else kept[0]


# =========================================================================
# Main pipeline
# =========================================================================

def build_watertight_mesh(
    input_mesh: trimesh.Trimesh,
    voxel_resolution: int = 256,
    closing_iterations: int = 1,
    smooth_iterations: int = 1,
    min_voxels_per_thin_dim: int = 3,
    max_grid_dim: int = 512,
    target_faces: Optional[int] = None,
    json_path: Optional[str] = None,
    invert_normals: bool = False,
) -> trimesh.Trimesh:
    print("\n" + "=" * 60)
    print("WATERTIGHT OBB MESH FOR SDF (v5)")
    print("=" * 60)

    start_time = time.time()
    bounds = input_mesh.bounds
    extent = bounds[1] - bounds[0]

    print(f"\nInput: {len(input_mesh.vertices):,} verts, "
          f"{len(input_mesh.faces):,} faces")
    print(f"Bounds: [{bounds[0]}] -> [{bounds[1]}]")
    print(f"Extent: {extent}")

    # Voxel size
    print("\n[1/4] Computing voxel size...")
    voxel_size = compute_adaptive_voxel_size(
        input_mesh, voxel_resolution, min_voxels_per_thin_dim, max_grid_dim
    )

    # Occupancy grid
    print("\n[2/4] Building occupancy grid...")
    occupancy, min_bound, voxel_size = create_occupancy_grid(
        input_mesh, voxel_size, padding=0.05,
        closing_iterations=closing_iterations,
        json_path=json_path,
    )

    # Marching cubes on occupancy
    print("\n[3/4] Extracting watertight surface...")
    mesh = extract_occupied_surface(
        occupancy, min_bound, voxel_size,
        smooth_iterations=smooth_iterations,
        invert_normals=invert_normals,
    )

    # Cleanup
    print("\n[4/4] Cleanup...")
    mesh = cleanup_mesh(mesh)

    if target_faces and len(mesh.faces) > target_faces:
        print(f"  Decimating -> {target_faces:,} faces...")
        mesh = mesh.simplify_quadric_decimation(target_faces)
        print(f"  Result: {len(mesh.faces):,} faces")

    # Final stats
    print(f"\n  Final mesh: {len(mesh.vertices):,} verts, "
          f"{len(mesh.faces):,} faces")
    print(f"  Watertight: {mesh.is_watertight}")
    print(f"  Volume: {abs(mesh.volume):.4f} m³")
    print(f"\nDone in {time.time() - start_time:.1f}s")
    print("=" * 60 + "\n")
    return mesh


def main():
    scene_id = "09c1414f1b"

    parser = argparse.ArgumentParser(
        description="Build watertight mesh from OBBs for SDF computation"
    )
    parser.add_argument("--input", "-i", type=str,
                        default=f"outputs/scannetpp/{scene_id}/obb_mesh.ply")
    parser.add_argument("--output", "-o", type=str,
                        default=f"outputs/scannetpp/{scene_id}/free_space_{scene_id}.ply")
    parser.add_argument("--json", "-j", type=str, default=None,
                        help="Scene-graph JSON (most accurate OBB source)")
    parser.add_argument("--resolution", "-r", type=int, default=256)
    parser.add_argument("--closing", type=int, default=1)
    parser.add_argument("--smooth", type=int, default=1,
                        help="Smoothing iterations (default: 1, keep low to "
                             "preserve sharp OBB edges)")
    parser.add_argument("--min-voxels-thin", type=int, default=3)
    parser.add_argument("--max-grid-dim", type=int, default=512)
    parser.add_argument("--target-faces", type=int, default=None)
    parser.add_argument("--invert", action="store_true",
                        help="Invert normals (SDF > 0 inside objects)")
    args = parser.parse_args()

    print(f"Loading mesh from {args.input}...")
    input_mesh = trimesh.load(args.input, force='mesh')
    print(f"  {len(input_mesh.vertices):,} vertices, "
          f"{len(input_mesh.faces):,} faces")

    mesh = build_watertight_mesh(
        input_mesh,
        voxel_resolution=args.resolution,
        closing_iterations=args.closing,
        smooth_iterations=args.smooth,
        min_voxels_per_thin_dim=args.min_voxels_thin,
        max_grid_dim=args.max_grid_dim,
        target_faces=args.target_faces,
        json_path=args.json,
        invert_normals=args.invert,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(out))
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()