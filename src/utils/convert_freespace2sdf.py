#!/usr/bin/env python3
"""
Convert PLY mesh to SDF field and save to .npz file.
Uses MeshLib for fast SDF computation.

Output format matches DifferentiableMeshSDF cache format:
- sdf_grid: (D, H, W) float32 array
- grid_min: (3,) float32 array  
- grid_max: (3,) float32 array
- resolution: int

SDF convention:
- Positive = inside mesh (safe)
- Negative = outside mesh (collision)
"""

import numpy as np
import trimesh
from pathlib import Path
import argparse


def load_ply_mesh(ply_path: str) -> trimesh.Trimesh:
    """Load mesh from PLY file."""
    print(f"Loading PLY file: {ply_path}")
    mesh = trimesh.load(ply_path, force='mesh')

    print(f"Loaded mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    print(f"Bounds: {mesh.bounds}")
    print(f"Watertight: {mesh.is_watertight}")

    return mesh


def trimesh_to_meshlib(mesh: trimesh.Trimesh):
    """Convert trimesh mesh to MeshLib mesh."""
    from meshlib import mrmeshnumpy as mn

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    mr_mesh = mn.meshFromFacesVerts(faces, vertices)
    return mr_mesh


def ensure_outward_normals(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Ensure mesh normals point outward (away from interior).
    For a free space mesh, normals should point INTO the room.

    We check by computing the signed volume - if negative, normals are inverted.
    """
    if mesh.is_watertight:
        volume = mesh.volume
        if volume < 0:
            print(f"  Mesh has inverted normals (volume={volume:.4f}), flipping...")
            mesh.invert()
        else:
            print(f"  Mesh normals OK (volume={volume:.4f})")
    else:
        print(f"  Mesh not watertight, cannot verify normals")

    return mesh


def close_mesh_holes(mesh: trimesh.Trimesh, max_hole_edges: int = 500) -> trimesh.Trimesh:
    """
    Close holes in mesh (e.g., door openings) by filling boundary loops.

    Args:
        mesh: Input mesh with holes
        max_hole_edges: Maximum edges in a hole to fill (skip very large holes)

    Returns:
        Mesh with holes filled
    """
    print(f"  Detecting and closing holes...")

    edges = mesh.edges_unique

    face_adjacency_edges = mesh.face_adjacency_edges
    interior_edges_set = set(map(tuple, np.sort(face_adjacency_edges, axis=1)))

    boundary_edges = []
    for i, edge in enumerate(edges):
        edge_tuple = tuple(sorted(edge))
        if edge_tuple not in interior_edges_set:
            boundary_edges.append(edge)

    if not boundary_edges:
        print(f"    No boundary edges found, mesh appears closed")
        return mesh

    boundary_edges = np.array(boundary_edges)
    print(f"    Found {len(boundary_edges)} boundary edges")

    # Group boundary edges into loops
    edge_dict = {}
    for e in boundary_edges:
        for v in e:
            if v not in edge_dict:
                edge_dict[v] = []
            edge_dict[v].append(e)

    # Trace boundary loops
    used_edges = set()
    loops = []

    for start_edge in boundary_edges:
        edge_key = (min(start_edge), max(start_edge))
        if edge_key in used_edges:
            continue

        loop = [start_edge[0], start_edge[1]]
        used_edges.add(edge_key)

        while True:
            current_v = loop[-1]
            found_next = False

            for edge in edge_dict.get(current_v, []):
                edge_key = (min(edge), max(edge))
                if edge_key in used_edges:
                    continue

                next_v = edge[0] if edge[1] == current_v else edge[1]

                if next_v == loop[0] and len(loop) > 2:
                    used_edges.add(edge_key)
                    loops.append(loop)
                    found_next = True
                    break
                elif next_v not in loop:
                    loop.append(next_v)
                    used_edges.add(edge_key)
                    found_next = True
                    break

            if not found_next:
                break

    print(f"    Found {len(loops)} boundary loops")

    if not loops:
        return mesh

    # Fill each hole with a fan triangulation
    new_faces = list(mesh.faces)
    vertices = mesh.vertices.copy()

    for i, loop in enumerate(loops):
        if len(loop) > max_hole_edges:
            print(f"    Skipping loop {i} with {len(loop)} edges (too large)")
            continue

        loop_vertices = vertices[loop]
        centroid = loop_vertices.mean(axis=0)

        centroid_idx = len(vertices)
        vertices = np.vstack([vertices, centroid])

        for j in range(len(loop)):
            v1 = loop[j]
            v2 = loop[(j + 1) % len(loop)]
            new_faces.append([centroid_idx, v1, v2])

        print(f"    Filled loop {i} with {len(loop)} edges")

    filled_mesh = trimesh.Trimesh(vertices=vertices, faces=np.array(new_faces))
    filled_mesh.fix_normals()

    print(f"  Hole filling complete: {len(mesh.faces)} -> {len(filled_mesh.faces)} faces")
    print(f"  Watertight after filling: {filled_mesh.is_watertight}")

    return filled_mesh


def compute_unsigned_distance_field(
    mesh: trimesh.Trimesh,
    grid_min: np.ndarray,
    grid_max: np.ndarray,
    resolution: int = 128,
) -> np.ndarray:
    """Compute unsigned distance field using MeshLib."""
    from meshlib import mrmeshpy as mm
    from meshlib import mrmeshnumpy as mn

    mr_mesh = trimesh_to_meshlib(mesh)

    grid_size = grid_max - grid_min
    voxel_size = grid_size / (resolution - 1)

    params = mm.MeshToDistanceVolumeParams()
    params.vol.origin = mm.Vector3f(float(grid_min[0]), float(grid_min[1]), float(grid_min[2]))
    params.vol.voxelSize = mm.Vector3f(float(voxel_size[0]), float(voxel_size[1]), float(voxel_size[2]))
    params.vol.dimensions = mm.Vector3i(resolution, resolution, resolution)

    params.dist.signMode = mm.SignDetectionMode.Unsigned
    max_dist = np.linalg.norm(grid_size) * 0.5
    params.dist.maxDistSq = float(max_dist * max_dist)

    volume = mm.meshToDistanceVolume(mr_mesh, params)
    udf_grid = mn.getNumpy3Darray(volume)

    return udf_grid.astype(np.float32)


def flood_fill_from_interior(
    udf_grid: np.ndarray,
    grid_min: np.ndarray,
    grid_max: np.ndarray,
    interior_point: np.ndarray,
    surface_threshold: float = 0.01,
    room_bounds: tuple = None,
) -> np.ndarray:
    """
    Flood-fill from a known interior point to identify the room interior.

    Args:
        udf_grid: Unsigned distance field
        grid_min: Grid minimum coordinates
        grid_max: Grid maximum coordinates
        interior_point: A point known to be inside the room (world coordinates)
        surface_threshold: Distance threshold to consider as "on surface" (barrier)
        room_bounds: Optional (min, max) bounds to clip results - prevents leakage through doors

    Returns:
        Boolean mask where True = inside room (safe), False = walls/outside
    """
    from scipy import ndimage

    resolution = udf_grid.shape[0]

    # Convert world point to grid indices
    grid_size = grid_max - grid_min
    normalized = (interior_point - grid_min) / grid_size
    grid_idx = (normalized * (resolution - 1)).astype(int)
    grid_idx = np.clip(grid_idx, 0, resolution - 1)

    print(f"    Interior point (world): {interior_point}")
    print(f"    Interior point (grid): {grid_idx}")
    print(f"    UDF at interior point: {udf_grid[grid_idx[0], grid_idx[1], grid_idx[2]]:.4f}")

    # Create barrier mask: points very close to surface block flood fill
    barrier_mask = udf_grid < surface_threshold

    # If room_bounds specified, also treat outside bounds as barrier
    if room_bounds is not None:
        room_min, room_max = room_bounds
        print(f"    Applying room bounds clip: {room_min} to {room_max}")

        x = np.linspace(grid_min[0], grid_max[0], resolution)
        y = np.linspace(grid_min[1], grid_max[1], resolution)
        z = np.linspace(grid_min[2], grid_max[2], resolution)
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')

        outside_bounds = (
            (xx < room_min[0]) | (xx > room_max[0]) |
            (yy < room_min[1]) | (yy > room_max[1]) |
            (zz < room_min[2]) | (zz > room_max[2])
        )
        barrier_mask = barrier_mask | outside_bounds
        print(f"    Bounds mask blocks {outside_bounds.sum()} additional voxels")

    # Initialize inside mask
    inside_seed = np.zeros_like(udf_grid, dtype=bool)
    inside_seed[grid_idx[0], grid_idx[1], grid_idx[2]] = True

    if barrier_mask[grid_idx[0], grid_idx[1], grid_idx[2]]:
        print(f"    WARNING: Interior point is on surface, searching nearby...")
        found = False
        for offset in range(1, 10):
            if found:
                break
            for dx in [-offset, 0, offset]:
                for dy in [-offset, 0, offset]:
                    for dz in [-offset, 0, offset]:
                        ni = np.clip(grid_idx + np.array([dx, dy, dz]), 0, resolution - 1)
                        if not barrier_mask[ni[0], ni[1], ni[2]]:
                            inside_seed[ni[0], ni[1], ni[2]] = True
                            print(f"    Found valid seed at offset ({dx}, {dy}, {dz})")
                            found = True
                            break
                    if found:
                        break
                if found:
                    break

    # Flood fill
    traversable = ~barrier_mask
    structure = ndimage.generate_binary_structure(3, 1)  # 6-connectivity

    inside_mask = inside_seed.copy()
    prev_count = 0
    iteration = 0
    max_iterations = resolution * 3

    while True:
        dilated = ndimage.binary_dilation(inside_mask, structure=structure)
        inside_mask = dilated & traversable

        current_count = inside_mask.sum()
        iteration += 1

        if current_count == prev_count or iteration >= max_iterations:
            break
        prev_count = current_count

        if iteration % 50 == 0:
            print(f"    Flood fill iteration {iteration}, inside voxels: {current_count}")

    print(f"  Flood fill completed in {iteration} iterations")
    print(f"  Inside (room) voxels: {inside_mask.sum()}, Outside (walls) voxels: {(~inside_mask).sum()}")

    return inside_mask


def compute_sdf_grid_meshlib(
    mesh: trimesh.Trimesh,
    grid_min: np.ndarray,
    grid_max: np.ndarray,
    resolution: int = 128,
    interior_point: np.ndarray = None,
    room_bounds: tuple = None,
) -> np.ndarray:
    """
    Compute SDF grid for free-space mesh (hollow shell with wall thickness).

    Uses unsigned distance + flood-fill from interior point to correctly
    identify the room interior vs wall space.

    SDF convention:
    - Positive = inside room (safe for camera)
    - Negative = walls or outside (collision)

    Args:
        mesh: The free-space mesh (hollow shell)
        grid_min/max: Grid bounds
        resolution: Grid resolution
        interior_point: A point known to be inside the room. If None, uses mesh centroid.
        room_bounds: Optional (min, max) tuple to clip flood-fill - prevents leakage through doors
    """
    from meshlib import mrmeshpy as mm
    from meshlib import mrmeshnumpy as mn

    print(f"  Mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
    print(f"  Watertight: {mesh.is_watertight}")

    # Always try to close holes (door openings)
    mesh = close_mesh_holes(mesh)

    # Use mesh centroid as interior point if not specified
    if interior_point is None:
        interior_point = mesh.centroid
        print(f"  Using mesh centroid as interior point: {interior_point}")

    grid_size = grid_max - grid_min
    voxel_size = (grid_size / (resolution - 1)).max()

    # Compute unsigned distance field
    print(f"  Computing unsigned distance field...")
    mr_mesh = trimesh_to_meshlib(mesh)
    voxel_size_vec = grid_size / (resolution - 1)

    params = mm.MeshToDistanceVolumeParams()
    params.vol.origin = mm.Vector3f(float(grid_min[0]), float(grid_min[1]), float(grid_min[2]))
    params.vol.voxelSize = mm.Vector3f(float(voxel_size_vec[0]), float(voxel_size_vec[1]), float(voxel_size_vec[2]))
    params.vol.dimensions = mm.Vector3i(resolution, resolution, resolution)
    params.dist.signMode = mm.SignDetectionMode.Unsigned

    max_dist = np.linalg.norm(grid_size) * 0.5
    params.dist.maxDistSq = float(max_dist * max_dist)

    volume = mm.meshToDistanceVolume(mr_mesh, params)
    udf_grid = mn.getNumpy3Darray(volume).astype(np.float32)

    print(f"  UDF range: [{udf_grid.min():.4f}, {udf_grid.max():.4f}]")

    # Surface threshold
    surface_threshold = voxel_size * 0.5
    print(f"  Surface threshold: {surface_threshold:.4f}")

    # Flood-fill from interior point to find room interior
    print(f"  Flood-filling from interior point...")
    inside_mask = flood_fill_from_interior(
        udf_grid, grid_min, grid_max, interior_point, surface_threshold, room_bounds
    )

    # Create signed distance field
    # Inside room = positive, walls/outside = negative
    sdf_grid = udf_grid.copy()
    sdf_grid[~inside_mask] = -sdf_grid[~inside_mask]

    print(f"  SDF computed! Shape: {sdf_grid.shape}")
    print(f"  SDF range: [{sdf_grid.min():.4f}, {sdf_grid.max():.4f}]")

    inside_count = inside_mask.sum()
    total_count = inside_mask.size
    print(f"  Inside (room) ratio: {100*inside_count/total_count:.1f}%")

    return sdf_grid


def main():
    scene_id = "09c1414f1b"
    parser = argparse.ArgumentParser(description='Convert PLY mesh to SDF')
    parser.add_argument('--input', type=str, default=f"outputs/scannetpp/{scene_id}/obb_mesh.ply",
                        help='Path to input PLY file')
    parser.add_argument('--output', type=str, default=f"outputs/scannetpp/{scene_id}/{scene_id}_sdf_res128.npz",
                        help='Path to output .npz file')
    parser.add_argument('--resolution', type=int, default=128,
                        help='SDF grid resolution')
    parser.add_argument('--padding', type=float, default=0.5,
                        help='Padding around mesh bounds')
    parser.add_argument('--invert_sign', action='store_true',
                        help='Invert SDF sign (positive=collision, negative=safe)')
    parser.add_argument('--interior_point', type=float, nargs=3, default=None,
                        metavar=('X', 'Y', 'Z'),
                        help='Known interior point in world coordinates (default: mesh centroid)')
    parser.add_argument('--room_bounds_min', type=float, nargs=3, default=None,
                        metavar=('X', 'Y', 'Z'),
                        help='Room bounds minimum (clips flood-fill to prevent door leakage)')
    parser.add_argument('--room_bounds_max', type=float, nargs=3, default=None,
                        metavar=('X', 'Y', 'Z'),
                        help='Room bounds maximum (clips flood-fill to prevent door leakage)')
    args = parser.parse_args()

    # Load PLY mesh
    mesh = load_ply_mesh(args.input)

    # Compute grid bounds with padding
    grid_min = mesh.bounds[0] - args.padding
    grid_max = mesh.bounds[1] + args.padding
    print(f"\nGrid bounds (with {args.padding} padding):")
    print(f"  min: {grid_min}")
    print(f"  max: {grid_max}")

    # Compute SDF
    print(f"\nComputing SDF with resolution {args.resolution}³...")
    interior_point = np.array(args.interior_point) if args.interior_point else None

    # Room bounds to clip flood-fill
    room_bounds = None
    if args.room_bounds_min and args.room_bounds_max:
        room_bounds = (np.array(args.room_bounds_min), np.array(args.room_bounds_max))
        print(f"Using manual room bounds: {room_bounds[0]} to {room_bounds[1]}")

    sdf_grid = compute_sdf_grid_meshlib(mesh, grid_min, grid_max,
                                         resolution=args.resolution,
                                         interior_point=interior_point,
                                         room_bounds=room_bounds)

    # Optionally invert sign
    if args.invert_sign:
        sdf_grid = -sdf_grid
        print(f"Inverted SDF sign. New range: [{sdf_grid.min():.4f}, {sdf_grid.max():.4f}]")

    # Determine output path
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save SDF
    np.savez_compressed(
        output_path,
        sdf_grid=sdf_grid,
        grid_min=grid_min.astype(np.float32),
        grid_max=grid_max.astype(np.float32),
        resolution=args.resolution,
    )
    print(f"\nSaved SDF to: {output_path}")

    # Also save mesh as PLY for visualization
    mesh_output_path = output_path.with_suffix('.ply')
    mesh.export(mesh_output_path)
    print(f"Saved mesh to: {mesh_output_path}")

    # Summary
    print(f"\n{'='*50}")
    print(f"Summary")
    print(f"{'='*50}")
    print(f"Input: {args.input}")
    print(f"Output: {output_path}")
    print(f"Mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    print(f"Watertight: {mesh.is_watertight}")
    print(f"SDF shape: {sdf_grid.shape}")
    print(f"SDF range: [{sdf_grid.min():.4f}, {sdf_grid.max():.4f}]")
    print(f"Grid min: {grid_min}")
    print(f"Grid max: {grid_max}")
    print(f"Memory: {sdf_grid.nbytes / 1024 / 1024:.1f} MB")


if __name__ == '__main__':
    main()


