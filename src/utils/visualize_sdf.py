"""
SDF Visualization Tools
=======================

This script provides tools to visualize precomputed SDF grids:
1. Export to OBJ mesh (for Blender, MeshLab, etc.)
2. Export to volumetric formats (VDB, NRRD)
3. Direct visualization with matplotlib/pyvista

Usage:
    # Export to OBJ for Blender
    python visualize_sdf.py --sdf_path ./sdf_cache/scene_0001_sdf_res64.npz --output scene_sdf.obj
    
    # Export multiple isosurfaces
    python visualize_sdf.py --sdf_path ./sdf_cache/scene_0001_sdf_res64.npz --output scene_sdf.obj --levels 0.0 0.1 0.2
    
    # Visualize with matplotlib (quick preview)
    python visualize_sdf.py --sdf_path ./sdf_cache/scene_0001_sdf_res64.npz --preview
"""

import numpy as np
import argparse
from pathlib import Path


def load_sdf(sdf_path: str) -> dict:
    """Load SDF from .npz file."""
    data = np.load(sdf_path)
    return {
        'sdf_grid': data['sdf_grid'],
        'grid_min': data['grid_min'],
        'grid_max': data['grid_max'],
        'resolution': int(data['resolution']),
    }


def sdf_to_mesh_marching_cubes(
    sdf_grid: np.ndarray,
    grid_min: np.ndarray,
    grid_max: np.ndarray,
    level: float = 0.0,
) -> tuple:
    """
    Convert SDF to mesh using marching cubes.
    
    Args:
        sdf_grid: 3D array of SDF values
        grid_min: Minimum corner of the grid in world coordinates
        grid_max: Maximum corner of the grid in world coordinates
        level: Isosurface level (0.0 = surface, positive = inside, negative = outside)
        
    Returns:
        vertices: (N, 3) array of vertex positions
        faces: (M, 3) array of triangle indices
    """
    try:
        from skimage import measure
    except ImportError:
        raise ImportError("scikit-image required: pip install scikit-image")
    
    # Marching cubes extracts isosurface
    # For SDF: level=0 gives the surface
    vertices, faces, normals, values = measure.marching_cubes(
        sdf_grid, 
        level=level,
        spacing=(
            (grid_max[0] - grid_min[0]) / (sdf_grid.shape[0] - 1),
            (grid_max[1] - grid_min[1]) / (sdf_grid.shape[1] - 1),
            (grid_max[2] - grid_min[2]) / (sdf_grid.shape[2] - 1),
        ),
    )
    
    # Offset vertices to world coordinates
    vertices = vertices + grid_min
    
    return vertices, faces


def export_to_obj(
    vertices: np.ndarray,
    faces: np.ndarray,
    output_path: str,
    flip_normals: bool = False,
):
    """Export mesh to OBJ format."""
    with open(output_path, 'w') as f:
        f.write(f"# SDF Isosurface Mesh\n")
        f.write(f"# Vertices: {len(vertices)}\n")
        f.write(f"# Faces: {len(faces)}\n\n")
        
        # Write vertices
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        
        f.write("\n")
        
        # Write faces (OBJ uses 1-indexed)
        for face in faces:
            if flip_normals:
                f.write(f"f {face[2]+1} {face[1]+1} {face[0]+1}\n")
            else:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    
    print(f"Exported mesh to {output_path}")
    print(f"  Vertices: {len(vertices)}")
    print(f"  Faces: {len(faces)}")


def export_to_ply(
    vertices: np.ndarray,
    faces: np.ndarray,
    output_path: str,
    vertex_colors: np.ndarray = None,
):
    """Export mesh to PLY format with optional vertex colors."""
    try:
        import trimesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        if vertex_colors is not None:
            mesh.visual.vertex_colors = vertex_colors
        mesh.export(output_path)
        print(f"Exported mesh to {output_path}")
    except ImportError:
        # Fallback: write PLY manually
        with open(output_path, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(vertices)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write(f"element face {len(faces)}\n")
            f.write("property list uchar int vertex_indices\n")
            f.write("end_header\n")
            
            for v in vertices:
                f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            
            for face in faces:
                f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
        
        print(f"Exported mesh to {output_path}")


def export_multiple_isosurfaces(
    sdf_data: dict,
    output_dir: str,
    levels: list = [0.0, 0.1, 0.2, 0.3, -0.1, -0.2],
    format: str = 'obj',
):
    """Export multiple isosurface levels."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    sdf_grid = sdf_data['sdf_grid']
    grid_min = sdf_data['grid_min']
    grid_max = sdf_data['grid_max']
    
    for level in levels:
        try:
            vertices, faces = sdf_to_mesh_marching_cubes(
                sdf_grid, grid_min, grid_max, level=level
            )
            
            level_str = f"{level:.2f}".replace('-', 'neg').replace('.', 'p')
            output_path = output_dir / f"sdf_level_{level_str}.{format}"
            
            if format == 'obj':
                # Flip normals for negative levels (outside surface)
                export_to_obj(vertices, faces, str(output_path), flip_normals=(level < 0))
            else:
                export_to_ply(vertices, faces, str(output_path))
                
        except Exception as e:
            print(f"Failed to extract level {level}: {e}")


def visualize_sdf_slices(sdf_data: dict, output_path: str = None):
    """Visualize SDF as 2D slices using matplotlib."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required: pip install matplotlib")
    
    sdf_grid = sdf_data['sdf_grid']
    resolution = sdf_data['resolution']
    
    # Create figure with slices along each axis
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Middle slices
    mid = resolution // 2
    
    # XY slice (Z = mid)
    im0 = axes[0, 0].imshow(sdf_grid[:, :, mid].T, cmap='RdBu', origin='lower')
    axes[0, 0].set_title(f'XY Slice (Z={mid})')
    axes[0, 0].set_xlabel('X')
    axes[0, 0].set_ylabel('Y')
    plt.colorbar(im0, ax=axes[0, 0])
    
    # XZ slice (Y = mid)
    im1 = axes[0, 1].imshow(sdf_grid[:, mid, :].T, cmap='RdBu', origin='lower')
    axes[0, 1].set_title(f'XZ Slice (Y={mid})')
    axes[0, 1].set_xlabel('X')
    axes[0, 1].set_ylabel('Z')
    plt.colorbar(im1, ax=axes[0, 1])
    
    # YZ slice (X = mid)
    im2 = axes[0, 2].imshow(sdf_grid[mid, :, :].T, cmap='RdBu', origin='lower')
    axes[0, 2].set_title(f'YZ Slice (X={mid})')
    axes[0, 2].set_xlabel('Y')
    axes[0, 2].set_ylabel('Z')
    plt.colorbar(im2, ax=axes[0, 2])
    
    # Contour plots showing zero-level (surface)
    axes[1, 0].contour(sdf_grid[:, :, mid].T, levels=[0], colors='black')
    axes[1, 0].contourf(sdf_grid[:, :, mid].T, levels=20, cmap='RdBu', alpha=0.7)
    axes[1, 0].set_title(f'XY Contour (Z={mid})')
    
    axes[1, 1].contour(sdf_grid[:, mid, :].T, levels=[0], colors='black')
    axes[1, 1].contourf(sdf_grid[:, mid, :].T, levels=20, cmap='RdBu', alpha=0.7)
    axes[1, 1].set_title(f'XZ Contour (Y={mid})')
    
    axes[1, 2].contour(sdf_grid[mid, :, :].T, levels=[0], colors='black')
    axes[1, 2].contourf(sdf_grid[mid, :, :].T, levels=20, cmap='RdBu', alpha=0.7)
    axes[1, 2].set_title(f'YZ Contour (X={mid})')
    
    plt.suptitle(f'SDF Visualization (Resolution: {resolution}³)\nBlue=Inside (positive), Red=Outside (negative)')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved slice visualization to {output_path}")
    else:
        plt.show()


def visualize_sdf_3d_pyvista(sdf_data: dict, output_path: str = None):
    """Visualize SDF in 3D using PyVista."""
    try:
        import pyvista as pv
    except ImportError:
        raise ImportError("pyvista required: pip install pyvista")
    
    sdf_grid = sdf_data['sdf_grid']
    grid_min = sdf_data['grid_min']
    grid_max = sdf_data['grid_max']
    resolution = sdf_data['resolution']
    
    # Create uniform grid
    grid = pv.ImageData(
        dimensions=(resolution, resolution, resolution),
        spacing=(
            (grid_max[0] - grid_min[0]) / (resolution - 1),
            (grid_max[1] - grid_min[1]) / (resolution - 1),
            (grid_max[2] - grid_min[2]) / (resolution - 1),
        ),
        origin=grid_min,
    )
    
    # Add SDF values
    grid.point_data['sdf'] = sdf_grid.flatten(order='F')
    
    # Create plotter
    plotter = pv.Plotter()
    
    # Add isosurface at level 0 (the actual surface)
    surface = grid.contour([0.0], scalars='sdf')
    plotter.add_mesh(surface, color='lightblue', opacity=1, label='Surface (SDF=0)')
    
    # Optionally add safety margin surface
    try:
        margin = 0.1
        margin_surface = grid.contour([margin], scalars='sdf')
        plotter.add_mesh(margin_surface, color='green', opacity=0.8, label=f'Safety margin (SDF={margin})')
    except:
        pass
    
    plotter.add_legend()
    plotter.add_axes()
    plotter.camera.view_angle = 90  # wider FOV
    
    if output_path:
        plotter.screenshot(output_path)
        print(f"Saved 3D visualization to {output_path}")
    else:
        plotter.show()


def create_blender_script(sdf_path: str, obj_path: str, output_script: str):
    """Create a Blender Python script to import and visualize the SDF mesh."""
    script = f'''"""
Blender SDF Visualization Script
================================
Run this script in Blender's scripting workspace.

Usage in Blender:
1. Open Blender
2. Go to Scripting workspace
3. Open this script
4. Click "Run Script"
"""

import bpy
import os

# Clear existing mesh objects
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.object.select_by_type(type='MESH')
bpy.ops.object.delete()

# Import the SDF mesh
obj_path = r"{obj_path}"

if os.path.exists(obj_path):
    bpy.ops.wm.obj_import(filepath=obj_path)
    
    # Get the imported object
    obj = bpy.context.selected_objects[0]
    
    # Create a new material with transparency
    mat = bpy.data.materials.new(name="SDF_Material")
    mat.use_nodes = True
    mat.blend_method = 'BLEND'
    
    # Get the principled BSDF node
    bsdf = mat.node_tree.nodes.get('Principled BSDF')
    if bsdf:
        bsdf.inputs['Base Color'].default_value = (0.2, 0.5, 0.8, 1.0)  # Blue
        bsdf.inputs['Alpha'].default_value = 0.5  # Semi-transparent
        bsdf.inputs['Roughness'].default_value = 0.3
    
    # Assign material to object
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)
    
    # Enable smooth shading
    bpy.ops.object.shade_smooth()
    
    # Frame the object in view
    bpy.ops.view3d.view_all(center=True)
    
    print(f"Successfully imported SDF mesh from {{obj_path}}")
else:
    print(f"Error: File not found: {{obj_path}}")

# Optional: Import original collision mesh for comparison
# Uncomment and modify the path below:
# collision_mesh_path = r"path/to/collision_mesh.obj"
# if os.path.exists(collision_mesh_path):
#     bpy.ops.wm.obj_import(filepath=collision_mesh_path)
#     obj2 = bpy.context.selected_objects[0]
#     mat2 = bpy.data.materials.new(name="Collision_Material")
#     mat2.use_nodes = True
#     bsdf2 = mat2.node_tree.nodes.get('Principled BSDF')
#     bsdf2.inputs['Base Color'].default_value = (0.8, 0.2, 0.2, 1.0)  # Red
#     bsdf2.inputs['Alpha'].default_value = 0.3
#     obj2.data.materials.append(mat2)
'''
    
    with open(output_script, 'w') as f:
        f.write(script)
    
    print(f"Created Blender script: {output_script}")
    print(f"  Open Blender -> Scripting -> Open script -> Run")


def main():
    scene_id = "0d2ee665be" #[09c1414f1b, 2c7c10379b, 0eba3981c9, 1cefb55d50, 0cf2e9402d, 0d2ee665be, 5f99900f09, 21d970d8de, 6115eddb86, 0a7cc12c0e, 00a231a370, 0b031f3119, 0dce89ab21, 00dd871005, 0e75f3c4d9, 0f0191b10b, 1b75758486, 1bb93d185e, 1c7a683c92, 4e0b8cbd33]

    parser = argparse.ArgumentParser(description='Visualize SDF grid')
    parser.add_argument('--sdf_path', type=str, default=f"data/ScanNetpp/scenes/{scene_id}/dslr/sdf/{scene_id}_sdf_res128.npz", help='Path to SDF .npz file')
    parser.add_argument('--output', type=str, default=None, help='Output file path')
    parser.add_argument('--format', type=str, default='obj', choices=['obj', 'ply'], help='Output mesh format')
    parser.add_argument('--level', type=float, default=0.0, help='Isosurface level (0=surface)')
    parser.add_argument('--levels', type=float, nargs='+', default=None, help='Multiple isosurface levels')
    parser.add_argument('--preview', action='store_true', help='Show matplotlib preview')
    parser.add_argument('--preview3d', action='store_true', help='Show 3D PyVista preview')
    parser.add_argument('--blender_script', type=str, default=None, help='Generate Blender import script')
    parser.add_argument('--flip_normals', action='store_true', help='Flip mesh normals')
    
    args = parser.parse_args()
    
    # Load SDF
    print(f"Loading SDF from {args.sdf_path}")
    sdf_data = load_sdf(args.sdf_path)
    print(f"  Resolution: {sdf_data['resolution']}³")
    print(f"  Bounds: {sdf_data['grid_min']} to {sdf_data['grid_max']}")
    print(f"  SDF range: [{sdf_data['sdf_grid'].min():.3f}, {sdf_data['sdf_grid'].max():.3f}]")
    
    # Preview
    if args.preview:
        output_img = args.output.replace('.obj', '_slices.png').replace('.ply', '_slices.png') if args.output else None
        visualize_sdf_slices(sdf_data, output_img)
        return
    
    if args.preview3d:
        output_img = args.output.replace('.obj', '_3d.png').replace('.ply', '_3d.png') if args.output else None
        visualize_sdf_3d_pyvista(sdf_data, output_img)
        return
    
    # Export mesh
    if args.output:
        if args.levels:
            # Export multiple levels
            output_dir = Path(args.output).parent / 'sdf_levels'
            export_multiple_isosurfaces(sdf_data, str(output_dir), args.levels, args.format)
        else:
            # Export single level
            vertices, faces = sdf_to_mesh_marching_cubes(
                sdf_data['sdf_grid'],
                sdf_data['grid_min'],
                sdf_data['grid_max'],
                level=args.level,
            )
            
            if args.format == 'obj':
                export_to_obj(vertices, faces, args.output, args.flip_normals)
            else:
                export_to_ply(vertices, faces, args.output)
        
        # Generate Blender script if requested
        if args.blender_script:
            create_blender_script(args.sdf_path, args.output, args.blender_script)
        elif args.output.endswith('.obj'):
            # Auto-generate Blender script
            blender_script = args.output.replace('.obj', '_blender.py')
            create_blender_script(args.sdf_path, args.output, blender_script)


if __name__ == '__main__':
    main()