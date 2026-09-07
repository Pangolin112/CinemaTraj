"""
Build a combined triangle mesh from oriented bounding boxes (OBBs) defined in a
scene-graph JSON file and save it for downstream SDF computation.

New JSON format stores objects as a flat dict mapping object_id to a list of
10 floats: [cx, cy, cz, sx, sy, sz, qx, qy, qz, qw]

Usage:
    python build_obb_mesh.py --input scene_graph.json \
                             --output outputs/scannetpp/sdf/obb_mesh.ply
"""

import argparse
import json
import os

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation


def qxyzw_to_rotation_matrix(qxyzw: list) -> np.ndarray:
    """Convert a (qx, qy, qz, qw) quaternion to a 3×3 rotation matrix."""
    qx, qy, qz, qw = qxyzw
    return Rotation.from_quat([qx, qy, qz, qw]).as_matrix()  # scipy uses (x,y,z,w)


def obb_to_mesh(center: list, size: list, qxyzw: list) -> trimesh.Trimesh:
    """
    Create a box mesh for one OBB.

    Parameters
    ----------
    center : [cx, cy, cz]
    size   : [sx, sy, sz]  – full extents along the local axes
    qxyzw  : [qx, qy, qz, qw]

    Returns
    -------
    trimesh.Trimesh  – the oriented box as a watertight mesh
    """
    box = trimesh.creation.box(extents=size)

    R = qxyzw_to_rotation_matrix(qxyzw)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = center

    box.apply_transform(T)
    return box


SKIP_CLASSES = {"door_frame"}


def build_scene_mesh(scene_json: dict) -> trimesh.Trimesh:
    """
    Iterate over all objects in the scene graph and merge their OBBs into one
    combined mesh.

    Expected JSON structure:
        {
          "obb_format": ["cx","cy","cz","sx","sy","sz","qx","qy","qz","qw"],
          "objects": {
              "table_0": [cx, cy, cz, sx, sy, sz, qx, qy, qz, qw],
              ...
          },
          ...
        }
    """
    objects = scene_json["objects"]

    meshes = []
    for obj_id, obj_data in objects.items():
        # Extract class name by stripping the trailing _N index
        obj_class = obj_id.rsplit("_", 1)[0]
        if obj_class in SKIP_CLASSES:
            continue

        obb_values = obj_data["obb"]
        cx, cy, cz, sx, sy, sz, qx, qy, qz, qw = obb_values
        mesh = obb_to_mesh(
            center=[cx, cy, cz],
            size=[sx, sy, sz],
            qxyzw=[qx, qy, qz, qw],
        )
        meshes.append(mesh)

    combined = trimesh.util.concatenate(meshes)
    print(f"Combined mesh: {len(meshes)} OBBs → "
          f"{len(combined.vertices)} verts, {len(combined.faces)} faces")
    return combined


def main():
    scene_id = "09c1414f1b"
    parser = argparse.ArgumentParser(description="OBB → mesh builder")
    parser.add_argument("--input", "-i",
                        default=f"data/ScanNetpp/scenes/{scene_id}/dslr/sg/{scene_id}-simple.json",
                        help="Path to the scene-graph JSON file")
    parser.add_argument("--output", "-o",
                        default=f"outputs/scannetpp/{scene_id}/obb_mesh.ply",
                        help="Where to save the combined mesh (default: .ply)")
    args = parser.parse_args()

    with open(args.input, "r") as f:
        scene_json = json.load(f)

    mesh = build_scene_mesh(scene_json)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    mesh.export(args.output)
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()