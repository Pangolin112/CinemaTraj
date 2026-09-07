"""
Cinematographic Camera Trajectory Generation in 3D Scenes
=========================================================
Implementation based on Wu (2025), UC Berkeley MS Thesis.

Pipeline:
    1. Scene Reconstruction (COLMAP)
    2. CLIP-based Keyframe Selection
    3. Trajectory Generation and Refinement (point cloud collision avoidance)
    4. Rendering via gsplat (3D Gaussian Splatting)

No nerfstudio dependency. Uses gsplat for rendering and point cloud/mesh
for collision avoidance.

Usage:
    python main.py --data_dir /path/to/dslr --prompt "zoom in on the mug, pan to the bear"
    python main.py --data_dir /path/to/dslr --gsplat_model_path /path/to/model.ply --prompt "..."
    python main.py --data_dir /path/to/dslr --point_cloud_path /path/to/mesh.ply --prompt "..."
"""

import argparse
import os
import json
import numpy as np
from pathlib import Path

from scene_reconstruction import SceneReconstructor
from keyframe_selection import CLIPKeyframeSelector
from trajectory_generation import TrajectoryGenerator
from rendering import GsplatRenderer, CameraPathExporter
from utils import (
    load_colmap_cameras,
    load_colmap_images,
    parse_prompt_segments,
    save_trajectory,
    visualize_trajectory_3d,
)


def load_point_cloud(data_dir, colmap_model_dir, points3d, point_cloud_path=None):
    """
    Load scene point cloud from various sources for collision checking.
    """
    ply_candidates = []
    if point_cloud_path:
        ply_candidates.append(Path(point_cloud_path))

    ply_candidates.extend([
        data_dir / "point_cloud.ply",
        data_dir / "output" / "point_cloud" / "iteration_30000" / "point_cloud.ply",
        data_dir / "sparse" / "0" / "points3D.ply",
        data_dir.parent / "mesh" / "mesh_aligned_0.05.ply",
        data_dir.parent / "scans" / "mesh_aligned_0.05.ply",
        data_dir.parent / "mesh_aligned_0.05.ply",
    ])
    if colmap_model_dir:
        ply_candidates.append(Path(colmap_model_dir) / "points3D.ply")

    for ply_path in ply_candidates:
        if ply_path is None or not ply_path.exists():
            continue
        try:
            import trimesh
            mesh_or_pc = trimesh.load(str(ply_path), process=False)
            if hasattr(mesh_or_pc, 'vertices'):
                pts = np.array(mesh_or_pc.vertices, dtype=np.float32)
                if len(pts) > 500000:
                    indices = np.random.choice(len(pts), 500000, replace=False)
                    pts = pts[indices]
                print(f"  Loaded point cloud from {ply_path}: {len(pts)} points")
                return pts
        except ImportError:
            try:
                from plyfile import PlyData
                ply_data = PlyData.read(str(ply_path))
                vertex = ply_data['vertex']
                pts = np.stack([
                    np.array(vertex['x'], dtype=np.float32),
                    np.array(vertex['y'], dtype=np.float32),
                    np.array(vertex['z'], dtype=np.float32),
                ], axis=-1)
                if len(pts) > 500000:
                    indices = np.random.choice(len(pts), 500000, replace=False)
                    pts = pts[indices]
                print(f"  Loaded point cloud from {ply_path}: {len(pts)} points")
                return pts
            except (ImportError, Exception) as e:
                print(f"  Could not load {ply_path}: {e}")
        except Exception as e:
            print(f"  Could not load {ply_path}: {e}")

    # Fallback: COLMAP sparse points
    if points3d and len(points3d) > 0:
        scene_points = np.array([p.xyz for p in points3d.values()], dtype=np.float32)
        if len(scene_points) > 10:
            print(f"  Using COLMAP sparse point cloud: {len(scene_points)} points")
            return scene_points

    return None


def main(args):
    print("=" * 70)
    print("Cinematographic Camera Trajectory Generation in 3D Scenes")
    print("=" * 70)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Stage 1: Scene Reconstruction
    # =========================================================================
    print("\n[Stage 1] Scene Reconstruction")
    reconstructor = SceneReconstructor(
        data_dir=data_dir,
        output_dir=output_dir / "colmap",
        colmap_executable=args.colmap_path,
    )

    if args.colmap_model_dir:
        print(f"  Loading existing COLMAP model from: {args.colmap_model_dir}")
        cameras, images, points3d = reconstructor.load_existing(args.colmap_model_dir)
    else:
        print(f"  Running COLMAP on images in: {data_dir}")
        cameras, images, points3d = reconstructor.reconstruct()

    print(f"  Reconstructed {len(images)} camera poses, {len(points3d)} 3D points")

    # =========================================================================
    # Stage 2: Keyframe Selection via CLIP
    # =========================================================================
    print("\n[Stage 2] CLIP-based Keyframe Selection")
    prompt = args.prompt
    print(f"  Prompt: \"{prompt}\"")

    segments = parse_prompt_segments(prompt)
    print(f"  Parsed {len(segments)} instruction segments:")
    for i, seg in enumerate(segments):
        print(f"    [{i+1}] {seg}")

    selector = CLIPKeyframeSelector(
        model_name=args.clip_model,
        device=args.device,
    )

    # Load training images
    if args.images_subdir:
        img_dir = data_dir / args.images_subdir
    else:
        img_dir = data_dir

    image_paths = []
    if img_dir.exists():
        for ext in ["*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG"]:
            image_paths.extend(img_dir.glob(ext))
        if not image_paths:
            print(f"  No images via flat glob in {img_dir}, trying recursive...")
            for ext in ["jpg", "JPG", "jpeg", "JPEG", "png", "PNG"]:
                image_paths.extend(img_dir.rglob(f"*.{ext}"))
    else:
        print(f"  Warning: Image directory does not exist: {img_dir}")

    if not image_paths and images:
        print(f"  Attempting to locate images from COLMAP image names...")
        for img_info in images.values():
            candidate = data_dir / img_info.name
            if candidate.exists():
                image_paths.append(candidate)
            else:
                basename = Path(img_info.name).name
                for search_dir in [img_dir, data_dir, data_dir / "images"]:
                    cand = search_dir / basename
                    if cand.exists():
                        image_paths.append(cand)
                        break
    image_paths = sorted(set(image_paths))

    print(f"  Found {len(image_paths)} training images")
    if image_paths:
        print(f"    Sample: {image_paths[0]}")
    else:
        print(f"  ERROR: No images found. Searched: {img_dir}")
        return

    keyframes = selector.select_keyframes(
        image_paths=image_paths,
        camera_poses=images,
        prompt_segments=segments,
        top_k=args.top_k_per_segment,
    )

    print(f"  Selected {len(keyframes)} keyframes:")
    for kf in keyframes:
        print(f"    Segment: \"{kf['segment']}\" -> {kf['image_name']} "
              f"(similarity: {kf['similarity']:.4f})")

    keyframe_info_path = output_dir / "keyframes.json"
    with open(keyframe_info_path, "w") as f:
        json.dump(
            [
                {
                    "segment": kf["segment"],
                    "image_name": kf["image_name"],
                    "similarity": float(kf["similarity"]),
                    "position": kf["position"].tolist(),
                    "rotation": kf["rotation"].tolist(),
                }
                for kf in keyframes
            ],
            f, indent=2,
        )

    # Save anchors.json for eval.py coverage metric
    anchors_out = {"anchors": [], "method": "cctg_clip"}
    for i, kf in enumerate(keyframes):
        pos = kf["position"]
        rot = kf["rotation"]
        c2w = np.eye(4)
        rot_np = np.array(rot)
        pos_np = np.array(pos)
        if rot_np.shape == (3, 3):
            c2w[:3, :3] = rot_np
        c2w[:3, 3] = pos_np[:3]
        anchors_out["anchors"].append({
            "anchor_idx": i,
            "c2w": c2w.tolist(),
            "position": pos_np.tolist(),
            "rotation": rot_np.tolist(),
            "text": kf.get("segment", ""),
            "similarity": float(kf.get("similarity", -1.0)),
            "image_name": kf.get("image_name", ""),
        })
    anchors_file = output_dir / "anchors.json"
    with open(anchors_file, "w") as f:
        json.dump(anchors_out, f, indent=2)
    print(f"  Saved {len(anchors_out['anchors'])} anchor poses -> {anchors_file}")

    # =========================================================================
    # Stage 3: Trajectory Generation and Refinement
    # =========================================================================
    print("\n[Stage 3] Trajectory Generation & Refinement")

    scene_points = load_point_cloud(
        data_dir=data_dir,
        colmap_model_dir=args.colmap_model_dir,
        points3d=points3d,
        point_cloud_path=args.point_cloud_path,
    )

    trajectory_gen = TrajectoryGenerator(
        point_cloud=scene_points,
        safety_threshold=args.safety_threshold,
        max_refinement_iters=args.max_refinement_iters,
        num_interpolation_steps=args.num_interpolation_steps,
        device=args.device,
    )

    keyframe_positions = np.array([kf["position"] for kf in keyframes])
    keyframe_rotations = np.array([kf["rotation"] for kf in keyframes])

    # Get intrinsics from COLMAP camera
    intrinsics = None
    if keyframes[0].get("intrinsics"):
        intrinsics = keyframes[0]["intrinsics"]
    elif cameras:
        cam = list(cameras.values())[0]
        intrinsics = {
            "fx": float(cam.params[0]),
            "fy": float(cam.params[1]) if len(cam.params) > 1 else float(cam.params[0]),
            "cx": float(cam.params[2]) if len(cam.params) > 2 else cam.width / 2.0,
            "cy": float(cam.params[3]) if len(cam.params) > 3 else cam.height / 2.0,
            "w": int(cam.width),
            "h": int(cam.height),
        }

    raw_trajectory = trajectory_gen.interpolate_trajectory(
        positions=keyframe_positions,
        rotations=keyframe_rotations,
    )
    print(f"  Generated raw trajectory with {len(raw_trajectory['positions'])} frames")

    has_pc = trajectory_gen.pc_checker is not None
    if has_pc:
        print(f"  Running collision refinement using point cloud KD-tree...")
        refined_trajectory = trajectory_gen.refine_trajectory(
            trajectory=raw_trajectory,
            points3d=scene_points,
        )
        print(f"  Refined trajectory: {refined_trajectory['num_corrections']} corrections applied")
    else:
        print("  [Warning] No collision geometry available")
        print("    Provide --point_cloud_path or a COLMAP model with 3D points")
        refined_trajectory = raw_trajectory
        refined_trajectory["num_corrections"] = 0

    # Save trajectory
    traj_path = output_dir / "trajectory.json"
    save_trajectory(refined_trajectory, traj_path)
    print(f"  Saved trajectory to {traj_path}")

    # Export camera path for external renderers
    if intrinsics:
        cam_path = output_dir / "camera_path.json"
        w = intrinsics.get("w", 1920)
        h = intrinsics.get("h", 1080)
        CameraPathExporter.export_gsplat_path(
            trajectory=refined_trajectory,
            intrinsics=intrinsics,
            output_path=cam_path,
            image_width=w,
            image_height=h,
            fps=args.fps,
        )

    # =========================================================================
    # Visualization
    # =========================================================================
    print("\n[Visualization]")
    vis_path = output_dir / "trajectory_visualization.html"
    visualize_trajectory_3d(
        trajectory=refined_trajectory,
        keyframe_positions=keyframe_positions,
        points3d=points3d,
        save_path=vis_path,
    )
    print(f"  Saved 3D visualization to {vis_path}")

    # =========================================================================
    # Rendering (optional, requires gsplat + trained 3DGS model)
    # =========================================================================
    if args.render and args.gsplat_model_path:
        print("\n[Rendering]")
        w = intrinsics.get("w", 1920) if intrinsics else 1920
        h = intrinsics.get("h", 1080) if intrinsics else 1080
        renderer = GsplatRenderer(
            model_path=args.gsplat_model_path,
            device=args.device,
            image_width=w,
            image_height=h,
        )
        video_path = output_dir / "rendered_video.mp4"
        renderer.render_trajectory(
            trajectory=refined_trajectory,
            intrinsics=intrinsics,
            output_path=video_path,
            fps=args.fps,
        )
        print(f"  Rendered video saved to {video_path}")

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cinematographic Camera Trajectory Generation in 3D Scenes"
    )
    scene_name = "0f25f24a4f" # [09c1414f1b, 0f25f24a4f, 0eba3981c9, 1cefb55d50, 0cf2e9402d, 0d2ee665be, 5f99900f09, 21d970d8de, 6115eddb86, 0a7cc12c0e, 00a231a370, 0b031f3119, 0dce89ab21, 00dd871005, 0e75f3c4d9, 0f0191b10b, 1b75758486, 1bb93d185e, 1c7a683c92, 4e0b8cbd33]
    data_dir = "data/ScanNetpp/scenes"
    gs_dir = "data/ScanNetpp/scenes"
    # Data paths
    parser.add_argument("--data_dir", type=str, default=f"{data_dir}/{scene_name}/dslr",
                        help="Directory containing input images")
    parser.add_argument("--images_subdir", type=str, default="images",
                        help="Subdirectory within data_dir containing images")
    parser.add_argument("--output_dir", type=str, default=f"outputs/baselines/Berkeley/{scene_name}",
                        help="Output directory")
    parser.add_argument("--colmap_path", type=str, default="colmap",
                        help="Path to COLMAP executable")
    parser.add_argument("--colmap_model_dir", type=str, default=f"{data_dir}/{scene_name}/dslr/sparse/0",
                        help="Path to existing COLMAP model (skip reconstruction)")
    parser.add_argument("--gsplat_model_path", type=str, default=f"{data_dir}/{scene_name}/dslr/ply/point_cloud.ply",
                        help="Path to trained 3DGS model (.ply or .pt) for rendering")
    parser.add_argument("--point_cloud_path", type=str, default=f"{data_dir}/{scene_name}/dslr/scans/mesh_aligned_0.05.ply",
                        help="Path to .ply file (point cloud or mesh) for collision checking")

    # Prompt
    parser.add_argument("--prompt", type=str, default="Close up of the sink, pan to the computer screen, zoom in to the board, pan to the red sofa.",  #"Close up of the refrigerator, pan to the sink, zoom in to TV, pan to the sofa.", #"Close up of the refrigerator, pan to the sink, zoom in to wash machine, pan to the range hood.",
                        help="Natural language camera instruction prompt")

    # CLIP settings
    parser.add_argument("--clip_model", type=str, default="ViT-B/32",
                        help="CLIP model variant")
    parser.add_argument("--top_k_per_segment", type=int, default=1,
                        help="Number of keyframes per prompt segment")

    # Trajectory settings
    parser.add_argument("--num_interpolation_steps", type=int, default=60,
                        help="Number of interpolated frames between keyframes")
    parser.add_argument("--safety_threshold", type=float, default=0.3,
                        help="Minimum distance threshold for collision avoidance")
    parser.add_argument("--max_refinement_iters", type=int, default=50,
                        help="Maximum iterations for trajectory refinement")

    # Rendering
    parser.add_argument("--render", action="store_false",
                        help="Render video from trajectory")
    parser.add_argument("--fps", type=int, default=24,
                        help="Output video frame rate")

    # Device
    parser.add_argument("--device", type=str, default="cuda",
                        help="Compute device (cuda/cpu)")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)