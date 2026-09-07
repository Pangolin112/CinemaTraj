"""GSCinema: end-to-end camera-trajectory generation pipeline.

Usage:
    python pipeline.py                        # uses config/config_scannetpp.yaml
    python pipeline.py --config my_cfg.yaml   # custom config

Supports two scene formats:
  - "scene_graph": New format with OBB-based bounding boxes, string IDs,
    and PLY collision mesh built from OBBs (ScanNet++ pipeline).
  - "legacy": Old InteriorGS format with labels.json, integer IDs,
    and USD collision mesh.
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
import yaml

# ---------------------------------------------------------------------------
# Project bootstrap
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.user_prompt_translator.prompt_translator import (
    run_from_scene_file,
    run_from_scene_graph,
)
from src.user_prompt_translator.validate_response import (
    validate_and_clean_response,
    print_full_validation_report,
)
from src.subtitle_voiceover_generator.video_renderer import (
    RenderConfig,
    TrajectoryRenderer,
)
from src.trajectory_optimizer.trajectory_optimizer import (
    TrajectoryCombiner,
    optimize_trajectory_result,
)
from src.parametric_trajectory_builder.trajectory_executor import (
    TrajectoryExecutor,
)
from src.utils.import_visualize_ply_gsplat_bbox_video_poses_optimized_poses import (
    visualize_assets,
)

from src.utils.create_mesh_from_obbs import (
    build_scene_mesh,
)

from src.utils.convert_freespace2sdf import (
    load_ply_mesh,
    compute_sdf_grid_meshlib,
)

from src.subtitle_voiceover_generator.video_renderer import (
    generate_subtitle_track,
)

from src.subtitle_voiceover_generator.subtitle_voiceover import (
    run_dual_pipeline
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Preferred PLY filenames in priority order.
# The first one that exists on disk will be used.
PLY_PREFERENCE_ORDER = [
    "point_cloud_30000.ply",
    "point_cloud.ply",
]


def _resolve_ply_path(ply_dir: Path, configured_filename: str) -> Path:
    """
    Resolve the actual PLY file to use.

    Priority:
      1. point_cloud_30000.ply  (higher-iteration, better quality)
      2. The configured filename  (default: point_cloud.ply, always exists)

    If point_cloud_30000.ply exists, use it regardless of what was configured.
    Otherwise fall back to the configured filename.
    """
    # First try the preference list
    for preferred in PLY_PREFERENCE_ORDER:
        candidate = ply_dir / preferred
        if candidate.exists():
            if preferred != configured_filename:
                print(f"  ↑ Using higher-quality PLY: {preferred}")
            return candidate

    # Final fallback: whatever was configured (may not exist yet)
    return ply_dir / configured_filename


@dataclass
class ScenePaths:
    """Derived asset paths for a single scene."""

    scene_graph: Path       # scene graph JSON (new format) or labels.json (legacy)
    ply: Path               # 3DGS point cloud
    mesh: Path              # collision mesh (PLY from OBBs or USD)
    sdf_dir: Path           # directory for cached SDF artefacts
    format: str             # "scene_graph" or "legacy"

    @classmethod
    def from_config(cls, cfg: dict) -> "ScenePaths":
        """Build paths from config, auto-detecting format."""
        fmt = cfg.get("format", "scene_graph")
        scene_id = cfg["scene_id"]

        if fmt == "scene_graph":
            data_root = Path(cfg["data_root"])
            sg_subdir = cfg.get("scene_graph_subdir", f"dslr/sg/sg_03_18")
            sg_filename = cfg.get("scene_graph_filename", f"{scene_id}-simple.json")
            ply_subdir = cfg.get("ply_subdir", "dslr/ply")
            ply_filename = cfg.get("ply_filename", "point_cloud.ply")

            # Resolve PLY: prefer point_cloud_30000.ply if it exists
            ply_dir = data_root / scene_id / ply_subdir
            ply_path = _resolve_ply_path(ply_dir, ply_filename)

            # Output directory: use as-is if skip_scene_subdir, else append scene_id
            if cfg.get("skip_scene_subdir", False):
                output_dir = Path(cfg["output_dir"])
            else:
                output_dir = Path(cfg["output_dir"]) / scene_id

            # SDF / mesh directory: use sdf.cache_dir if provided,
            # otherwise fall back to output_dir
            sdf_cache_dir = cfg.get("sdf", {}).get("cache_dir")
            if sdf_cache_dir:
                sdf_dir = Path(sdf_cache_dir)
                mesh_path = sdf_dir / "obb_mesh.ply"
            else:
                sdf_dir = output_dir
                mesh_path = output_dir / "obb_mesh.ply"

            return cls(
                scene_graph=data_root / scene_id / sg_subdir / sg_filename,
                ply=ply_path,
                mesh=mesh_path,
                sdf_dir=sdf_dir,
                format="scene_graph",
            )
        else:
            # Legacy InteriorGS format
            data_root = Path(cfg["data_root"])
            mesh_stem = scene_id.split("_")[1]

            if cfg.get("skip_scene_subdir", False):
                output_dir = Path(cfg["output_dir"])
            else:
                output_dir = Path(cfg["output_dir"]) / scene_id

            return cls(
                scene_graph=data_root / f"compressed/{scene_id}/labels.json",
                ply=data_root / f"decompressed/{scene_id}/{scene_id}.ply",
                mesh=data_root / f"collision_mesh/{mesh_stem}/{mesh_stem}_collision.usd",
                sdf_dir=output_dir / f"sdf/{scene_id}",
                format="legacy",
            )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
class GSCinema:
    """Prompt → mesh → SDF → trajectory → optimisation → subtitle → render → visualise."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        scene_id = cfg["scene_id"]

        # Output directory: use as-is if skip_scene_subdir, else append scene_id
        if cfg.get("skip_scene_subdir", False):
            self.output_dir = Path(cfg["output_dir"])
        else:
            self.output_dir = Path(cfg["output_dir"]) / scene_id
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.paths = ScenePaths.from_config(cfg)

    # ── public ────────────────────────────────────────────────────────
    def run(self) -> None:
        cfg = self.cfg
        traj_cfg = cfg.get("trajectory", {})
        sdf_cfg = cfg.get("sdf", {})
        opt_cfg = cfg.get("optimisation", {})
        sub_cfg = cfg.get("subtitles", {})
        rnd_cfg = cfg.get("render", {})
        vis_cfg = cfg.get("visualise", {})

        # 0. Build OBB mesh (scene_graph format) + SDF
        sdf_resolution = sdf_cfg.get("resolution", 128)
        self._ensure_mesh_and_sdf(sdf_cfg, sdf_resolution)

        # 1. LLM: prompt → structured camera instructions + viewing prefs
        llm_output, user_specified_order, viewing_preferences = (
            self._prompt_to_instructions()
        )

        # 2. Per-segment trajectory generation
        # Build SDF npz path (already computed in _ensure_mesh_and_sdf)
        sdf_npz = str(self.paths.sdf_dir / f"{cfg['scene_id']}_sdf_res{sdf_resolution}.npz")
 
        executor = TrajectoryExecutor(
            project_root=cfg.get("project_root", str(PROJECT_ROOT)),
            bbox_path=str(self.paths.scene_graph),
            mesh_path=str(self.paths.mesh),
            workspace=str(self.output_dir),
            viewing_preferences=viewing_preferences,
            # NEW: SDF-based collision for anchor selection
            sdf_npz_path=sdf_npz,
            collision_margin=opt_cfg.get("anchor_collision_margin", 0.10),
            debug_candidates=opt_cfg.get("debug_candidates", False),
        )

        fps = rnd_cfg.get("fps", 30.0)
        combiner = TrajectoryCombiner(
            transition_frames=traj_cfg.get("transition_frames", 30),
            fps=fps,
        )

        result = executor.execute_from_llm_output(llm_output, base_name="segments_executor")

        # Reorder decision
        reorder_policy = traj_cfg.get("reorder", "auto")
        if reorder_policy == "always":
            should_reorder = True
        elif reorder_policy == "never":
            should_reorder = False
        else:  # "auto"
            should_reorder = not user_specified_order

        if should_reorder:
            print("  ↻ Reordering segments (user did not specify order)")
            result = combiner.reorder_segments_nearest_neighbour(result)
        else:
            print("  ✓ Keeping original order (user specified sequence)")

        # 3. Collision-aware optimisation
        with open(self.paths.scene_graph, 'r') as f:
            scene_json = json.load(f)

        result = optimize_trajectory_result(
            result, executor, str(self.paths.ply), executor.mesh,
            sdf_resolution=sdf_resolution,
            sdf_cache_dir=str(self.paths.sdf_dir),
            scene_name=cfg["scene_id"],
            verbose=opt_cfg.get("verbose", True),
            scene_json=scene_json,  # >>> NEW: enables cross-room routing
        )

        # 4. Combine & smooth
        combined, frame_mappings = combiner.combine_trajectories(result)
        smoothed = combiner.smooth_trajectory(
            combined,
            window_size=traj_cfg.get("smooth_window", 5),
        )
        self._save_trajectory_artifacts(combiner, smoothed, result, frame_mappings)
        print(f"\n✓ Final trajectory: {len(smoothed)} poses, {smoothed.timestamps[-1]:.1f}s")

        # 5. Subtitles
        if rnd_cfg.get("show_subtitles"):
            sub_track = self._generate_subtitles(result, frame_mappings, combiner, sub_cfg)
        else:
            sub_track = None

        # 6. Render video
        self._render_video(smoothed, sub_track, rnd_cfg)

        # 6.5 Subtitle and voiceover from video
        fmt = rnd_cfg.get("output_format", "mp4")
        output_path = self.output_dir / f"rendered_video.{fmt}"
        run_dual_pipeline(
            video_path=output_path,
            output_dir=self.output_dir / "dual_narration",
            api_key=cfg["api_key"],
            base_url="https://api.openai.com/v1",
            tts_api_key=cfg["api_key"],
            styles=["english"],
            sample_fps=sub_cfg.get("sample_fps", 2),
            vision_model="gpt-4o",
            tts_model="gpt-4o-mini-tts",
            bgm_path=os.environ.get("CINEMATRAJ_BGM"),
            bgm_vol=0.0,
        )

        # 7. Interactive visualisation
        if vis_cfg.get("enabled", True):
            self._visualize(vis_cfg.get("port", 8080))

    # ── Mesh + SDF creation ──────────────────────────────────────────
    def _ensure_mesh_and_sdf(self, sdf_cfg: dict, resolution: int) -> None:
        """
        For scene_graph format:
          1. Build OBB mesh from scene graph JSON → obb_mesh.ply
          2. Compute SDF from that mesh → <scene_id>_sdf_res<R>.npz

        For legacy format:
          - Mesh already exists (USD), just compute SDF if missing.
        """

        sdf_dir = self.paths.sdf_dir
        sdf_dir.mkdir(parents=True, exist_ok=True)

        scene_id = self.cfg["scene_id"]
        npz_path = sdf_dir / f"{scene_id}_sdf_res{resolution}.npz"

        # --- Step 1: Build OBB mesh if scene_graph format ---
        if self.paths.format == "scene_graph":
            mesh_path = self.paths.mesh
            if not mesh_path.exists() or sdf_cfg.get("force_rebuild", False):
                print("\n" + "=" * 60)
                print("BUILDING OBB MESH FROM SCENE GRAPH")
                print("=" * 60)

                with open(self.paths.scene_graph, 'r') as f:
                    scene_json = json.load(f)

                mesh = build_scene_mesh(scene_json)
                mesh_path.parent.mkdir(parents=True, exist_ok=True)
                mesh.export(str(mesh_path))
                print(f"✓ OBB mesh saved → {mesh_path}")
            else:
                print(f"✓ OBB mesh exists: {mesh_path}")

        # --- Step 2: SDF grid ---
        if npz_path.exists() and not sdf_cfg.get("force_rebuild", False):
            print(f"✓ SDF cache hit: {npz_path}")
            return

        print("\n" + "=" * 60)
        print("COMPUTING SDF GRID")
        print("=" * 60)

        mesh = load_ply_mesh(str(self.paths.mesh))

        padding = sdf_cfg.get("padding", 0.5)
        grid_min = mesh.bounds[0] - padding
        grid_max = mesh.bounds[1] + padding

        # Optional interior point and room bounds from config
        interior_point = None
        if "interior_point" in sdf_cfg:
            interior_point = np.array(sdf_cfg["interior_point"], dtype=np.float32)

        room_bounds = None
        if "room_bounds_min" in sdf_cfg and "room_bounds_max" in sdf_cfg:
            room_bounds = (
                np.array(sdf_cfg["room_bounds_min"], dtype=np.float32),
                np.array(sdf_cfg["room_bounds_max"], dtype=np.float32),
            )

        print(f"Building SDF grid ({resolution}³)...")
        sdf_grid = compute_sdf_grid_meshlib(
            mesh, grid_min, grid_max,
            resolution=resolution,
            interior_point=interior_point,
            room_bounds=room_bounds,
        )

        np.savez_compressed(
            npz_path,
            sdf_grid=sdf_grid,
            grid_min=grid_min.astype(np.float32),
            grid_max=grid_max.astype(np.float32),
            resolution=resolution,
        )
        print(f"✓ SDF saved → {npz_path} "
              f"(range [{sdf_grid.min():.3f}, {sdf_grid.max():.3f}], "
              f"{sdf_grid.nbytes / 1024 / 1024:.1f} MB)")

    # ── LLM ──────────────────────────────────────────────────────────
    def _prompt_to_instructions(self) -> tuple:
        """
        Returns:
            (llm_output, user_specified_order, viewing_preferences)
        """
        if self.paths.format == "scene_graph":
            llm_result, objects_summary = run_from_scene_graph(
                user_words=self.cfg["prompt"],
                scene_graph_path=str(self.paths.scene_graph),
                api_key=self.cfg["api_key"],
            )
        else:
            llm_result, objects_summary = run_from_scene_file(
                user_words=self.cfg["prompt"],
                labels_path=str(self.paths.scene_graph),
                api_key=self.cfg["api_key"],
            )

        final = validate_and_clean_response(
            llm_result["parsed"],
            objects_summary,
            auto_correct=True,
            delete_nonexistent=True,
        )
        print_full_validation_report(final)

        parsed = llm_result["parsed"] or {}
        user_specified_order = parsed.get("user_specified_order", False)
        print(f"  LLM reports user_specified_order = {user_specified_order}")

        viewing_preferences = parsed.get("viewing_preferences", {})
        if viewing_preferences:
            print(f"  Viewing preferences for {len(viewing_preferences)} objects:")
            for obj, prefs in viewing_preferences.items():
                print(f"    {obj}: elevation={prefs.get('elevation', '?')}, "
                      f"distance={prefs.get('distance', '?')}, "
                      f"placement={prefs.get('placement', '?')}")
        else:
            print("  No viewing preferences from LLM — will use geometry fallback")

        return final["cleaned_response"], user_specified_order, viewing_preferences

    # ── Subtitles ────────────────────────────────────────────────────
    def _generate_subtitles(self, result, frame_mappings, combiner, sub_cfg):
        return generate_subtitle_track(
            result,
            fps=self.cfg.get("render", {}).get("fps", 30.0),
            transition_frames=self.cfg.get("trajectory", {}).get("transition_frames", 30),
            api_key=self.cfg.get("api_key"),
            ply_path=str(self.paths.ply),
            device=sub_cfg.get("device", "cuda"),
            language=sub_cfg.get("language", "en"),
        )

    # ── Trajectory artefacts ─────────────────────────────────────────
    def _save_trajectory_artifacts(self, combiner, smoothed, result, frame_mappings) -> None:
        combiner.save_trajectory(smoothed, self.output_dir / "combined_trajectory.json")
        combiner.save_anchors(result.all_anchors, self.output_dir / "anchors.json")
        combiner.save_frame_mappings(frame_mappings, self.output_dir / "frame_mappings.json")
        combiner.visualize_trajectory(
            smoothed,
            self.output_dir / "combined_trajectory.png",
            title="Scene Tour",
        )

    # ── Render ───────────────────────────────────────────────────────
    def _render_video(self, smoothed, sub_track, rnd_cfg: dict) -> None:
        renderer = TrajectoryRenderer(
            ply_path=str(self.paths.ply),
            bbox_json_path=str(self.paths.scene_graph),
        )
        fmt = rnd_cfg.get("output_format", "mp4")
        config = RenderConfig(
            width=rnd_cfg.get("width", 1920),
            height=rnd_cfg.get("height", 1080),
            fps=rnd_cfg.get("fps", 30.0),
            fov_y=rnd_cfg.get("fov_y", 60.0),
            gaussian_scale=rnd_cfg.get("gaussian_scale", 1.0),
            show_bboxes=rnd_cfg.get("show_bboxes", False),
            output_format=fmt,
            show_subtitles=rnd_cfg.get("show_subtitles", True),
        )
        output_path = self.output_dir / f"rendered_video.{fmt}"
        renderer.render_trajectory(smoothed, str(output_path), config, subtitle_track=sub_track)

    # ── Visualise ────────────────────────────────────────────────────
    def _visualize(self, port: int) -> None:
        visualize_assets(
            str(self.paths.ply),
            str(self.paths.scene_graph),
            str(self.paths.mesh),
            str(self.output_dir / "combined_trajectory.json"),
            None,
            str(self.output_dir / "anchors.json"),
            port,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="GSCinema pipeline")
    parser.add_argument(
        "--config", type=str, default="config/config_scannetpp.yaml",
        help="Path to YAML config",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    GSCinema(cfg).run()


if __name__ == "__main__":
    main()