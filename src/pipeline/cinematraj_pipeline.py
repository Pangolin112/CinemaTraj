"""GSCinema: end-to-end camera-trajectory generation pipeline.

Usage:
    python pipeline.py                                  # full pipeline
    python pipeline.py --config my_cfg.yaml             # custom config
    python pipeline.py --config ablation_cfg.yaml       # ablation study

Ablation study modes (configured in YAML under `ablation:`):
  - no_anchor_determinator:    Replace AnchorDeterminator with CLIP-based selection
  - no_parametric_trajectories: Replace parametric builder with GenDoP 6-DoF diffusion
  - no_sdf_optimization:       Replace SDF optimizer with point-cloud KD-tree refinement

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Preferred PLY filenames in priority order.
PLY_PREFERENCE_ORDER = [
    "point_cloud_30000.ply",
    "point_cloud.ply",
]


def _resolve_ply_path(ply_dir: Path, configured_filename: str) -> Path:
    for preferred in PLY_PREFERENCE_ORDER:
        candidate = ply_dir / preferred
        if candidate.exists():
            if preferred != configured_filename:
                print(f"  ↑ Using higher-quality PLY: {preferred}")
            return candidate
    return ply_dir / configured_filename


@dataclass
class ScenePaths:
    """Derived asset paths for a single scene."""
    scene_graph: Path
    ply: Path
    mesh: Path
    sdf_dir: Path
    format: str

    @classmethod
    def from_config(cls, cfg: dict) -> "ScenePaths":
        fmt = cfg.get("format", "scene_graph")
        scene_id = cfg["scene_id"]

        if fmt == "scene_graph":
            data_root = Path(cfg["data_root"])
            sg_subdir = cfg.get("scene_graph_subdir", f"dslr/sg/sg_03_18")
            sg_filename = cfg.get("scene_graph_filename", f"{scene_id}-simple.json")
            ply_subdir = cfg.get("ply_subdir", "dslr/ply")
            ply_filename = cfg.get("ply_filename", "point_cloud.ply")

            ply_dir = data_root / scene_id / ply_subdir
            ply_path = _resolve_ply_path(ply_dir, ply_filename)

            if cfg.get("skip_scene_subdir", False):
                output_dir = Path(cfg["output_dir"])
            else:
                output_dir = Path(cfg["output_dir"]) / scene_id

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
# Ablation Helpers
# ---------------------------------------------------------------------------

@dataclass
class AblationConfig:
    """Parsed ablation flags from config."""
    no_anchor_determinator: bool = False
    no_parametric_trajectories: bool = False
    no_sdf_optimization: bool = False

    # CLIP anchor adapter settings (for no_anchor_determinator)
    clip_images_dir: str = ""
    clip_colmap_dir: str = ""
    clip_model_name: str = "ViT-B/32"

    # GenDoP settings (for no_parametric_trajectories)
    gendop_resume: str = ""
    gendop_cond_mode: str = "text"

    @classmethod
    def from_config(cls, cfg: dict) -> "AblationConfig":
        abl = cfg.get("ablation", {})
        if not abl:
            return cls()

        scene_id = cfg.get("scene_id", "")
        data_root = cfg.get("data_root", "")

        # Default paths for CLIP adapter
        default_images_dir = str(
            Path(data_root) / scene_id / "dslr" / "images"
        ) if data_root else ""
        default_colmap_dir = str(
            Path(data_root) / scene_id / "dslr" / "sparse" / "0"
        ) if data_root else ""

        return cls(
            no_anchor_determinator=abl.get("no_anchor_determinator", False),
            no_parametric_trajectories=abl.get("no_parametric_trajectories", False),
            no_sdf_optimization=abl.get("no_sdf_optimization", False),
            clip_images_dir=abl.get("clip_images_dir", default_images_dir),
            clip_colmap_dir=abl.get("clip_colmap_dir", default_colmap_dir),
            clip_model_name=abl.get("clip_model_name", "ViT-B/32"),
            gendop_resume=abl.get("gendop_resume", ""),
            gendop_cond_mode=abl.get("gendop_cond_mode", "text"),
        )

    @property
    def any_active(self) -> bool:
        return (
            self.no_anchor_determinator
            or self.no_parametric_trajectories
            or self.no_sdf_optimization
        )

    def describe(self) -> str:
        active = []
        if self.no_anchor_determinator:
            active.append("w/o Anchor Determinator (→ CLIP-based)")
        if self.no_parametric_trajectories:
            active.append("w/o Parametric Trajectories (→ GenDoP 6-DoF)")
        if self.no_sdf_optimization:
            active.append("w/o SDF-based Optimization (→ KD-tree)")
        if not active:
            return "Full pipeline (no ablation)"
        return "ABLATION: " + " + ".join(active)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
class GSCinema:
    """Prompt → mesh → SDF → trajectory → optimisation → subtitle → render → visualise."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        # Keep the key out of the configs: an unset/empty `api_key` falls back
        # to the environment for both the LLM planner and the VLM captioner.
        cfg["api_key"] = cfg.get("api_key") or os.environ.get("OPENAI_API_KEY")
        scene_id = cfg["scene_id"]

        if cfg.get("skip_scene_subdir", False):
            self.output_dir = Path(cfg["output_dir"])
        else:
            self.output_dir = Path(cfg["output_dir"]) / scene_id
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.paths = ScenePaths.from_config(cfg)
        self.ablation = AblationConfig.from_config(cfg)

    # ── public ────────────────────────────────────────────────────────
    def run(self) -> None:
        cfg = self.cfg
        traj_cfg = cfg.get("trajectory", {})
        sdf_cfg = cfg.get("sdf", {})
        opt_cfg = cfg.get("optimisation", {})
        sub_cfg = cfg.get("subtitles", {})
        rnd_cfg = cfg.get("render", {})
        vis_cfg = cfg.get("visualise", {})

        # ── Print ablation status ──
        print("\n" + "=" * 60)
        print(self.ablation.describe())
        print("=" * 60)

        # 0. Always build mesh + SDF (needed for anchor determinator)
        sdf_resolution = sdf_cfg.get("resolution", 128)
        self._ensure_mesh_and_sdf(sdf_cfg, sdf_resolution)

        # 1. LLM: prompt → structured camera instructions + viewing prefs
        llm_output, user_specified_order, viewing_preferences = (
            self._prompt_to_instructions()
        )

        # 2. Build executor (with or without anchor ablation)
        sdf_npz = str(
            self.paths.sdf_dir / f"{cfg['scene_id']}_sdf_res{sdf_resolution}.npz"
        )

        executor = self._build_executor(
            viewing_preferences=viewing_preferences,
            sdf_npz=sdf_npz,
            opt_cfg=opt_cfg,
        )

        fps = rnd_cfg.get("fps", 30.0)
        combiner = TrajectoryCombiner(
            transition_frames=traj_cfg.get("transition_frames", 30),
            fps=fps,
        )

        # 3. Per-segment trajectory generation
        #    (with or without parametric trajectory ablation)
        if self.ablation.no_parametric_trajectories:
            result = self._execute_with_gendop(executor, llm_output)
        else:
            result = executor.execute_from_llm_output(
                llm_output, base_name="segments_executor"
            )

        # Reorder decision
        reorder_policy = traj_cfg.get("reorder", "auto")
        if reorder_policy == "always":
            should_reorder = True
        elif reorder_policy == "never":
            should_reorder = False
        else:
            should_reorder = not user_specified_order

        if should_reorder:
            print("  ↻ Reordering segments (user did not specify order)")
            result = combiner.reorder_segments_nearest_neighbour(result)
        else:
            print("  ✓ Keeping original order (user specified sequence)")

        # 4. Collision-aware optimisation
        if self.ablation.no_sdf_optimization:
            # Ablation 3: Parametric optimization with 3DGS density only (no SDF)
            # Same parametric optimizer, but mesh=None disables:
            #   - _mesh_sdf_cost (SDF collision avoidance)
            #   - _occlusion_cost (SDF ray-marching occlusion)
            # Only _collision_cost (Gaussian density) remains active.
            with open(self.paths.scene_graph, 'r') as f:
                scene_json = json.load(f)

            result = optimize_trajectory_result(
                result, executor, str(self.paths.ply),
                mesh=None,  # ← disables SDF, uses density field only
                sdf_resolution=sdf_resolution,
                sdf_cache_dir=str(self.paths.sdf_dir),
                scene_name=cfg["scene_id"],
                verbose=opt_cfg.get("verbose", True),
                scene_json=scene_json,
            )
        elif self.ablation.no_parametric_trajectories:
            # Ablation 2: Direct pose SDF optimization (no parametric re-creation)
            from src.trajectory_optimizer.direct_pose_optimizer import (
                optimize_trajectory_result_direct,
            )
            with open(self.paths.scene_graph, 'r') as f:
                scene_json = json.load(f)

            result = optimize_trajectory_result_direct(
                result, executor, str(self.paths.ply), executor.mesh,
                sdf_resolution=sdf_resolution,
                sdf_cache_dir=str(self.paths.sdf_dir),
                scene_name=cfg["scene_id"],
                verbose=opt_cfg.get("verbose", True),
                scene_json=scene_json,
            )
        else:
            # Full pipeline: parametric SDF optimization
            with open(self.paths.scene_graph, 'r') as f:
                scene_json = json.load(f)

            result = optimize_trajectory_result(
                result, executor, str(self.paths.ply), executor.mesh,
                sdf_resolution=sdf_resolution,
                sdf_cache_dir=str(self.paths.sdf_dir),
                scene_name=cfg["scene_id"],
                verbose=opt_cfg.get("verbose", True),
                scene_json=scene_json,
            )

        # 5. Combine & smooth
        combined, frame_mappings = combiner.combine_trajectories(result)
        smoothed = combiner.smooth_trajectory(
            combined, window_size=traj_cfg.get("smooth_window", 5),
        )
        self._save_trajectory_artifacts(combiner, smoothed, result, frame_mappings)
        print(f"\n✓ Final trajectory: {len(smoothed)} poses, "
              f"{smoothed.timestamps[-1]:.1f}s")

        # 6. Subtitles
        if rnd_cfg.get("show_subtitles"):
            sub_track = self._generate_subtitles(
                result, frame_mappings, combiner, sub_cfg
            )
        else:
            sub_track = None

        # 7. Render video
        self._render_video(smoothed, sub_track, rnd_cfg)

        # 8. Interactive visualisation
        if vis_cfg.get("enabled", True):
            self._visualize(vis_cfg.get("port", 8080))

    # ── Executor construction ────────────────────────────────────────

    def _build_executor(self, viewing_preferences, sdf_npz, opt_cfg):
        """
        Build the TrajectoryExecutor with the appropriate anchor determinator.

        If ablation.no_anchor_determinator is set, the executor's
        anchor_determinator is replaced with a CLIPAnchorAdapter.
        """
        cfg = self.cfg

        # Build base executor (always needed for parsing, coord conversion, etc.)
        executor = TrajectoryExecutor(
            project_root=cfg.get("project_root", str(PROJECT_ROOT)),
            bbox_path=str(self.paths.scene_graph),
            mesh_path=str(self.paths.mesh),
            workspace=str(self.output_dir),
            viewing_preferences=viewing_preferences,
            sdf_npz_path=sdf_npz if not self.ablation.no_sdf_optimization else None,
            collision_margin=opt_cfg.get("anchor_collision_margin", 0.10),
            debug_candidates=opt_cfg.get("debug_candidates", False),
        )

        # ── Ablation 1: Replace AnchorDeterminator with CLIP-based ──
        if self.ablation.no_anchor_determinator:
            print("\n" + "=" * 60)
            print("ABLATION: Replacing AnchorDeterminator with CLIP-based selection")
            print("=" * 60)

            from src.utils.ablation_utils import CLIPAnchorAdapter

            clip_adapter = CLIPAnchorAdapter(
                bboxes=executor.bboxes,
                mesh=executor.mesh,
                images_dir=self.ablation.clip_images_dir,
                colmap_model_dir=self.ablation.clip_colmap_dir,
                clip_model_name=self.ablation.clip_model_name,
                output_dir=str(self.output_dir),
            )

            # Monkey-patch the executor's anchor_determinator
            executor.anchor_determinator = clip_adapter
            # Clear anchor cache so new adapter is used
            executor._anchor_cache = {}

            print("  ✓ AnchorDeterminator replaced with CLIPAnchorAdapter")

        return executor

    # ── GenDoP trajectory execution ──────────────────────────────────

    def _execute_with_gendop(self, executor, llm_output):
        """
        Execute trajectory plan using GenDoP instead of parametric builder.

        Ablation 2: w/o Parametric Trajectories
        """
        print("\n" + "=" * 60)
        print("ABLATION: Using GenDoP 6-DoF generation (no parametric trajectories)")
        print("=" * 60)

        from src.utils.ablation_utils import GenDoPTrajectoryExecutor

        gendop_config = {}
        if self.ablation.gendop_resume:
            gendop_config["resume"] = self.ablation.gendop_resume
            gendop_config["cond_mode"] = self.ablation.gendop_cond_mode

        gendop_executor = GenDoPTrajectoryExecutor(
            base_executor=executor,
            gendop_config=gendop_config,
        )

        # Parse the LLM output using the base executor's parser
        plan = executor.parse_llm_output(llm_output)

        # Execute plan, but use GenDoP for trajectory generation
        from src.parametric_trajectory_builder.trajectory_executor import (
            AnchorCommand,
            AtomTrajCommand,
            TrajectoryResult,
            TrajectorySegment,
        )

        result = TrajectoryResult()
        current_anchor = None
        pending_anchors = []
        traj_count = 0

        output_dir = Path(executor.project_root) / executor.workspace / "segments_executor"
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== Executing Plan with GenDoP: {len(plan.commands)} commands ===")

        for i, cmd in enumerate(plan.commands):
            if isinstance(cmd, AnchorCommand):
                # Anchors are still determined by the (possibly ablated) anchor system
                anchor = executor.get_anchor(cmd.object_id, cmd.object_label)
                result.all_anchors.append(anchor)

                if current_anchor is None:
                    current_anchor = anchor
                else:
                    pending_anchors.append(anchor)

            elif isinstance(cmd, AtomTrajCommand):
                if current_anchor is None:
                    continue

                end_anchor = None
                if cmd.movement_category == "transitional" and pending_anchors:
                    end_anchor = pending_anchors.pop(0)

                traj_count += 1
                try:
                    # Use GenDoP executor instead of parametric
                    output = gendop_executor.execute_tools(
                        cmd, current_anchor, end_anchor
                    )
                    result.trajectory_outputs.append(output)

                    output_path = (
                        output_dir
                        / f"step_{traj_count:02d}_{cmd.movement_type}.npy"
                    )
                    np.save(output_path, output["c2w"])
                    result.trajectory_paths.append(output_path)
                except Exception as e:
                    print(f"  ERROR: {e}")
                    import traceback
                    traceback.print_exc()
                    output = None

                segment = TrajectorySegment(
                    start_anchor=current_anchor,
                    end_anchor=end_anchor,
                    movement_type=cmd.movement_type,
                    movement_category=cmd.movement_category,
                    arc_angle=cmd.arc_angle,
                    trajectory_output=output,
                )
                result.segments.append(segment)

                if end_anchor is not None:
                    current_anchor = end_anchor

        print(f"\n=== Done: {len(result.segments)} segments, "
              f"{len(result.all_anchors)} anchors ===")
        return result

    # ── Mesh + SDF creation ──────────────────────────────────────────

    def _ensure_mesh_only(self, sdf_cfg: dict) -> None:
        """Build OBB mesh but skip SDF computation (for ablation)."""
        if self.paths.format == "scene_graph":
            mesh_path = self.paths.mesh
            if not mesh_path.exists() or sdf_cfg.get("force_rebuild", False):
                print("\n" + "=" * 60)
                print("BUILDING OBB MESH FROM SCENE GRAPH (SDF skipped — ablation)")
                print("=" * 60)
                with open(self.paths.scene_graph, 'r') as f:
                    scene_json = json.load(f)
                mesh = build_scene_mesh(scene_json)
                mesh_path.parent.mkdir(parents=True, exist_ok=True)
                mesh.export(str(mesh_path))
                print(f"✓ OBB mesh saved → {mesh_path}")
            else:
                print(f"✓ OBB mesh exists: {mesh_path}")
            print("  (SDF computation skipped — ablation: no_sdf_optimization)")
        else:
            print("  Legacy format — mesh already exists, SDF skipped (ablation)")

    def _ensure_mesh_and_sdf(self, sdf_cfg: dict, resolution: int) -> None:
        sdf_dir = self.paths.sdf_dir
        sdf_dir.mkdir(parents=True, exist_ok=True)

        scene_id = self.cfg["scene_id"]
        npz_path = sdf_dir / f"{scene_id}_sdf_res{resolution}.npz"

        # Step 1: Build OBB mesh if scene_graph format
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

        # Step 2: SDF grid
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
            transition_frames=self.cfg.get("trajectory", {}).get(
                "transition_frames", 30
            ),
            api_key=self.cfg.get("api_key"),
            ply_path=str(self.paths.ply),
            device=sub_cfg.get("device", "cuda"),
            language=sub_cfg.get("language", "en"),
        )

    # ── Trajectory artefacts ─────────────────────────────────────────
    def _save_trajectory_artifacts(
        self, combiner, smoothed, result, frame_mappings
    ) -> None:
        combiner.save_trajectory(
            smoothed, self.output_dir / "combined_trajectory.json"
        )
        combiner.save_anchors(
            result.all_anchors, self.output_dir / "anchors.json"
        )
        combiner.save_frame_mappings(
            frame_mappings, self.output_dir / "frame_mappings.json"
        )
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
        renderer.render_trajectory(
            smoothed, str(output_path), config, subtitle_track=sub_track
        )

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
        "--config",
        type=str,
        default="config/config_scannetpp.yaml",
        help="Path to YAML config",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    GSCinema(cfg).run()


if __name__ == "__main__":
    main()