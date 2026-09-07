"""
GSCinema Ablation Study Runner (Benchmark-Integrated)
=====================================================

Runs the GSCinema pipeline under four conditions (full + 3 ablations)
across multiple scenes and prompt difficulty levels, reusing the
benchmark infrastructure for prompt generation, scene discovery,
and result tracking.

Ablation conditions:
  - full:           GSCinema (all components)
  - no_anchor:      w/o Anchor Determinator  → CLIP-based selection
  - no_parametric:  w/o Parametric Trajectories → GenDoP 6-DoF diffusion
  - no_sdf:         w/o SDF-based Optimization → KD-tree refinement

Workflow:
  1. Discover scenes (reuses benchmark_prompt_generator.discover_scenes).
  2. Generate/load prompts at specified difficulty levels.
  3. For each (scene, level, prompt, condition), run the pipeline.
  4. Collect summary JSON with timing and per-condition stats.

Usage:
    # Run all conditions on all scenes
    python run_ablation_study.py --config config/benchmark_scannetpp.yaml

    # Specific scenes and conditions
    python run_ablation_study.py --config config/benchmark_scannetpp.yaml \\
        --scenes 09c1414f1b 0f25f24a4f \\
        --conditions full no_anchor no_sdf \\
        --levels high medium

    # Resume (skips completed runs via done.marker files)
    python run_ablation_study.py --config config/benchmark_scannetpp.yaml --resume
"""

import argparse
import copy
import json
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

# ---------------------------------------------------------------------------
# Project bootstrap
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.benchmark_prompt_generator import (
    discover_scenes,
    generate_all_prompts,
)


# ---------------------------------------------------------------------------
# Ablation condition definitions
# ---------------------------------------------------------------------------
ABLATION_CONDITIONS = {
    "full": {
        "label": "GSCinema (Full)",
        "short": "Full",
        "ablation": {
            "no_anchor_determinator": False,
            "no_parametric_trajectories": False,
            "no_sdf_optimization": False,
        },
    },
    "no_anchor": {
        "label": "w/o Anchor Determinator",
        "short": "-Anchor",
        "ablation": {
            "no_anchor_determinator": True,
            "no_parametric_trajectories": False,
            "no_sdf_optimization": False,
        },
    },
    "no_parametric": {
        "label": "w/o Parametric Trajectories",
        "short": "-Param",
        "ablation": {
            "no_anchor_determinator": False,
            "no_parametric_trajectories": True,
            "no_sdf_optimization": False,
        },
    },
    "no_sdf": {
        "label": "w/o SDF-based Optimization",
        "short": "-SDF",
        "ablation": {
            "no_anchor_determinator": False,
            "no_parametric_trajectories": False,
            "no_sdf_optimization": True,
        },
    },
}


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------
@dataclass
class AblationRunResult:
    """Result of a single (scene, level, prompt, condition) run."""
    scene_id: str
    level: str
    prompt_idx: int
    prompt: str
    condition: str
    condition_label: str
    success: bool
    error: Optional[str] = None
    elapsed_seconds: float = 0.0
    output_dir: str = ""
    n_trajectory_poses: int = 0
    n_objects_visited: int = 0


@dataclass
class AblationSummary:
    """Aggregate summary of the ablation study."""
    total_runs: int = 0
    successful: int = 0
    failed: int = 0
    total_time_seconds: float = 0.0
    results: List[dict] = field(default_factory=list)
    stats_by_condition: Dict[str, Dict[str, Dict]] = field(default_factory=dict)

    def ensure_stats(self, condition: str, level: str):
        if condition not in self.stats_by_condition:
            self.stats_by_condition[condition] = {}
        if level not in self.stats_by_condition[condition]:
            self.stats_by_condition[condition][level] = {
                "total": 0, "success": 0, "fail": 0,
            }


# ---------------------------------------------------------------------------
# Single run wrapper
# ---------------------------------------------------------------------------
def run_single(
    scene_id: str,
    prompt: str,
    level: str,
    prompt_idx: int,
    condition_key: str,
    cfg_base: dict,
    output_dir: Path,
) -> AblationRunResult:
    """Run the GSCinema pipeline for a single (scene, prompt, condition) triple."""
    from src.pipeline.cinematraj_pipeline import GSCinema

    condition = ABLATION_CONDITIONS[condition_key]

    result = AblationRunResult(
        scene_id=scene_id,
        level=level,
        prompt_idx=prompt_idx,
        prompt=prompt,
        condition=condition_key,
        condition_label=condition["label"],
        success=False,
        output_dir=str(output_dir),
    )

    # Build per-run config
    cfg = copy.deepcopy(cfg_base)
    cfg["scene_id"] = scene_id
    cfg["prompt"] = prompt
    cfg["output_dir"] = str(output_dir)
    cfg["skip_scene_subdir"] = True

    # Inject ablation flags
    cfg["ablation"] = copy.deepcopy(condition["ablation"])

    # Propagate ablation-specific settings from base config
    abl_extra = cfg_base.get("ablation_settings", {})
    if abl_extra:
        for key in ("clip_images_dir", "clip_colmap_dir", "clip_model_name",
                     "gendop_resume", "gendop_cond_mode"):
            if key in abl_extra:
                cfg["ablation"][key] = abl_extra[key]

    # Shared SDF cache per scene
    data_root = cfg.get("data_root", "")
    shared_sdf_dir = Path(data_root) / scene_id / "dslr" / "sdf"
    shared_sdf_dir.mkdir(parents=True, exist_ok=True)
    if "sdf" not in cfg:
        cfg["sdf"] = {}
    cfg["sdf"]["cache_dir"] = str(shared_sdf_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    (output_dir / "condition.txt").write_text(
        f"{condition_key}: {condition['label']}", encoding="utf-8"
    )

    t0 = time.time()
    try:
        pipeline = GSCinema(cfg)
        pipeline.run()
        result.success = True

        traj_file = output_dir / "combined_trajectory.json"
        if traj_file.exists():
            with open(traj_file) as f:
                traj_data = json.load(f)
            if isinstance(traj_data, dict):
                result.n_trajectory_poses = len(traj_data.get("frames", []))

        anchors_file = output_dir / "anchors.json"
        if anchors_file.exists():
            with open(anchors_file) as f:
                anchors_data = json.load(f)
            if isinstance(anchors_data, dict):
                result.n_objects_visited = len(anchors_data.get("anchors", []))
            elif isinstance(anchors_data, list):
                result.n_objects_visited = len(anchors_data)

    except Exception as e:
        result.error = f"{type(e).__name__}: {e}"
        error_log = output_dir / "error.log"
        with open(error_log, "w") as f:
            traceback.print_exc(file=f)
        print(f"    ✗ {result.error}")

    result.elapsed_seconds = time.time() - t0
    return result


# ---------------------------------------------------------------------------
# Main ablation loop
# ---------------------------------------------------------------------------
def run_ablation_study(cfg: dict) -> AblationSummary:
    """Run the full ablation study."""
    data_root = cfg["data_root"]
    output_root = Path(cfg.get("output_root", "outputs/ablation_study"))
    api_key = cfg.get("api_key", os.environ.get("OPENAI_API_KEY", ""))
    max_scenes = cfg.get("max_scenes", 200)
    n_per_level = cfg.get("n_prompts_per_level", 2)
    use_llm = cfg.get("use_llm_prompts", False)
    seed = cfg.get("prompt_seed", 42)
    fmt = cfg.get("format", "scene_graph")
    sg_subdir = cfg.get("scene_graph_subdir", "dslr/sg")
    include_id_levels = cfg.get("include_id_levels", True)
    resume = cfg.get("resume", True)

    default_levels = (
        ["high", "medium", "medium_id", "low", "low_id"]
        if include_id_levels
        else ["high", "medium", "low"]
    )
    levels_to_run = cfg.get("levels", default_levels)
    conditions_to_run = cfg.get("conditions", list(ABLATION_CONDITIONS.keys()))

    output_root.mkdir(parents=True, exist_ok=True)

    # --- Discover scenes ---
    print("=" * 70)
    print("GSCINEMA ABLATION STUDY")
    print("=" * 70)
    print(f"Format:     {fmt}")
    print(f"Conditions: {[ABLATION_CONDITIONS[c]['short'] for c in conditions_to_run]}")
    print(f"Levels:     {levels_to_run}")

    explicit_scenes = cfg.get("scenes")
    if explicit_scenes:
        scene_ids = explicit_scenes
        print(f"Scenes:     {len(scene_ids)} (explicit list)")
    else:
        scene_ids = discover_scenes(
            data_root, fmt=fmt, sg_subdir=sg_subdir, max_scenes=max_scenes,
        )
        print(f"Scenes:     {len(scene_ids)} (discovered)")

    # --- Generate/load prompts ---
    prompts_file = cfg.get("prompts_file")
    if prompts_file and Path(prompts_file).exists():
        print(f"Loading prompts from {prompts_file}")
        with open(prompts_file) as f:
            all_prompts = json.load(f)
    else:
        print(f"Generating prompts (use_llm={use_llm}, n_per_level={n_per_level})")
        all_prompts = generate_all_prompts(
            data_root=data_root, fmt=fmt, sg_subdir=sg_subdir,
            max_scenes=max_scenes, api_key=api_key, use_llm=use_llm,
            n_per_level=n_per_level, seed=seed,
            include_id_levels=include_id_levels,
        )
        prompts_save_path = output_root / "ablation_prompts.json"
        with open(prompts_save_path, "w") as f:
            json.dump(all_prompts, f, indent=2)
        print(f"Saved prompts -> {prompts_save_path}")

    # --- Count total runs ---
    total_runs = 0
    for sid in scene_ids:
        if sid not in all_prompts:
            continue
        for level in levels_to_run:
            n_prompts = len(all_prompts[sid].get(level, []))
            total_runs += n_prompts * len(conditions_to_run)

    print(f"\nTotal pipeline runs: {total_runs}")
    print("=" * 70)

    # --- Build base config ---
    cfg_base = {
        "data_root": data_root,
        "api_key": api_key,
        "format": fmt,
        "project_root": cfg.get("project_root", str(PROJECT_ROOT)),
        "scene_graph_subdir": sg_subdir,
        "ply_subdir": cfg.get("ply_subdir", "dslr/ply"),
        "ply_filename": cfg.get("ply_filename", "point_cloud.ply"),
        "sdf": cfg.get("sdf", {"resolution": 128}),
        "render": cfg.get("render", {}),
        "subtitles": cfg.get("subtitles", {}),
        "trajectory": cfg.get("trajectory", {}),
        "optimisation": cfg.get("optimisation", {}),
        "visualise": cfg.get("visualise", {"enabled": False}),
        "ablation_settings": cfg.get("ablation_settings", {}),
    }

    # --- Run ---
    summary = AblationSummary()
    run_idx = 0

    for scene_idx, scene_id in enumerate(scene_ids):
        if scene_id not in all_prompts:
            print(f"\n[Scene {scene_idx+1}/{len(scene_ids)}] {scene_id} — no prompts, skipping")
            continue

        scene_prompts = all_prompts[scene_id]

        for level in levels_to_run:
            prompts = scene_prompts.get(level, [])

            for prompt_idx, prompt in enumerate(prompts):
                for condition_key in conditions_to_run:
                    run_idx += 1
                    condition = ABLATION_CONDITIONS[condition_key]

                    run_output = (
                        output_root / condition_key / scene_id
                        / level / f"prompt_{prompt_idx:02d}"
                    )

                    summary.ensure_stats(condition_key, level)

                    done_marker = run_output / "done.marker"
                    if resume and done_marker.exists():
                        print(
                            f"[{run_idx}/{total_runs}] "
                            f"{condition['short']:>8s} | {scene_id} | {level} | p{prompt_idx} "
                            f"— already done, skipping"
                        )
                        summary.total_runs += 1
                        summary.successful += 1
                        summary.stats_by_condition[condition_key][level]["total"] += 1
                        summary.stats_by_condition[condition_key][level]["success"] += 1
                        continue

                    print(
                        f"\n[{run_idx}/{total_runs}] "
                        f"{condition['short']:>8s} | {scene_id} | {level} | prompt {prompt_idx}"
                    )
                    print(f"  Prompt: {prompt[:120]}{'...' if len(prompt) > 120 else ''}")

                    result = run_single(
                        scene_id=scene_id, prompt=prompt, level=level,
                        prompt_idx=prompt_idx, condition_key=condition_key,
                        cfg_base=cfg_base, output_dir=run_output,
                    )

                    if result.success:
                        done_marker.parent.mkdir(parents=True, exist_ok=True)
                        done_marker.write_text("done")

                    summary.total_runs += 1
                    summary.total_time_seconds += result.elapsed_seconds
                    if result.success:
                        summary.successful += 1
                        summary.stats_by_condition[condition_key][level]["success"] += 1
                    else:
                        summary.failed += 1
                        summary.stats_by_condition[condition_key][level]["fail"] += 1
                    summary.stats_by_condition[condition_key][level]["total"] += 1
                    summary.results.append(asdict(result))

                    status = "✓" if result.success else "✗"
                    print(
                        f"  {status} {result.elapsed_seconds:.1f}s | "
                        f"poses={result.n_trajectory_poses} | "
                        f"objects={result.n_objects_visited}"
                    )

                    _save_summary(summary, output_root / "ablation_summary.json")

    _print_summary_table(summary, conditions_to_run, levels_to_run)
    _save_summary(summary, output_root / "ablation_summary.json")
    print(f"\n✓ Summary saved -> {output_root / 'ablation_summary.json'}")

    return summary


def _save_summary(summary: AblationSummary, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(asdict(summary), f, indent=2)


def _print_summary_table(
    summary: AblationSummary,
    conditions: List[str],
    levels: List[str],
):
    """Print a formatted summary table."""
    print("\n" + "=" * 70)
    print("ABLATION STUDY RESULTS")
    print("=" * 70)
    print(f"Total runs: {summary.total_runs}  |  "
          f"Success: {summary.successful}  |  "
          f"Failed: {summary.failed}  |  "
          f"Time: {summary.total_time_seconds:.0f}s "
          f"({summary.total_time_seconds / 3600:.1f}h)")
    print()

    col_w = 14
    header = f"{'Condition':<28}"
    for level in levels:
        header += f"  {level:>{col_w}}"
    header += f"  {'TOTAL':>{col_w}}"
    print(header)
    print("-" * len(header))

    for cond in conditions:
        cond_stats = summary.stats_by_condition.get(cond, {})
        label = ABLATION_CONDITIONS[cond]["label"]
        row = f"{label:<28}"

        cond_total = 0
        cond_success = 0
        for level in levels:
            stats = cond_stats.get(level, {"total": 0, "success": 0})
            t = stats["total"]
            s = stats["success"]
            cond_total += t
            cond_success += s
            if t > 0:
                rate = s / t * 100
                cell = f"{s}/{t} ({rate:.0f}%)"
            else:
                cell = "—"
            row += f"  {cell:>{col_w}}"

        if cond_total > 0:
            rate = cond_success / cond_total * 100
            total_cell = f"{cond_success}/{cond_total} ({rate:.0f}%)"
        else:
            total_cell = "—"
        row += f"  {total_cell:>{col_w}}"
        print(row)

    print()
    for cond in conditions:
        cond_results = [
            r for r in summary.results
            if r.get("condition") == cond and r.get("success")
        ]
        if cond_results:
            times = [r["elapsed_seconds"] for r in cond_results]
            avg_t = sum(times) / len(times)
            label = ABLATION_CONDITIONS[cond]["short"]
            print(f"  {label:>8s}: avg {avg_t:.1f}s/run  "
                  f"(min={min(times):.0f}s, max={max(times):.0f}s, "
                  f"n={len(times)})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="GSCinema Ablation Study Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_ablation_study.py --config config/benchmark_scannetpp.yaml
  python run_ablation_study.py --config config/benchmark_scannetpp.yaml \\
      --scenes 09c1414f1b 0f25f24a4f --conditions full no_anchor no_sdf
  python run_ablation_study.py --config config/benchmark_scannetpp.yaml --levels high
""",
    )
    parser.add_argument("--config", type=str, default="config/ablation_scannetpp.yaml")
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument("--conditions", nargs="*", default=None,
                        choices=list(ABLATION_CONDITIONS.keys()))
    parser.add_argument("--levels", nargs="*", default=None)
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--max_scenes", type=int, default=None)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no_resume", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
    else:
        print(f"Config file {config_path} not found — using CLI args only")
        cfg = {}

    if args.scenes:
        cfg["scenes"] = args.scenes
    if args.conditions:
        cfg["conditions"] = args.conditions
    if args.levels:
        cfg["levels"] = args.levels
    if args.output_root:
        cfg["output_root"] = args.output_root
    if args.max_scenes is not None:
        cfg["max_scenes"] = args.max_scenes
    if args.no_resume:
        cfg["resume"] = False

    if "data_root" not in cfg:
        parser.error("data_root is required (via config file)")
    if "output_root" not in cfg:
        cfg["output_root"] = "outputs/ablation_study"

    run_ablation_study(cfg)


if __name__ == "__main__":
    main()