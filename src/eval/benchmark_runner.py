"""
GSCinema Benchmark Runner (v2 — Scene Graph Support)
=====================================================

Runs the GSCinema/TrajScene pipeline across multiple scenes and prompt
difficulty levels for systematic evaluation.

Supports both legacy InteriorGS format and the new ScanNet++ scene-graph
format.  Also supports the new medium_id / low_id prompt levels that
include explicit object IDs.

Workflow:
  1. Discover scenes under data_root (format-aware).
  2. Generate (or load) prompts at five levels:
     high, medium, medium_id, low, low_id.
  3. For each (scene, level, prompt) triple, run the full pipeline and
     save results under output_root/<scene_id>/<level>/<prompt_idx>/.
  4. Collect a summary JSON with timing, success/failure, and paths.

Usage:
    python benchmark_runner.py --config config/benchmark_scannetpp.yaml
"""

import argparse
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
    generate_prompts_for_scene,
    get_scene_graph_path,
)


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------
@dataclass
class RunResult:
    """Result of a single pipeline run."""
    scene_id: str
    level: str          # high / medium / medium_id / low / low_id
    prompt_idx: int
    prompt: str
    success: bool
    error: Optional[str] = None
    elapsed_seconds: float = 0.0
    output_dir: str = ""
    n_trajectory_poses: int = 0
    n_objects_visited: int = 0


@dataclass
class BenchmarkSummary:
    """Aggregate summary of a full benchmark run."""
    total_runs: int = 0
    successful: int = 0
    failed: int = 0
    total_time_seconds: float = 0.0
    results: List[dict] = field(default_factory=list)
    stats_by_level: Dict[str, Dict] = field(default_factory=lambda: {
        level: {"total": 0, "success": 0, "fail": 0}
        for level in ("high", "medium", "medium_id", "low", "low_id")
    })



# ---------------------------------------------------------------------------
# Pipeline wrapper
# ---------------------------------------------------------------------------
def run_single(
    scene_id: str,
    prompt: str,
    level: str,
    prompt_idx: int,
    cfg_base: dict,
    output_dir: Path,
) -> RunResult:
    """
    Run the GSCinema pipeline for a single (scene, prompt) pair.

    Wrapped in try/except so one failure doesn't kill the benchmark.
    """
    # Late import so heavy deps only load once needed
    from src.pipeline.cinematraj_pipeline_base import GSCinema

    result = RunResult(
        scene_id=scene_id,
        level=level,
        prompt_idx=prompt_idx,
        prompt=prompt,
        success=False,
        output_dir=str(output_dir),
    )

    # Build per-run config
    cfg = dict(cfg_base)
    cfg["scene_id"] = scene_id
    cfg["prompt"] = prompt
    cfg["output_dir"] = str(output_dir)
    # The output path already includes the scene_id
    # (e.g. outputs/.../00a231a370/high/prompt_00),
    # so tell the pipeline NOT to append another scene_id subdirectory.
    cfg["skip_scene_subdir"] = True

    # --- Shared SDF cache ---
    # Store SDF .npz and OBB mesh once per scene under:
    #   data_root/<scene_id>/dslr/sdf/
    # instead of duplicating in every prompt's output directory.
    data_root = cfg.get("data_root", "")
    shared_sdf_dir = Path(data_root) / scene_id / "dslr" / "sdf"
    shared_sdf_dir.mkdir(parents=True, exist_ok=True)
    if "sdf" not in cfg:
        cfg["sdf"] = {}
    cfg["sdf"]["cache_dir"] = str(shared_sdf_dir)

    # Disable interactive visualisation in batch mode
    if "visualise" not in cfg:
        cfg["visualise"] = {}
    cfg["visualise"]["enabled"] = False

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the prompt text so each run is self-contained
    (output_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

    t0 = time.time()
    try:
        pipeline = GSCinema(cfg)
        pipeline.run()
        result.success = True

        # Try to read trajectory length from output
        traj_file = output_dir / "combined_trajectory.json"
        if traj_file.exists():
            with open(traj_file) as f:
                traj_data = json.load(f)
            if isinstance(traj_data, dict):
                # save_trajectory() writes a nerfstudio-style dict keyed on
                # "frames"; "positions" is the older flat layout.
                result.n_trajectory_poses = len(
                    traj_data.get("frames", traj_data.get("positions", []))
                )
            elif isinstance(traj_data, list):
                result.n_trajectory_poses = len(traj_data)

        # Try to read anchors
        anchors_file = output_dir / "anchors.json"
        if anchors_file.exists():
            with open(anchors_file) as f:
                anchors_data = json.load(f)
            if isinstance(anchors_data, dict):
                anchors_data = anchors_data.get("anchors", [])
            if isinstance(anchors_data, list):
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
# Main benchmark loop
# ---------------------------------------------------------------------------
def run_benchmark(cfg: dict) -> BenchmarkSummary:
    """
    Run the full benchmark: discover scenes → generate prompts → run pipeline.
    """
    data_root = cfg["data_root"]
    output_root = Path(cfg["output_root"])
    api_key = cfg.get("api_key", os.environ.get("OPENAI_API_KEY", ""))
    max_scenes = cfg.get("max_scenes", 200)
    n_per_level = cfg.get("n_prompts_per_level", 2)
    use_llm = cfg.get("use_llm_prompts", False)
    seed = cfg.get("prompt_seed", 42)
    fmt = cfg.get("format", "scene_graph")
    sg_subdir = cfg.get("scene_graph_subdir", "dslr/sg")
    include_id_levels = cfg.get("include_id_levels", True)

    # Default levels depend on whether ID levels are included
    default_levels = ["high", "medium", "medium_id", "low", "low_id"] if include_id_levels \
        else ["high", "medium", "low"]
    levels_to_run = cfg.get("levels", default_levels)

    output_root.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Discover scenes ---
    print("=" * 70)
    print("GSCINEMA BENCHMARK")
    print("=" * 70)
    print(f"Format: {fmt}")

    explicit_scenes = cfg.get("scenes")
    if explicit_scenes:
        scene_ids = explicit_scenes
        print(f"Scenes: {len(scene_ids)} (explicit list)")
    else:
        scene_ids = discover_scenes(data_root, fmt=fmt, sg_subdir=sg_subdir,
                                    max_scenes=max_scenes)
        print(f"Scenes discovered: {len(scene_ids)}")

    # --- Step 2: Generate or load prompts ---
    prompts_file = cfg.get("prompts_file")
    if prompts_file and Path(prompts_file).exists():
        print(f"Loading prompts from {prompts_file}")
        with open(prompts_file) as f:
            all_prompts = json.load(f)
    else:
        print(f"Generating prompts (use_llm={use_llm}, n_per_level={n_per_level})")
        all_prompts = generate_all_prompts(
            data_root=data_root,
            fmt=fmt,
            sg_subdir=sg_subdir,
            max_scenes=max_scenes,
            api_key=api_key,
            use_llm=use_llm,
            n_per_level=n_per_level,
            seed=seed,
            include_id_levels=include_id_levels,
        )
        if explicit_scenes:
            # generate_all_prompts() rediscovers every scene under data_root.
            # Keep only the requested ones: downstream runners (the baselines)
            # take their scene list from this file.
            all_prompts = {s: p for s, p in all_prompts.items() if s in scene_ids}
        prompts_save_path = output_root / "benchmark_prompts.json"
        with open(prompts_save_path, "w") as f:
            json.dump(all_prompts, f, indent=2)
        print(f"Saved prompts → {prompts_save_path}")

    # --- Step 3: Count total runs ---
    total_runs = 0
    for sid in scene_ids:
        if sid not in all_prompts:
            continue
        for level in levels_to_run:
            total_runs += len(all_prompts[sid].get(level, []))

    print(f"\nTotal pipeline runs: {total_runs}")
    print(f"Levels: {levels_to_run}")
    print("=" * 70)

    # --- Step 4: Build base config ---
    # PLY filename: the pipeline's _resolve_ply_path() will try
    # point_cloud_30000.ply first and fall back to point_cloud.ply,
    # so we pass point_cloud.ply as the safe default here.
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
        "visualise": {"enabled": False},
    }

    # --- Step 5: Run ---
    # Initialise summary with all levels
    summary = BenchmarkSummary()
    for level in levels_to_run:
        if level not in summary.stats_by_level:
            summary.stats_by_level[level] = {"total": 0, "success": 0, "fail": 0}

    run_idx = 0

    for scene_idx, scene_id in enumerate(scene_ids):
        if scene_id not in all_prompts:
            print(f"[{scene_idx+1}/{len(scene_ids)}] {scene_id} — no prompts, skipping")
            continue

        scene_prompts = all_prompts[scene_id]

        for level in levels_to_run:
            prompts = scene_prompts.get(level, [])

            for prompt_idx, prompt in enumerate(prompts):
                run_idx += 1
                run_output = (
                    output_root / scene_id / level / f"prompt_{prompt_idx:02d}"
                )

                # Skip if already completed
                done_marker = run_output / "done.marker"
                if done_marker.exists():
                    print(
                        f"[{run_idx}/{total_runs}] {scene_id}/{level}/{prompt_idx} "
                        f"— already done, skipping"
                    )
                    summary.total_runs += 1
                    summary.successful += 1
                    summary.stats_by_level[level]["total"] += 1
                    summary.stats_by_level[level]["success"] += 1
                    continue

                print(
                    f"\n[{run_idx}/{total_runs}] "
                    f"{scene_id} | {level} | prompt {prompt_idx}"
                )
                print(f"  Prompt: {prompt[:120]}{'...' if len(prompt) > 120 else ''}")

                result = run_single(
                    scene_id=scene_id,
                    prompt=prompt,
                    level=level,
                    prompt_idx=prompt_idx,
                    cfg_base=cfg_base,
                    output_dir=run_output,
                )

                if result.success:
                    done_marker.parent.mkdir(parents=True, exist_ok=True)
                    done_marker.write_text("done")

                summary.total_runs += 1
                summary.total_time_seconds += result.elapsed_seconds
                if result.success:
                    summary.successful += 1
                    summary.stats_by_level[level]["success"] += 1
                else:
                    summary.failed += 1
                    summary.stats_by_level[level]["fail"] += 1
                summary.stats_by_level[level]["total"] += 1
                summary.results.append(asdict(result))

                status = "✓" if result.success else "✗"
                print(
                    f"  {status} {result.elapsed_seconds:.1f}s | "
                    f"poses={result.n_trajectory_poses} | "
                    f"objects={result.n_objects_visited}"
                )

                _save_summary(summary, output_root / "benchmark_summary.json")

    # --- Final summary ---
    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
    print(f"Total:     {summary.total_runs}")
    print(f"Success:   {summary.successful}")
    print(f"Failed:    {summary.failed}")
    print(f"Time:      {summary.total_time_seconds:.1f}s "
          f"({summary.total_time_seconds / 3600:.1f}h)")
    for level in levels_to_run:
        s = summary.stats_by_level[level]
        print(f"  {level:12s}: {s['success']}/{s['total']} success")

    _save_summary(summary, output_root / "benchmark_summary.json")
    print(f"\n✓ Summary saved → {output_root / 'benchmark_summary.json'}")

    return summary


def _save_summary(summary: BenchmarkSummary, path: Path):
    with open(path, "w") as f:
        json.dump(asdict(summary), f, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="GSCinema Benchmark Runner")
    parser.add_argument("--config", type=str, default="config/benchmark_scannetpp.yaml")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--max_scenes", type=int, default=None)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--format", type=str, default=None, choices=["scene_graph", "legacy"])
    parser.add_argument(
        "--levels", type=str, nargs="+", default=None,
        help="Which levels to run (e.g. --levels high medium low_id)",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
    else:
        print(f"Config file {config_path} not found — using CLI args only")
        cfg = {}

    # CLI overrides
    if args.data_root:
        cfg["data_root"] = args.data_root
    if args.output_root:
        cfg["output_root"] = args.output_root
    if args.max_scenes is not None:
        cfg["max_scenes"] = args.max_scenes
    if args.api_key:
        cfg["api_key"] = args.api_key
    if args.format:
        cfg["format"] = args.format
    if args.levels:
        cfg["levels"] = args.levels

    if "data_root" not in cfg:
        parser.error("--data_root is required (via config or CLI)")
    if "output_root" not in cfg:
        cfg["output_root"] = str(Path(cfg["data_root"]).parent / "benchmark_outputs")

    run_benchmark(cfg)


if __name__ == "__main__":
    main()