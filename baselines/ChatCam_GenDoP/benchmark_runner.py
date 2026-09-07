"""
Benchmark Runner — ChatCam + GenDoP Baseline
=============================================

Runs the ChatCam+GenDoP baseline across multiple scenes using low_id
prompts from GSCinema's benchmark_prompts.json.  The "(id: xxx)" tags
are stripped before feeding to ChatCam since it uses CLIP for anchoring
and object IDs would be unfair.

Usage:
    python benchmark_runner_chatcam.py --config config/benchmark_chatcam.yaml

    # Or with CLI overrides:
    python benchmark_runner_chatcam.py \
        --data_root /path/to/ScanNetpp/gsplat_20_scenes \
        --prompts_file outputs/benchmark_scannetpp/benchmark_prompts.json \
        --output_root outputs/benchmark_chatcam
"""

import argparse
import json
import os
import re
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

# ---------------------------------------------------------------------------
# Project bootstrap — adjust if ChatCam lives elsewhere
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
# ChatCam modules are expected on sys.path (e.g. baselines/ChatCam_GenDoP/)
CHATCAM_DIR = PROJECT_ROOT / "baselines" / "ChatCam_GenDoP"
sys.path.insert(0, str(CHATCAM_DIR))


# ---------------------------------------------------------------------------
# PLY resolution — prefer higher-iteration point cloud
# ---------------------------------------------------------------------------
_PLY_PREFERENCE_ORDER = [
    "point_cloud_30000.ply",
    "point_cloud.ply",
]


def _resolve_ply_path(ply_dir: Path, fallback: str = "point_cloud.ply") -> Path:
    """Return the best available PLY: point_cloud_30000.ply > point_cloud.ply."""
    for name in _PLY_PREFERENCE_ORDER:
        candidate = ply_dir / name
        if candidate.exists():
            return candidate
    return ply_dir / fallback


# ---------------------------------------------------------------------------
# Prompt cleaning — strip (id: xxx) tags
# ---------------------------------------------------------------------------
_ID_TAG_RE = re.compile(r"\s*\(id:\s*[A-Za-z0-9_]+\)")


def strip_id_tags(prompt: str) -> str:
    """
    Remove all '(id: object_id)' tags from a prompt.

    Example:
        "pan left from the coffee machine (id: coffee_machine_0)"
        → "pan left from the coffee machine"
    """
    return _ID_TAG_RE.sub("", prompt).strip()


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------
@dataclass
class RunResult:
    scene_id: str
    prompt_idx: int
    prompt_original: str
    prompt_cleaned: str
    success: bool
    error: Optional[str] = None
    elapsed_seconds: float = 0.0
    output_dir: str = ""
    n_trajectory_frames: int = 0


@dataclass
class BenchmarkSummary:
    total_runs: int = 0
    successful: int = 0
    failed: int = 0
    total_time_seconds: float = 0.0
    results: List[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------
def run_single_chatcam(
    scene_id: str,
    prompt_cleaned: str,
    prompt_idx: int,
    prompt_original: str,
    cfg: dict,
    output_dir: Path,
) -> RunResult:
    """Run ChatCam+GenDoP for a single (scene, prompt) pair."""
    from chatcam import ChatCam

    result = RunResult(
        scene_id=scene_id,
        prompt_idx=prompt_idx,
        prompt_original=prompt_original,
        prompt_cleaned=prompt_cleaned,
        success=False,
        output_dir=str(output_dir),
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save both prompts for reference
    (output_dir / "prompt_original.txt").write_text(prompt_original, encoding="utf-8")
    (output_dir / "prompt.txt").write_text(prompt_cleaned, encoding="utf-8")

    data_root = cfg["data_root"]
    scene_dir = Path(data_root) / scene_id / "dslr"

    # Resolve best available PLY for gsplat rendering
    ply_dir = Path(data_root) / scene_id / "dslr" / "ply"
    gsplat_ply = _resolve_ply_path(ply_dir)

    t0 = time.time()
    try:
        chatcam = ChatCam(
            scene_dir=str(scene_dir),
            openai_api_key=cfg.get("api_key", os.environ.get("OPENAI_API_KEY", "")),
            llm_model=cfg.get("llm_model", "gpt-4.1"),
            clip_model=cfg.get("clip_model", "ViT-B/32"),
            device=cfg.get("device", "cuda"),
            point_cloud_path=cfg.get("point_cloud_path"),
            gsplat_model_path=str(gsplat_ply),
            render=cfg.get("render", True),
            refine_anchors=cfg.get("refine_anchors", False),
            gendop_convention=cfg.get("gendop_convention", "opengl"),
            fps=cfg.get("fps", 24),
        )

        chatcam_result = chatcam.process_instruction(prompt_cleaned, str(output_dir))
        result.success = True

        # Try to get frame count
        traj = chatcam_result.get("trajectory", {})
        c2ws = traj.get("c2ws")
        if c2ws is not None:
            result.n_trajectory_frames = len(c2ws)

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
    data_root = cfg["data_root"]
    output_root = Path(cfg["output_root"])
    prompts_file = cfg["prompts_file"]
    source_level = cfg.get("source_level", "low_id")

    output_root.mkdir(parents=True, exist_ok=True)

    # Load prompts
    print("=" * 70)
    print("CHATCAM + GENDOP BENCHMARK")
    print("=" * 70)
    print(f"Source level: {source_level} (IDs will be stripped)")

    with open(prompts_file) as f:
        all_prompts = json.load(f)

    # Filter to scenes that exist on disk
    scene_ids = sorted(
        sid for sid in all_prompts
        if (Path(data_root) / sid / "dslr").exists()
        and source_level in all_prompts[sid]
        and all_prompts[sid][source_level]
    )
    print(f"Scenes with prompts: {len(scene_ids)}")

    # Count total runs
    total_runs = sum(len(all_prompts[sid][source_level]) for sid in scene_ids)
    print(f"Total runs: {total_runs}")
    print("=" * 70)

    summary = BenchmarkSummary()
    run_idx = 0

    for scene_idx, scene_id in enumerate(scene_ids):
        prompts = all_prompts[scene_id][source_level]

        for prompt_idx, prompt_original in enumerate(prompts):
            run_idx += 1
            prompt_cleaned = strip_id_tags(prompt_original)

            run_output = output_root / scene_id / "low" / f"prompt_{prompt_idx:02d}"

            # Skip if done
            done_marker = run_output / "done.marker"
            if done_marker.exists():
                print(f"[{run_idx}/{total_runs}] {scene_id}/low/{prompt_idx} — already done")
                summary.total_runs += 1
                summary.successful += 1
                continue

            print(f"\n[{run_idx}/{total_runs}] {scene_id} | prompt {prompt_idx}")
            print(f"  Original: {prompt_original[:120]}{'...' if len(prompt_original) > 120 else ''}")
            print(f"  Cleaned:  {prompt_cleaned[:120]}{'...' if len(prompt_cleaned) > 120 else ''}")

            result = run_single_chatcam(
                scene_id=scene_id,
                prompt_cleaned=prompt_cleaned,
                prompt_idx=prompt_idx,
                prompt_original=prompt_original,
                cfg=cfg,
                output_dir=run_output,
            )

            if result.success:
                done_marker.parent.mkdir(parents=True, exist_ok=True)
                done_marker.write_text("done")

            summary.total_runs += 1
            summary.total_time_seconds += result.elapsed_seconds
            if result.success:
                summary.successful += 1
            else:
                summary.failed += 1
            summary.results.append(asdict(result))

            status = "✓" if result.success else "✗"
            print(f"  {status} {result.elapsed_seconds:.1f}s | frames={result.n_trajectory_frames}")

            # Incremental save
            _save_summary(summary, output_root / "benchmark_summary.json")

    # Final
    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
    print(f"Total: {summary.total_runs}  Success: {summary.successful}  Failed: {summary.failed}")
    print(f"Time: {summary.total_time_seconds:.1f}s ({summary.total_time_seconds / 3600:.1f}h)")

    _save_summary(summary, output_root / "benchmark_summary.json")
    print(f"✓ Summary → {output_root / 'benchmark_summary.json'}")

    return summary


def _save_summary(summary: BenchmarkSummary, path: Path):
    with open(path, "w") as f:
        json.dump(asdict(summary), f, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="ChatCam+GenDoP Benchmark Runner")
    parser.add_argument("--config", type=str, default="config/benchmark_chatcam.yaml")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--prompts_file", type=str, default=None)
    parser.add_argument("--api_key", type=str, default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {}

    if args.data_root:
        cfg["data_root"] = args.data_root
    if args.output_root:
        cfg["output_root"] = args.output_root
    if args.prompts_file:
        cfg["prompts_file"] = args.prompts_file
    if args.api_key:
        cfg["api_key"] = args.api_key

    if "data_root" not in cfg:
        parser.error("--data_root required")
    if "prompts_file" not in cfg:
        parser.error("--prompts_file required")
    if "output_root" not in cfg:
        cfg["output_root"] = "outputs/benchmark_chatcam"

    run_benchmark(cfg)


if __name__ == "__main__":
    main()