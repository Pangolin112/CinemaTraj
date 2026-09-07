#!/usr/bin/env python3
"""
run_eval.py — Evaluate GSCinema + baselines on one or more scenes.

Usage (run from code/gscinema/):
    python src/eval/run_eval.py --scene 09c1414f1b
    python src/eval/run_eval.py --scene 09c1414f1b,scene2
    python src/eval/run_eval.py --scene 09c1414f1b --methods gscinema,chatcam
    python src/eval/run_eval.py --scene 09c1414f1b --metrics collision,coverage
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

# ─── Resolve paths ───────────────────────────────────────────────────────
# CWD = where the user runs from (e.g. code/gscinema/)
# All relative paths (benchmark_dir, output) are relative to CWD.
CWD = Path.cwd()
SCRIPT_DIR = Path(__file__).resolve().parent

# Find eval.py: same directory as this script, or one level up
EVAL_PY = None
for candidate in [
    SCRIPT_DIR / "eval.py",
    SCRIPT_DIR.parent / "eval.py",
    SCRIPT_DIR / "evaluation" / "eval.py",
]:
    if candidate.exists():
        EVAL_PY = candidate.resolve()
        break
if EVAL_PY is None:
    print(f"ERROR: eval.py not found near {SCRIPT_DIR}")
    sys.exit(1)

# ─── Defaults (edit these to match your setup) ───────────────────────────
DATA_ROOT = "data/ScanNetpp"

METHODS = {
    "gscinema": {
        "benchmark_dir": "outputs/benchmark_scannetpp",
        "level": "low_id",
    },
    "chatcam": {
        "benchmark_dir": "outputs/benchmark_chatcam",
        "level": "low",
    },
    "cctg": {
        "benchmark_dir": "outputs/benchmark_cctg",
        "level": "low",
    },
    # ── Ablation conditions ──
    "gscinema_full": {
        "benchmark_dir": "outputs/ablation_study/full",
        "level": "low_id",
    },
    "gscinema_no_anchor": {
        "benchmark_dir": "outputs/ablation_study/no_anchor",
        "level": "low_id",
    },
    "gscinema_no_parametric": {
        "benchmark_dir": "outputs/ablation_study/no_parametric",
        "level": "low_id",
    },
    "gscinema_no_sdf": {
        "benchmark_dir": "outputs/ablation_study/no_sdf",
        "level": "low_id",
    },
}

def run_eval(method_name, method_cfg, scene, data_root, metrics, device, extra_args):
    # Resolve benchmark_dir relative to CWD
    bench_dir = (CWD / method_cfg["benchmark_dir"]).resolve()
    if not bench_dir.exists():
        print(f"  ⚠ {method_name}: benchmark_dir not found: {bench_dir}")
        return None

    # Output path inside the benchmark dir
    if scene:
        output_path = bench_dir / f"eval_{method_name}_{scene}.json"
    else:
        output_path = bench_dir / f"eval_{method_name}.json"

    cmd = [
        sys.executable, str(EVAL_PY),
        "--benchmark_dir", str(bench_dir),
        "--data_root", data_root,
        "--level", method_cfg["level"],
        "--output", str(output_path),
        "--device", device,
    ]
    if scene:
        cmd += ["--scene", scene]
    if metrics:
        cmd += ["--metrics", metrics]
    cmd += extra_args

    print(f"\n{'='*60}")
    print(f"  {method_name.upper()}")
    print(f"  bench_dir: {bench_dir}")
    print(f"  level:     {method_cfg['level']}")
    print(f"  scene:     {scene or 'all'}")
    print(f"  output:    {output_path}")
    print(f"{'='*60}")

    # Run eval.py with CWD = caller's working directory
    result = subprocess.run(cmd, cwd=str(CWD))

    if result.returncode != 0:
        print(f"  ✗ {method_name} failed (exit code {result.returncode})")
        return None

    if output_path.exists():
        print(f"  ✓ {method_name} -> {output_path}")
        return output_path
    else:
        print(f"  ⚠ {method_name} finished but output not found at {output_path}")
        return None


def print_comparison(result_paths):
    """Print a side-by-side comparison table of all methods."""
    results = {}
    for method, path in result_paths.items():
        if path is None or not Path(path).exists():
            continue
        with open(path) as f:
            data = json.load(f)
        results[method] = data.get("summary", data)

    if not results:
        print("\nNo results to compare.")
        return

    metrics_keys = [
        ("motion_mse_translation", "Motion MSE (trans)", ".6f"),
        ("motion_mse_rotation", "Motion MSE (rot)", ".4f"),
        ("avg_clatr_score", "CLaTr Score", ".4f"),
        ("avg_collision_rate", "Collision Rate", ".4f"),
        ("avg_occlusion_rate", "Occlusion Rate", ".4f"),
        ("avg_coverage", "Object Coverage", ".4f"),
    ]

    methods = list(results.keys())
    col_w = max(16, max(len(m) for m in methods) + 2)

    print(f"\n{'='*70}")
    print("  COMPARISON")
    print(f"{'='*70}")
    header = f"  {'Metric':<24s}" + "".join(f"{m:>{col_w}s}" for m in methods)
    print(header)
    print(f"  {'-'*24}" + "".join(f"{'-'*col_w}" for _ in methods))

    for key, label, fmt in metrics_keys:
        row = f"  {label:<24s}"
        for m in methods:
            val = results[m].get(key, -1)
            if isinstance(val, (int, float)) and val >= 0:
                row += f"{val:{col_w}{fmt}}"
            else:
                row += f"{'n/a':>{col_w}s}"
        print(row)

    # Per-run count
    row = f"  {'Runs':<24s}"
    for m in methods:
        total = results[m].get("total_runs", "?")
        ok = results[m].get("successful", "?")
        row += f"{f'{ok}/{total}':>{col_w}s}"
    print(row)
    print(f"{'='*70}")


def main():
    p = argparse.ArgumentParser(description="Evaluate GSCinema + baselines")
    p.add_argument("--scene", default=None,
                   help="Scene ID(s), comma-separated. Default: 09c1414f1b")
    p.add_argument("--methods", default=None,
                   help="Methods to evaluate, comma-separated. "
                        f"Available: {','.join(METHODS.keys())}. Default: all.")
    p.add_argument("--data_root", default=DATA_ROOT,
                   help="Root directory for scene data (SDF, labels, GT)")
    p.add_argument("--metrics", default="all",
                   help="Comma-separated metrics or 'all'")
    p.add_argument("--device", default="cuda")
    # Forward extra args to eval.py
    p.add_argument("--clatr_dir", default="CLaTr")
    p.add_argument("--clatr_checkpoint", default="checkpoints/clatr-e100.ckpt")
    p.add_argument("--n_resample", type=int, default=100)
    p.add_argument("--center_fraction", type=float, default=0.5)
    p.add_argument("--occlusion_n_steps", type=int, default=32)
    p.add_argument("--occlusion_margin", type=float, default=0.15)
    a = p.parse_args()

    # Build extra args to forward
    extra = []
    for k in ["clatr_dir", "clatr_checkpoint", "n_resample", "center_fraction",
              "occlusion_n_steps", "occlusion_margin"]:
        extra += [f"--{k}", str(getattr(a, k))]

    # Select methods
    if a.methods:
        method_names = [m.strip() for m in a.methods.split(",")]
        for m in method_names:
            if m not in METHODS:
                print(f"Unknown method: {m}. Available: {list(METHODS.keys())}")
                sys.exit(1)
    else:
        method_names = list(METHODS.keys())

    metrics_arg = a.metrics if a.metrics != "all" else None

    print(f"eval.py:   {EVAL_PY}")
    print(f"CWD:       {CWD}")
    print(f"Methods:   {method_names}")
    print(f"Scene(s):  {a.scene or 'all'}")
    print(f"Data root: {a.data_root}")
    print(f"Device:    {a.device}")

    # Run each method
    result_paths = {}
    for name in method_names:
        cfg = METHODS[name]
        out = run_eval(name, cfg, a.scene, a.data_root, metrics_arg, a.device, extra)
        result_paths[name] = out

    # Print comparison
    print_comparison(result_paths)


if __name__ == "__main__":
    main()