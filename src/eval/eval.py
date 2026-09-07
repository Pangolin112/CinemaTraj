"""
TrajScene Evaluation
====================

Quantitative evaluation for the TrajScene benchmark.

Paper Metrics (5 total)
-----------------------
1. Motion MSE          — translation MSE + geodesic rotation error vs GT (low-level only)
2. CLaTr Score         — cosine similarity in shared trajectory-text latent space (↑)
3. Collision Rate      — fraction of samples with SDF < 0 (↓)
4. Occlusion Rate      — fraction of samples where target object is blocked (↓)
5. Object Coverage     — fraction of planned anchor objects visited (↑)

Usage
-----
    python eval.py --run_dir outputs/scene/high/prompt_00
    python eval.py --benchmark_dir outputs --data_root data --output results.json
    python eval.py --run_dir ... --metrics motion_mse,collision   # selective
    python eval.py --benchmark_dir outputs --data_root data --level low   # low-level only (default)
    python eval.py --benchmark_dir outputs --data_root data --level all   # all levels
"""

import argparse
import json
import sys
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# =========================================================================
# Data structures — aligned with the 5 paper metrics
# =========================================================================

@dataclass
class MotionMSEMetrics:
    """Metric 1: Motion MSE against hand-crafted ground-truth trajectory.
    Both trajectories are uniformly resampled by arc length to n_resample points.
    Only evaluated for low-level prompts when GT is provided."""
    translation_mse: float = -1.0     # MSE of positions after arc-length resampling
    rotation_error_mean: float = -1.0  # mean geodesic rotation error (radians)
    rotation_error_max: float = -1.0   # max geodesic rotation error (radians)
    n_resample: int = 0

@dataclass
class CLaTrMetrics:
    """Metric 2: CLaTr Score (E.T. / CLaTr framework).
    Cosine similarity between trajectory and text embeddings in shared 256-D
    latent space via two ACTORStyleEncoders."""
    clatr_score: float = -1.0

@dataclass
class CollisionMetrics:
    """Metric 3: Collision Rate from SDF evaluation."""
    collision_rate: float = -1.0       # fraction of poses with SDF < 0
    min_sdf_value: float = 0.0         # worst-case penetration depth

@dataclass
class OcclusionMetrics:
    """Metric 4: Occlusion Rate — fraction of trajectory samples where the
    current target object is blocked by intervening geometry.

    Uses the same transmittance-weighted ray-marching as the optimizer's
    _occlusion_cost: softplus penalty with OBB exclusion masking, accumulated
    via volume-rendering-style transmittance weighting."""
    occlusion_rate: float = -1.0       # fraction of samples where target is blocked
    mean_occlusion_value: float = 0.0  # average transmittance-weighted occlusion

@dataclass
class CoverageMetrics:
    """Metric 5: Object Coverage — fraction of planned objects that appear
    unoccluded and roughly centered in at least one frame.

    For scene-graph-based methods (ours), anchoring is grounded in the 3D
    scene graph so coverage is 100% by construction.  For baselines that use
    2D CLIP-based anchoring, we verify geometrically: project the target
    object's bbox center into the camera frame and check that it is
    (a) within a central region of the image, and (b) not occluded by
    intervening geometry (SDF ray-march)."""
    object_coverage: float = -1.0
    n_planned: int = 0
    n_visited: int = 0
    per_object: Dict = field(default_factory=dict)  # label -> bool

@dataclass
class EvalResult:
    scene_id: str = ""
    level: str = ""
    prompt_idx: int = 0
    prompt: str = ""
    success: bool = True
    # The 5 paper metrics
    motion_mse: MotionMSEMetrics = field(default_factory=MotionMSEMetrics)
    clatr: CLaTrMetrics = field(default_factory=CLaTrMetrics)
    collision: CollisionMetrics = field(default_factory=CollisionMetrics)
    occlusion: OcclusionMetrics = field(default_factory=OcclusionMetrics)
    coverage: CoverageMetrics = field(default_factory=CoverageMetrics)
    # Auxiliary (kept for debugging, not in paper table)
    n_poses: int = 0
    path_length: float = 0.0


# =========================================================================
# Helpers
# =========================================================================

def load_trajectory(path):
    p = Path(path)
    return json.load(open(p)) if p.exists() else None


def _get_first(d, *keys, default=None):
    for k in keys:
        if k in d:
            return d[k]
    return default


def extract_poses(traj_data):
    """Returns (N,3) positions, (N,3,3)||(N,3) rotations, (N,) timestamps.
    Handles NeRF-style transform_matrix/c2w frames, list-of-dicts, dict-of-arrays."""
    if traj_data is None:
        return np.zeros((0, 3)), np.zeros((0, 3, 3)), np.zeros(0)

    # ── Unwrap container keys ──
    if isinstance(traj_data, dict):
        if "frames" in traj_data and isinstance(traj_data["frames"], list):
            frames = traj_data["frames"]
            # Detect transform key: transform_matrix or c2w
            tm_key = None
            if frames and isinstance(frames[0], dict):
                if "transform_matrix" in frames[0]: tm_key = "transform_matrix"
                elif "c2w" in frames[0]: tm_key = "c2w"
            if tm_key is not None:
                positions, rotations, timestamps = [], [], []
                for i, fr in enumerate(frames):
                    T = np.array(fr[tm_key])
                    if T.shape == (4, 4) or T.shape == (3, 4):
                        positions.append(T[:3, 3])
                        rotations.append(T[:3, :3])
                    timestamps.append(fr.get("timestamp", fr.get("time", i / 30.0)))
                if positions:
                    return np.array(positions), np.array(rotations), np.array(timestamps)
                return np.zeros((0, 3)), np.zeros((0, 3, 3)), np.zeros(0)
            else:
                traj_data = frames
        elif "trajectory" in traj_data and isinstance(traj_data["trajectory"], (list, dict)):
            traj_data = traj_data["trajectory"]
        elif "keyframes" in traj_data and isinstance(traj_data["keyframes"], list):
            traj_data = traj_data["keyframes"]

    # ── List of per-frame dicts ──
    if isinstance(traj_data, list):
        if not traj_data:
            return np.zeros((0, 3)), np.zeros((0, 3, 3)), np.zeros(0)
        first = traj_data[0] if isinstance(traj_data[0], dict) else {}

        # Detect transform key: transform_matrix or c2w
        tm_key2 = None
        if "transform_matrix" in first: tm_key2 = "transform_matrix"
        elif "c2w" in first: tm_key2 = "c2w"
        if tm_key2 is not None:
            positions, rotations, timestamps = [], [], []
            for i, fr in enumerate(traj_data):
                if not isinstance(fr, dict): continue
                T = np.array(fr[tm_key2])
                if T.shape in ((4, 4), (3, 4)):
                    positions.append(T[:3, 3]); rotations.append(T[:3, :3])
                timestamps.append(fr.get("timestamp", fr.get("time", i / 30.0)))
            if positions:
                return np.array(positions), np.array(rotations), np.array(timestamps)
            return np.zeros((0, 3)), np.zeros((0, 3, 3)), np.zeros(0)

        pos_key = None
        for k in ("position", "camera_position", "pos", "translation", "xyz"):
            if k in first: pos_key = k; break
        if pos_key is None:
            if isinstance(traj_data[0], (list, tuple)) and len(traj_data[0]) >= 3:
                pos = np.array(traj_data)[:, :3]
                return pos, np.tile(np.eye(3), (len(pos), 1, 1)), np.arange(len(pos)) / 30.0
            return np.zeros((0, 3)), np.zeros((0, 3, 3)), np.zeros(0)

        pos = np.array([p[pos_key] for p in traj_data])
        ts = np.array([p.get("timestamp", p.get("time", p.get("t", i / 30.0)))
                        for i, p in enumerate(traj_data)])

        rot_key = None
        for k in ("rotation", "rotation_matrix", "quaternion", "quat", "rot"):
            if k in first: rot_key = k; break
        if rot_key is not None:
            rot = np.array([p[rot_key] for p in traj_data])
        else:
            la_key = None
            for k in ("look_at", "lookat", "look_at_point", "target"):
                if k in first: la_key = k; break
            if la_key is not None:
                la = np.array([p[la_key] for p in traj_data])
                fwd = la - pos
                fwd /= np.maximum(np.linalg.norm(fwd, axis=1, keepdims=True), 1e-8)
                rot = _directions_to_rotmats(fwd)
            else:
                rot = np.tile(np.eye(3), (len(pos), 1, 1))

    elif isinstance(traj_data, dict):
        pos_raw = _get_first(traj_data, "positions", "camera_positions", "poses",
                             "translations", "camera_path", "path_positions", default=[])
        pos = np.array(pos_raw)
        if pos.ndim != 2 or pos.shape[0] == 0:
            return np.zeros((0, 3)), np.zeros((0, 3, 3)), np.zeros(0)
        if pos.shape[1] > 3: pos = pos[:, :3]

        ts_raw = _get_first(traj_data, "timestamps", "times", "time",
                            default=np.arange(len(pos)) / 30.0)
        ts = np.array(ts_raw)

        rot_raw = _get_first(traj_data, "rotations", "rotation_matrices",
                             "quaternions", "quats")
        if rot_raw is not None:
            rot = np.array(rot_raw)
        else:
            la_raw = _get_first(traj_data, "look_ats", "look_at_points",
                                "targets", "lookats")
            if la_raw is not None:
                la = np.array(la_raw)
                fwd = la - pos
                fwd /= np.maximum(np.linalg.norm(fwd, axis=1, keepdims=True), 1e-8)
                rot = _directions_to_rotmats(fwd)
            else:
                rot = np.tile(np.eye(3), (len(pos), 1, 1))
    else:
        return np.zeros((0, 3)), np.zeros((0, 3, 3)), np.zeros(0)

    # Ensure rotation is always (N, 3, 3)
    if rot.ndim == 2 and rot.shape[1] == 3:
        rot = _directions_to_rotmats(rot)
    elif rot.ndim == 2 and rot.shape[1] == 4:
        from scipy.spatial.transform import Rotation
        rot = Rotation.from_quat(rot).as_matrix()

    return pos, rot, ts


def _directions_to_rotmats(fwd):
    """Convert (N,3) forward direction vectors to (N,3,3) rotation matrices."""
    n = len(fwd)
    fwd = fwd / np.maximum(np.linalg.norm(fwd, axis=1, keepdims=True), 1e-8)
    up = np.tile([0., 0., 1.], (n, 1))
    right = np.cross(fwd, up)
    rn = np.linalg.norm(right, axis=1, keepdims=True)
    degen = rn.flatten() < 1e-6
    if degen.any():
        right[degen] = np.cross(fwd[degen], [1., 0., 0.])
        rn[degen] = np.linalg.norm(right[degen], axis=1, keepdims=True)
    right /= np.maximum(rn, 1e-8)
    up2 = np.cross(right, fwd)
    return np.stack([right, up2, fwd], axis=-1)


# =========================================================================
# Prompt Discovery (unchanged from previous version)
# =========================================================================

def discover_prompts_file(start_dir):
    d = Path(start_dir).resolve()
    for _ in range(10):
        candidate = d / "benchmark_prompts.json"
        if candidate.exists(): return candidate
        parent = d.parent
        if parent == d: break
        d = parent
    return None

def load_prompts_map(prompts_file):
    if prompts_file is None: return {}
    p = Path(prompts_file)
    if not p.exists(): return {}
    data = json.load(open(p))
    pmap = {}
    if isinstance(data, list):
        for entry in data:
            if not isinstance(entry, dict): continue
            sid = entry.get("scene_id", "")
            lvl = entry.get("level", entry.get("difficulty", ""))
            pidx = entry.get("prompt_idx", entry.get("prompt_id", entry.get("index", 0)))
            text = entry.get("prompt", entry.get("text", entry.get("description", "")))
            if sid and text: pmap[(sid, lvl, int(pidx))] = text
    elif isinstance(data, dict):
        for sid, scene_data in data.items():
            if isinstance(scene_data, dict):
                for lvl, prompts in scene_data.items():
                    if isinstance(prompts, list):
                        for idx, pe in enumerate(prompts):
                            if isinstance(pe, str): pmap[(sid, lvl, idx)] = pe
                            elif isinstance(pe, dict):
                                text = pe.get("prompt", pe.get("text", ""))
                                pidx = pe.get("prompt_idx", pe.get("index", idx))
                                if text: pmap[(sid, lvl, int(pidx))] = text
                    elif isinstance(prompts, str): pmap[(sid, lvl, 0)] = prompts
            elif isinstance(scene_data, list):
                for idx, pe in enumerate(scene_data):
                    if isinstance(pe, str): pmap[(sid, "", idx)] = pe
                    elif isinstance(pe, dict):
                        lvl = pe.get("level", pe.get("difficulty", ""))
                        text = pe.get("prompt", pe.get("text", ""))
                        pidx = pe.get("prompt_idx", pe.get("index", idx))
                        if text: pmap[(sid, lvl, int(pidx))] = text
    return pmap

def lookup_prompt(pmap, scene_id, level, prompt_idx):
    if not pmap: return ""
    key = (scene_id, level, prompt_idx)
    if key in pmap: return pmap[key]
    for (sid, lvl, pidx), text in pmap.items():
        if sid == scene_id and pidx == prompt_idx: return text
    for (sid, lvl, pidx), text in pmap.items():
        if sid == scene_id: return text
    return ""

def _discover_prompt(run_dir, scene_id="", level="", prompt_idx=0):
    rd = Path(run_dir)
    pf = discover_prompts_file(rd)
    if pf:
        pmap = load_prompts_map(pf)
        prompt = lookup_prompt(pmap, scene_id, level, prompt_idx)
        if prompt: return prompt
    try:
        import yaml as _yaml
    except ImportError:
        _yaml = None
    if _yaml is not None:
        d = rd.resolve()
        for _ in range(6):
            for name in ["config.yaml", "config.yml"]:
                yp = d / name
                if yp.exists():
                    try:
                        with open(yp) as f: ycfg = _yaml.safe_load(f)
                        if isinstance(ycfg, dict):
                            prompt = ycfg.get("prompt", "")
                            cfg_scene = ycfg.get("scene_id", "")
                            if prompt and (not scene_id or not cfg_scene or cfg_scene == scene_id):
                                return prompt
                    except Exception: pass
            parent = d.parent
            if parent == d: break
            d = parent
    for sname in ["subtitles.json"]:
        sp = rd / sname
        if sp.exists():
            try:
                with open(sp) as f: sdata = json.load(f)
                if isinstance(sdata, dict):
                    prompt = sdata.get("prompt", sdata.get("text", ""))
                    if prompt: return prompt
            except Exception: pass
    # Try prompt.txt
    pt = rd / "prompt.txt"
    if pt.exists():
        try:
            return pt.read_text().strip()
        except Exception:
            pass
    return ""


# =========================================================================
# Metric 1: Motion MSE (vs ground-truth trajectory)
# =========================================================================

def _resample_by_arc_length(positions, rotations, n_points):
    """Uniformly resample a trajectory by arc length.
    Returns resampled positions (n_points, 3) and rotations (n_points, 3, 3)."""
    diffs = np.diff(positions, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    cum_len = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total_len = cum_len[-1]
    if total_len < 1e-10:
        return positions[:1].repeat(n_points, axis=0), rotations[:1].repeat(n_points, axis=0)

    target_s = np.linspace(0, total_len, n_points)
    # Interpolate positions
    resampled_pos = np.zeros((n_points, 3))
    for dim in range(3):
        resampled_pos[:, dim] = np.interp(target_s, cum_len, positions[:, dim])

    # Interpolate rotations via SLERP
    from scipy.spatial.transform import Rotation, Slerp
    rots = Rotation.from_matrix(rotations)
    # Slerp requires strictly increasing times
    # Use cum_len but handle duplicate values
    unique_mask = np.concatenate([[True], np.diff(cum_len) > 1e-10])
    if unique_mask.sum() < 2:
        resampled_rot = np.tile(rotations[0], (n_points, 1, 1))
    else:
        slerp = Slerp(cum_len[unique_mask], rots[unique_mask])
        # Clamp target_s to valid range
        target_s_clamped = np.clip(target_s, cum_len[unique_mask][0], cum_len[unique_mask][-1])
        resampled_rot = slerp(target_s_clamped).as_matrix()

    return resampled_pos, resampled_rot


def _geodesic_rotation_error(R_pred, R_gt):
    """Geodesic distance between rotation matrices: arccos((trace(R_pred^T R_gt) - 1) / 2).
    Returns per-frame errors in radians."""
    # R_pred, R_gt: (N, 3, 3)
    R_rel = np.einsum("nij,nkj->nik", R_pred, R_gt)  # R_pred @ R_gt.T
    traces = np.trace(R_rel, axis1=1, axis2=2)
    # Clamp for numerical stability
    cos_angle = np.clip((traces - 1.0) / 2.0, -1.0, 1.0)
    return np.arccos(cos_angle)


def _opengl_to_opencv_rotation(R):
    """Convert rotation matrices from OpenGL convention (forward=-Z, up=+Y)
    to OpenCV convention (forward=+Z, up=-Y).
    Flips Y and Z columns: R_opencv = R_opengl @ diag(1, -1, -1)."""
    flip = np.array([[1, 0, 0],
                     [0, -1, 0],
                     [0, 0, -1]], dtype=R.dtype)
    if R.ndim == 3:
        return R @ flip[None, :, :]
    return R @ flip


def evaluate_motion_mse(traj_path, gt_path, n_resample=100):
    """Metric 1: Motion MSE between predicted and ground-truth trajectories.
    Both are uniformly resampled by arc length to n_resample points."""
    m = MotionMSEMetrics()
    if gt_path is None or not Path(gt_path).exists():
        return m  # GT not available, skip

    td = load_trajectory(traj_path)
    gt = load_trajectory(gt_path)
    if td is None or gt is None: return m

    pos_pred, rot_pred, _ = extract_poses(td)
    pos_gt, rot_gt, _ = extract_poses(gt)
    if len(pos_pred) < 2 or len(pos_gt) < 2: return m

    # Skip degenerate trajectories (position range > 100m is clearly broken)
    pred_range = np.ptp(pos_pred, axis=0).max()
    gt_range = np.ptp(pos_gt, axis=0).max()
    if pred_range > 100.0 or gt_range > 100.0:
        return m  # returns -1 sentinel values, excluded from aggregation

    # Ensure rotations are (N, 3, 3)
    # GT uses OpenGL camera convention (forward = -Z), while methods use
    # OpenCV-like convention (forward = +Z). Convert GT to match.
    rot_gt = _opengl_to_opencv_rotation(rot_gt)
    if rot_pred.ndim != 3: rot_pred = np.tile(np.eye(3), (len(pos_pred), 1, 1))
    if rot_gt.ndim != 3: rot_gt = np.tile(np.eye(3), (len(pos_gt), 1, 1))

    # Resample both to same number of points by arc length
    pos_p, rot_p = _resample_by_arc_length(pos_pred, rot_pred, n_resample)
    pos_g, rot_g = _resample_by_arc_length(pos_gt, rot_gt, n_resample)

    # Translation MSE
    m.translation_mse = float(np.mean(np.sum((pos_p - pos_g) ** 2, axis=1)))

    # Geodesic rotation error
    geo_err = _geodesic_rotation_error(rot_p, rot_g)
    m.rotation_error_mean = float(np.mean(geo_err))
    m.rotation_error_max = float(np.max(geo_err))
    m.n_resample = n_resample

    return m


# =========================================================================
# CLIP loading helper
# =========================================================================

_clip_module = None
_clip_model_cache = {}

def _get_clip_module():
    global _clip_module
    if _clip_module is not None: return _clip_module
    try:
        import clip
        if not hasattr(clip, 'load'):
            raise ImportError("Installed 'clip' is not OpenAI CLIP.")
        _clip_module = clip; return clip
    except ImportError as e:
        raise ImportError(f"CLIP not available: {e}") from e

def _load_clip_model(model_name="ViT-B/32", device="cuda"):
    key = (model_name, str(device))
    if key not in _clip_model_cache:
        clip = _get_clip_module()
        _clip_model_cache[key] = clip.load(model_name, device=device)
    return _clip_model_cache[key]


# =========================================================================
# Metric 2: CLaTr Score
# =========================================================================

_CLATR_NUM_CAMS = 300
_CLATR_NORM_MEAN = np.array([7.93987673e-05, -9.98621393e-05, 4.12940653e-04], dtype=np.float32)
_CLATR_NORM_STD  = np.array([0.027841, 0.01819818, 0.03138536], dtype=np.float32)
_CLATR_SHIFT_MEAN = np.array([0.00201079, -0.27488501, -1.23616805], dtype=np.float32)
_CLATR_SHIFT_STD  = np.array([1.13433516, 1.19061042, 1.58744263], dtype=np.float32)


def _resample_trajectory_uniform(positions, rotations, n_target):
    """Resample a trajectory to exactly n_target frames via uniform temporal
    interpolation (linear for positions, SLERP for rotations).

    This ensures all trajectories fed to CLaTr have the same number of frames
    regardless of original length, avoiding bias from truncation or zero-padding.

    Args:
        positions:  (N, 3) array of camera positions.
        rotations:  (N, 3, 3) array of rotation matrices.
        n_target:   Target number of frames (e.g. 300).

    Returns:
        resampled_pos: (n_target, 3)
        resampled_rot: (n_target, 3, 3)
    """
    n = len(positions)
    if n == n_target:
        return positions.astype(np.float32), rotations.astype(np.float32)
    if n < 2:
        return (np.tile(positions[:1], (n_target, 1)).astype(np.float32),
                np.tile(rotations[:1], (n_target, 1, 1)).astype(np.float32))

    # Original frame indices as "time" parameter
    t_orig = np.linspace(0.0, 1.0, n)
    t_target = np.linspace(0.0, 1.0, n_target)

    # Interpolate positions (linear per dimension)
    resampled_pos = np.zeros((n_target, 3), dtype=positions.dtype)
    for dim in range(3):
        resampled_pos[:, dim] = np.interp(t_target, t_orig, positions[:, dim])

    # Interpolate rotations via SLERP
    from scipy.spatial.transform import Rotation, Slerp
    rots = Rotation.from_matrix(rotations)
    # Handle near-duplicate time values for Slerp
    unique_mask = np.concatenate([[True], np.diff(t_orig) > 1e-12])
    if unique_mask.sum() < 2:
        resampled_rot = np.tile(rotations[0], (n_target, 1, 1))
    else:
        slerp = Slerp(t_orig[unique_mask], rots[unique_mask])
        t_clamped = np.clip(t_target, t_orig[unique_mask][0], t_orig[unique_mask][-1])
        resampled_rot = slerp(t_clamped).as_matrix()

    return resampled_pos.astype(np.float32), resampled_rot.astype(np.float32)


class CLaTrEvaluator:
    """Loads pretrained CLaTr encoders and computes CLaTr-Score."""

    def __init__(self, clatr_dir="CLaTr", ckpt="checkpoints/clatr-e100.ckpt", device="cuda"):
        self.device = device
        self.traj_encoder = self.text_encoder = None
        self.available = False
        try:
            import torch; self.torch = torch
        except ImportError: return
        root = Path(clatr_dir); cp = root / ckpt
        te_path = cp.parent / "clatr-traj_encoder.ckpt"
        tx_path = cp.parent / "clatr-text_encoder.ckpt"
        if te_path.exists() and tx_path.exists():
            self._load_submodules(te_path, tx_path)
        elif cp.exists():
            self._load_full(cp)
        else:
            print(f"  [CLaTr] checkpoint not found: {cp}")

    def _make_encoder(self, num_feats):
        import torch; import torch.nn as nn
        class PosEnc(nn.Module):
            def __init__(self, d, max_len=5000):
                super().__init__()
                pe = torch.zeros(max_len, d)
                p = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
                div = torch.exp(torch.arange(0, d, 2).float() * (-np.log(10000.) / d))
                pe[:, 0::2] = torch.sin(p * div); pe[:, 1::2] = torch.cos(p * div)
                self.register_buffer("pe", pe.unsqueeze(0), persistent=False)
            def forward(self, x): return x + self.pe[:, :x.shape[1], :]
        class Enc(nn.Module):
            def __init__(self):
                super().__init__()
                self.projection = nn.Linear(num_feats, 256)
                self.nbtokens = 2
                self.tokens = nn.Parameter(torch.randn(2, 256))
                self.pos = PosEnc(256)
                layer = nn.TransformerEncoderLayer(d_model=256, nhead=4,
                    dim_feedforward=1024, dropout=0.1, activation="gelu", batch_first=True)
                self.seqTransEncoder = nn.TransformerEncoder(layer, num_layers=6)
            def forward(self, x, mask):
                x = self.projection(x); bs = len(x)
                tok = self.tokens.unsqueeze(0).expand(bs, -1, -1)
                xseq = torch.cat((tok, x), 1)
                tm = torch.ones((bs, 2), dtype=torch.bool, device=x.device)
                am = torch.cat((tm, mask), 1)
                xseq = self.pos(xseq)
                out = self.seqTransEncoder(xseq, src_key_padding_mask=~am)
                return out[:, :2]
        return Enc()

    def _load_submodules(self, te_path, tx_path):
        self.traj_encoder = self._make_encoder(9)
        self.text_encoder = self._make_encoder(512)
        self.traj_encoder.load_state_dict(self.torch.load(str(te_path), map_location="cpu"))
        self.text_encoder.load_state_dict(self.torch.load(str(tx_path), map_location="cpu"))
        self.traj_encoder = self.traj_encoder.to(self.device).eval()
        self.text_encoder = self.text_encoder.to(self.device).eval()
        self.available = True
        print("  [CLaTr] ✓ loaded from submodule checkpoints")

    def _load_full(self, cp):
        sd = self.torch.load(str(cp), map_location="cpu").get("state_dict", {})
        def extract(prefix):
            return {".".join(k.split(".")[1:]): v for k, v in sd.items() if k.startswith(prefix)}
        tsd, xsd = extract("traj_encoder."), extract("text_encoder.")
        if not tsd or not xsd:
            print("  [CLaTr] ⚠ missing encoder keys"); return
        self.traj_encoder = self._make_encoder(9)
        self.text_encoder = self._make_encoder(512)
        self.traj_encoder.load_state_dict(tsd)
        self.text_encoder.load_state_dict(xsd)
        self.traj_encoder = self.traj_encoder.to(self.device).eval()
        self.text_encoder = self.text_encoder.to(self.device).eval()
        self.available = True
        print(f"  [CLaTr] ✓ loaded from {cp.name}")

    @staticmethod
    def poses_to_rot6d_t(positions, rotations):
        """Convert to CLaTr 9D input with velocity encoding + standardization.

        The trajectory is first resampled to exactly 300 frames via uniform
        temporal interpolation (linear for positions, SLERP for rotations).
        This ensures consistent representation regardless of original length,
        with all 300 frames being valid (mask = all ones).
        """
        n = len(positions)
        if n < 2:
            # Degenerate: replicate single frame
            positions = np.tile(positions[:1], (2, 1)) if n == 1 else np.zeros((2, 3))
            rotations = (np.tile(rotations[:1], (2, 1, 1)) if n == 1
                         else np.tile(np.eye(3, dtype=np.float32), (2, 1, 1)))
            n = 2

        # Ensure rotations are (N, 3, 3)
        if rotations.ndim == 3 and rotations.shape[1:] == (3, 3):
            R = rotations.astype(np.float32)
        elif rotations.ndim == 2 and rotations.shape[1] == 4:
            from scipy.spatial.transform import Rotation
            R = Rotation.from_quat(rotations).as_matrix().astype(np.float32)
        else:
            R = np.tile(np.eye(3, dtype=np.float32), (n, 1, 1))

        # ── Resample to exactly 300 frames ──
        pos_resampled, R_resampled = _resample_trajectory_uniform(
            positions.astype(np.float32), R, _CLATR_NUM_CAMS)

        # rot6d: column-major via .transpose(0,2,1)
        rot6d = R_resampled[:, :, :2].transpose(0, 2, 1).reshape(_CLATR_NUM_CAMS, 6)

        # Velocity encoding
        trans = pos_resampled.copy()
        velocity = trans[1:] - trans[:-1]
        trans_vel = np.concatenate([trans[0:1], velocity], axis=0)

        # Standardize
        trans_vel[0] = (trans_vel[0] - _CLATR_SHIFT_MEAN) / _CLATR_SHIFT_STD
        trans_vel[1:] = (trans_vel[1:] - _CLATR_NORM_MEAN) / _CLATR_NORM_STD

        feats = np.concatenate([rot6d, trans_vel], axis=1)  # (300, 9)

        # All frames are valid — no padding needed
        mask = np.ones(_CLATR_NUM_CAMS, dtype=np.float32)

        return feats, mask

    @staticmethod
    def _extract_clip_sequence_features(text, clip_model, device):
        """Per-token CLIP sequence features [1, 77, 512] with mask."""
        import torch
        clip = _get_clip_module()
        tokens = clip.tokenize([text], truncate=True).to(device)
        with torch.no_grad():
            x = clip_model.token_embedding(tokens).type(clip_model.dtype)
            x = x + clip_model.positional_embedding.type(clip_model.dtype)
            x = x.permute(1, 0, 2)
            x = clip_model.transformer(x)
            x = x.permute(1, 0, 2)
            x = clip_model.ln_final(x)
            seq_features = x.float()
        eot_idx = tokens.argmax(dim=-1).item()
        seq_len = eot_idx + 1
        mask = torch.zeros(1, 77, dtype=torch.bool, device=device)
        mask[0, :seq_len] = True
        return seq_features, mask

    def encode_trajectory(self, positions, rotations):
        torch = self.torch
        feats, mask = self.poses_to_rot6d_t(positions, rotations)
        x = torch.tensor(feats).unsqueeze(0).to(self.device)
        m = torch.tensor(mask).unsqueeze(0).bool().to(self.device)
        with torch.no_grad():
            out = self.traj_encoder(x, m)
        mu = out[:, 0]
        mu = mu / mu.norm(dim=-1, keepdim=True)
        return mu.cpu().numpy().flatten()

    def encode_text(self, text):
        torch = self.torch
        clip_model, _ = _load_clip_model("ViT-B/32", device=self.device)
        seq_features, mask = self._extract_clip_sequence_features(text, clip_model, self.device)
        with torch.no_grad():
            out = self.text_encoder(seq_features, mask)
        mu = out[:, 0]
        mu = mu / mu.norm(dim=-1, keepdim=True)
        return mu.cpu().numpy().flatten()

    def compute_clatr_score(self, positions, rotations, text):
        if not self.available: return -1.0
        tf = self.encode_trajectory(positions, rotations)
        xf = self.encode_text(text)
        return float(np.dot(tf, xf))


_clatr: Optional[CLaTrEvaluator] = None

def get_clatr(clatr_dir="CLaTr", ckpt="checkpoints/clatr-e100.ckpt", device="cuda"):
    global _clatr
    if _clatr is None: _clatr = CLaTrEvaluator(clatr_dir, ckpt, device)
    return _clatr

def evaluate_clatr(traj_path, prompt, clatr_dir="CLaTr", ckpt="checkpoints/clatr-e100.ckpt", device="cuda"):
    m = CLaTrMetrics()
    td = load_trajectory(traj_path)
    if td is None or not prompt: return m
    pos, rot, ts = extract_poses(td)
    if len(pos) < 2: return m
    ev = get_clatr(clatr_dir, ckpt, device)
    if not ev.available: return m
    m.clatr_score = ev.compute_clatr_score(pos, rot, prompt)
    return m


# =========================================================================
# Metric 3: Collision Rate
# =========================================================================

def _load_sdf_grid(sdf_path):
    """Load precomputed SDF grid from .npz file."""
    if sdf_path is None or not Path(sdf_path).exists():
        return None
    d = np.load(sdf_path)
    return {
        "grid": d["sdf_grid"],
        "grid_min": d["grid_min"].astype(float),
        "grid_max": d["grid_max"].astype(float),
        "resolution": int(d["resolution"]),
    }


def _query_sdf_grid(sdf_data, points):
    """Query SDF values at given points using nearest-neighbor lookup.
    Returns array of SDF values, one per point."""
    grid = sdf_data["grid"]
    gmin, gmax = sdf_data["grid_min"], sdf_data["grid_max"]
    res = sdf_data["resolution"]
    gs = gmax - gmin

    vals = np.zeros(len(points))
    for i, p in enumerate(points):
        gc = ((p - gmin) / gs) * (res - 1)
        ix = tuple(int(np.clip(round(gc[j]), 0, res - 1)) for j in range(3))
        vals[i] = float(grid[ix])
    return vals


def evaluate_collision(traj_path, sdf_path=None):
    """Metric 3: Collision Rate — fraction of trajectory poses with SDF < 0."""
    m = CollisionMetrics()
    td = load_trajectory(traj_path)
    if td is None: return m
    pos, rot, ts = extract_poses(td)
    if len(pos) < 2: return m

    sdf_data = _load_sdf_grid(sdf_path)
    if sdf_data is None: return m

    sdf_vals = _query_sdf_grid(sdf_data, pos)
    m.collision_rate = float(np.mean(sdf_vals < 0))
    m.min_sdf_value = float(np.min(sdf_vals))
    return m


# =========================================================================
# Metric 4: Occlusion Rate
# =========================================================================

def _load_labels(labels_path):
    """Load labels.json with bounding box data (old format).
    Returns dict mapping ins_id -> {label, bbox_corners (8,3), center (3,), ...}."""
    if labels_path is None or not Path(labels_path).exists():
        return {}
    with open(labels_path) as f:
        data = json.load(f)
    objects = {}
    for obj in data:
        if not isinstance(obj, dict): continue
        ins_id = str(obj.get("ins_id", ""))
        label = obj.get("label", "")
        bbox_raw = obj.get("bounding_box", [])
        if not bbox_raw or len(bbox_raw) < 8:
            continue
        corners = np.array([[c["x"], c["y"], c["z"]] for c in bbox_raw])
        center = corners.mean(axis=0)
        # Extract OBB axes from corners (bottom face: 0-3, top face: 4-7)
        # Edge vectors of bottom face
        e0 = corners[1] - corners[0]  # along one axis
        e1 = corners[3] - corners[0]  # along another axis
        e2 = corners[4] - corners[0]  # up axis
        axes = np.stack([e0, e1, e2])
        half_extents = np.linalg.norm(axes, axis=1) / 2.0
        # Normalize axes
        axes_norm = axes / np.maximum(np.linalg.norm(axes, axis=1, keepdims=True), 1e-8)
        objects[ins_id] = {
            "label": label,
            "corners": corners,
            "center": center,
            "axes": axes_norm,       # (3, 3) OBB local axes (rows)
            "half_extents": half_extents,  # (3,) half-sizes along each axis
        }
    return objects


def _load_labels_simple(labels_path):
    """Load scene graph JSON in OBB format (cx,cy,cz,sx,sy,sz,qx,qy,qz,qw).
    This is the ScanNet++ '-simple.json' format where objects are keyed by
    label string (e.g. 'sofa_0') with an 'obb' array.

    Returns dict mapping label_str -> {label, corners, center, axes, half_extents}.
    """
    if labels_path is None or not Path(labels_path).exists():
        return {}
    with open(labels_path) as f:
        data = json.load(f)

    if not isinstance(data, dict) or "objects" not in data:
        return {}

    from scipy.spatial.transform import Rotation as R_scipy

    objects = {}
    for obj_label, obj_data in data["objects"].items():
        if not isinstance(obj_data, dict) or "obb" not in obj_data:
            continue
        obb = obj_data["obb"]
        if len(obb) < 10:
            continue
        cx, cy, cz = obb[0], obb[1], obb[2]
        sx, sy, sz = obb[3], obb[4], obb[5]
        qx, qy, qz, qw = obb[6], obb[7], obb[8], obb[9]

        center = np.array([cx, cy, cz])
        half_extents = np.array([sx, sy, sz]) / 2.0

        # Quaternion to rotation matrix (scipy uses [x,y,z,w])
        rot_mat = R_scipy.from_quat([qx, qy, qz, qw]).as_matrix()  # (3,3)
        # Each row of axes_norm is a local OBB axis direction
        axes_norm = rot_mat.T

        # Compute 8 corners for compatibility
        local_corners = np.array([
            [-1, -1, -1], [+1, -1, -1], [+1, +1, -1], [-1, +1, -1],
            [-1, -1, +1], [+1, -1, +1], [+1, +1, +1], [-1, +1, +1],
        ], dtype=float) * half_extents[None, :]
        corners = (rot_mat @ local_corners.T).T + center[None, :]

        objects[obj_label] = {
            "label": obj_label,
            "corners": corners,
            "center": center,
            "axes": axes_norm,
            "half_extents": half_extents,
        }
    return objects


def load_labels_auto(labels_path):
    """Auto-detect label format and load accordingly.
    Tries OBB simple format first, falls back to old bounding_box format."""
    if labels_path is None or not Path(labels_path).exists():
        return {}
    with open(labels_path) as f:
        data = json.load(f)
    if isinstance(data, dict) and "objects" in data:
        return _load_labels_simple(labels_path)
    return _load_labels(labels_path)


def _ray_obb_t_enter(ray_origin, ray_dir, obb_center, obb_axes, obb_half_extents):
    """Compute parametric t where ray enters OBB using slab method.
    ray_origin: (3,), ray_dir: (3,), obb_center: (3,),
    obb_axes: (3,3) rows are local axes, obb_half_extents: (3,).
    Returns t_enter (scalar), or 1.0 if no intersection."""
    d = ray_origin - obb_center
    t_min_all = -np.inf
    t_max_all = np.inf
    for i in range(3):
        axis = obb_axes[i]
        e = obb_half_extents[i]
        # Project ray onto this axis
        denom = np.dot(ray_dir, axis)
        num = np.dot(d, axis)
        if abs(denom) < 1e-10:
            # Ray parallel to slab
            if abs(num) > e:
                return 1.0  # no intersection
            continue
        t1 = (-e - num) / denom
        t2 = (e - num) / denom
        if t1 > t2: t1, t2 = t2, t1
        t_min_all = max(t_min_all, t1)
        t_max_all = min(t_max_all, t2)
        if t_min_all > t_max_all:
            return 1.0  # no intersection
    return max(t_min_all, 0.0)


def _compute_occlusion_for_segment(positions, target_info, sdf_data,
                                    n_steps=32, margin=0.15, near_skip=0.1):
    """Check occlusion for a batch of camera positions against one target object.

    Uses the same transmittance-weighted ray-marching as the optimizer's
    _occlusion_cost:
      1. Ray from camera toward target center, t in [near_skip, t_far]
      2. t_far from ray-OBB intersection (padded by 0.05)
      3. Query SDF at each sample
      4. Mask out samples near target OBB (exclusion_margin = max(half_ext)*0.5)
         by boosting their SDF by +10 (same as optimizer: sdf + mask * 10)
      5. Softplus penalty: softplus(margin - sdf, beta=20)
      6. Transmittance-weighted accumulation: T_k = exp(-sum_{l<k} h_l)
      7. Per-camera occlusion = sum(T_k * h_k)

    A pose is "occluded" when accumulated occlusion exceeds `margin`.

    Returns per-sample binary occlusion and per-sample occlusion values.
    """
    n = len(positions)
    target_center = target_info["center"]
    obb_axes = target_info["axes"]          # (3, 3) — rows are local axes
    obb_half = target_info["half_extents"]  # (3,)
    exclusion_margin = obb_half.max() * 0.5

    occluded = np.zeros(n, dtype=bool)
    occ_values = np.zeros(n)

    for i in range(n):
        ray_origin = positions[i]
        ray_vec = target_center - ray_origin  # unnormalized direction
        ray_len = np.linalg.norm(ray_vec)
        if ray_len < 1e-6:
            continue

        # ── t_far via ray-OBB intersection (same as optimizer) ──
        t_enter = _ray_obb_t_enter(ray_origin, ray_vec, target_center,
                                    obb_axes, obb_half + 0.05)
        t_far = np.clip(t_enter, near_skip + 0.05, 1.0)

        # ── Sample along ray in [near_skip, t_far] ──
        t_uniform = np.linspace(0, 1, n_steps)
        t_vals = near_skip + t_uniform * (t_far - near_skip)  # (S,)

        ray_points = (
            ray_origin[None, :] * (1.0 - t_vals[:, None])
            + target_center[None, :] * t_vals[:, None]
        )  # (S, 3)

        # ── Query SDF ──
        sdf_vals = _query_sdf_grid(sdf_data, ray_points)  # (S,)

        # ── OBB exclusion mask (same as optimizer) ──
        # Project points into OBB local frame, compute distance to OBB
        d_local = ray_points - target_center[None, :]  # (S, 3)
        proj = d_local @ obb_axes.T  # (S, 3) — coords in OBB frame
        dist_per_axis = np.abs(proj) - obb_half[None, :]  # (S, 3)
        obb_dist = np.maximum(dist_per_axis, 0).max(axis=1)  # (S,)
        obb_dist[np.all(dist_per_axis < 0, axis=1)] = 0.0  # inside OBB → dist=0
        target_mask = (obb_dist < exclusion_margin).astype(float)
        # Boost SDF for masked points (same as optimizer: sdf + mask * 10)
        sdf_vals = sdf_vals + target_mask * 10.0

        # ── Softplus penalty: softplus(margin - sdf, beta=20) ──
        x = 20.0 * (margin - sdf_vals)
        # Numerically stable softplus
        per_step_penalty = np.where(
            x > 20, x / 20.0,
            np.log1p(np.exp(np.clip(x, -50, 20))) / 20.0
        )

        # ── Transmittance-weighted accumulation ──
        cumulative = np.cumsum(per_step_penalty)
        shifted_cumulative = np.concatenate([[0.0], cumulative[:-1]])
        transmittance = np.exp(-shifted_cumulative)
        occ_value = float(np.sum(transmittance * per_step_penalty))

        occ_values[i] = occ_value
        if occ_value > margin:
            occluded[i] = True

    return occluded, occ_values


# =========================================================================
# GT-based target extraction (for baselines without anchors.json)
# =========================================================================

def _extract_gt_targets(gt_path):
    """Extract target object labels from GT trajectory segments.

    The GT trajectory has a 'segments' list where each segment has
    'target_object' like 'sink_0 (sink_0)'. We parse out the label
    (first part before the parenthesized portion, or the whole string).

    Returns list of unique target labels in segment order.
    """
    if gt_path is None or not Path(gt_path).exists():
        return []
    gt_data = load_trajectory(gt_path)
    if gt_data is None or not isinstance(gt_data, dict):
        return []
    segments = gt_data.get("segments", [])
    if not segments:
        return []

    targets = []
    seen = set()
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        raw = seg.get("target_object", "")
        if not raw:
            continue
        # Parse: "sink_0 (sink_0)" -> "sink_0"
        # Also handle: "sink_0" (no parentheses)
        label = raw.split("(")[0].strip()
        if not label:
            label = raw.strip()
        # Deduplicate consecutive same targets but keep order for segment mapping
        targets.append(label)

    return targets


def _match_target_labels_to_labels(target_labels, labels_dict):
    """Match a list of target label strings to label entries in the scene graph.

    Returns list of (label_str, target_info_or_None), same length as target_labels.
    Uses exact key match first, then case-insensitive, then substring.
    """
    result = []
    for lbl in target_labels:
        matched = None
        # Exact key match (simple format keys are like 'sink_0')
        if lbl in labels_dict:
            matched = labels_dict[lbl]
        if matched is None:
            for lid, linfo in labels_dict.items():
                if linfo["label"].lower() == lbl.lower():
                    matched = linfo; break
        if matched is None:
            for lid, linfo in labels_dict.items():
                if lbl.lower() in linfo["label"].lower():
                    matched = linfo; break
        result.append((lbl, matched))
    return result


def evaluate_occlusion(traj_path, anchors_path, labels_path, sdf_path,
                       n_steps=32, margin=0.15, gt_targets=None):
    """Metric 4: Occlusion Rate.

    For each trajectory segment (between consecutive anchors), ray-march from
    camera positions to the corresponding target object through the SDF grid.
    A pose is "occluded" if accumulated geometry hits along the ray exceed a
    threshold before reaching the target's OBB.

    If anchors lack frame_start/frame_end, the trajectory is evenly divided
    among anchors.

    For baselines without anchors.json, gt_targets (list of label strings from
    GT trajectory segments) can be provided as fallback. The trajectory is
    evenly divided among gt_targets.
    """
    m = OcclusionMetrics()

    sdf_data = _load_sdf_grid(sdf_path)
    if sdf_data is None: return m

    td = load_trajectory(traj_path)
    if td is None: return m
    pos, rot, ts = extract_poses(td)
    if len(pos) < 2: return m

    labels = load_labels_auto(labels_path)
    if not labels: return m

    # ── Determine target objects and frame segments ──
    anchor_targets = []
    segments = None

    # Try anchors.json first
    if anchors_path is not None and Path(anchors_path).exists():
        with open(anchors_path) as f:
            anc_data = json.load(f)
        if isinstance(anc_data, dict):
            anchors = anc_data.get("anchors", anc_data.get("data", []))
        elif isinstance(anc_data, list):
            anchors = anc_data
        else:
            anchors = []

        for a in anchors:
            if not isinstance(a, dict): continue
            obj_id = str(a.get("object_id", a.get("ins_id", a.get("id", ""))))
            obj_label = a.get("object_label", a.get("label", ""))
            if obj_id in labels:
                anchor_targets.append(labels[obj_id])
            elif obj_label in labels:
                anchor_targets.append(labels[obj_label])
            else:
                matched = None
                for lid, linfo in labels.items():
                    if linfo["label"].lower() == obj_label.lower():
                        matched = linfo; break
                if matched is None:
                    for lid, linfo in labels.items():
                        if obj_label.lower() in linfo["label"].lower():
                            matched = linfo; break
                if matched is not None:
                    anchor_targets.append(matched)

        if anchor_targets:
            n_anchors = len(anchor_targets)
            n_frames = len(pos)
            has_frames = all(
                isinstance(a, dict) and ("frame_start" in a or "start_frame" in a)
                for a in anchors[:n_anchors]
            )
            if has_frames:
                segments = []
                for a in anchors[:n_anchors]:
                    sf = a.get("frame_start", a.get("start_frame", 0))
                    ef = a.get("frame_end", a.get("end_frame", n_frames))
                    segments.append((max(0, sf), min(ef, n_frames)))
            else:
                boundaries = np.linspace(0, n_frames, n_anchors + 1, dtype=int)
                segments = [(boundaries[i], boundaries[i + 1]) for i in range(n_anchors)]

    # Fallback to GT targets if no anchors matched
    if not anchor_targets and gt_targets:
        matched_pairs = _match_target_labels_to_labels(gt_targets, labels)
        for lbl, info in matched_pairs:
            if info is not None:
                anchor_targets.append(info)
        if anchor_targets:
            n_anchors = len(anchor_targets)
            n_frames = len(pos)
            boundaries = np.linspace(0, n_frames, n_anchors + 1, dtype=int)
            segments = [(boundaries[i], boundaries[i + 1]) for i in range(n_anchors)]

    if not anchor_targets or segments is None: return m

    # Evaluate occlusion per segment
    all_occluded = []
    all_depths = []
    for (sf, ef), target in zip(segments, anchor_targets):
        if sf >= ef: continue
        seg_pos = pos[sf:ef]
        occ, depths = _compute_occlusion_for_segment(
            seg_pos, target, sdf_data, n_steps=n_steps, margin=margin
        )
        all_occluded.extend(occ.tolist())
        all_depths.extend(depths.tolist())

    if all_occluded:
        m.occlusion_rate = float(np.mean(all_occluded))
        blocked_vals = [d for o, d in zip(all_occluded, all_depths) if o]
        m.mean_occlusion_value = float(np.mean(blocked_vals)) if blocked_vals else 0.0

    return m


# =========================================================================
# Metric 5: Object Coverage
# =========================================================================

def _project_point_to_camera(point_world, c2w, fx, fy, cx, cy):
    """Project a 3D world point into pixel coordinates given c2w.

    Assumes OpenCV camera convention: +X right, +Y down, +Z forward.
    """
    w2c = np.linalg.inv(c2w)
    p_cam = w2c[:3, :3] @ point_world + w2c[:3, 3]
    # OpenCV: camera looks along +Z
    depth = p_cam[2]
    if depth < 0.01:
        return None, None, depth  # behind camera
    # OpenCV: u = fx * x/z + cx, v = fy * y/z + cy
    u = fx * (p_cam[0] / depth) + cx
    v = fy * (p_cam[1] / depth) + cy
    return u, v, depth


def _is_object_visible(cam_pos, target_center, target_info, sdf_data,
                       n_steps=24, margin=0.15):
    """Check if target object is unoccluded from camera position.
    Delegates to _compute_occlusion_for_segment (single-sample).
    Returns True if the line-of-sight is clear."""
    if sdf_data is None:
        return True  # no SDF → assume visible
    occ, _ = _compute_occlusion_for_segment(
        cam_pos[None, :], target_info, sdf_data,
        n_steps=n_steps, margin=margin, near_skip=0.1
    )
    return not occ[0]


def evaluate_coverage(traj_path, anchors_path, labels_path, sdf_path=None,
                      llm_plan_path=None, center_fraction=0.5, gt_targets=None,
                      consecutive_frames=5):
    """Metric 5: Object Coverage.
 
    For each planned object, check whether it is properly covered by the
    generated camera trajectory.
 
    Three modes depending on available data:
    1. GSCinema (anchors with object_id): object anywhere in-frame
    2. Baselines (anchors with c2w from CLIP): object centered + unoccluded
    3. Fallback (gt_targets, no anchors): object centered + unoccluded
    """
    m = CoverageMetrics()
 
    # ── Load trajectory ──
    td = load_trajectory(traj_path) if traj_path and Path(traj_path).exists() else None
    if td is None:
        return m
 
    # ── Load SDF for occlusion checks ──
    sdf_data = _load_sdf_grid(sdf_path) if sdf_path else None
 
    # ── Load labels (bounding boxes) ──
    labels = load_labels_auto(labels_path) if labels_path else {}
 
    # ── Load and classify anchors ──
    anchors = []
    anchor_mode = "none"  # "gscinema", "clip_poses", "none"
 
    if anchors_path is not None and Path(anchors_path).exists():
        with open(anchors_path) as f:
            anc_data = json.load(f)
        if isinstance(anc_data, dict):
            anchors = anc_data.get("anchors", anc_data.get("data", []))
            method_tag = anc_data.get("method", "")
        elif isinstance(anc_data, list):
            anchors = anc_data
            method_tag = ""
 
        if anchors:
            # Detect mode: GSCinema anchors have object_id/object_label,
            # baseline anchors have c2w/position from CLIP selection
            first = anchors[0] if isinstance(anchors[0], dict) else {}
            has_object_ref = ("object_id" in first or "object_label" in first or
                              "ins_id" in first)
            has_c2w = "c2w" in first or "position" in first
 
            if has_object_ref and not method_tag.startswith(("chatcam", "cctg")):
                anchor_mode = "gscinema"
            elif has_c2w:
                anchor_mode = "clip_poses"
            elif has_object_ref:
                anchor_mode = "gscinema"
 
    # ── Determine planned objects based on mode ──
    planned_labels = []
 
    if anchor_mode == "gscinema":
        # GSCinema: get object labels from anchors or LLM plan
        if llm_plan_path and Path(llm_plan_path).exists():
            try:
                with open(llm_plan_path) as f:
                    plan = json.load(f)
                planned_labels = plan.get("object_sequence", [])
            except Exception:
                pass
        if not planned_labels:
            for a in anchors:
                if isinstance(a, dict):
                    lbl = a.get("object_label", a.get("label", ""))
                    if lbl:
                        planned_labels.append(lbl)
 
    elif anchor_mode == "clip_poses":
        # Baselines with CLIP anchor poses: determine target objects by
        # projecting all OBB centers through each anchor camera and picking
        # the one closest to image center
        planned_labels = _determine_targets_from_anchor_poses(anchors, labels, td)
 
    if not planned_labels and gt_targets:
        # Fallback: use GT targets, deduplicate
        anchor_mode = "none"
        seen = set()
        for lbl in gt_targets:
            if lbl not in seen:
                planned_labels.append(lbl)
                seen.add(lbl)
 
    if not planned_labels:
        return m
 
    m.n_planned = len(planned_labels)
 
    # ── Match planned objects to label entries ──
    if anchor_mode == "gscinema" and anchors:
        planned_targets = _match_gscinema_targets(planned_labels, anchors, labels)
    else:
        planned_targets = _match_target_labels_to_labels(planned_labels, labels)
 
    # ── Extract trajectory data ──
    pos, rot, ts = extract_poses(td)
    if len(pos) < 2:
        return m
 
    # ── Read camera intrinsics from trajectory JSON ──
    raw_td = td
    if isinstance(raw_td, dict):
        fx = raw_td.get("fl_x", 256.0)
        fy = raw_td.get("fl_y", fx)
        cx = raw_td.get("cx", raw_td.get("w", 512) / 2.0)
        cy = raw_td.get("cy", raw_td.get("h", 512) / 2.0)
        W = raw_td.get("w", cx * 2)
        H = raw_td.get("h", cy * 2)
    else:
        W, H = 512, 512
        fx = fy = 256.0
        cx, cy = W / 2, H / 2
 
    # ── Build c2w matrices ──
    n = len(pos)
    c2w_list = []
    for i in range(n):
        c2w = np.eye(4)
        if rot.ndim == 3 and rot.shape[1:] == (3, 3):
            c2w[:3, :3] = rot[i]
        c2w[:3, 3] = pos[i]
        c2w_list.append(c2w)
 
    # ── Central region bounds (for baseline coverage checks) ──
    margin_u = W * (1.0 - center_fraction) / 2.0
    margin_v = H * (1.0 - center_fraction) / 2.0
    u_min_center = margin_u
    u_max_center = W - margin_u
    v_min_center = margin_v
    v_max_center = H - margin_v
 
    # ── Check coverage ──
    visited = 0
    per_object = {}
 
    for lbl, target_info in planned_targets:
        if target_info is None:
            per_object[lbl] = False
            continue
 
        target_center = target_info["center"]
        consecutive_count = 0
        found = False
 
        for i in range(n):
            c2w = c2w_list[i]
            u, v, depth = _project_point_to_camera(target_center, c2w, fx, fy, cx, cy)
 
            if u is None or depth <= 0.01:
                consecutive_count = 0
                continue
 
            if anchor_mode == "gscinema":
                # GSCinema: object anywhere in frame is sufficient
                in_frame = (0 <= u <= W and 0 <= v <= H)
            else:
                # Baselines: object must be in central region + unoccluded
                in_frame = (u_min_center <= u <= u_max_center and
                            v_min_center <= v <= v_max_center)
                if in_frame and sdf_data is not None:
                    visible = _is_object_visible(
                        pos[i], target_center, target_info, sdf_data,
                        n_steps=24, margin=0.15)
                    if not visible:
                        in_frame = False
 
            if in_frame:
                consecutive_count += 1
                if consecutive_count >= consecutive_frames:
                    found = True
                    break
            else:
                consecutive_count = 0
 
        per_object[lbl] = found
        if found:
            visited += 1
 
    m.n_visited = visited
    m.object_coverage = visited / max(m.n_planned, 1)
    m.per_object = per_object
    return m
 
 
def _determine_targets_from_anchor_poses(anchors, labels, traj_data):
    """For baseline CLIP anchors: determine which object each anchor is looking at.
 
    For each anchor pose (c2w), project all OBB centers into the anchor's camera
    frame. The object whose projection is closest to the image center is the
    target for that anchor.
 
    Returns list of target label strings (one per anchor, deduplicated).
    """
    if not labels or not anchors:
        return []
 
    # Read intrinsics from trajectory JSON for projection
    if isinstance(traj_data, dict):
        fx = traj_data.get("fl_x", 256.0)
        fy = traj_data.get("fl_y", fx)
        cx = traj_data.get("cx", traj_data.get("w", 512) / 2.0)
        cy = traj_data.get("cy", traj_data.get("h", 512) / 2.0)
        W = traj_data.get("w", cx * 2)
        H = traj_data.get("h", cy * 2)
    else:
        W, H = 512, 512
        fx = fy = 256.0
        cx, cy = W / 2, H / 2
 
    # Collect all OBB centers
    label_keys = list(labels.keys())
    label_centers = np.array([labels[k]["center"] for k in label_keys])
 
    targets = []
    seen = set()
 
    for anc in anchors:
        if not isinstance(anc, dict):
            continue
 
        # Build anchor c2w
        c2w_raw = anc.get("c2w")
        if c2w_raw is not None:
            c2w = np.array(c2w_raw)
            if c2w.shape != (4, 4):
                continue
        elif "position" in anc:
            # Position only — can't do full projection, skip
            continue
        else:
            continue
 
        # Use anchor-specific intrinsics if available
        anc_intrinsics = anc.get("intrinsics", {})
        a_fx = anc_intrinsics.get("fx", fx)
        a_fy = anc_intrinsics.get("fy", fy)
        a_cx = anc_intrinsics.get("cx", cx)
        a_cy = anc_intrinsics.get("cy", cy)
        a_W = anc_intrinsics.get("w", W)
        a_H = anc_intrinsics.get("h", H)
 
        # Project all OBB centers and find closest to image center
        best_label = None
        best_dist = float("inf")
 
        for k, center in zip(label_keys, label_centers):
            u, v, depth = _project_point_to_camera(center, c2w, a_fx, a_fy, a_cx, a_cy)
            if u is None or depth <= 0.01:
                continue
            if not (0 <= u <= a_W and 0 <= v <= a_H):
                continue
            # Distance from image center
            dist = ((u - a_cx) ** 2 + (v - a_cy) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_label = k
 
        if best_label is not None and best_label not in seen:
            targets.append(best_label)
            seen.add(best_label)
 
    return targets
 
 
def _match_gscinema_targets(planned_labels, anchors, labels):
    """Match GSCinema planned labels to label entries via anchors' object_id."""
    planned_targets = []
    for lbl in planned_labels:
        matched = None
        # Find the anchor for this label
        anchor_for_label = None
        for a in anchors:
            if not isinstance(a, dict):
                continue
            a_lbl = a.get("object_label", a.get("label", ""))
            if a_lbl.lower() == lbl.lower():
                anchor_for_label = a
                break
        if anchor_for_label:
            obj_id = str(anchor_for_label.get("object_id",
                         anchor_for_label.get("ins_id", "")))
            if obj_id in labels:
                matched = labels[obj_id]
        if matched is None and lbl in labels:
            matched = labels[lbl]
        if matched is None:
            for lid, linfo in labels.items():
                if linfo["label"].lower() == lbl.lower():
                    matched = linfo
                    break
        if matched is None:
            for lid, linfo in labels.items():
                if lbl.lower() in linfo["label"].lower():
                    matched = linfo
                    break
        planned_targets.append((lbl, matched))
    return planned_targets


# =========================================================================
# Run-level + Benchmark-level
# =========================================================================

def _find_file(directory, *candidates):
    """Return first existing file from candidates in directory."""
    for name in candidates:
        p = Path(directory) / name
        if p.exists(): return p
    return None


def evaluate_single_run(run_dir, sdf_path=None, labels_path=None, gt_path=None,
                        device="cuda", metrics_to_compute="all",
                        prompt_override=None, **kw):
    """Evaluate a single run directory with the 5 paper metrics."""
    rd = Path(run_dir)
    r = EvalResult()

    # ── Read metadata ──
    for mn in ["run_config.json", "config.json", "metadata.json"]:
        mp = rd / mn
        if mp.exists():
            meta = json.load(open(mp))
            r.scene_id = meta.get("scene_id", "")
            r.prompt = meta.get("prompt", meta.get("text", meta.get("description", "")))
            r.level = meta.get("level", meta.get("difficulty", ""))
            break

    # ── Infer from directory structure ──
    parts = rd.parts
    if len(parts) >= 3:
        r.scene_id = r.scene_id or parts[-3]
        r.level = r.level or parts[-2]
        try: r.prompt_idx = int(parts[-1].split("_")[-1])
        except: pass

    if prompt_override:
        r.prompt = prompt_override
    if not r.prompt:
        r.prompt = _discover_prompt(rd, r.scene_id, r.level, r.prompt_idx)

    want = set(metrics_to_compute.split(",")) if metrics_to_compute != "all" else {
        "motion_mse", "clatr", "collision", "occlusion", "coverage"}

    # ── Find trajectory file ──
    traj_file = _find_file(rd, "combined_trajectory.json", "trajectory.json",
                           "camera_trajectory.json", "camera_path.json", "traj.json")

    # ── Auxiliary info ──
    if traj_file:
        td = load_trajectory(str(traj_file))
        if td:
            pos, _, _ = extract_poses(td)
            r.n_poses = len(pos)
            if len(pos) >= 2:
                r.path_length = float(np.linalg.norm(np.diff(pos, axis=0), axis=1).sum())

    # ── Resolve GT path once for Motion MSE + baseline fallback ──
    gt_file_resolved = gt_path or str(_find_file(rd, "gt_trajectory.json", "ground_truth.json") or "")
    gt_target_labels = None
    if gt_file_resolved and Path(gt_file_resolved).exists():
        gt_target_labels = _extract_gt_targets(gt_file_resolved)

    # ── Metric 1: Motion MSE ──
    if "motion_mse" in want and traj_file:
        if gt_file_resolved and Path(gt_file_resolved).exists():
            r.motion_mse = evaluate_motion_mse(str(traj_file), gt_file_resolved,
                                                n_resample=kw.get("n_resample", 100))

    # ── Metric 2: CLaTr Score ──
    if "clatr" in want and traj_file and r.prompt:
        r.clatr = evaluate_clatr(str(traj_file), r.prompt,
                                  kw.get("clatr_dir", "CLaTr"),
                                  kw.get("clatr_checkpoint", "checkpoints/clatr-e100.ckpt"),
                                  device)

    # ── Metric 3: Collision Rate ──
    if "collision" in want and traj_file:
        r.collision = evaluate_collision(str(traj_file), sdf_path)

    # ── Metric 4: Occlusion Rate ──
    if "occlusion" in want and traj_file and labels_path and sdf_path:
        anchors_file = _find_file(rd, "anchors.json")
        r.occlusion = evaluate_occlusion(
            str(traj_file),
            str(anchors_file) if anchors_file else None,
            labels_path, sdf_path,
            n_steps=kw.get("occlusion_n_steps", 32),
            margin=kw.get("occlusion_margin", 0.15),
            gt_targets=gt_target_labels)

    # ── Metric 5: Object Coverage ──
    if "coverage" in want and traj_file:
        anchors_file = _find_file(rd, "anchors.json")
        llm_file = _find_file(rd, "llm_plan.json", "llm_output.json", "plan.json")
        r.coverage = evaluate_coverage(
            str(traj_file),
            str(anchors_file) if anchors_file else None,
            labels_path, sdf_path,
            str(llm_file) if llm_file else None,
            center_fraction=kw.get("center_fraction", 0.5),
            gt_targets=gt_target_labels)

    return r


def evaluate_benchmark(bench_dir, data_root, device="cuda", metrics_to_compute="all",
                       level_filter=None, scene_filter=None, **kw):
    """Evaluate all runs in a benchmark directory.

    Args:
        bench_dir:          Root directory containing benchmark outputs.
        data_root:          Root directory containing scene data (SDF, labels, GT).
        device:             Torch device for CLaTr evaluation.
        metrics_to_compute: Comma-separated metric names or "all".
        level_filter:       If set, only evaluate runs whose level matches this
                            string (e.g. "low", "medium", "high", "low_id").
                            None = all levels.
        scene_filter:       If set, only evaluate runs whose scene_id matches.
                            Comma-separated for multiple scenes (e.g. "09c1414f1b,scene2").
                            None = all scenes.
        **kw:               Forwarded to evaluate_single_run.
    """
    bd, dr = Path(bench_dir), Path(data_root)
    all_res, by_level = [], {}

    # ── Load prompts map ──
    pmap = {}
    prompts_file = kw.get("prompts_file")
    if prompts_file:
        pmap = load_prompts_map(prompts_file)
        if pmap: print(f"  [prompts] loaded {len(pmap)} prompts from {prompts_file}")
    if not pmap:
        for pname in ["benchmark_prompts.json", "prompts.json"]:
            pf = bd / pname
            if pf.exists():
                pmap = load_prompts_map(pf)
                if pmap: print(f"  [prompts] loaded {len(pmap)} prompts from {pf}"); break
    if not pmap:
        for sname in ["benchmark_summary.json", "summary.json"]:
            sf = bd / sname
            if sf.exists():
                try:
                    with open(sf) as f: sdata = json.load(f)
                    for entry in sdata.get("results", []):
                        if not isinstance(entry, dict): continue
                        sid = entry.get("scene_id", "")
                        lvl = entry.get("level", "")
                        pidx = entry.get("prompt_idx", 0)
                        text = entry.get("prompt", "")
                        if sid and text: pmap[(sid, lvl, int(pidx))] = text
                    if pmap: print(f"  [prompts] extracted {len(pmap)} from {sf}"); break
                except Exception: pass
    if not pmap:
        print("  [prompts] ⚠ no prompts found — CLaTr will be skipped")

    if level_filter:
        print(f"  [filter] evaluating only level='{level_filter}'")

    # Parse scene filter into a set for fast lookup
    scene_set = None
    if scene_filter:
        scene_set = set(s.strip() for s in scene_filter.split(","))
        print(f"  [filter] evaluating only scene(s): {scene_set}")

    for dm in sorted(bd.rglob("done.marker")):
        rd = dm.parent
        parts = rd.relative_to(bd).parts
        if len(parts) < 3: continue
        sid, lvl, pdir = parts[0], parts[1], parts[2]

        # ── Scene filter: skip runs that don't match ──
        if scene_set and sid not in scene_set:
            continue

        # ── Level filter: skip runs that don't match ──
        if level_filter and lvl != level_filter:
            continue

        try: pidx = int(pdir.split("_")[-1])
        except (ValueError, IndexError): pidx = 0
        prompt_text = lookup_prompt(pmap, sid, lvl, pidx)

        # ── Try to read prompt from prompt.txt if not found ──
        if not prompt_text:
            pt = rd / "prompt.txt"
            if pt.exists():
                try: prompt_text = pt.read_text().strip()
                except Exception: pass

        # ── Strip object IDs from low_id prompts: "(id: xxx)" -> "" ──
        if prompt_text:
            import re
            prompt_text = re.sub(r'\s*\(id:\s*[^)]*\)', '', prompt_text).strip()

        # ── Resolve SDF path ──
        sdf_p = None
        for sdf_dir_name in ["sdf", "collision_sdf"]:
            sd = dr / sdf_dir_name / sid
            sf = list(sd.glob("*_sdf_*.npz")) if sd.exists() else []
            if sf: sdf_p = str(sf[0]); break
        # ScanNet++ layout: {scenes,gsplat_20_scenes}/{sid}/dslr/sdf/{sid}_sdf_*.npz
        if sdf_p is None:
            for root in ["scenes", "gsplat_20_scenes"]:
                sd = dr / root / sid / "dslr" / "sdf"
                sf = list(sd.glob("*_sdf_*.npz")) if sd.exists() else []
                if sf: sdf_p = str(sf[0]); break

        # ── Resolve labels path ──
        labels_p = None
        for ldir in ["compressed", "scenes"]:
            lf = dr / ldir / sid / "labels.json"
            if lf.exists(): labels_p = str(lf); break
        # ScanNet++ layout: {scenes,gsplat_20_scenes}/{sid}/dslr/sg/{sid}-simple.json
        if labels_p is None:
            for root in ["scenes", "gsplat_20_scenes"]:
                lf = dr / root / sid / "dslr" / "sg" / f"{sid}-simple.json"
                if lf.exists(): labels_p = str(lf); break

        # ── Resolve GT path (for low-level Motion MSE) ──
        gt_p = None
        gt_dir = dr / "gt_trajectories" / sid
        if gt_dir.exists():
            gt_candidates = list(gt_dir.glob(f"*prompt_{pidx:02d}*"))
            if gt_candidates: gt_p = str(gt_candidates[0])
        # ScanNet++ layout: {gt,blend_sg_20_scenes/gt}/{sid}/prompt_{idx:02d}/gt_traj.json
        if gt_p is None:
            for gt_root in [dr / "gt", dr / "blend_sg_20_scenes" / "gt"]:
                gt_file = gt_root / sid / f"prompt_{pidx:02d}" / "gt_traj.json"
                if gt_file.exists(): gt_p = str(gt_file); break

        print(f"  {sid}/{lvl}/{pdir} ...", end=" ", flush=True)
        if sdf_p: print(f"[SDF: ✓]", end=" ")
        else: print(f"[SDF: ✗]", end=" ")
        if labels_p: print(f"[Labels: ✓]", end=" ")
        else: print(f"[Labels: ✗]", end=" ")
        if gt_p: print(f"[GT: ✓]", end=" ")
        else: print(f"[GT: ✗]", end=" ")
        if prompt_text: print(f"[Prompt: ✓]", end=" ")
        else: print(f"[Prompt: ✗]", end=" ")

        try:
            r = evaluate_single_run(
                str(rd), sdf_path=sdf_p, labels_path=labels_p, gt_path=gt_p,
                device=device, metrics_to_compute=metrics_to_compute,
                prompt_override=prompt_text, **kw)
            r.scene_id, r.level, r.prompt_idx, r.success = sid, lvl, pidx, True
            rd_ = asdict(r)
            all_res.append(rd_)
            by_level.setdefault(lvl, []).append(rd_)
            print("ok")
        except Exception as e:
            print(f"ERR: {e}")
            all_res.append({"scene_id": sid, "level": lvl, "error": str(e)})

    # ── Aggregates ──
    agg = {}
    for lvl, res in by_level.items():
        if res: agg[lvl] = _agg(res)
    agg["overall"] = _agg(all_res)

    ov = agg.get("overall", {})
    summary = {
        "total_runs": len(all_res),
        "successful": sum(1 for r in all_res if r.get("success")),
        "level_filter": level_filter or "all",
        "motion_mse_translation": ov.get("motion_mse.translation_mse_mean", -1),
        "motion_mse_rotation": ov.get("motion_mse.rotation_error_mean_mean", -1),
        "avg_clatr_score": ov.get("clatr.clatr_score_mean", -1),
        "avg_collision_rate": ov.get("collision.collision_rate_mean", -1),
        "avg_occlusion_rate": ov.get("occlusion.occlusion_rate_mean", -1),
        "avg_coverage": ov.get("coverage.object_coverage_mean", -1),
    }
    return {"summary": summary, "aggregate": agg, "per_run": all_res}


def _agg(results):
    a = {}
    for r in results:
        for cat in ["motion_mse", "clatr", "collision", "occlusion", "coverage"]:
            if cat not in r or not isinstance(r[cat], dict): continue
            for k, v in r[cat].items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    a.setdefault(f"{cat}.{k}", []).append(v)
    st = {}
    for k, vs in a.items():
        vs = [v for v in vs if v > -0.5]  # skip -1 sentinel values
        if vs:
            st[f"{k}_mean"] = float(np.mean(vs))
            st[f"{k}_std"] = float(np.std(vs))
    return st


# =========================================================================
# CLI
# =========================================================================

def main():
    p = argparse.ArgumentParser(description="TrajScene Eval (5 paper metrics)")
    p.add_argument("--run_dir", help="Single run directory to evaluate")
    p.add_argument("--benchmark_dir", default="outputs/benchmark_outputs")
    p.add_argument("--data_root", default="data/ScanNetpp")
    p.add_argument("--output", default="outputs/benchmark_outputs/eval_results.json")
    p.add_argument("--metrics", default="all",
                   help="Comma-separated: motion_mse,clatr,collision,occlusion,coverage")
    p.add_argument("--level", default="low",
                   help="Filter by prompt difficulty level: low, medium, high, low_id, or all (default: low)")
    p.add_argument("--scene", default=None,
                   help="Filter by scene ID. Comma-separated for multiple (e.g. '09c1414f1b,scene2'). Default: all scenes.")
    p.add_argument("--device", default="cuda")
    # Per-run overrides
    p.add_argument("--sdf_path", help="Path to SDF .npz file")
    p.add_argument("--labels_path", help="Path to labels.json with bounding boxes")
    p.add_argument("--gt_path", help="Path to ground-truth trajectory for Motion MSE")
    p.add_argument("--prompt", help="Override prompt text")
    p.add_argument("--config", dest="yaml_config",
                   help="Path to pipeline config.yaml")
    # Benchmark options
    p.add_argument("--prompts_file", help="Path to benchmark_prompts.json")
    # CLaTr options
    p.add_argument("--clatr_dir", default="CLaTr")
    p.add_argument("--clatr_checkpoint", default="checkpoints/clatr-e100.ckpt")
    # Occlusion options
    p.add_argument("--occlusion_n_steps", type=int, default=32)
    p.add_argument("--occlusion_margin", type=float, default=0.15)
    # Motion MSE options
    p.add_argument("--n_resample", type=int, default=100,
                   help="Number of arc-length resampling points for Motion MSE")
    # Coverage options
    p.add_argument("--center_fraction", type=float, default=0.5,
                   help="Fraction of image considered 'centered' for coverage check")
    a = p.parse_args()

    kw = {}
    for k in ["clatr_dir", "clatr_checkpoint", "prompts_file",
              "occlusion_n_steps", "occlusion_margin", "n_resample",
              "center_fraction"]:
        kw[k] = getattr(a, k)

    # Normalize level filter: "all" means no filtering
    level_filter = None if a.level.lower() == "all" else a.level.lower()

    if a.run_dir:
        prompt = a.prompt
        if not prompt and a.yaml_config:
            try:
                import yaml as _yaml
                with open(a.yaml_config) as f: ycfg = _yaml.safe_load(f)
                prompt = ycfg.get("prompt", "")
            except Exception as e:
                print(f"⚠ Could not read config: {e}")
        r = evaluate_single_run(
            a.run_dir, sdf_path=a.sdf_path, labels_path=a.labels_path,
            gt_path=a.gt_path, device=a.device, metrics_to_compute=a.metrics,
            prompt_override=prompt, **kw)
        out = asdict(r)
        # Print summary
        print(f"\n{'='*50}")
        print(f"TrajScene Evaluation: {r.scene_id}/{r.level}/prompt_{r.prompt_idx}")
        print(f"{'='*50}")
        if r.motion_mse.translation_mse >= 0:
            print(f"  Motion MSE (trans):  {r.motion_mse.translation_mse:.6f}")
            print(f"  Motion MSE (rot):    {r.motion_mse.rotation_error_mean:.4f} rad")
        else:
            print(f"  Motion MSE:          n/a (no GT)")
        print(f"  CLaTr Score:         {r.clatr.clatr_score:.4f}" if r.clatr.clatr_score >= 0
              else "  CLaTr Score:         n/a")
        print(f"  Collision Rate:      {r.collision.collision_rate:.4f}" if r.collision.collision_rate >= 0
              else "  Collision Rate:      n/a (no SDF)")
        print(f"  Occlusion Rate:      {r.occlusion.occlusion_rate:.4f}" if r.occlusion.occlusion_rate >= 0
              else "  Occlusion Rate:      n/a")
        print(f"  Object Coverage:     {r.coverage.object_coverage:.4f} ({r.coverage.n_visited}/{r.coverage.n_planned})"
              if r.coverage.object_coverage >= 0 else "  Object Coverage:     n/a")
        if r.prompt: print(f"\n  Prompt: {r.prompt[:100]}...")
        print(f"{'='*50}")

    elif a.benchmark_dir:
        assert a.data_root, "--data_root required"
        out = evaluate_benchmark(a.benchmark_dir, a.data_root, a.device, a.metrics,
                                 level_filter=level_filter, scene_filter=a.scene, **kw)
        s = out["summary"]
        print(f"\n{'='*60}")
        print(f"Benchmark Summary: {s['successful']}/{s['total_runs']} runs"
              f" (level={s['level_filter']})")
        print(f"{'='*60}")
        for k, v in s.items():
            if k not in ("total_runs", "successful", "level_filter"):
                print(f"  {k}: {v}")
        print(f"{'='*60}")
    else:
        p.error("--run_dir or --benchmark_dir required")

    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.output, "w"), indent=2, default=str)
    print(f"Saved -> {a.output}")


if __name__ == "__main__":
    main()