#!/usr/bin/env python3
"""
Visualize GSCinema / ChatCam / CCTG / GT trajectories for one scene,
with the scene mesh as reference geometry.

Outputs an interactive HTML file with 3D trajectory + forward direction arrows
+ scene mesh (subsampled point cloud for performance).

Usage (run from code/gscinema/):
    python visualize_traj_conventions.py --scene 09c1414f1b --prompt 0
    python visualize_traj_conventions.py --scene 09c1414f1b --prompt 0 --mesh path/to/mesh.ply
    python visualize_traj_conventions.py --scene 09c1414f1b --prompt 0 --max_mesh_points 100000
"""

import argparse
import json
import struct
import sys
import numpy as np
from pathlib import Path

# ─── Paths (edit if needed) ─────────────────────────────────────────────
CWD = Path.cwd()
DATA_ROOT = Path("data/ScanNetpp")

TRAJ_SOURCES = {
    "gscinema": {
        "base": CWD / "outputs/benchmark_scannetpp",
        "level": "low_id",
    },
    "chatcam": {
        "base": CWD / "outputs/benchmark_chatcam",
        "level": "low",
    },
    "cctg": {
        "base": CWD / "outputs/benchmark_cctg",
        "level": "low",
    },
}

TRAJ_FILENAMES = [
    "combined_trajectory.json",
    "trajectory.json",
    "camera_trajectory.json",
    "camera_path.json",
    "traj.json",
]

# Candidate mesh file locations (relative to DATA_ROOT)
MESH_CANDIDATES = [
    "gsplat_20_scenes/{scene}/dslr/mesh/{scene}_mesh.ply",
    "gsplat_20_scenes/{scene}/dslr/mesh/mesh.ply",
    "gsplat_20_scenes/{scene}/dslr/mesh/{scene}.ply",
    "gsplat_20_scenes/{scene}/mesh/{scene}_mesh.ply",
    "gsplat_20_scenes/{scene}/mesh/mesh.ply",
    "scenes/{scene}/mesh.ply",
    "scenes/{scene}/{scene}_mesh.ply",
    "compressed/{scene}/mesh.ply",
    "{scene}/mesh.ply",
    "{scene}/dslr/mesh.ply",
]

# ─── PLY Loader ──────────────────────────────────────────────────────────

def load_ply(path, max_points=50000):
    """Load PLY file, return subsampled vertices (N,3) and colors (N,3) in [0,1].
    Supports ASCII and binary_little_endian formats."""
    path = Path(path)
    if not path.exists():
        return None, None

    print(f"    Loading PLY: {path.name} ...", end=" ", flush=True)

    with open(path, "rb") as f:
        # Parse header
        header_lines = []
        while True:
            line = f.readline().decode("ascii", errors="replace").strip()
            header_lines.append(line)
            if line == "end_header":
                break

        fmt = "ascii"
        n_vertices = 0
        props = []
        in_vertex = False

        for line in header_lines:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "format":
                fmt = parts[1]
            elif parts[0] == "element":
                if parts[1] == "vertex":
                    n_vertices = int(parts[2])
                    in_vertex = True
                else:
                    in_vertex = False
            elif parts[0] == "property" and in_vertex:
                if len(parts) >= 3 and parts[1] != "list":
                    ptype = parts[1]
                    pname = parts[2]
                    props.append((pname, ptype))

        prop_names = [p[0] for p in props]
        xi = prop_names.index("x") if "x" in prop_names else None
        yi = prop_names.index("y") if "y" in prop_names else None
        zi = prop_names.index("z") if "z" in prop_names else None

        has_color = False
        ri = gi = bi = None
        if "red" in prop_names:
            ri = prop_names.index("red")
            gi = prop_names.index("green")
            bi = prop_names.index("blue")
            has_color = True

        if xi is None or yi is None or zi is None:
            print("no xyz properties found")
            return None, None

        type_map = {
            "float": ("f", 4), "float32": ("f", 4),
            "double": ("d", 8), "float64": ("d", 8),
            "uchar": ("B", 1), "uint8": ("B", 1),
            "char": ("b", 1), "int8": ("b", 1),
            "short": ("h", 2), "int16": ("h", 2),
            "ushort": ("H", 2), "uint16": ("H", 2),
            "int": ("i", 4), "int32": ("i", 4),
            "uint": ("I", 4), "uint32": ("I", 4),
        }

        step = max(1, n_vertices // max_points)
        keep_mask = set(range(0, n_vertices, step))
        n_keep = len(keep_mask)

        positions = np.zeros((n_keep, 3), dtype=np.float32)
        colors = np.zeros((n_keep, 3), dtype=np.float32) if has_color else None
        out_idx = 0

        if fmt == "ascii":
            for v_idx in range(n_vertices):
                line = f.readline().decode("ascii", errors="replace").strip()
                if v_idx not in keep_mask:
                    continue
                vals = line.split()
                positions[out_idx] = [float(vals[xi]), float(vals[yi]), float(vals[zi])]
                if has_color:
                    r_val = float(vals[ri])
                    g_val = float(vals[gi])
                    b_val = float(vals[bi])
                    if r_val > 1.0 or g_val > 1.0 or b_val > 1.0:
                        colors[out_idx] = [r_val / 255.0, g_val / 255.0, b_val / 255.0]
                    else:
                        colors[out_idx] = [r_val, g_val, b_val]
                out_idx += 1

        elif fmt in ("binary_little_endian", "binary_big_endian"):
            endian = "<" if fmt == "binary_little_endian" else ">"
            fmt_parts = []
            for pname, ptype in props:
                if ptype in type_map:
                    fmt_parts.append(type_map[ptype][0])
                else:
                    print(f"unknown type {ptype}")
                    return None, None
            struct_fmt = endian + "".join(fmt_parts)
            vertex_size = struct.calcsize(struct_fmt)

            for v_idx in range(n_vertices):
                raw = f.read(vertex_size)
                if len(raw) < vertex_size:
                    break
                if v_idx not in keep_mask:
                    continue
                vals = struct.unpack(struct_fmt, raw)
                positions[out_idx] = [vals[xi], vals[yi], vals[zi]]
                if has_color:
                    r_val = float(vals[ri])
                    g_val = float(vals[gi])
                    b_val = float(vals[bi])
                    if r_val > 1.0 or g_val > 1.0 or b_val > 1.0:
                        colors[out_idx] = [r_val / 255.0, g_val / 255.0, b_val / 255.0]
                    else:
                        colors[out_idx] = [r_val, g_val, b_val]
                out_idx += 1
        else:
            print(f"unsupported format: {fmt}")
            return None, None

    positions = positions[:out_idx]
    if colors is not None:
        colors = colors[:out_idx]

    print(f"{n_vertices} vertices -> {out_idx} subsampled")
    return positions, colors


def find_mesh(scene, data_root, mesh_override=None):
    """Find mesh file for a scene."""
    if mesh_override:
        p = Path(mesh_override)
        if p.exists():
            return p
        print(f"  Mesh override not found: {p}")

    dr = Path(data_root)
    for pattern in MESH_CANDIDATES:
        p = dr / pattern.format(scene=scene)
        if p.exists():
            return p

    # Glob fallback
    for base in [dr / "gsplat_20_scenes" / scene, dr / "scenes" / scene, dr / scene]:
        if base.exists():
            plys = list(base.rglob("*.ply"))
            for ply in plys:
                if "mesh" in ply.name.lower():
                    return ply
            if plys:
                return plys[0]
    return None


# ─── Trajectory Extraction ───────────────────────────────────────────────

def _directions_to_rotmats(fwd):
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


def extract_poses(traj_data):
    """Returns (N,3) positions, (N,3,3) rotations."""
    if traj_data is None:
        return np.zeros((0, 3)), np.zeros((0, 3, 3))

    if isinstance(traj_data, dict):
        if "frames" in traj_data and isinstance(traj_data["frames"], list):
            frames = traj_data["frames"]
            tm_key = None
            if frames and isinstance(frames[0], dict):
                if "transform_matrix" in frames[0]: tm_key = "transform_matrix"
                elif "c2w" in frames[0]: tm_key = "c2w"
            if tm_key is not None:
                positions, rotations = [], []
                for fr in frames:
                    T = np.array(fr[tm_key])
                    if T.shape in ((4, 4), (3, 4)):
                        positions.append(T[:3, 3])
                        rotations.append(T[:3, :3])
                if positions:
                    return np.array(positions), np.array(rotations)
                return np.zeros((0, 3)), np.zeros((0, 3, 3))
            else:
                traj_data = frames
        elif "trajectory" in traj_data:
            traj_data = traj_data["trajectory"]
        elif "keyframes" in traj_data:
            traj_data = traj_data["keyframes"]

    # ── Dict-of-arrays format (CCTG: {"positions": [...], "look_ats": [...]}) ──
    if isinstance(traj_data, dict):
        pos_raw = None
        for k in ("positions", "camera_positions", "poses", "translations",
                  "camera_path", "path_positions"):
            if k in traj_data:
                pos_raw = traj_data[k]; break
        if pos_raw is not None:
            pos = np.array(pos_raw)
            if pos.ndim == 2 and pos.shape[0] > 0:
                if pos.shape[1] > 3:
                    pos = pos[:, :3]
                rot_raw = None
                for k in ("rotations", "rotation_matrices", "quaternions", "quats"):
                    if k in traj_data:
                        rot_raw = traj_data[k]; break
                if rot_raw is not None:
                    rot = np.array(rot_raw)
                    if rot.ndim == 2 and rot.shape[1] == 4:
                        from scipy.spatial.transform import Rotation
                        rot = Rotation.from_quat(rot).as_matrix()
                    elif rot.ndim == 2 and rot.shape[1] == 3:
                        rot = _directions_to_rotmats(rot)
                else:
                    la_raw = None
                    for k in ("look_ats", "look_at_points", "targets", "lookats"):
                        if k in traj_data:
                            la_raw = traj_data[k]; break
                    if la_raw is not None:
                        la = np.array(la_raw)
                        fwd = la - pos
                        fwd /= np.maximum(np.linalg.norm(fwd, axis=1, keepdims=True), 1e-8)
                        rot = _directions_to_rotmats(fwd)
                    else:
                        rot = np.tile(np.eye(3), (len(pos), 1, 1))
                return pos, rot

    if isinstance(traj_data, list):
        if not traj_data:
            return np.zeros((0, 3)), np.zeros((0, 3, 3))
        first = traj_data[0] if isinstance(traj_data[0], dict) else {}

        tm_key = None
        if "transform_matrix" in first: tm_key = "transform_matrix"
        elif "c2w" in first: tm_key = "c2w"
        if tm_key is not None:
            positions, rotations = [], []
            for fr in traj_data:
                if not isinstance(fr, dict): continue
                T = np.array(fr[tm_key])
                if T.shape in ((4, 4), (3, 4)):
                    positions.append(T[:3, 3])
                    rotations.append(T[:3, :3])
            if positions:
                return np.array(positions), np.array(rotations)
            return np.zeros((0, 3)), np.zeros((0, 3, 3))

        pos_key = None
        for k in ("position", "camera_position", "pos", "translation", "xyz"):
            if k in first: pos_key = k; break
        if pos_key is None:
            if isinstance(traj_data[0], (list, tuple)) and len(traj_data[0]) >= 3:
                pos = np.array(traj_data)[:, :3]
                return pos, np.tile(np.eye(3), (len(pos), 1, 1))
            return np.zeros((0, 3)), np.zeros((0, 3, 3))

        pos = np.array([p[pos_key] for p in traj_data])

        rot_key = None
        for k in ("rotation", "rotation_matrix", "quaternion", "quat", "rot"):
            if k in first: rot_key = k; break
        if rot_key is not None:
            rot = np.array([p[rot_key] for p in traj_data])
            if rot.ndim == 2 and rot.shape[1] == 4:
                from scipy.spatial.transform import Rotation
                rot = Rotation.from_quat(rot).as_matrix()
            elif rot.ndim == 2 and rot.shape[1] == 3:
                rot = _directions_to_rotmats(rot)
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
        return pos, rot

    return np.zeros((0, 3)), np.zeros((0, 3, 3))


def find_traj_file(directory):
    for name in TRAJ_FILENAMES:
        p = Path(directory) / name
        if p.exists(): return p
    return None


def load_and_extract(method, scene, prompt_idx):
    """Load trajectory for a method+scene+prompt, return dict with positions/forwards."""
    if method == "gt":
        gt_file = DATA_ROOT / "blend_sg_20_scenes" / "gt" / scene / f"prompt_{prompt_idx:02d}" / "gt_traj.json"
        if not gt_file.exists():
            print(f"  GT not found: {gt_file}")
            return None
        with open(gt_file) as f:
            data = json.load(f)
        pos, rot = extract_poses(data)
        if len(pos) == 0:
            if "segments" in data:
                all_pos, all_rot = [], []
                for seg in data["segments"]:
                    sp, sr = extract_poses(seg)
                    if len(sp) > 0:
                        all_pos.append(sp)
                        all_rot.append(sr)
                if all_pos:
                    pos = np.concatenate(all_pos)
                    rot = np.concatenate(all_rot)
    else:
        cfg = TRAJ_SOURCES[method]
        run_dir = cfg["base"] / scene / cfg["level"] / f"prompt_{prompt_idx:02d}"
        if not run_dir.exists():
            print(f"  {method} run_dir not found: {run_dir}")
            return None
        traj_file = find_traj_file(run_dir)
        if traj_file is None:
            print(f"  {method} traj file not found in {run_dir}")
            return None
        with open(traj_file) as f:
            data = json.load(f)
        pos, rot = extract_poses(data)

    if len(pos) < 2:
        print(f"  {method}: only {len(pos)} poses extracted")
        return None

    forward_z = rot[:, :, 2]
    forward_y = rot[:, :, 1]
    forward_x = rot[:, :, 0]

    step = max(1, len(pos) // 50)
    idx = list(range(0, len(pos), step))

    return {
        "n_poses": int(len(pos)),
        "pos_min": pos.min(axis=0).tolist(),
        "pos_max": pos.max(axis=0).tolist(),
        "pos_mean": pos.mean(axis=0).tolist(),
        "path": pos.tolist(),
        "positions": pos[idx].tolist(),
        "forward_z": forward_z[idx].tolist(),
        "forward_neg_z": (-forward_z[idx]).tolist(),
        "forward_y": forward_y[idx].tolist(),
        "forward_x": forward_x[idx].tolist(),
    }


def generate_html(scene, prompt_idx):
    """Generate the HTML template (data injected separately)."""
    return r"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Trajectory Convention Check — """ + scene + " prompt_" + f"{prompt_idx:02d}" + r"""</title>
<style>
body { margin:0; overflow:hidden; font-family: monospace; background: #1a1a2e; color: #eee; }
canvas { display:block; }
#info { position:absolute; top:10px; left:10px; background:rgba(0,0,0,0.8); padding:12px;
        border-radius:8px; font-size:13px; max-width:380px; z-index:10; }
#info h3 { margin:0 0 8px; }
.legend-item { margin: 3px 0; }
.legend-dot { display:inline-block; width:12px; height:12px; border-radius:50%;
              margin-right:6px; vertical-align:middle; }
#controls { position:absolute; bottom:10px; left:10px; background:rgba(0,0,0,0.8);
            padding:10px; border-radius:8px; font-size:12px; z-index:10; }
label { margin-right: 12px; cursor: pointer; display: inline-block; }
#controls input[type=range] { width: 80px; vertical-align: middle; }
</style>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
</head><body>
<div id="info">
<h3>Trajectory Convention Check</h3>
<p>Scene: <b>""" + scene + r"""</b> &nbsp; Prompt: <b>""" + f"{prompt_idx:02d}" + r"""</b></p>
<p>Arrows = camera <b>+Z axis</b>. Toggle -Z / Y to compare conventions.<br>
Large sphere = start, small sphere = end.</p>
<div id="legend"></div>
<div id="stats" style="margin-top:8px; font-size:11px;"></div>
</div>
<div id="controls">
<label><input type="checkbox" id="showPosZ" checked> +Z arrows</label>
<label><input type="checkbox" id="showNegZ"> -Z arrows</label>
<label><input type="checkbox" id="showY"> Y-up arrows</label>
<label><input type="checkbox" id="showMesh" checked> Mesh</label>
<br style="margin-bottom:4px">
<label>Point size: <input type="range" id="ptSize" min="1" max="8" step="0.5" value="2"></label>
<label>Mesh opacity: <input type="range" id="meshOpacity" min="0" max="100" step="5" value="60"></label>
</div>
<script>
const DATA = __TRAJ_DATA__;
const MESH = __MESH_DATA__;

const COLORS = {
    "gt": 0xffd700,
    "gscinema": 0x00ff88,
    "chatcam": 0xff6b6b,
    "cctg": 0x6bc5ff,
};
const COLOR_NAMES = {
    "gt": "#ffd700",
    "gscinema": "#00ff88",
    "chatcam": "#ff6b6b",
    "cctg": "#6bc5ff",
};

// Setup
const scene3 = new THREE.Scene();
scene3.background = new THREE.Color(0x1a1a2e);
const camera = new THREE.PerspectiveCamera(60, window.innerWidth/window.innerHeight, 0.01, 200);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

// Lights
scene3.add(new THREE.AmbientLight(0xffffff, 0.6));
const dl = new THREE.DirectionalLight(0xffffff, 0.8);
dl.position.set(5, 10, 7);
scene3.add(dl);

// Grid
const grid = new THREE.GridHelper(20, 40, 0x444466, 0x333355);
scene3.add(grid);

// Axes helper at origin (R=X, G=Y, B=Z)
const axes = new THREE.AxesHelper(1);
scene3.add(axes);

// Groups for toggle
const posZGroup = new THREE.Group(); scene3.add(posZGroup);
const negZGroup = new THREE.Group(); scene3.add(negZGroup);
const yGroup = new THREE.Group(); scene3.add(yGroup);
const meshGroup = new THREE.Group(); scene3.add(meshGroup);
negZGroup.visible = false;
yGroup.visible = false;

// ─── Render mesh as colored point cloud ───
let pointsMaterial = null;
if (MESH && MESH.positions && MESH.positions.length > 0) {
    const n = MESH.positions.length;
    const geom = new THREE.BufferGeometry();
    const posArr = new Float32Array(n * 3);
    const colArr = new Float32Array(n * 3);
    const hasColor = MESH.colors && MESH.colors.length === n;

    for (let i = 0; i < n; i++) {
        posArr[i*3]   = MESH.positions[i][0];
        posArr[i*3+1] = MESH.positions[i][1];
        posArr[i*3+2] = MESH.positions[i][2];
        if (hasColor) {
            colArr[i*3]   = MESH.colors[i][0];
            colArr[i*3+1] = MESH.colors[i][1];
            colArr[i*3+2] = MESH.colors[i][2];
        } else {
            colArr[i*3] = colArr[i*3+1] = colArr[i*3+2] = 0.6;
        }
    }
    geom.setAttribute('position', new THREE.BufferAttribute(posArr, 3));
    geom.setAttribute('color', new THREE.BufferAttribute(colArr, 3));

    pointsMaterial = new THREE.PointsMaterial({
        size: 0.02,
        vertexColors: true,
        opacity: 0.6,
        transparent: true,
        sizeAttenuation: true,
    });
    const points = new THREE.Points(geom, pointsMaterial);
    meshGroup.add(points);

    document.getElementById('info').innerHTML +=
        '<div style="margin-top:4px; font-size:11px;">Mesh: ' + n.toLocaleString() + ' pts</div>';
}

// ─── Trajectories ───
let allPts = [];
const legendEl = document.getElementById('legend');
const statsEl = document.getElementById('stats');
let statsHtml = '';

for (const [method, d] of Object.entries(DATA)) {
    if (!d) continue;
    const color = COLORS[method] || 0xffffff;
    const colorName = COLOR_NAMES[method] || "#fff";

    legendEl.innerHTML += '<div class="legend-item"><span class="legend-dot" style="background:' +
        colorName + '"></span>' + method + ' (' + d.n_poses + ' frames)</div>';
    statsHtml += '<b>' + method + '</b>: [' + d.pos_min.map(function(v){return v.toFixed(2);}).join(', ') +
        '] .. [' + d.pos_max.map(function(v){return v.toFixed(2);}).join(', ') + ']<br>';

    // Path line
    const pathPts = d.path.map(function(p){ return new THREE.Vector3(p[0], p[1], p[2]); });
    allPts.push.apply(allPts, pathPts);
    const lineGeom = new THREE.BufferGeometry().setFromPoints(pathPts);
    const lineMat = new THREE.LineBasicMaterial({ color: color, linewidth: 2, opacity: 0.9, transparent: true });
    scene3.add(new THREE.Line(lineGeom, lineMat));

    // Start marker (larger sphere)
    const startGeom = new THREE.SphereGeometry(0.05, 16, 16);
    const startMat = new THREE.MeshLambertMaterial({ color: color });
    const startMesh = new THREE.Mesh(startGeom, startMat);
    startMesh.position.set(d.path[0][0], d.path[0][1], d.path[0][2]);
    scene3.add(startMesh);

    // End marker (smaller, slightly transparent)
    const endGeom = new THREE.SphereGeometry(0.035, 16, 16);
    const endMat = new THREE.MeshLambertMaterial({ color: color, opacity: 0.5, transparent: true });
    const endMesh = new THREE.Mesh(endGeom, endMat);
    var lastPt = d.path[d.path.length - 1];
    endMesh.position.set(lastPt[0], lastPt[1], lastPt[2]);
    scene3.add(endMesh);

    // Direction arrows
    var arrowLen = 0.15;
    for (var i = 0; i < d.positions.length; i++) {
        var p = d.positions[i];
        var origin = new THREE.Vector3(p[0], p[1], p[2]);

        // +Z
        var fz = d.forward_z[i];
        var dirZ = new THREE.Vector3(fz[0], fz[1], fz[2]).normalize();
        posZGroup.add(new THREE.ArrowHelper(dirZ, origin, arrowLen, color, arrowLen*0.3, arrowLen*0.15));

        // -Z
        var fnz = d.forward_neg_z[i];
        var dirNZ = new THREE.Vector3(fnz[0], fnz[1], fnz[2]).normalize();
        var aNZ = new THREE.ArrowHelper(dirNZ, origin, arrowLen, color, arrowLen*0.3, arrowLen*0.15);
        aNZ.line.material.opacity = 0.4; aNZ.line.material.transparent = true;
        negZGroup.add(aNZ);

        // Y
        var fy = d.forward_y[i];
        var dirY = new THREE.Vector3(fy[0], fy[1], fy[2]).normalize();
        yGroup.add(new THREE.ArrowHelper(dirY, origin, arrowLen*0.7, 0xaaaaaa, arrowLen*0.2, arrowLen*0.1));
    }
}

statsEl.innerHTML = statsHtml;

// Camera positioning from all content
const bbox = new THREE.Box3();
allPts.forEach(function(p){ bbox.expandByPoint(p); });
if (MESH && MESH.bbox_min && MESH.bbox_min.length === 3) {
    bbox.expandByPoint(new THREE.Vector3(MESH.bbox_min[0], MESH.bbox_min[1], MESH.bbox_min[2]));
    bbox.expandByPoint(new THREE.Vector3(MESH.bbox_max[0], MESH.bbox_max[1], MESH.bbox_max[2]));
}
var center = new THREE.Vector3();
bbox.getCenter(center);
var sceneSize = bbox.getSize(new THREE.Vector3()).length();
camera.position.copy(center).add(new THREE.Vector3(sceneSize*0.8, sceneSize*0.6, sceneSize*0.8));
camera.lookAt(center);

// Orbit controls (left-drag = orbit, right-drag = pan, scroll = zoom)
var isDragging = false, isRightDrag = false, prevMouse = {x:0, y:0};
var spherical = new THREE.Spherical().setFromVector3(camera.position.clone().sub(center));
var orbitCenter = center.clone();

renderer.domElement.addEventListener('contextmenu', function(e){ e.preventDefault(); });
renderer.domElement.addEventListener('mousedown', function(e){
    isDragging = true;
    isRightDrag = (e.button === 2);
    prevMouse = {x: e.clientX, y: e.clientY};
});
renderer.domElement.addEventListener('mouseup', function(){ isDragging = false; isRightDrag = false; });
renderer.domElement.addEventListener('mousemove', function(e){
    if (!isDragging) return;
    var dx = e.clientX - prevMouse.x;
    var dy = e.clientY - prevMouse.y;

    if (isRightDrag) {
        var panSpeed = 0.002 * spherical.radius;
        var right = new THREE.Vector3();
        var up = new THREE.Vector3();
        right.setFromMatrixColumn(camera.matrixWorld, 0);
        up.setFromMatrixColumn(camera.matrixWorld, 1);
        var panDelta = right.multiplyScalar(-dx * panSpeed).add(up.multiplyScalar(dy * panSpeed));
        orbitCenter.add(panDelta);
        camera.position.add(panDelta);
    } else {
        spherical.theta -= dx * 0.005;
        spherical.phi = Math.max(0.1, Math.min(Math.PI - 0.1, spherical.phi - dy * 0.005));
        camera.position.setFromSpherical(spherical).add(orbitCenter);
    }
    camera.lookAt(orbitCenter);
    prevMouse = {x: e.clientX, y: e.clientY};
});
renderer.domElement.addEventListener('wheel', function(e){
    spherical.radius *= (1 + e.deltaY * 0.001);
    spherical.radius = Math.max(0.1, Math.min(100, spherical.radius));
    camera.position.setFromSpherical(spherical).add(orbitCenter);
    camera.lookAt(orbitCenter);
});

// UI toggles
document.getElementById('showPosZ').addEventListener('change', function(e){ posZGroup.visible = e.target.checked; });
document.getElementById('showNegZ').addEventListener('change', function(e){ negZGroup.visible = e.target.checked; });
document.getElementById('showY').addEventListener('change', function(e){ yGroup.visible = e.target.checked; });
document.getElementById('showMesh').addEventListener('change', function(e){ meshGroup.visible = e.target.checked; });
document.getElementById('ptSize').addEventListener('input', function(e){
    if (pointsMaterial) pointsMaterial.size = parseFloat(e.target.value) * 0.01;
});
document.getElementById('meshOpacity').addEventListener('input', function(e){
    if (pointsMaterial) pointsMaterial.opacity = parseFloat(e.target.value) / 100;
});

function animate() {
    requestAnimationFrame(animate);
    renderer.render(scene3, camera);
}
animate();

window.addEventListener('resize', function(){
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});
</script>
</body></html>"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", default="09c1414f1b")
    p.add_argument("--prompt", type=int, default=0)
    p.add_argument("--output", default=None, help="Output HTML file path")
    p.add_argument("--mesh", default="data/ScanNetpp/scenes/09c1414f1b/dslr/scans/mesh_aligned_0.05.ply", help="Path to mesh file (PLY). Auto-detected if not set.")
    p.add_argument("--max_mesh_points", type=int, default=50000,
                   help="Max points to subsample from mesh (default: 50000)")
    p.add_argument("--dump_json", default=None, help="Also dump raw trajectory data as JSON")
    a = p.parse_args()

    print(f"Scene: {a.scene}, Prompt: {a.prompt:02d}")

    # ── Load mesh ──
    print(f"\n  Finding mesh...")
    mesh_path = find_mesh(a.scene, DATA_ROOT, a.mesh)
    mesh_data = {"positions": [], "colors": [], "bbox_min": [], "bbox_max": []}
    if mesh_path:
        print(f"  Found: {mesh_path}")
        verts, colors = load_ply(str(mesh_path), max_points=a.max_mesh_points)
        if verts is not None and len(verts) > 0:
            mesh_data["positions"] = verts.tolist()
            mesh_data["colors"] = colors.tolist() if colors is not None else []
            mesh_data["bbox_min"] = verts.min(axis=0).tolist()
            mesh_data["bbox_max"] = verts.max(axis=0).tolist()
            print(f"    Mesh bbox: {[f'{v:.2f}' for v in mesh_data['bbox_min']]} .. {[f'{v:.2f}' for v in mesh_data['bbox_max']]}")
    else:
        print(f"  No mesh found for scene {a.scene}")
        print(f"  Use --mesh /path/to/mesh.ply to specify manually")

    # ── Load trajectories ──
    print(f"\nLoading trajectories...")
    all_data = {}
    for method in ["gt", "gscinema", "chatcam", "cctg"]:
        print(f"\n  Loading {method}...")
        data = load_and_extract(method, a.scene, a.prompt)
        if data:
            print(f"    ✓ {data['n_poses']} poses, range: {[f'{v:.2f}' for v in data['pos_min']]} .. {[f'{v:.2f}' for v in data['pos_max']]}")
        else:
            print(f"    ✗ not found or empty")
        all_data[method] = data

    if a.dump_json:
        dump_out = {"trajectories": all_data, "mesh_info": {
            "bbox_min": mesh_data.get("bbox_min"),
            "bbox_max": mesh_data.get("bbox_max"),
            "n_points": len(mesh_data.get("positions", [])),
        }}
        with open(a.dump_json, "w") as f:
            json.dump(dump_out, f, indent=2)
        print(f"\nDumped JSON -> {a.dump_json}")

    # ── Subsample paths for HTML ──
    html_data = {}
    for method, d in all_data.items():
        if d is None:
            html_data[method] = None
            continue
        hd = dict(d)
        path = d["path"]
        if len(path) > 200:
            step = len(path) // 200
            hd["path"] = path[::step]
        html_data[method] = hd

    # ── Generate HTML ──
    output_path = a.output or f"traj_convention_check_{a.scene}_p{a.prompt:02d}.html"
    html = generate_html(a.scene, a.prompt)
    html = html.replace("__TRAJ_DATA__", json.dumps(html_data))
    html = html.replace("__MESH_DATA__", json.dumps(mesh_data))

    with open(output_path, "w") as f:
        f.write(html)

    file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"\nHTML visualization -> {output_path} ({file_size_mb:.1f} MB)")
    print(f"Open in browser to inspect.")
    print(f"\nControls:")
    print(f"  Left-drag:   orbit")
    print(f"  Right-drag:  pan")
    print(f"  Scroll:      zoom")
    print(f"  Checkboxes:  toggle +Z/-Z/Y arrows, mesh visibility")
    print(f"  Sliders:     point size, mesh opacity")


if __name__ == "__main__":
    main()