# CinemaTraj: Composing Atomic Camera Trajectories for 3D Scenes with LLM Agents

<p align="center">
  <a href="https://cinematraj.github.io/"><img src="https://img.shields.io/badge/%F0%9F%8C%90%20Project%20Page-CinemaTraj-1a73e8?style=for-the-badge" alt="Project Page"></a>
  <a href="https://arxiv.org/abs/2607.26910"><img src="https://img.shields.io/badge/arXiv-2607.26910-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv"></a>
</p>

Given a 3D scene and a natural language prompt, **CinemaTraj** decomposes the request into a sequence of cinematographic movements -- orbit, crane, dolly, pan, tilt, zoom, arc -- grounded in a 3D scene graph, and plans a collision-free camera trajectory through the scene. The resulting trajectory is rendered into a cinematic video with synchronized voiceover and subtitles.

<p align="center">
  <img src="assets/demo.gif" width="100%" alt="Side-by-side comparison with baselines">
  <br>
  <em>Rendered trajectories and camera paths for the same prompt &mdash; CCTG and
  ChatCam&nbsp;+&nbsp;GenDoP leave the free space and lose their targets, while ours
  stays collision-free and visits every requested object in order.</em>
</p>

## Pipeline Overview

```
User Prompt ──> User Prompt Translator ──> Anchor Selector ──> Parametric Atomic Trajectory Builder
                        |                       |                          |
                   (LLM agent +            (score-based              (orbit, dolly, crane,
                    scene graph)          viewpoint selection)        pan, tilt, zoom, arc)
                                                                           |
                                                                           v
                    Cinematic Video  <──  Subtitle & Voiceover  <──  Trajectory Optimizer
                                           Generator               (SDF collision + occlusion)
```

| Module | Paper Section | Description |
|--------|---------------|-------------|
| **User Prompt Translator** | &sect;3.2 | LLM agent equipped with a cinematographic toolset decomposes free-form text into atomic camera commands |
| **Anchor Selector** | &sect;3.2.1 / &sect;3.3 | Selects collision/occlusion-free viewpoints per target object via face-normal-biased OBB sampling |
| **Parametric Atomic Trajectory Builder** | &sect;3.3 | Instantiates each atomic command as a parametric trajectory &tau;(t; &theta;<sub>fix</sub>, &theta;<sub>free</sub>) |
| **Trajectory Optimizer** | &sect;3.4 | Two-pass gradient descent minimizing SDF-based collision (C<sub>sdf</sub>) and occlusion (C<sub>occl</sub>) costs |
| **Subtitle & Voiceover Generator** | &sect;3.5 | Renders 3DGS frames, generates captions via VLM, synthesizes TTS voiceover |

---

## 1. Installation

### Prerequisites

- Linux, Python 3.10
- An NVIDIA GPU with a CUDA toolkit installed (`nvcc` on `PATH`)
- `ffmpeg` on `PATH` (used for video muxing and the TTS voiceover)
- An OpenAI API key (GPT-4.1 for the planner, GPT-4o-mini for captions, `tts-1-hd` for voiceover)

### Environment

```bash
conda create -y -n cinematraj python=3.10
conda activate cinematraj

# Install the torch build matching your GPU *first*. cu128 shown here; it is
# required for Blackwell cards (RTX 50xx, sm_120).
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git     # CLaTr metric + no_anchor ablation
```

**gsplat builds from source.** There is no prebuilt wheel for recent
torch/CUDA combinations, and the published wheels stop at cu124, which does not
cover sm_120. gsplat JIT-compiles its CUDA kernels on the first render (~45 s,
cached afterwards). `nvcc` must accept your host compiler — CUDA 12.8 supports
GCC ≤ 14, so on a system with a newer default GCC install one alongside it:

```bash
conda install -y -c conda-forge gxx_linux-64=13
export CUDA_HOME=/usr/local/cuda-12.8
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
export CUDAHOSTCXX=$CXX
export TORCH_CUDA_ARCH_LIST="12.0"        # 8.6 for RTX 30xx, 8.9 for 40xx
```

### API key

Every entry point reads `OPENAI_API_KEY` from the environment; the `api_key`
field in the configs is left `null` and only needs setting if you prefer to
keep the key in the file.

```bash
export OPENAI_API_KEY="sk-..."
```

---

## 2. Data

Download the 20-scene ScanNet++ evaluation subset (~11 GB) from Google Drive:

**[CinemaTraj_ScanNetpp_20 &rarr; Google Drive](https://drive.google.com/drive/folders/1QS-pO7DXQXaPS0JYglh5K6iXFX7PuYUO?usp=sharing)**

Then place (or symlink) it at `data/ScanNetpp` inside the repository:

```bash
mkdir -p data
ln -s /path/to/CinemaTraj_ScanNetpp_20 data/ScanNetpp
```

The configs use paths relative to the repository root, so **run every command
from the repository root**. The expected layout is:

```
data/ScanNetpp/
├── scenes/<scene_id>/dslr/
│   ├── sg/<scene_id>-simple.json      scene graph (OBBs, rooms, placement flags)
│   ├── ply/point_cloud_30000.ply      3D Gaussian Splatting reconstruction
│   └── sdf/<scene_id>_sdf_res128.npz  precomputed SDF grid + obb_mesh.ply
└── gt/<scene_id>/prompt_{00,01}/      hand-authored Blender GT for Motion MSE
```

See `data/ScanNetpp/README.md` for the scene list, licence terms and known
caveats. The `no_anchor` ablation additionally needs the DSLR frames and COLMAP
poses (`dslr/images/`, `dslr/sparse/0/`) from the official ScanNet++ release;
the other conditions run from the packaged subset alone.

### Model checkpoints

```bash
bash CLaTr/download_checkpoints.sh          # CLaTr encoders — needed for the CLaTr Score
```

This writes `CLaTr/checkpoints/clatr-{e100,text_encoder,traj_encoder,traj_decoder}.ckpt`,
which is where `eval.py` looks by default. Only the two encoders are actually
loaded — the CLaTr model itself is reimplemented inline in `src/eval/eval.py`,
so no CLaTr source checkout is required. The weights are not redistributed
here; they come from the E.T. authors' release.

For the `no_parametric` ablation, download GenDoP's `text_directorial.safetensors`
from [huggingface.co/Dubhe-zmc/GenDoP](https://huggingface.co/Dubhe-zmc/GenDoP/blob/main/checkpoints/text_directorial.safetensors)
into `third_party/GenDoP/checkpoints/`. Its first run additionally downloads
Stable Diffusion 2.1 from the HuggingFace hub (~5 GB) — GenDoP instantiates the
full pipeline only to take its CLIP text encoder — so budget network and disk
for that.

`cinegpt_wrapper.py` catches model-loading failures and continues on an
analytical fallback, which still renders a video — so after a `no_parametric`
run, check that the log does **not** contain
`CineGPT: Could not load GenDoP model`, otherwise the numbers are not the
ablation you intended. Leaving `gendop_resume` empty selects that fallback
deliberately and needs no checkpoint.

---

## 3. Quick start

Run the pipeline on a single scene:

```bash
python src/pipeline/cinematraj_pipeline.py --config config/config_scannetpp.yaml
```

Edit `scene_id` and `prompt` in `config/config_scannetpp.yaml` to change the
scene or request. Outputs land in `outputs/scannetpp/<scene_id>/`:

| File | Contents |
|---|---|
| `rendered_video.mp4` | 3DGS render along the final trajectory |
| `combined_trajectory.json` | Final camera poses (nerfstudio-style `frames`) |
| `combined_trajectory.png` | Top-down trajectory plot |
| `anchors.json` | Selected anchor viewpoint per target object |
| `segments_executor/*.npy` | Per-segment c2w matrices before composition |
| `obb_mesh.ply`, `*_sdf_res128.npz` | Collision geometry built from the scene graph |

Set `render.show_subtitles: true` to burn in VLM-generated captions, and
`visualise.enabled: true` to serve an interactive viser scene on port 8080.

A full single-scene run takes roughly a minute end to end on an RTX 5090, of
which the 1080p render dominates; planning plus optimization alone is ~20 s.

---

## 4. Reproducing the paper

### 4.1 Prompts

Prompts are generated by template sampling with a fixed seed, so the benchmark
is deterministic. The paper's three specificity levels map onto the config
level names as:

| Paper | Config level | Used for |
|---|---|---|
| Fully-specified | `low_id` | Table 1 (main comparison + ablation) |
| Partially-specified | `medium_id` | Table B.1 (supplementary) |
| Open-ended | `high` | Table B.1 (supplementary) |

With `prompt_seed: 42` and `n_prompts_per_level: 2`, the generated `low_id`
prompts match the hand-authored ground-truth trajectories in
`data/ScanNetpp/gt/<scene_id>/prompt_{00,01}/` one-to-one — do not change
either value or Motion MSE will compare against the wrong reference.

### 4.2 Main results (Table 1)

```bash
# 1. Generate trajectories for all 20 scenes, fully-specified prompts
python src/eval/benchmark_runner.py --config config/benchmark_scannetpp.yaml --levels low_id

# 2. Score them
python src/eval/eval.py \
    --benchmark_dir outputs/benchmark_scannetpp \
    --data_root data/ScanNetpp \
    --level low_id
```

### 4.3 Ablations (Table 1, lower section)

```bash
python src/eval/ablation_study_runner.py --config config/ablation_scannetpp.yaml --levels low_id
python src/eval/eval.py --benchmark_dir outputs/ablation_study --data_root data/ScanNetpp --level low_id
```

The four conditions are `full`, `no_anchor` (CLIP keyframe selection),
`no_parametric` (GenDoP 6-DoF generation) and `no_sdf` (3DGS density only);
restrict them with `--conditions`, and restrict scenes with `--scenes`.

### 4.4 Supplementary results (Table B.1)

```bash
python src/eval/ablation_study_runner.py --config config/ablation_scannetpp.yaml --levels medium_id high
```

### 4.5 Baselines (Table 1, upper section)

Both baselines read the prompts generated by the main benchmark, so run
&sect;4.2 first. IDs are stripped automatically, since neither baseline is
given the scene graph.

```bash
python baselines/ChatCam_GenDoP/benchmark_runner.py --config config/benchmark_chatcam.yaml
python baselines/Berkeley/benchmark_runner.py       --config config/benchmark_cctg.yaml

# note: --level low, not low_id — the runners strip the IDs and write into a
# "low" directory, so low_id would match nothing
python src/eval/eval.py --benchmark_dir outputs/benchmark_chatcam --data_root data/ScanNetpp --level low
python src/eval/eval.py --benchmark_dir outputs/benchmark_cctg    --data_root data/ScanNetpp --level low
```

The baselines need ScanNet++ assets that the packaged subset does **not**
carry, because they reconstruct the scene themselves rather than reading the
scene graph:

| Baseline | Additional per-scene data |
|---|---|
| ChatCam + GenDoP | `dslr/sparse/0/` (COLMAP points), GenDoP checkpoint |
| CCTG | `dslr/images/`, `dslr/sparse/0/`, `dslr/scans/mesh_aligned_0.05.ply` |

Obtain them from the official ScanNet++ release and place them under the
matching `data/ScanNetpp/scenes/<scene_id>/` directories.

### 4.6 Metrics

`src/eval/eval.py` reports the five metrics of &sect;4.1:

| Metric | Direction | Note |
|---|---|---|
| Motion MSE | ↓ | Only defined for `low_id`; needs the GT trajectories |
| CLaTr Score | ↑ | **Raw cosine similarity** — the paper reports it ×100 |
| Collision Rate | ↓ | Fraction of samples with SDF < 0 |
| Occlusion Rate | ↓ | Fraction of samples where the target is blocked |
| Object Coverage | ↑ | Fraction of planned anchors actually visited |

CLaTr is trained on fully-specified trajectory descriptions; on open-ended
prompts the similarity can go negative, which `eval.py` reports as `n/a` (the
paper's `N/A` entries).

---

## 5. Configuration

| Section | Parameters |
|---------|-----------|
| **Scene** | `scene_id`, `prompt`, `format` (`scene_graph` or `legacy`) |
| **Paths** | `data_root`, `project_root`, `output_dir` |
| **Trajectory** | `transition_frames`, `reorder`, `smooth_window` |
| **SDF** | `resolution`, `padding`, `force_rebuild`, `cache_dir` |
| **Optimisation** | `anchor_collision_margin`, `verbose` |
| **Render** | `width`, `height`, `fov_y`, `fps`, `gaussian_scale`, `show_subtitles` |
| **Subtitles** | `language` (en/zh/de/ja/fr/es/ko), `device` |

Optimizer hyperparameters (Adam, lr 1.0, 1000 iterations per segment) and the
anchor-scoring weights of Eq. 2 live in
`src/trajectory_optimizer/trajectory_optimizer.py` and
`src/anchor_selector/anchor_selector.py` respectively.

### Scene format support

- **Scene Graph (ScanNet++)** — JSON with string IDs (`"table_0"`), oriented bounding boxes (OBBs), room connectivity
- **Legacy (InteriorGS)** — `labels.json` with integer IDs, 8-corner bbox arrays, USD collision mesh

---

## 6. Project structure

```
CinemaTraj/
├── config/                              # YAML configs
│   ├── config_scannetpp.yaml            # Single-scene pipeline
│   ├── benchmark_scannetpp.yaml         # Benchmark over all scenes
│   ├── ablation_scannetpp.yaml          # Ablation conditions
│   └── benchmark_{chatcam,cctg}.yaml    # Baseline benchmarks
├── src/
│   ├── pipeline/cinematraj_pipeline.py  # Primary entry point
│   ├── user_prompt_translator/          # §3.2 LLM planner + response validator
│   ├── anchor_selector/                 # §3.2.1 score-based viewpoint selection
│   ├── parametric_trajectory_builder/   # §3.3 atomic trajectory types + executor
│   ├── trajectory_optimizer/            # §3.4 SDF collision/occlusion optimizer
│   ├── subtitle_voiceover_generator/    # §3.5 3DGS render, VLM captions, TTS
│   ├── eval/                            # benchmark, ablation and metric runners
│   └── utils/                           # mesh/SDF construction, viser viewer
├── baselines/                           # ChatCam+GenDoP and CCTG adapters
├── third_party/GenDoP/                  # vendored GenDoP (no_parametric ablation)
├── CLaTr/                               # CLaTr checkpoint fetcher (metric only)
└── requirements.txt
```

## License

MIT — see [LICENSE](LICENSE). Vendored third-party code under `baselines/` and
`third_party/` keeps its original license. ScanNet++ data is governed by its own
terms.
