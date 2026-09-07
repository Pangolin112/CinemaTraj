# ChatCam: Empowering Camera Control through Conversational AI

Implementation of the ChatCam pipeline from Liu et al. (2024), HKUST & Dartmouth College.

## Architecture

```
User Instruction
       │
       ▼
┌──────────────────┐
│   LLM Agent      │  GPT-4 parses instruction → observation/reasoning/plan
│   (GPT-4)        │  Decomposes into tool calls: CineGPT + Anchor Determinator
└────────┬─────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌──────────────────┐
│CineGPT │ │Anchor Determinator│
│(GenDoP)│ │  CLIP + Refine    │
└───┬────┘ └────────┬─────────┘
    │               │
    │  Atomic        │  Anchor
    │  Trajectories  │  Camera Poses
    │               │
    └───────┬───────┘
            ▼
   ┌────────────────────┐
   │Trajectory Composer  │  Affine transforms to align segments with anchors
   │                     │  Smooth blending at segment junctions
   └────────┬────────────┘
            ▼
    Final Camera Trajectory → Render Video (3DGS/NeRF)
```

## Project Structure

```
chatcam/
├── chatcam.py              # Main pipeline orchestrator
├── llm_agent.py            # GPT-4 agent with ChatCam system prompt
├── anchor_determinator.py  # CLIP initial selector + gradient refinement (Eq. 4-6)
├── cinegpt_wrapper.py      # GenDoP wrapper + analytical fallback
├── trajectory_composer.py  # Affine alignment + smooth blending
├── trajectory_utils.py     # I/O, visualization, Nerfstudio export
└── README.md
```

## Usage

### Full Pipeline (with models)
```bash
export OPENAI_API_KEY="sk-..."

python chatcam.py \
  --scene_dir /path/to/scene \
  --instruction "Starting from the sofa, pan to the window, then zoom in on the flowers" \
  --cinegpt_resume /path/to/gendop_checkpoint.safetensors \
  --output_dir ./output
```

### Interactive Mode
```bash
python chatcam.py --scene_dir /path/to/scene --interactive
```

### Demo (no GPU/API required)
```bash
python demo.py
```

### Render with Nerfstudio
```bash
ns-render camera-path \
  --load-config /path/to/nerfstudio/config.yml \
  --camera-path-filename output/camera_path.json \
  --output-path output/video.mp4
```

## Key Components

### LLM Agent (Section 3.3)
Uses GPT-4 with a carefully designed system prompt to:
- **Observe**: Summarize the user's request
- **Reason**: Identify trajectory descriptions vs. anchor points
- **Plan**: Generate ordered tool calls to CineGPT and Anchor Determinator

Falls back to local heuristic parsing when no API key is available.

### CineGPT / GenDoP (Section 3.1)
GPT-based autoregressive model that tokenizes camera trajectories via VQ-VAE
and generates them from text. Camera parameters per frame:
- Rotation R (S² × S² representation)
- Translation t
- Intrinsics K (focal length, principal point)
- Velocity (global duration parameter)

### Anchor Determinator (Section 3.2)
Two-stage anchor finding:

**Initial Selection (Eq. 4):**
```
i_anchor = argmax_i  f_image(I_i) · f_text(T) / (||f_image(I_i)|| · ||f_text(T)||)
```

**Refinement (Eq. 5-6):**
```
min_c  L_anchor(c) = -f_image(R(c)) · f_text(T) / (||...|| · ||...||)
c_{t+1} = c_t - η ∇_c L_anchor(c_t)
```

### Trajectory Composition
Combines sub-trajectories by:
1. Applying affine transforms to align start/end with anchors
2. SLERP rotation interpolation at segment junctions
3. Smooth blending over configurable overlap windows
