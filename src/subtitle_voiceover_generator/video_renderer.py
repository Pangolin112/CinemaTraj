# src/subtitle_voiceover_generator/video_renderer.py

"""
Video Renderer with LLM-Generated Subtitles
============================================

Pipeline:
  1. For each unique anchor in the TrajectoryResult, render the anchor's
     viewpoint image using the 3DGS renderer.
  2. Send each image to GPT-4o (vision) with context (object label, movement
     type) and get back a rich, cinematic description.
  3. Build a SubtitleTrack where object-level and transition segments use
     the LLM descriptions instead of generic labels.
  4. Burn the subtitles onto the rendered video frames.

NEW: Per-frame focal_multiplier support for zoom_in_out / zoom_out_in.
UPDATED: Supports new scene graph JSON format with OBB-based bounding boxes.
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import base64
import io
import numpy as np
import torch
import json
import cv2
from tqdm import tqdm
from typing import Optional, List, Dict, Union, Tuple, Any
from dataclasses import dataclass, field
from plyfile import PlyData
from gsplat import rasterization

from src.trajectory_optimizer.trajectory_optimizer import (
    CameraTrajectory, CameraPose, CameraIntrinsics, TrajectoryCombiner,
)

CODEC = "mp4v" # mp4v does not support browser display, but avc1 is not available for this opencv version so we postprocess the video

# =============================================================================
# Subtitle Data Structures
# =============================================================================

@dataclass
class SubtitleEntry:
    """A single subtitle spanning a frame range."""
    start_frame: int
    end_frame: int            # exclusive
    text: str
    movement_type: str        # e.g. 'orbit_half', 'arc', 'static'
    movement_category: str    # 'object-level', 'transitional', or 'transition'
    object_label: str = ""


@dataclass
class SubtitleTrack:
    """Full subtitle track for a video."""
    entries: List[SubtitleEntry] = field(default_factory=list)
    fps: float = 30.0
    total_frames: int = 0

    def get_subtitle_at_frame(self, frame_idx: int) -> Optional[SubtitleEntry]:
        for entry in self.entries:
            if entry.start_frame <= frame_idx < entry.end_frame:
                return entry
        return None

    def save_srt(self, path: Union[str, Path]):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            for i, entry in enumerate(self.entries, 1):
                t0 = entry.start_frame / self.fps
                t1 = (entry.end_frame - 1) / self.fps
                f.write(f"{i}\n{_fmt_srt(t0)} --> {_fmt_srt(t1)}\n{entry.text}\n\n")
        print(f"Saved SRT to {path}")

    def save_json(self, path: Union[str, Path]):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            'fps': self.fps,
            'total_frames': self.total_frames,
            'entries': [
                {
                    'start_frame': e.start_frame,
                    'end_frame': e.end_frame,
                    'start_time': round(e.start_frame / self.fps, 3),
                    'end_time': round(e.end_frame / self.fps, 3),
                    'text': e.text,
                    'movement_type': e.movement_type,
                    'movement_category': e.movement_category,
                    'object_label': e.object_label,
                }
                for e in self.entries
            ],
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved subtitle JSON to {path}")


def _fmt_srt(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    ms = int((sec % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def load_subtitle_track(path: Union[str, Path]) -> SubtitleTrack:
    with open(path) as f:
        data = json.load(f)
    track = SubtitleTrack(fps=data.get('fps', 30.0),
                          total_frames=data.get('total_frames', 0))
    for e in data.get('entries', []):
        track.entries.append(SubtitleEntry(
            start_frame=e['start_frame'], end_frame=e['end_frame'],
            text=e['text'], movement_type=e.get('movement_type', ''),
            movement_category=e.get('movement_category', ''),
            object_label=e.get('object_label', ''),
        ))
    return track


# =============================================================================
# Bounding Box Loader — supports both formats
# =============================================================================

def _load_bboxes_for_renderer(json_path: str) -> Tuple[list, str]:
    """
    Load bounding boxes for the renderer.

    Returns:
        (objects_list, format_type)

    For scene_graph format, each item has:
        'id', 'label', 'corners' (8x3 numpy), 'center'
    For legacy format, each item has:
        'ins_id', 'label', 'bounding_box' (list of 8 dicts)
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    if isinstance(data, dict) and "objects" in data:
        from scipy.spatial.transform import Rotation

        objects_dict = data["objects"]
        objects_list = []
        for obj_id, obj_data in objects_dict.items():
            obb = obj_data["obb"]
            center = np.array(obb[0:3], dtype=np.float64)
            size = np.array(obb[3:6], dtype=np.float64)
            qxyzw = np.array(obb[6:10], dtype=np.float64)

            R = Rotation.from_quat(qxyzw).as_matrix()
            half = size / 2.0
            signs = np.array([
                [-1, -1, -1], [-1, -1,  1], [-1,  1, -1], [-1,  1,  1],
                [ 1, -1, -1], [ 1, -1,  1], [ 1,  1, -1], [ 1,  1,  1],
            ], dtype=np.float64)
            corners = (R @ (signs * half).T).T + center

            label = obj_id.rsplit("_", 1)[0]
            objects_list.append({
                'id': obj_id,
                'label': label,
                'corners': corners,
                'center': center,
            })
        return objects_list, 'scene_graph'
    else:
        return data, 'legacy'


# =============================================================================
# LLM-Based Anchor Description
# =============================================================================

SUPPORTED_LANGUAGES = {
    "en": "English",
    "zh": "Chinese (Simplified)",
    "de": "German",
    "ja": "Japanese",
    "fr": "French",
    "es": "Spanish",
    "ko": "Korean",
}

DESCRIPTION_SYSTEM_PROMPTS = {
    "en": """You are a concise scene narrator for a cinematic camera tour of an indoor environment.

Given an image rendered from a camera viewpoint and context about what the camera is doing, produce a SHORT, vivid subtitle (max 12 words) that describes what the viewer sees or what the camera is doing.

Rules:
- Max 12 words. Shorter is better.
- Use present tense, active voice.
- Be specific to what's visible in the image, not generic.
- Don't start with "The camera..." — describe the scene or action naturally.
- Match the tone to the movement: orbiting → emphasize perspective change; static → emphasize details; move_in → emphasize approaching/revealing; arc → emphasize transition.
- If you see specific objects, textures, colors, or spatial relationships, mention them.
- Respond ONLY in English.
""",
    "zh": """你是一位简洁的场景解说员，为室内环境的电影级镜头之旅配字幕。

根据给定的摄像机视角渲染图像和镜头运动信息，生成一条简短、生动的字幕（最多15个中文字），描述观众看到的画面或镜头的动作。

规则：
- 最多15个中文字，越短越好。
- 使用现在时，主动语态。
- 描述画面中具体可见的内容，不要泛泛而谈。
- 不要以"镜头..."开头——用自然的方式描述场景或动作。
- 根据运动类型调整语气：环绕→强调视角变化；静止→强调细节；推进→强调接近/展现；弧线→强调过渡。
- 如果看到具体的物体、纹理、颜色或空间关系，请提及。
- 只用简体中文回答。
""",
}

_GENERIC_SYSTEM_PROMPT_TEMPLATE = """You are a concise scene narrator for a cinematic camera tour of an indoor environment.

Given an image rendered from a camera viewpoint and context about what the camera is doing, produce a SHORT, vivid subtitle (max 12 words) that describes what the viewer sees or what the camera is doing.

Rules:
- Max 12 words. Shorter is better.
- Use present tense, active voice.
- Be specific to what's visible in the image, not generic.
- Don't start with "The camera..." — describe the scene or action naturally.
- Match the tone to the movement: orbiting → emphasize perspective change; static → emphasize details; move_in → emphasize approaching/revealing; arc → emphasize transition.
- If you see specific objects, textures, colors, or spatial relationships, mention them.
- Respond ONLY in {language_name}.
"""

DESCRIPTION_USER_TEMPLATES = {
    "en": """Movement: {movement_type} ({movement_category})
Target object: {object_label}
{extra_context}

Write a short subtitle (max 12 words) for this camera viewpoint.""",
    "zh": """镜头运动：{movement_type}（{movement_category}）
目标物体：{object_label}
{extra_context}

为这个镜头视角写一条简短字幕（最多15个中文字）。""",
}

_GENERIC_USER_TEMPLATE = """Movement: {movement_type} ({movement_category})
Target object: {object_label}
{extra_context}

Write a short subtitle (max 12 words) in {language_name} for this camera viewpoint."""


def _get_system_prompt(language: str) -> str:
    if language in DESCRIPTION_SYSTEM_PROMPTS:
        return DESCRIPTION_SYSTEM_PROMPTS[language]
    lang_name = SUPPORTED_LANGUAGES.get(language, language)
    return _GENERIC_SYSTEM_PROMPT_TEMPLATE.format(language_name=lang_name)


def _get_user_template(language: str) -> str:
    if language in DESCRIPTION_USER_TEMPLATES:
        return DESCRIPTION_USER_TEMPLATES[language]
    lang_name = SUPPORTED_LANGUAGES.get(language, language)
    return _GENERIC_USER_TEMPLATE.replace("{language_name}", lang_name)


def _encode_image_to_base64(image_bgr: np.ndarray, max_side: int = 512) -> str:
    h, w = image_bgr.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        image_bgr = cv2.resize(image_bgr, (int(w * scale), int(h * scale)))
    _, buf = cv2.imencode('.jpg', image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf).decode('utf-8')


def describe_anchor_view(
    image_bgr: np.ndarray,
    movement_type: str,
    movement_category: str,
    object_label: str,
    api_key: str,
    model: str = "gpt-4o-mini",
    extra_context: str = "",
    language: str = "en",
) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    b64 = _encode_image_to_base64(image_bgr)
    user_template = _get_user_template(language)
    user_text = user_template.format(
        movement_type=movement_type,
        movement_category=movement_category,
        object_label=object_label,
        extra_context=extra_context,
    )
    system_prompt = _get_system_prompt(language)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64}",
                                "detail": "low",
                            },
                        },
                    ],
                },
            ],
            max_tokens=60,
            temperature=0.4,
        )
        text = response.choices[0].message.content.strip()
        text = text.strip('"\'')
        return text
    except Exception as e:
        print(f"  [LLM description failed: {e}] — using fallback")
        return _fallback_description(movement_type, object_label, language)


def _fallback_description(movement_type: str, object_label: str,
                          language: str = "en") -> str:
    _FALLBACK_TEMPLATES = {
        "en": {
            'orbit_full': "Circling around the {obj}",
            'orbit_half': "Orbiting the {obj}",
            'orbit_quarter': "Sweeping past the {obj}",
            'pan_left': "Panning left across the {obj}",
            'pan_right': "Panning right across the {obj}",
            'move_in': "Approaching the {obj}",
            'move_out': "Pulling back from the {obj}",
            'zoom_in_out': "Zooming in on the {obj}",
            'zoom_out_in': "Wide-angle view of the {obj}",
            'static': "Resting on the {obj}",
            'crane': "Rising above the {obj}",
            'tilt_up': "Tilting up from the {obj}",
            'tilt_down': "Tilting down to the {obj}",
            'arc': "Gliding toward the {obj}",
            '_transition_interp': "Moving toward the {obj}",
            '_default': "Viewing the {obj}",
        },
        "zh": {
            'orbit_full': "环绕{obj}一周",
            'orbit_half': "环绕{obj}",
            'orbit_quarter': "掠过{obj}",
            'pan_left': "向左扫视{obj}",
            'pan_right': "向右扫视{obj}",
            'move_in': "靠近{obj}",
            'move_out': "远离{obj}",
            'zoom_in_out': "聚焦{obj}细节",
            'zoom_out_in': "广角展现{obj}",
            'static': "注视{obj}",
            'crane': "俯瞰{obj}",
            'tilt_up': "仰望{obj}上方",
            'tilt_down': "俯视{obj}",
            'arc': "滑向{obj}",
            '_transition_interp': "移向{obj}",
            '_default': "观察{obj}",
        },
    }
    templates = _FALLBACK_TEMPLATES.get(language, _FALLBACK_TEMPLATES["en"])
    tmpl = templates.get(movement_type, templates.get('_default', "Viewing the {obj}"))
    obj = object_label or ("场景" if language == "zh" else "scene")
    return tmpl.format(obj=obj)


# =============================================================================
# Anchor Image Rendering
# =============================================================================

def render_anchor_image(
    anchor,
    gaussians: Dict[str, torch.Tensor],
    device: str,
    width: int = 512,
    height: int = 512,
    fov_y: float = 60.0,
) -> np.ndarray:
    combiner = TrajectoryCombiner()
    pose = combiner.anchor_to_pose(anchor)

    c2w = np.eye(4)
    c2w[:3, :3] = pose.rotation
    c2w[:3, 3] = pose.position
    w2c = np.linalg.inv(c2w)
    viewmat = torch.tensor(w2c, dtype=torch.float32, device=device)

    fov_y_rad = np.radians(fov_y)
    fy = height / (2 * np.tan(fov_y_rad / 2))
    fx = fy
    K = torch.tensor([[fx, 0, width / 2],
                       [0, fy, height / 2],
                       [0, 0, 1]], dtype=torch.float32, device=device)

    with torch.no_grad():
        renders, _, _ = rasterization(
            means=gaussians['means'], quats=gaussians['quats'],
            scales=gaussians['scales'], opacities=gaussians['opacities'],
            colors=gaussians['colors'], viewmats=viewmat[None], Ks=K[None],
            width=width, height=height, packed=False, render_mode="RGB",
        )
    image = renders[0].clamp(0, 1)
    image_np = (image.cpu().numpy() * 255).astype(np.uint8)
    return cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)


# =============================================================================
# Subtitle Generation with LLM Descriptions
# =============================================================================

def generate_subtitle_track(
    result,
    fps: float = 30.0,
    transition_frames: int = 30,
    include_transitions: bool = True,
    api_key: Optional[str] = None,
    ply_path: Optional[str] = None,
    device: str = "cuda",
    llm_model: str = "gpt-4o-mini",
    render_width: int = 512,
    render_height: int = 512,
    fov_y: float = 60.0,
    save_anchor_images_dir: Optional[str] = None,
    language: str = "en",
) -> SubtitleTrack:
    lang_name = SUPPORTED_LANGUAGES.get(language, language)
    print(f"Generating subtitles in {lang_name} (code: {language})")

    track = SubtitleTrack(fps=fps)
    current_frame = 0

    use_llm = api_key is not None and ply_path is not None
    gaussians = None
    if use_llm:
        print("Loading Gaussians for anchor image rendering...")
        gaussians = _load_3dgs_ply_minimal(ply_path, device)
        if save_anchor_images_dir:
            anchor_dir = Path(save_anchor_images_dir)
            if anchor_dir.exists():
                import shutil
                shutil.rmtree(anchor_dir)
            anchor_dir.mkdir(parents=True, exist_ok=True)

    anchor_desc_cache: Dict[str, str] = {}

    def _get_description(anchor, movement_type, movement_category, extra_context=""):
        obj_label = getattr(anchor, 'object_label', '') if anchor else ''
        cache_key = f"{getattr(anchor, 'object_id', '?')}_{movement_type}"

        if cache_key in anchor_desc_cache:
            return anchor_desc_cache[cache_key]

        if use_llm and anchor is not None:
            image = render_anchor_image(
                anchor, gaussians, device, render_width, render_height, fov_y)
            if save_anchor_images_dir:
                img_path = Path(save_anchor_images_dir) / f"{cache_key}.jpg"
                cv2.imwrite(str(img_path), image)
            desc = describe_anchor_view(
                image, movement_type, movement_category, obj_label,
                api_key, llm_model, extra_context, language=language)
            print(f"  [{cache_key}] → \"{desc}\"")
        else:
            desc = _fallback_description(movement_type, obj_label, language=language)

        anchor_desc_cache[cache_key] = desc
        return desc

    prev_segment = None
    for i, seg in enumerate(result.segments):
        traj_out = seg.trajectory_output
        if traj_out is not None and 'c2w' in traj_out:
            n_seg_frames = len(traj_out['c2w'])
        else:
            n_seg_frames = transition_frames

        if include_transitions and i > 0 and transition_frames > 0:
            prev_label = ""
            cur_label = ""
            if prev_segment is not None and prev_segment.start_anchor:
                prev_label = getattr(prev_segment.start_anchor, 'object_label', '')
            if seg.start_anchor:
                cur_label = getattr(seg.start_anchor, 'object_label', '')

            dest_anchor = seg.start_anchor
            trans_desc = _get_description(
                dest_anchor, '_transition_interp', 'transition',
                extra_context=f"Camera is transitioning from {prev_label} toward {cur_label}.",
            )
            track.entries.append(SubtitleEntry(
                start_frame=current_frame,
                end_frame=current_frame + transition_frames,
                text=trans_desc,
                movement_type='_transition_interp',
                movement_category='transition',
                object_label=cur_label,
            ))
            current_frame += transition_frames

        obj_label = getattr(seg.start_anchor, 'object_label', '') if seg.start_anchor else ''
        end_label = getattr(seg.end_anchor, 'object_label', '') if seg.end_anchor else ''
        movement_cat = getattr(seg, 'movement_category', 'object-level')

        if movement_cat == 'transitional' and seg.end_anchor is not None:
            desc_anchor = seg.end_anchor
            extra = f"Camera is moving from {obj_label} toward {end_label}."
        else:
            desc_anchor = seg.start_anchor
            extra = ""

        text = _get_description(desc_anchor, seg.movement_type, movement_cat, extra)

        track.entries.append(SubtitleEntry(
            start_frame=current_frame,
            end_frame=current_frame + n_seg_frames,
            text=text,
            movement_type=seg.movement_type,
            movement_category=movement_cat,
            object_label=obj_label,
        ))
        current_frame += n_seg_frames
        prev_segment = seg

    track.total_frames = current_frame
    print(f"Generated {len(track.entries)} subtitle entries, {current_frame} total frames")
    return track


def _load_3dgs_ply_minimal(path: str, device: str) -> Dict[str, torch.Tensor]:
    plydata = PlyData.read(str(path))
    v = plydata['vertex']
    xyz = np.stack([v['x'], v['y'], v['z']], axis=1)
    SH_C0 = 0.28209479177387814
    colors = np.clip(0.5 + SH_C0 * np.stack([v['f_dc_0'], v['f_dc_1'], v['f_dc_2']], axis=1), 0, 1)
    opacities = 1 / (1 + np.exp(-v['opacity']))
    scales = np.exp(np.stack([v['scale_0'], v['scale_1'], v['scale_2']], axis=1))
    rots = np.stack([v['rot_0'], v['rot_1'], v['rot_2'], v['rot_3']], axis=1)
    rots = rots / np.linalg.norm(rots, axis=1, keepdims=True)
    dev = device if torch.cuda.is_available() else 'cpu'
    return {
        'means': torch.tensor(xyz, dtype=torch.float32, device=dev),
        'colors': torch.tensor(colors, dtype=torch.float32, device=dev),
        'opacities': torch.tensor(opacities, dtype=torch.float32, device=dev),
        'scales': torch.tensor(scales, dtype=torch.float32, device=dev),
        'quats': torch.tensor(rots, dtype=torch.float32, device=dev),
    }


# =============================================================================
# Subtitle Rendering (burn-in) — PIL-based for full Unicode/CJK support
# =============================================================================

from PIL import Image, ImageDraw, ImageFont

_FONT_SEARCH_PATHS = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/System/Library/Fonts/Helvetica.ttc",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/arial.ttf",
]

_font_cache: Dict[int, ImageFont.FreeTypeFont] = {}


def _find_system_font() -> Optional[str]:
    for p in _FONT_SEARCH_PATHS:
        if Path(p).exists():
            return p
    return None


def _get_font(size: int) -> ImageFont.FreeTypeFont:
    if size in _font_cache:
        return _font_cache[size]
    font_path = _find_system_font()
    if font_path:
        try:
            font = ImageFont.truetype(font_path, size)
            _font_cache[size] = font
            return font
        except Exception:
            pass
    font = ImageFont.load_default()
    _font_cache[size] = font
    return font


@dataclass
class SubtitleStyle:
    font_size: int = 28
    badge_font_size: int = 18
    color: Tuple[int, int, int] = (255, 255, 255)
    shadow_color: Tuple[int, int, int] = (0, 0, 0)
    shadow_offset: int = 2
    bg_color: Tuple[int, int, int] = (0, 0, 0)
    bg_alpha: float = 0.55
    margin_bottom: int = 40
    padding_x: int = 20
    padding_y: int = 12
    show_movement_badge: bool = True
    badge_color: Tuple[int, int, int] = (180, 180, 180)


def _pil_text_size(draw, text, font):
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def draw_subtitle_on_frame(frame, entry, style=None, frame_idx=0,
                           total_frames=0, show_progress=True):
    if style is None:
        style = SubtitleStyle()
    h, w = frame.shape[:2]

    if show_progress and total_frames > 0:
        bar_h = 3
        progress = frame_idx / max(total_frames - 1, 1)
        frame[h - bar_h:h, :int(w * progress)] = (100, 200, 100)

    if entry is None:
        return frame

    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
    overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    main_font = _get_font(style.font_size)
    badge_font = _get_font(style.badge_font_size)

    text = entry.text
    tw, th = _pil_text_size(draw, text, main_font)

    badge_text = ""
    btw, bth = 0, 0
    if style.show_movement_badge and entry.movement_type and not entry.movement_type.startswith('_'):
        badge_text = entry.movement_type.replace('_', ' ')
        btw, bth = _pil_text_size(draw, badge_text, badge_font)

    total_text_h = th + (bth + 6 if badge_text else 0)
    box_w = max(tw, btw) + 2 * style.padding_x
    box_h = total_text_h + 2 * style.padding_y
    x0 = (w - box_w) // 2
    y0 = h - style.margin_bottom - box_h
    x1 = x0 + box_w
    y1 = y0 + box_h

    bg_rgba = style.bg_color + (int(255 * style.bg_alpha),)
    draw.rectangle([(x0, y0), (x1, y1)], fill=bg_rgba)

    text_x = x0 + (box_w - tw) // 2
    text_y = y0 + style.padding_y

    draw.text((text_x + style.shadow_offset, text_y + style.shadow_offset),
              text, font=main_font, fill=style.shadow_color + (220,))
    draw.text((text_x, text_y), text, font=main_font, fill=style.color + (255,))

    if badge_text:
        badge_x = x0 + (box_w - btw) // 2
        badge_y = text_y + th + 6
        draw.text((badge_x, badge_y), badge_text, font=badge_font,
                  fill=style.badge_color + (255,))

    composited = Image.alpha_composite(pil_img, overlay).convert("RGB")
    result = cv2.cvtColor(np.array(composited), cv2.COLOR_RGB2BGR)
    np.copyto(frame, result)
    return frame


# =============================================================================
# Render Config & Renderer
# =============================================================================

@dataclass
class RenderConfig:
    width: int = 1920
    height: int = 1080
    fps: float = 30.0
    fov_y: float = 60.0
    gaussian_scale: float = 1.0
    show_bboxes: bool = False
    bbox_line_width: int = 2
    output_format: str = "mp4"
    codec: str = CODEC
    show_subtitles: bool = True
    subtitle_style: Optional[SubtitleStyle] = None
    show_progress_bar: bool = True


class TrajectoryRenderer:
    def __init__(self, ply_path, bbox_json_path=None, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"Loading Gaussians from {ply_path}...")
        self.gaussians = _load_3dgs_ply_minimal(str(ply_path), self.device)
        print(f"Loaded {self.gaussians['means'].shape[0]} Gaussians")

        self.bboxes = None
        self._bbox_format = None
        if bbox_json_path:
            self.bboxes, self._bbox_format = _load_bboxes_for_renderer(bbox_json_path)
            print(f"Loaded {len(self.bboxes)} bounding boxes (format: {self._bbox_format})")

    def _render_gaussians(self, viewmat, K, width, height, gaussian_scale=1.0):
        g = self.gaussians.copy()
        if gaussian_scale != 1.0:
            g['scales'] = self.gaussians['scales'] * gaussian_scale
        renders, _, _ = rasterization(
            means=g['means'], quats=g['quats'], scales=g['scales'],
            opacities=g['opacities'], colors=g['colors'],
            viewmats=viewmat[None], Ks=K[None],
            width=width, height=height, packed=False, render_mode="RGB",
        )
        return (renders[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)

    def _project_point(self, point, viewmat, K):
        ph = np.append(point, 1.0)
        pc = viewmat @ ph
        if pc[2] <= 0:
            return None
        pp = K @ pc[:3]
        return pp[:2] / pp[2]

    def _draw_bbox_on_image(self, image, corners, viewmat, K,
                            color=(0, 255, 0), line_width=2):
        """
        Draw OBB wireframe on image.

        Args:
            corners: (8, 3) numpy array of world-space corners
        """
        proj = [self._project_point(c, viewmat, K) for c in corners]

        # Edge connectivity for obb_to_corners sign ordering
        edges = [
            (0, 2), (2, 6), (6, 4), (4, 0),  # bottom
            (1, 3), (3, 7), (7, 5), (5, 1),  # top
            (0, 1), (2, 3), (4, 5), (6, 7),  # vertical
        ]

        h, w = image.shape[:2]
        m = 100
        for s, e in edges:
            p1, p2 = proj[s], proj[e]
            if p1 is not None and p2 is not None:
                pt1, pt2 = (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1]))
                if (-m < pt1[0] < w+m and -m < pt1[1] < h+m and
                    -m < pt2[0] < w+m and -m < pt2[1] < h+m):
                    cv2.line(image, pt1, pt2, color, line_width)
        return image

    def _draw_bbox_on_image_legacy(self, image, bbox_points, viewmat, K,
                                   color=(0, 255, 0), line_width=2):
        """Legacy: draw bbox from list of 8 dicts with x,y,z."""
        corners = np.array([[p['x'], p['y'], p['z']] for p in bbox_points])
        proj = [self._project_point(c, viewmat, K) for c in corners]
        edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
        h, w = image.shape[:2]
        m = 100
        for s, e in edges:
            p1, p2 = proj[s], proj[e]
            if p1 is not None and p2 is not None:
                pt1, pt2 = (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1]))
                if (-m < pt1[0] < w+m and -m < pt1[1] < h+m and
                    -m < pt2[0] < w+m and -m < pt2[1] < h+m):
                    cv2.line(image, pt1, pt2, color, line_width)
        return image

    def _get_label_color(self, label):
        cmap = {
            'door': (100,100,255), 'window': (255,100,100), 'wardrobe': (100,255,100),
            'curtain': (100,200,255), 'cabinet': (100,200,100),
            'table': (255,255,100), 'chair': (100,255,255), 'sofa': (255,150,200),
            'tv': (200,150,255), 'refrigerator': (150,255,200),
            'kitchen_counter': (200,200,150), 'sink': (100,200,255),
            'picture': (255,100,255), 'plant': (50,200,50),
        }
        return cmap.get(label, (200, 200, 200))

    def pose_to_matrices(self, pose, width, height, fx=None, fy=None):
        c2w = np.eye(4); c2w[:3, :3] = pose.rotation; c2w[:3, 3] = pose.position
        w2c = np.linalg.inv(c2w)
        viewmat = torch.tensor(w2c, dtype=torch.float32, device=self.device)
        if fx is None: fx = width
        if fy is None: fy = fx
        K = torch.tensor([[fx,0,width/2],[0,fy,height/2],[0,0,1]],
                         dtype=torch.float32, device=self.device)
        return viewmat, K, w2c

    def render_frame(self, pose, config, fx=None, fy=None):
        viewmat, K, w2c_np = self.pose_to_matrices(pose, config.width, config.height, fx, fy)
        K_np = K.cpu().numpy()
        with torch.no_grad():
            image = self._render_gaussians(viewmat, K, config.width, config.height,
                                           config.gaussian_scale)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if config.show_bboxes and self.bboxes:
            if self._bbox_format == 'scene_graph':
                for item in self.bboxes:
                    color = self._get_label_color(item.get('label', ''))
                    self._draw_bbox_on_image(
                        image, item['corners'], w2c_np, K_np,
                        color, config.bbox_line_width)
            else:
                for item in self.bboxes:
                    if 'bounding_box' not in item:
                        continue
                    color = self._get_label_color(item.get('label', ''))
                    self._draw_bbox_on_image_legacy(
                        image, item['bounding_box'], w2c_np, K_np,
                        color, config.bbox_line_width)
        return image

    def render_trajectory(self, trajectory, output_path, config=None,
                          subtitle_track=None):
        if config is None:
            config = RenderConfig()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fov_y_rad = np.radians(config.fov_y)
        base_fy = config.height / (2 * np.tan(fov_y_rad / 2))
        base_fx = base_fy

        n_frames = len(trajectory)
        print(f"Rendering {n_frames} frames at {config.width}x{config.height}...")
        if subtitle_track:
            print(f"  Subtitle track: {len(subtitle_track.entries)} entries")

        has_focal = (hasattr(trajectory, 'focal_multipliers')
                     and trajectory.focal_multipliers is not None
                     and len(trajectory.focal_multipliers) == n_frames)
        if has_focal:
            zoom_count = int(np.sum(np.abs(trajectory.focal_multipliers - 1.0) > 1e-6))
            if zoom_count > 0:
                print(f"  Applying per-frame focal zoom to {zoom_count}/{n_frames} frames "
                      f"(range [{trajectory.focal_multipliers.min():.2f}×, "
                      f"{trajectory.focal_multipliers.max():.2f}×])")

        sub_style = config.subtitle_style or SubtitleStyle()

        if config.output_format == 'frames':
            fdir = output_path.parent / f"{output_path.stem}_frames"
            fdir.mkdir(parents=True, exist_ok=True)
            for i, pose in enumerate(tqdm(trajectory.poses, desc="Rendering frames")):
                fm = float(trajectory.focal_multipliers[i]) if has_focal else 1.0
                frame = self.render_frame(pose, config, base_fx * fm, base_fy * fm)
                if config.show_subtitles and subtitle_track:
                    entry = subtitle_track.get_subtitle_at_frame(i)
                    draw_subtitle_on_frame(frame, entry, sub_style, i, n_frames,
                                           config.show_progress_bar)
                cv2.imwrite(str(fdir / f"frame_{i:06d}.png"), frame)
            print(f"Saved {n_frames} frames to {fdir}")
        else:
            fourcc = cv2.VideoWriter_fourcc(*config.codec)
            out = cv2.VideoWriter(str(output_path), fourcc, config.fps,
                                  (config.width, config.height))
            for i, pose in enumerate(tqdm(trajectory.poses, desc="Rendering video")):
                fm = float(trajectory.focal_multipliers[i]) if has_focal else 1.0
                frame = self.render_frame(pose, config, base_fx * fm, base_fy * fm)
                if config.show_subtitles and subtitle_track:
                    entry = subtitle_track.get_subtitle_at_frame(i)
                    draw_subtitle_on_frame(frame, entry, sub_style, i, n_frames,
                                           config.show_progress_bar)
                out.write(frame)
            out.release()
            # Re-encode to H.264 for browser compatibility
            import subprocess
            h264_path = str(output_path) + ".h264.mp4"
            ret = subprocess.run([
                "ffmpeg", "-y", "-i", str(output_path),
                "-c:v", "libx264", "-crf", "23",
                "-c:a", "aac", "-movflags", "+faststart",
                h264_path
            ], capture_output=True)
            if ret.returncode == 0:
                import shutil
                shutil.move(h264_path, str(output_path))
                print(f"Saved video to {output_path} (H.264, {n_frames / config.fps:.1f}s)")
            else:
                print(f"Saved video to {output_path} ({n_frames / config.fps:.1f}s)")
                print(f"  ⚠ H.264 re-encode failed. Run manually:")
                print(f"  ffmpeg -y -i \"{output_path}\" -c:v libx264 -crf 23 -movflags +faststart \"{output_path}\"")


# =============================================================================
# Post-hoc subtitle burn-in
# =============================================================================

def burn_subtitles_onto_video(input_video, output_video, subtitle_track,
                              style=None, show_progress=True):
    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {input_video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    Path(output_video).parent.mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*CODEC), fps, (w, h))
    if style is None:
        style = SubtitleStyle()

    for idx in tqdm(range(total), desc="Burning subtitles"):
        ret, frame = cap.read()
        if not ret:
            break
        entry = subtitle_track.get_subtitle_at_frame(idx)
        draw_subtitle_on_frame(frame, entry, style, idx, total, show_progress)
        out.write(frame)
    cap.release()
    out.release()
    print(f"Saved to {output_video}")


# =============================================================================
# Trajectory loader
# =============================================================================

def load_trajectory_from_json(json_path):
    with open(json_path) as f:
        data = json.load(f)
    intrinsics = CameraIntrinsics(
        width=data.get('w', 512.0), height=data.get('h', 512.0),
        fx=data.get('fl_x', 256.0), fy=data.get('fl_y', 256.0),
        cx=data.get('cx', 256.0), cy=data.get('cy', 256.0),
    )
    poses, positions, rotations = [], [], []
    focal_multipliers = []
    for i, frame in enumerate(data.get('frames', [])):
        T = np.array(frame['transform_matrix'])
        rot, pos = T[:3, :3], T[:3, 3]
        poses.append(CameraPose(pos.copy(), rot.copy(), i / 30.0))
        positions.append(pos); rotations.append(rot)
        focal_multipliers.append(frame.get('focal_multiplier', 1.0))

    fm_array = np.array(focal_multipliers, dtype=np.float64)
    has_zoom = np.any(np.abs(fm_array - 1.0) > 1e-6)

    traj = CameraTrajectory(
        poses=poses, timestamps=np.arange(len(poses)) / 30.0,
        positions=np.array(positions) if positions else np.array([]),
        rotations=np.array(rotations) if rotations else np.array([]),
        intrinsics=intrinsics,
    )
    if has_zoom:
        traj.focal_multipliers = fm_array
    return traj


# =============================================================================
# Main
# =============================================================================

def main():
    import argparse

    scene_id = "09c1414f1b"

    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", default=f"data/ScanNetpp/scenes/{scene_id}/dslr/ply/point_cloud_30000.ply")
    parser.add_argument("--trajectory", default=f"outputs/scannetpp/{scene_id}/combined_trajectory.json")
    parser.add_argument("--output", default=f"outputs/scannetpp/{scene_id}/rendered_video.mp4")
    parser.add_argument("--bbox", default=f"data/ScanNetpp/scenes/{scene_id}/dslr/sg/{scene_id}-simple.json")
    parser.add_argument("--subtitle-json", default=None)
    parser.add_argument("--language", default="en",
        choices=list(SUPPORTED_LANGUAGES.keys()),
        help=f"Subtitle language: {', '.join(f'{k}={v}' for k,v in SUPPORTED_LANGUAGES.items())}")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fov", type=float, default=60.0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--show-bboxes", action="store_true")
    parser.add_argument("--no-subtitles", action="store_true")
    parser.add_argument("--gaussian-scale", type=float, default=1.0)
    parser.add_argument("--format", default="mp4", choices=["mp4", "avi", "frames"])
    args = parser.parse_args()

    renderer = TrajectoryRenderer(ply_path=args.ply, bbox_json_path=args.bbox)
    trajectory = load_trajectory_from_json(args.trajectory)
    print(f"Loaded trajectory: {len(trajectory)} poses")

    sub_track = None
    if args.subtitle_json:
        sub_track = load_subtitle_track(args.subtitle_json)

    config = RenderConfig(
        width=args.width, height=args.height, fps=args.fps, fov_y=args.fov,
        gaussian_scale=args.gaussian_scale, show_bboxes=args.show_bboxes,
        output_format=args.format, show_subtitles=not args.no_subtitles,
    )
    renderer.render_trajectory(trajectory, args.output, config, sub_track)


if __name__ == "__main__":
    main()