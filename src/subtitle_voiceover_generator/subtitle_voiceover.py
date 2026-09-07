"""
Dual-style subtitle & voiceover generator for TrajScene videos (v4).

KEY DESIGN (v4.1): Subtitles and voiceover are 1:1 SYNCHRONIZED.
  - Each time segment (~4s) gets exactly ONE subtitle AND one voiceover line.
  - Voiceover is a spoken expansion of the subtitle, tightly synced to what's
    on screen — no lag, no void, no desync.
  - TTS clips that exceed their window are DROPPED (never trimmed mid-sentence).

Generates up to THREE versions:
  Style A - Real Estate Agent (Chinese)
  Style B - Douyin Home Decor (Chinese)
  Style C - English Real Estate Agent

Dependencies: pip install openai opencv-python-headless numpy
Requires ffmpeg on PATH.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from openai import OpenAI


# ===================================================================
# Data classes - subtitles and voiceover are 1:1 SYNCHRONIZED
# ===================================================================
@dataclass
class SubtitleEntry:
    index: int
    start_sec: float
    end_sec: float
    text: str


@dataclass
class VoiceoverBlock:
    index: int
    start_sec: float
    end_sec: float
    text: str

    @property
    def duration(self) -> float:
        return self.end_sec - self.start_sec


@dataclass
class NarrationResult:
    subtitles: list[SubtitleEntry] = field(default_factory=list)
    voiceover_blocks: list[VoiceoverBlock] = field(default_factory=list)

    def save_srt(self, path: str | Path) -> None:
        lines: list[str] = []
        for e in self.subtitles:
            lines.append(str(e.index))
            lines.append(f"{_fmt_srt_ts(e.start_sec)} --> {_fmt_srt_ts(e.end_sec)}")
            lines.append(_strip_emoji(e.text))
            lines.append("")
        Path(path).write_text("\n".join(lines), encoding="utf-8")

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({
            "subtitles": [{"index": e.index, "start": e.start_sec, "end": e.end_sec, "text": e.text} for e in self.subtitles],
            "voiceover_blocks": [{"index": b.index, "start": b.start_sec, "end": b.end_sec, "text": b.text} for b in self.voiceover_blocks],
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    def save_script(self, path: str | Path, label: str = "") -> None:
        lines = ["=" * 60, label, "=" * 60, "", "-- SUBTITLES --"]
        for e in self.subtitles:
            lines.append(f"  [{_fmt_ts_short(e.start_sec)}-{_fmt_ts_short(e.end_sec)}] {e.text}")
        lines.extend(["", "-- VOICEOVER (1:1 per segment) --"])
        for b in self.voiceover_blocks:
            lines.extend([f"  Block {b.index} [{_fmt_ts_short(b.start_sec)}-{_fmt_ts_short(b.end_sec)}] ({b.duration:.1f}s)", f"    {b.text}", ""])
        Path(path).write_text("\n".join(lines), encoding="utf-8")

    def full_voiceover_text(self) -> str:
        return " ".join(b.text for b in self.voiceover_blocks)


def _fmt_srt_ts(seconds: float) -> str:
    h, m, s = int(seconds // 3600), int((seconds % 3600) // 60), int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def _fmt_ts_short(seconds: float) -> str:
    return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"

def _strip_emoji(text: str) -> str:
    return re.sub(r'[\U0001F300-\U0001F9FF\U0001FA00-\U0001FAFF\U00002702-\U000027B0\U0000FE00-\U0000FE0F\U0000200D\U000020E3\U00002600-\U000026FF\U0000231A-\U0000231B\U00002934-\U00002935\U000025AA-\U000025FE\U00002B05-\U00002B55\U00003030\U0000303D\U00003297\U00003299]+', '', text).strip()


# ===================================================================
# Voiceover length validation — sentence-boundary safe
# ===================================================================
def _count_spoken_units(text: str, lang: str) -> int:
    if lang == "zh":
        return len(re.findall(r'[\u4e00-\u9fff\u3400-\u4dbf]', text)) + len(re.findall(r'[a-zA-Z]+|\d+', text))
    return len(text.split())


def _truncate_at_sentence_boundary(text: str, max_units: int, lang: str) -> str:
    """Truncate text at the last COMPLETE sentence that fits within max_units.

    Unlike the old _truncate_voiceover, this NEVER cuts mid-clause.
    It only keeps whole sentences (ending with . ! ? or Chinese equivalents).
    If not even one sentence fits, returns empty string (block will be dropped).
    """
    if _count_spoken_units(text, lang) <= max_units:
        return text

    if lang == "zh":
        # Split on sentence-ending punctuation (period, !, ?)
        # NOT on commas — commas create clause fragments like "where..."
        sentences = re.split(r'(?<=[\u3002\uff01\uff1f])', text)
        sentences = [s for s in sentences if s.strip()]
    else:
        # Split on sentence-ending punctuation
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s for s in sentences if s.strip()]

    # Greedily add complete sentences
    result = ""
    for sent in sentences:
        candidate = result + sent if lang == "zh" else (result + " " + sent).strip()
        if _count_spoken_units(candidate, lang) > max_units:
            break
        result = candidate

    return result.strip()


def validate_voiceover_blocks(blocks, lang="zh", chars_per_sec_zh=4.0, words_per_sec_en=2.3):
    """Validate voiceover blocks. Truncate at sentence boundaries only.
    Returns list of block indices that should be dropped (empty after truncation)."""
    drop_indices = []
    for block in blocks:
        max_units = max(int(block.duration * (chars_per_sec_zh if lang == "zh" else words_per_sec_en)), 6)
        current = _count_spoken_units(block.text, lang)
        if current > max_units:
            truncated = _truncate_at_sentence_boundary(block.text, max_units, lang)
            if not truncated:
                # No complete sentence fits — mark for drop
                drop_indices.append(block.index)
                print(f"    Warning: Block {block.index} [{block.start_sec:.1f}-{block.end_sec:.1f}s] "
                      f"no complete sentence fits ({current} units, budget {max_units}) — will drop")
            else:
                new_count = _count_spoken_units(truncated, lang)
                print(f"    Warning: Block {block.index} [{block.start_sec:.1f}-{block.end_sec:.1f}s] "
                      f"trimmed at sentence boundary: {current} -> {new_count} units (budget: {max_units})")
                block.text = truncated
    return drop_indices


# ===================================================================
# Prompt texts — 1:1 voiceover per segment, self-contained sentences
# ===================================================================
_REAL_ESTATE_PROMPT = """你是一位经验丰富的高端房产经纪人，正在为VIP客户做一对一视频带看。
你会收到一段室内3D漫游视频，已按时间分成若干段（Segment），每段包含该时间区间的关键帧。

重要：每条字幕和旁白必须只描述其对应Segment中关键帧里实际可见的内容。
不要提前描述后面Segment的内容，不要描述帧中看不到的物体。

输出格式：

输出一个JSON对象，包含两个数组：

{
  "subtitles": [
    {"start": float, "end": float, "text": "短字幕文本"}
  ],
  "voiceover_blocks": [
    {"start": float, "end": float, "text": "口播旁白"}
  ]
}

字幕（subtitles）：
- 每条字幕的start/end必须与对应Segment的时间边界完全一致
- 精炼的屏幕文字，10-18个中文字
- 覆盖整个视频，每个Segment恰好一条字幕

旁白（voiceover_blocks）：
- 每个Segment恰好一条旁白，start/end与该Segment完全一致
- 旁白数量与字幕数量完全相同，一一对应
- 每条旁白是一个完整的句子，用句号结尾
- 绝对不要写半句话或以逗号、"的"、"和"、"在"等结尾

风格指南：
字幕：精炼、信息密度高，突出核心卖点
旁白：
- 自然口语，像跟客户面对面讲解
- 每条都是独立完整的句子，句号结尾
- 描述当前画面可见的内容

旁白字数规则：中文TTS语速约每秒4个字。
公式：旁白字数 <= (end - start) * 4
4秒的段落最多16个字。请写短小精悍的完整句子。

关键要求：
- 字幕和旁白数量相同，与Segment一一对应
- 每条旁白必须是语法完整的句子，以句号结尾
- 不要写从句、半句话或开放式结尾
- 请仅输出JSON对象，不要包含markdown代码块或任何其他文字"""

_DOUYIN_PROMPT = """你是抖音/小红书上百万粉丝的家居博主，正在拍一条沉浸式家居vlog。
你会收到一段室内3D漫游视频，已按时间分成若干段（Segment），每段包含该时间区间的关键帧。

重要：每条字幕和旁白必须只描述其对应Segment中关键帧里实际可见的内容。

输出格式：

输出一个JSON对象，包含两个数组：

{
  "subtitles": [{"start": float, "end": float, "text": "短字幕"}],
  "voiceover_blocks": [{"start": float, "end": float, "text": "口播文案"}]
}

字幕：每条字幕的start/end必须与对应Segment的时间边界完全一致，
口语化有hook感，8-16个中文字，不要emoji，每个Segment恰好一条

旁白：每个Segment恰好一条旁白，start/end与该Segment完全一致
旁白数量与字幕数量完全相同，一一对应
每条旁白必须是完整的句子，以句号或感叹号结尾

风格：
字幕：口语化，有情绪爆点
旁白：像跟粉丝聊天，情绪饱满

旁白字数：中文TTS约每秒4.5字。公式：字数 <= (end-start)*4.5
4秒的段落最多18个字。写短小有力的完整句子。

关键要求：字幕和旁白数量相同与Segment一一对应，
每条旁白必须是语法完整的句子不要半句话，
基于对应Segment关键帧的实际画面，仅输出JSON对象"""

_ENGLISH_PROMPT = """You are a seasoned luxury real estate agent giving a private video walkthrough to a VIP client.
You will receive an indoor 3D tour video divided into time Segments, each containing keyframes from that time range.

IMPORTANT: Each subtitle and voiceover must ONLY describe what is actually visible in the keyframes of its corresponding Segment.
Do NOT describe content from later Segments ahead of time.

OUTPUT FORMAT:

Output a JSON object with two arrays:

{
  "subtitles": [{"start": float, "end": float, "text": "short on-screen text"}],
  "voiceover_blocks": [{"start": float, "end": float, "text": "spoken narration"}]
}

Subtitles: Each subtitle's start/end must exactly match the corresponding Segment time boundaries.
Punchy highlight, 6-12 words. Exactly one per Segment.

Voiceover blocks: Exactly ONE per Segment, with start/end matching that Segment exactly.
The number of voiceover blocks MUST equal the number of subtitles — one-to-one.
Each voiceover line must be a COMPLETE, self-contained sentence ending with a period.
NEVER write a dangling clause like "where..." or "with its..." or "featuring...".

STYLE: Subtitles concise and highlight-driven.
Voiceover: warm, professional spoken English. Each line is a standalone sentence.

WORD COUNT: English TTS ~2.5 words/sec.
Formula: words <= (end-start) * 2.5
A 4-second segment allows max 10 words. Write short, complete sentences.

CRITICAL REQUIREMENTS:
- Subtitles and voiceover blocks must be the SAME count, one per Segment
- Every voiceover line must be a grammatically COMPLETE sentence ending with a period
- NEVER end a sentence with a comma, subordinate clause, or dangling modifier
- NEVER write things like: "We enter the living room, where..." (incomplete!)
- INSTEAD write: "This is the spacious living room." (complete!)
- Output ONLY the JSON object. No markdown, no extra text."""



# ===================================================================
# Style definitions
# ===================================================================
STYLE_CONFIGS = {
    "real_estate": {
        "label": "Real Estate Agent (Chinese)",
        "folder": "real_estate",
        "tts_voice": "onyx",
        "tts_speed": 1.14,
        "lang": "zh",
        "subtitle_style": "FontSize=24,FontName=Noto Sans CJK SC,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,Outline=2,Shadow=1,MarginV=50,Alignment=2",
        "system_prompt": _REAL_ESTATE_PROMPT,
    },
    "douyin": {
        "label": "Douyin Home Decor (Chinese)",
        "folder": "douyin",
        "tts_voice": "nova",
        "tts_speed": 1.26,
        "lang": "zh",
        "subtitle_style": "FontSize=28,FontName=Noto Sans CJK SC,PrimaryColour=&H0000FFFF,OutlineColour=&H00000000,Outline=3,Shadow=2,MarginV=55,Alignment=2,Bold=1",
        "system_prompt": _DOUYIN_PROMPT,
    },
    "english": {
        "label": "English Real Estate Walkthrough",
        "folder": "english",
        "tts_voice": "alloy",
        "tts_speed": 1.2,
        "lang": "en",
        "subtitle_style": "FontSize=24,FontName=Arial,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,Outline=2,Shadow=1,MarginV=50,Alignment=2",
        "system_prompt": _ENGLISH_PROMPT,
    },
}


# ===================================================================
# Keyframe extraction
# ===================================================================
def extract_keyframes(video_path, sample_fps=2.0, max_dim=512):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration = total_frames / video_fps
    num_samples = max(1, int(duration * sample_fps))
    sample_timestamps = np.linspace(0, duration, num_samples, endpoint=False)
    indices = np.clip((sample_timestamps * video_fps).astype(int), 0, total_frames - 1)
    keyframes = []
    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        h, w = frame.shape[:2]
        scale = min(max_dim / max(h, w), 1.0)
        if scale < 1.0:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        keyframes.append({"frame_idx": int(idx), "timestamp_sec": round(sample_timestamps[i], 2), "image_b64": base64.b64encode(buf.tobytes()).decode()})
    cap.release()
    print(f"    Extracted {len(keyframes)} frames at {sample_fps} fps from {duration:.1f}s video ({video_fps:.0f} fps source)")
    return keyframes, duration


# ===================================================================
# Vision model -> 1:1 subtitles + voiceover blocks
# ===================================================================
def _compute_segments(keyframes, duration, target_seg_sec=4.0):
    """Pre-compute time segments (~4s each) and assign keyframes to each."""
    n_segs = max(1, round(duration / target_seg_sec))
    seg_dur = duration / n_segs
    segments = []
    for i in range(n_segs):
        seg_start = round(i * seg_dur, 2)
        seg_end = round((i + 1) * seg_dur, 2) if i < n_segs - 1 else round(duration, 2)
        seg_frames = [kf for kf in keyframes if seg_start <= kf["timestamp_sec"] < seg_end]
        if i == n_segs - 1:
            seg_frames = [kf for kf in keyframes if kf["timestamp_sec"] >= seg_start]
        segments.append({"index": i + 1, "start": seg_start, "end": seg_end, "frames": seg_frames})
    return segments


def generate_narration(client, keyframes, duration, style_config, model="qwen3-omni-flash"):
    segments = _compute_segments(keyframes, duration, target_seg_sec=4.0)

    seg_summary_lines = []
    for seg in segments:
        n_frames = len(seg["frames"])
        seg_dur = seg["end"] - seg["start"]
        seg_summary_lines.append(
            f"  Segment {seg['index']}: {seg['start']:.1f}s - {seg['end']:.1f}s ({seg_dur:.1f}s, {n_frames} frames)"
        )
    seg_summary = "\n".join(seg_summary_lines)

    content = [{"type": "text", "text": (
        f"The video is {duration:.1f} seconds long, divided into {len(segments)} segments.\n"
        f"Each segment has a fixed time window. You must produce exactly ONE subtitle AND "
        f"ONE voiceover line per segment, using that segment's start/end times.\n"
        f"The voiceover must be a COMPLETE sentence (ending with a period). "
        f"Never write half-sentences or dangling clauses.\n\n"
        f"Segment layout:\n{seg_summary}\n\n"
        f"Below are the keyframes grouped by segment."
    )}]

    for seg in segments:
        seg_dur = seg["end"] - seg["start"]
        content.append({"type": "text", "text": (
            f"\n{'='*40}\n"
            f"Segment {seg['index']} ({seg['start']:.1f}s - {seg['end']:.1f}s, {seg_dur:.1f}s)\n"
            f"Write ONE subtitle + ONE voiceover for: start={seg['start']:.1f}, end={seg['end']:.1f}\n"
            f"Voiceover must be a COMPLETE sentence. Max ~{int(seg_dur * 2.5)} words.\n"
            f"{'='*40}"
        )})
        if seg["frames"]:
            for kf in seg["frames"]:
                content.append({"type": "text", "text": f"[t = {kf['timestamp_sec']:.1f}s]"})
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{kf['image_b64']}", "detail": "low"}})
        else:
            content.append({"type": "text", "text": "(no keyframe — interpolate from neighbors)"})

    print(f"    Calling {model} ({style_config['label']}) with {len(segments)} segments (1:1 mode) ...")
    _seg_info = ", ".join(f"{s['start']:.1f}-{s['end']:.1f}s({len(s['frames'])}f)" for s in segments)
    print(f"    Segments: {_seg_info}")
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": style_config["system_prompt"]}, {"role": "user", "content": content}],
        temperature=0.7, max_tokens=4096,
    )
    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    data = json.loads(raw)

    result = NarrationResult()
    for i, s in enumerate(data["subtitles"], start=1):
        result.subtitles.append(SubtitleEntry(index=i, start_sec=float(s["start"]), end_sec=float(s["end"]), text=str(s["text"])))
    for i, v in enumerate(data["voiceover_blocks"], start=1):
        result.voiceover_blocks.append(VoiceoverBlock(index=i, start_sec=float(v["start"]), end_sec=float(v["end"]), text=str(v["text"])))

    # Validate: sentence-boundary-safe truncation
    lang = style_config.get("lang", "zh")
    drop_indices = validate_voiceover_blocks(result.voiceover_blocks, lang=lang)
    if drop_indices:
        print(f"    Dropping {len(drop_indices)} voiceover blocks (no complete sentence fits)")
        result.voiceover_blocks = [b for b in result.voiceover_blocks if b.index not in set(drop_indices)]

    print(f"    Generated {len(result.subtitles)} subtitles + {len(result.voiceover_blocks)} voiceover blocks")
    for s in result.subtitles:
        print(f"      Sub [{_fmt_ts_short(s.start_sec)}-{_fmt_ts_short(s.end_sec)}] {s.text}")
    for b in result.voiceover_blocks:
        units = _count_spoken_units(b.text, lang)
        print(f"      VO  [{_fmt_ts_short(b.start_sec)}-{_fmt_ts_short(b.end_sec)}] ({b.duration:.1f}s, {units}u) {b.text}")
    return result


# ===================================================================
# Audio helpers
# ===================================================================
def _get_audio_duration(path):
    result = subprocess.run(["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", path], capture_output=True, text=True)
    return float(result.stdout.strip())

def _fit_audio_to_duration(input_path, output_path, target_sec, max_speedup=1.15):
    """Speed up audio to fit within target_sec. Only mild speedup allowed."""
    actual_dur = _get_audio_duration(input_path)
    if actual_dur <= target_sec + 0.1:
        if input_path != output_path:
            shutil.copy2(input_path, output_path)
        return 1.0
    ratio = actual_dur / target_sec
    if ratio <= max_speedup:
        subprocess.run(["ffmpeg", "-y", "-i", input_path, "-af", f"atempo={ratio:.4f}", output_path], check=True, capture_output=True)
        return ratio
    else:
        # Fallback (shouldn't happen — overlong blocks are pre-dropped)
        subprocess.run(["ffmpeg", "-y", "-i", input_path, "-af", f"atempo={max_speedup:.4f}", output_path], check=True, capture_output=True)
        return max_speedup

def _pad_audio_to_duration(input_path, output_path, target_sec):
    actual_dur = _get_audio_duration(input_path)
    if actual_dur < target_sec:
        pad_dur = target_sec - actual_dur
        subprocess.run(["ffmpeg", "-y", "-i", input_path, "-af", f"apad=pad_dur={pad_dur:.4f}", "-t", f"{target_sec:.4f}", "-ar", "44100", "-ac", "2", "-sample_fmt", "s16", output_path], check=True, capture_output=True)
    else:
        subprocess.run(["ffmpeg", "-y", "-i", input_path, "-t", f"{target_sec:.4f}", "-ar", "44100", "-ac", "2", "-sample_fmt", "s16", output_path], check=True, capture_output=True)

def _generate_silence_wav(output_path, duration_sec):
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo", "-t", f"{duration_sec:.4f}", "-c:a", "pcm_s16le", output_path], check=True, capture_output=True)


def generate_voiceover_single(client, result, output_path, voice="alloy", model="tts-1-hd", speed=1.0):
    narration = result.full_voiceover_text()
    print(f"    Synthesising full voiceover ({voice}) ...")
    response = client.audio.speech.create(model=model, voice=voice, input=narration, speed=speed, response_format="mp3")
    out = Path(output_path)
    response.stream_to_file(str(out))
    print(f"    Voiceover -> {out} ({out.stat().st_size / 1024:.0f} KB)")
    return out


def generate_voiceover_per_block(client, result, work_dir, voice="alloy", model="tts-1-hd", speed=1.0):
    """One TTS clip per voiceover block (1:1 with segments). Drift-free via pad+concat.

    Blocks whose TTS audio cannot fit within a mild 1.15x speedup are
    DROPPED entirely — never trimmed mid-sentence.
    """
    work = Path(work_dir)
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    padded_files = []
    blocks = list(result.voiceover_blocks)  # work on a copy
    n = len(blocks)
    print(f"    Generating {n} TTS blocks ({voice}, speed={speed}) ...")

    # ---------------------------------------------------------------
    # Two-pass: generate all TTS first, then drop blocks that overflow
    # ---------------------------------------------------------------
    max_speedup = 1.15
    raw_durations = {}
    for block in blocks:
        raw_path = work / f"raw_{block.index:04d}.mp3"
        response = client.audio.speech.create(
            model=model, voice=voice, input=block.text,
            speed=speed, response_format="mp3")
        response.stream_to_file(str(raw_path))
        raw_durations[block.index] = _get_audio_duration(str(raw_path))

    # Drop blocks whose TTS audio exceeds what mild speedup can handle
    kept_blocks = []
    for block in blocks:
        raw_dur = raw_durations[block.index]
        if raw_dur > block.duration * max_speedup:
            print(f"      Dropping voiceover block {block.index} "
                  f"[{block.start_sec:.1f}-{block.end_sec:.1f}s]: "
                  f"TTS ({raw_dur:.1f}s) > window ({block.duration:.1f}s) x {max_speedup}")
        else:
            kept_blocks.append(block)
    if len(kept_blocks) < len(blocks):
        print(f"    Dropped {len(blocks) - len(kept_blocks)}/{len(blocks)} "
              f"voiceover blocks (would be cut off)")
        blocks = kept_blocks
        kept_set = {bl.index for bl in blocks}
        result.voiceover_blocks = [b for b in result.voiceover_blocks if b.index in kept_set]
    n = len(blocks)

    if not blocks:
        # Edge case: all blocks dropped — return silence
        silence_path = str(work / "silence_full.wav")
        _generate_silence_wav(silence_path, 1.0)
        return silence_path

    # ---------------------------------------------------------------
    # Build timeline: silence gaps + voiced blocks
    # ---------------------------------------------------------------
    first_start = blocks[0].start_sec
    if first_start > 0.05:
        sp = str(work / "silence_lead.wav")
        _generate_silence_wav(sp, first_start)
        padded_files.append(sp)

    for idx, block in enumerate(blocks):
        raw_path = work / f"raw_{block.index:04d}.mp3"
        fit_path = work / f"fit_{block.index:04d}.mp3"
        pad_path = work / f"pad_{block.index:04d}.wav"

        # Insert silence for gap between this block and previous
        if idx > 0:
            gap = block.start_sec - blocks[idx - 1].end_sec
            if gap > 0.05:
                gp = str(work / f"gap_{block.index:04d}.wav")
                _generate_silence_wav(gp, gap)
                padded_files.append(gp)

        raw_dur = raw_durations[block.index]
        speedup = _fit_audio_to_duration(str(raw_path), str(fit_path),
                                          block.duration, max_speedup=max_speedup)
        _pad_audio_to_duration(str(fit_path), str(pad_path), block.duration)

        status = "ok" if speedup <= 1.0 else f"~{speedup:.2f}x"
        print(f"      [{idx+1}/{n}] blk{block.index} t={block.start_sec:.1f}-"
              f"{block.end_sec:.1f}s ({block.duration:.1f}s window, "
              f"{raw_dur:.1f}s audio) {status}")
        padded_files.append(str(pad_path))

    # Concatenate all segments
    combined_path = work / "voiceover_combined.mp3"
    n_files = len(padded_files)
    input_args = []
    for p in padded_files:
        input_args.extend(["-i", p])
    filter_parts = []
    for i in range(n_files):
        filter_parts.append(f"[{i}:a]aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo[a{i}]")
    filter_parts.append("".join(f"[a{i}]" for i in range(n_files)) + f"concat=n={n_files}:v=0:a=1[out]")

    cmd = ["ffmpeg", "-y"] + input_args + ["-filter_complex", ";".join(filter_parts), "-map", "[out]", "-c:a", "libmp3lame", "-q:a", "2", str(combined_path)]
    res = subprocess.run(cmd, capture_output=True)
    if res.returncode != 0:
        print(f"    ffmpeg stderr:\n{res.stderr.decode(errors='replace')}")
        res.check_returncode()

    final_dur = _get_audio_duration(str(combined_path))
    expected_dur = blocks[-1].end_sec if blocks else 0.0
    print(f"    Combined voiceover -> {combined_path}")
    print(f"    Duration: {final_dur:.2f}s (expected: {expected_dur:.2f}s, drift: {abs(final_dur - expected_dur):.3f}s)")
    return combined_path


# ===================================================================
# Background music mixing
# ===================================================================
def mix_voiceover_with_bgm(voiceover_path, bgm_path, output_path, video_duration,
                            bgm_vol=0.15, vo_vol=1.0, fade_in=2.0, fade_out=3.0):
    """Mix voiceover with background music.

    - BGM is looped to cover full video duration, with fade-in/out.
    - BGM volume is ducked so voiceover remains clearly audible.
    - Output is a single mixed audio file matching video_duration.
    """
    vo_exists = voiceover_path and Path(voiceover_path).exists()
    bgm_exists = bgm_path and Path(bgm_path).exists()

    if not bgm_exists:
        print(f"    BGM file not found: {bgm_path}, skipping music mix")
        return voiceover_path

    print(f"    Mixing background music (vol={bgm_vol}) with voiceover ...")
    print(f"      BGM: {bgm_path}")
    print(f"      Voiceover: {voiceover_path}")

    input_args = []
    filter_parts = []

    if vo_exists:
        input_args.extend(["-i", str(voiceover_path)])
        filter_parts.append(
            f"[0:a]apad=whole_dur={video_duration:.4f},"
            f"atrim=0:{video_duration:.4f},"
            f"volume={vo_vol:.2f}[vo]"
        )
    else:
        input_args.extend(["-f", "lavfi", "-i",
                           f"anullsrc=r=44100:cl=stereo"])
        filter_parts.append(
            f"[0:a]atrim=0:{video_duration:.4f}[vo]"
        )

    n_loops = max(1, math.ceil(video_duration / 30.0) + 1)
    input_args.extend(["-stream_loop", str(n_loops), "-i", str(bgm_path)])

    bgm_input_idx = 1
    fade_out_start = max(0.0, video_duration - fade_out)
    filter_parts.append(
        f"[{bgm_input_idx}:a]"
        f"atrim=0:{video_duration:.4f},"
        f"afade=t=in:st=0:d={fade_in:.2f},"
        f"afade=t=out:st={fade_out_start:.2f}:d={fade_out:.2f},"
        f"volume={bgm_vol:.2f}[bgm]"
    )

    filter_parts.append("[vo][bgm]amix=inputs=2:duration=first:normalize=0[mixed]")

    full_filter = ";".join(filter_parts)
    cmd = (
        ["ffmpeg", "-y"] + input_args +
        ["-filter_complex", full_filter,
         "-map", "[mixed]",
         "-c:a", "libmp3lame", "-q:a", "2",
         "-t", f"{video_duration:.4f}",
         str(output_path)]
    )

    res = subprocess.run(cmd, capture_output=True)
    if res.returncode != 0:
        print(f"    ffmpeg mix stderr:\n{res.stderr.decode(errors='replace')}")
        res.check_returncode()

    mixed_dur = _get_audio_duration(str(output_path))
    print(f"    Mixed audio -> {output_path} ({mixed_dur:.2f}s)")
    return str(output_path)


# ===================================================================
# Compose final video
# ===================================================================
def compose_final_video(video_path, srt_path, voiceover_path, output_path, subtitle_style, burn_subtitles=True):
    cmd = ["ffmpeg", "-y", "-i", video_path]
    audio_map = []
    if voiceover_path and Path(voiceover_path).exists():
        cmd.extend(["-i", voiceover_path])
        audio_map = ["-map", "0:v", "-map", "1:a", "-shortest"]
    if burn_subtitles and Path(srt_path).exists():
        safe_srt = srt_path.replace("\\", "/").replace(":", "\\:")
        cmd.extend(["-vf", f"subtitles='{safe_srt}':force_style='{subtitle_style}'"])
    cmd.extend(audio_map + ["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-c:a", "aac", "-b:a", "192k", str(output_path)])
    subprocess.run(cmd, check=True)
    out = Path(output_path)
    print(f"    Final video -> {out} ({out.stat().st_size / (1024*1024):.1f} MB)")
    return out


# ===================================================================
# Pipeline
# ===================================================================
def run_single_style(client, tts_client, style_key, video_path, keyframes, duration,
                     output_dir, vision_model="qwen3-omni-flash", tts_model="tts-1-hd",
                     per_segment_tts=True, burn_subtitles=True,
                     bgm_path=None, bgm_vol=0.15):
    cfg = STYLE_CONFIGS[style_key]
    style_dir = output_dir / cfg["folder"]
    if style_dir.exists():
        shutil.rmtree(style_dir)
    style_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  -- {cfg['label']} --")
    print("  [1] Generating narration (1:1 mode) ...")
    result = generate_narration(client, keyframes, duration, cfg, model=vision_model)

    srt_path, json_path, script_path = style_dir / "subtitles.srt", style_dir / "narration.json", style_dir / "narration_script.txt"
    result.save_srt(srt_path)
    result.save_json(json_path)
    result.save_script(script_path, label=cfg["label"])
    print(f"    SRT -> {srt_path}\n    JSON -> {json_path}\n    Script -> {script_path}")

    print("  [2] Generating voiceover ...")
    if per_segment_tts:
        vo_path = generate_voiceover_per_block(tts_client, result, str(style_dir / "tts_segments"), voice=cfg["tts_voice"], model=tts_model, speed=cfg.get("tts_speed", 1.0))
    else:
        vo_path = generate_voiceover_single(tts_client, result, str(style_dir / "voiceover.mp3"), voice=cfg["tts_voice"], model=tts_model, speed=cfg.get("tts_speed", 1.0))

    # Re-save after potential block drops
    result.save_json(json_path)
    result.save_script(script_path, label=cfg["label"])

    if bgm_path:
        print("  [2.5] Mixing with background music ...")
        mixed_path = str(style_dir / "voiceover_with_bgm.mp3")
        vo_path = mix_voiceover_with_bgm(
            str(vo_path), bgm_path, mixed_path, duration,
            bgm_vol=bgm_vol)

    print("  [3] Composing final video ...")
    return compose_final_video(video_path, str(srt_path), str(vo_path), str(style_dir / f"final_{cfg['folder']}.mp4"), cfg["subtitle_style"], burn_subtitles)


def run_dual_pipeline(video_path, output_dir, api_key, base_url=None,
                      tts_api_key=None, tts_base_url=None, styles=None,
                      sample_fps=2.0, vision_model="qwen3-omni-flash",
                      tts_model="tts-1-hd", per_segment_tts=True,
                      burn_subtitles=True, bgm_path=None, bgm_vol=0.8):
    if styles is None:
        styles = ["real_estate", "douyin", "english"]
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    vkw = {"api_key": api_key}
    if base_url:
        vkw["base_url"] = base_url
    client = OpenAI(**vkw)

    tkw = {"api_key": tts_api_key or api_key}
    if tts_base_url:
        tkw["base_url"] = tts_base_url
    elif not tts_api_key and base_url:
        tkw["base_url"] = base_url
    tts_client = OpenAI(**tkw)

    print(f"\n{'=' * 60}\nMULTI-STYLE NARRATION PIPELINE (v4.1 - 1:1 sync)\n{'=' * 60}")
    print(f"  Video: {video_path}  Vision: {vision_model}  TTS: {tts_model}  FPS: {sample_fps}")
    print(f"  Styles: {', '.join(STYLE_CONFIGS[s]['label'] for s in styles)}")
    if bgm_path:
        print(f"  BGM: {bgm_path} (vol={bgm_vol})")

    print("\n[0] Extracting keyframes ...")
    keyframes, duration = extract_keyframes(video_path, sample_fps=sample_fps)
    print(f"    Duration: {duration:.1f}s -> {len(keyframes)} frames")

    results = {}
    for i, sk in enumerate(styles, 1):
        print(f"\n{'-' * 60}\n  STYLE {i}/{len(styles)}: {STYLE_CONFIGS[sk]['label']}\n{'-' * 60}")
        results[sk] = run_single_style(
            client, tts_client, sk, video_path, keyframes, duration, out,
            vision_model, tts_model, per_segment_tts, burn_subtitles,
            bgm_path=bgm_path, bgm_vol=bgm_vol)

    print(f"\n{'=' * 60}\nDONE!\n{'=' * 60}")
    for sk, path in results.items():
        cfg = STYLE_CONFIGS[sk]
        sz = path.stat().st_size / (1024*1024) if path.exists() else 0
        print(f"  {cfg['label']}: {path} ({sz:.1f} MB)")
    return results


def main():
    p = argparse.ArgumentParser(description="Multi-style narration v4.1 (1:1 sync)")
    p.add_argument("--video", required=True)
    p.add_argument("--output", default="outputs/dual_narration")
    p.add_argument("--api_key", default=None)
    p.add_argument("--base_url", default=None)
    p.add_argument("--tts_api_key", default=None)
    p.add_argument("--tts_base_url", default=None)
    p.add_argument("--style", default=None, choices=["real_estate", "douyin", "english"])
    p.add_argument("--styles", default=None, nargs="+", choices=["real_estate", "douyin", "english"])
    p.add_argument("--sample_fps", type=float, default=2.0)
    p.add_argument("--vision_model", default="qwen3-omni-flash")
    p.add_argument("--tts_model", default="tts-1-hd", choices=["tts-1", "tts-1-hd"])
    p.add_argument("--single_tts", action="store_true")
    p.add_argument("--no_burn_subs", action="store_true")
    p.add_argument("--bgm", default=None, help="Path to background music file (mp3/wav)")
    p.add_argument("--bgm_vol", type=float, default=0.15,
                   help="Background music volume (0.0-1.0, default 0.15)")
    a = p.parse_args()
    api_key = a.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Provide --api_key or set OPENAI_API_KEY")
    styles = [a.style] if a.style else (a.styles or None)
    run_dual_pipeline(a.video, a.output, api_key, a.base_url, a.tts_api_key,
                      a.tts_base_url, styles, a.sample_fps, a.vision_model,
                      a.tts_model, not a.single_tts, not a.no_burn_subs,
                      bgm_path=a.bgm, bgm_vol=a.bgm_vol)

if __name__ == "__main__":
    main()