"""
Benchmark Prompt Generator (v2 — Scene Graph Support)
=====================================================

Generates camera trajectory prompts at multiple difficulty levels for each
scene in the dataset.  Supports both the legacy InteriorGS labels.json
format and the new ScanNet++ scene-graph JSON format.

Prompt levels:
  - **high-level**: Abstract/cinematic requests with NO object names and
    NO camera movement keywords.

  - **medium-level**: Mentions ≥3 specific objects by label (no IDs) and
    does NOT prescribe camera movements.

  - **medium-id-level**: Like medium, but each object mention includes its
    unique scene-graph ID in parentheses, e.g. "the computer screen
    (id: screen_2)".  This removes ambiguity when multiple objects share
    the same label.

  - **low-level**: Mentions ≥3 specific objects AND explicit camera
    movements.  Placement-aware.

  - **low-id-level**: Like low, but each object mention includes its
    unique scene-graph ID.

Usage:
    from benchmark_prompt_generator import generate_prompts_for_scene

    prompts = generate_prompts_for_scene(scene_path, fmt="scene_graph")
    # → {"high": [...], "medium": [...], "medium_id": [...],
    #    "low": [...], "low_id": [...]}
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

from openai import OpenAI


# ==========================================================================
# Object selection helpers
# ==========================================================================

# Labels that are uninteresting or too small to anchor a camera on.
SKIP_LABELS = {
    "wall", "floor", "ceiling", "room", "unknown", "misc", "object",
    "switch", "outlet", "plug", "wire", "cable", "pipe", "vent",
    "baseboard", "molding", "trim",
}

# ---------------------------------------------------------------------------
# Placement classification — infer where an object sits from its label
# ---------------------------------------------------------------------------

WALL_LABELS = {
    "painting", "tv", "television", "monitor", "mirror", "window", "door",
    "bookshelf", "bookshelves", "wardrobe", "cabinet", "radiator",
    "wall clock", "clock", "shelf", "shelves", "curtain", "curtains",
    "blinds", "poster", "picture", "picture frame", "frame",
    "thermostat", "light switch", "wall lamp", "sconce", "coat rack",
    "towel rack", "towel bar", "medicine cabinet", "wall art",
    "whiteboard", "bulletin board", "tapestry",
}

CEILING_LABELS = {
    "chandelier", "ceiling fan", "ceiling light", "pendant lamp",
    "pendant light", "smoke detector", "sprinkler", "overhead projector",
    "ceiling lamp", "recessed light", "track light",
}

FLOOR_WALL_LABELS = {
    "console table", "console", "shoe rack", "nightstand", "dresser",
    "sideboard", "buffet", "credenza", "desk", "toilet", "sink",
    "vanity", "filing cabinet", "bench",
    "reception", "table", "bed", "kitchen counter",
}


def infer_placement_from_label(label: str) -> str:
    """Infer placement from label string (legacy path)."""
    label_lower = label.strip().lower()
    for kw in CEILING_LABELS:
        if kw in label_lower:
            return "ceiling"
    for kw in WALL_LABELS:
        if kw in label_lower:
            return "wall"
    for kw in FLOOR_WALL_LABELS:
        if kw in label_lower:
            return "floor_wall"
    return "freestanding"


def _infer_flags_from_label(label: str) -> tuple:
    """
    Infer (against_wall, attached_to_ceiling) booleans from a label string.
    Used for the legacy path where we don't have scene-graph metadata.
    """
    placement = infer_placement_from_label(label)
    against_wall = placement in ("wall", "floor_wall")
    attached_to_ceiling = placement == "ceiling"
    return against_wall, attached_to_ceiling


def infer_placement_from_sg(obj_data: dict, label: str) -> str:
    """
    Infer placement from scene-graph flags (against_wall, attached_to_ceiling).

    Falls back to label-based heuristics when needed.
    """
    if obj_data.get("attached_to_ceiling", False):
        return "ceiling"
    if obj_data.get("against_wall", False):
        # Distinguish wall-mounted vs floor_wall using label
        label_lower = label.strip().lower()
        for kw in FLOOR_WALL_LABELS:
            if kw in label_lower:
                return "floor_wall"
        return "wall"
    return "freestanding"


def _extract_label_from_id(obj_id: str) -> str:
    """
    Extract the human-readable label from a scene-graph object ID.

    Examples:
        "coffee_machine_0"  → "coffee machine"
        "screen_2"          → "screen"
        "table_15"          → "table"
        "door_frame_1"      → "door frame"
    """
    # Split from the right — the last segment after '_' is the numeric index
    parts = obj_id.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0].replace("_", " ")
    return obj_id.replace("_", " ")


# ==========================================================================
# Scene loading — two formats
# ==========================================================================

def load_scene_objects_legacy(labels_path: str) -> List[Dict]:
    """Load legacy labels.json (InteriorGS format)."""
    with open(labels_path, "r") as f:
        labels = json.load(f)

    objects = []
    for obj in labels:
        label = obj.get("label", "").strip().lower()
        if not label or label in SKIP_LABELS:
            continue
        if "bounding_box" not in obj or not obj["bounding_box"]:
            continue
        against_wall, attached_to_ceiling = _infer_flags_from_label(obj["label"])
        objects.append({
            "id": str(obj["ins_id"]),
            "label": obj["label"],
            "placement": infer_placement_from_label(obj["label"]),
            "against_wall": against_wall,
            "attached_to_ceiling": attached_to_ceiling,
        })
    return objects


def load_scene_objects_scene_graph(sg_path: str) -> List[Dict]:
    """
    Load scene objects from the new scene-graph JSON (ScanNet++ format).

    Expected structure:
        {
            "scene": "...",
            "rooms": { "room_name": ["obj_id", ...], ... },
            "objects": { "obj_id": { "obb": [...], "against_wall": bool,
                                      "attached_to_ceiling": bool }, ... }
        }
    """
    with open(sg_path, "r") as f:
        sg = json.load(f)

    sg_objects = sg.get("objects", {})

    objects = []
    for obj_id, obj_data in sg_objects.items():
        label = _extract_label_from_id(obj_id)
        if label.lower() in SKIP_LABELS:
            continue
        # Skip structural elements
        if any(skip in label.lower() for skip in ("door frame", "door_frame")):
            continue

        against_wall = obj_data.get("against_wall", False)
        attached_to_ceiling = obj_data.get("attached_to_ceiling", False)
        placement = infer_placement_from_sg(obj_data, label)
        objects.append({
            "id": obj_id,               # e.g. "screen_2", "coffee_machine_0"
            "label": label,             # e.g. "screen", "coffee machine"
            "placement": placement,
            "against_wall": against_wall,
            "attached_to_ceiling": attached_to_ceiling,
        })

    return objects


def load_scene_objects(path: str, fmt: str = "scene_graph") -> List[Dict]:
    """Unified loader dispatching on format."""
    if fmt == "scene_graph":
        return load_scene_objects_scene_graph(path)
    else:
        return load_scene_objects_legacy(path)


# ==========================================================================
# Object selection
# ==========================================================================

def select_objects(
    objects: List[Dict],
    min_count: int = 3,
    max_count: int = 6,
    seed: Optional[int] = None,
    deduplicate_labels: bool = True,
    skip_ceiling: bool = False,
) -> List[Dict]:
    """
    Select a diverse subset of objects for prompt generation.

    Args:
        deduplicate_labels: If True, pick at most one object per label
            (base class name). This ensures no two chairs, no two tables, etc.
            When generating id-level prompts we set this to False since
            the ID disambiguates — but even then, class diversity is enforced
            separately when skip_ceiling is True (low-level prompts).
        skip_ceiling: If True, exclude objects with attached_to_ceiling=True.
            The 3DGS reconstruction near ceilings tends to have poor quality,
            so we avoid selecting ceiling-attached objects for benchmarking.
    """
    rng = random.Random(seed)

    pool = list(objects)

    # Filter out ceiling-attached objects (bad 3DGS quality near ceiling)
    if skip_ceiling:
        pool = [o for o in pool if not o.get("attached_to_ceiling", False)]

    # Deduplicate by base class label (e.g. only one "chair", one "table")
    if deduplicate_labels:
        seen_labels = set()
        unique = []
        for obj in pool:
            lbl = obj["label"].lower()
            if lbl not in seen_labels:
                seen_labels.add(lbl)
                unique.append(obj)
        pool = unique

    if len(pool) < min_count:
        return pool

    count = min(rng.randint(min_count, max_count), len(pool))
    return rng.sample(pool, count)


# ==========================================================================
# Camera movement vocabulary — flag-aware
# ==========================================================================

ALL_OBJECT_LEVEL_MOVEMENTS = [
    ("orbit_full", "do a full 360° orbit around"),
    ("orbit_half", "do a half orbit around"),
    ("orbit_quarter", "do a quarter orbit around"),
    ("pan_left", "pan left from"),
    ("pan_right", "pan right at"),
    ("move_in", "move in closer to"),
    ("move_out", "pull back from"),
    ("crane", "crane up above"),
    ("tilt_up", "tilt up from"),
    ("tilt_down", "tilt down toward"),
    ("static", "hold a static shot of"),
    ("zoom_in_out", "zoom in on"),
    ("zoom_out_in", "do a wide-angle zoom on"),
]

# Legacy placement-based forbidden sets (kept for backward compatibility)
PLACEMENT_FORBIDDEN = {
    "wall":       {"orbit_full", "orbit_half"},
    "ceiling":    {"crane"},
    "floor_wall": {"orbit_full"},
    "freestanding": set(),
}


def get_forbidden_movements_from_flags(
    against_wall: bool,
    attached_to_ceiling: bool,
) -> set:
    """
    Compute forbidden movements directly from the scene-graph boolean flags.

    Rules:
      - against_wall=True  → NEVER orbit_full (camera cannot go behind the wall)
      - attached_to_ceiling=True → NEVER crane (camera cannot rise above the ceiling)
    """
    forbidden = set()
    if against_wall:
        forbidden.add("orbit_full")
    if attached_to_ceiling:
        forbidden.add("crane")
    return forbidden


def get_allowed_movements(
    placement: str,
    against_wall: bool = False,
    attached_to_ceiling: bool = False,
) -> List[tuple]:
    """
    Get allowed movements for an object, using both placement heuristics
    AND the raw against_wall / attached_to_ceiling flags.

    The final forbidden set is the UNION of:
      1. PLACEMENT_FORBIDDEN[placement]  (label-based heuristics)
      2. Flag-based rules (against_wall → no orbit_full,
                           attached_to_ceiling → no crane)

    This ensures that even if an object's placement string doesn't capture
    all constraints (e.g. a curtain classified as "ceiling" but also
    against_wall), the raw flags still enforce the correct restrictions.
    """
    # Start with placement-based forbidden set
    forbidden = set(PLACEMENT_FORBIDDEN.get(placement, set()))
    # Layer on flag-based forbidden set
    forbidden |= get_forbidden_movements_from_flags(against_wall, attached_to_ceiling)

    return [(k, p) for k, p in ALL_OBJECT_LEVEL_MOVEMENTS if k not in forbidden]


# ---------------------------------------------------------------------------
# Movement type grouping — movements in the same group are considered the
# "same type" and should not repeat within a single low-level prompt.
# ---------------------------------------------------------------------------

MOVEMENT_TYPE_GROUPS = {
    "orbit_full":   "orbit",
    "orbit_half":   "orbit",
    "orbit_quarter": "orbit",
    "pan_left":     "pan",
    "pan_right":    "pan",
    "move_in":      "dolly",
    "move_out":     "dolly",
    "crane":        "crane",
    "tilt_up":      "tilt",
    "tilt_down":    "tilt",
    "static":       "static",
    "zoom_in_out":  "zoom",
    "zoom_out_in":  "zoom",
}


def _movement_type(movement_key: str) -> str:
    """Return the type-group name for a movement key."""
    return MOVEMENT_TYPE_GROUPS.get(movement_key, movement_key)


def pick_diverse_movement(
    allowed: List[tuple],
    used_types: set,
    rng: random.Random,
) -> tuple:
    """
    Pick a movement from `allowed` whose type hasn't been used yet.

    If all types in `allowed` are already used (rare, happens when there are
    more objects than movement types), fall back to any allowed movement.

    Args:
        allowed:    List of (key, phrase) tuples from get_allowed_movements.
        used_types: Set of movement-type strings already assigned in this prompt.
        rng:        Random instance for sampling.

    Returns:
        (movement_key, movement_phrase) tuple.  Also mutates `used_types` in-place.
    """
    # Filter to movements whose type is not yet used
    fresh = [(k, p) for k, p in allowed if _movement_type(k) not in used_types]
    if not fresh:
        # Fallback: all types exhausted — just pick any allowed movement
        fresh = allowed
    key, phrase = rng.choice(fresh)
    used_types.add(_movement_type(key))
    return key, phrase


# ==========================================================================
# Templates
# ==========================================================================

HIGH_LEVEL_TEMPLATES = [
    "Give me a cozy walkthrough of this space.",
    "Show me around this room as if I'm visiting for the first time.",
    "Create a cinematic tour of this interior.",
    "Walk me through this place like a real estate showing.",
    "I want to get a feel for the atmosphere of this room. Show me around.",
    "Imagine I'm an Airbnb guest arriving for the first time — give me a welcome tour.",
    "Create a smooth, professional walkthrough video of this space.",
    "Guide me through this room with a calm, exploratory pace.",
    "Film this room like a short architectural showcase.",
    "Take me on a leisurely tour, lingering on the most interesting spots.",
    "I want to see the highlights of this room in a slow, cinematic sweep.",
    "Give me an overview of this space, moving from one area to the next naturally.",
    "Show me what makes this room special — focus on the most eye-catching features.",
    "Create a dreamy, slow-motion tour of the interior.",
    "Walk through this room as if you're showing it to a design magazine.",
    "Capture the essence of this room in a short video tour.",
    "Give me a bird's-eye perspective of the layout, then come down for details.",
    "Tour this space like a documentary filmmaker would.",
    "Explore this room from corner to corner, spending time on interesting details.",
    "Show me the room in a way that highlights its spatial flow and design.",
]

MEDIUM_LEVEL_TEMPLATES = [
    "Show me the {obj_list}.",
    "I'd like to see the {obj_list} in this room.",
    "Take me on a tour starting from the {first}, passing by the {middle}, and ending at the {last}.",
    "Focus on the {obj_list} — those are the items I'm most interested in.",
    "Walk me through the room, making sure to cover the {obj_list}.",
    "Give me a good view of the {obj_list}.",
    "Start at the {first}, then move to the {middle}, and finish at the {last}.",
    "I want to see the {first} up close, then the {middle}, and finally the {last}.",
    "Explore the {obj_list} one by one.",
    "Highlight the {obj_list} in a smooth sequence.",
    "Sweep across the room, pausing at the {obj_list}.",
    "A quick tour focusing on the {obj_list}.",
    "Guide me from the {first} over to the {middle} and then to the {last}.",
    "Can you show me the {first}, the {middle}, and wrap up at the {last}?",
    "I'm curious about the {obj_list} — give me a closer look at each.",
]

# Medium-ID templates — same structure but each object has (id: xxx)
MEDIUM_ID_TEMPLATES = [
    "Show me {obj_list}.",
    "I'd like to see {obj_list} in this room.",
    "Take me on a tour starting from {first}, passing by {middle}, and ending at {last}.",
    "Focus on {obj_list} — those are the items I'm most interested in.",
    "Walk me through the room, making sure to cover {obj_list}.",
    "Give me a good view of {obj_list}.",
    "Start at {first}, then move to {middle}, and finish at {last}.",
    "I want to see {first} up close, then {middle}, and finally {last}.",
    "Explore {obj_list} one by one.",
    "Highlight {obj_list} in a smooth sequence.",
]

LOW_LEVEL_TEMPLATES = [
    "{steps_sentence}",
    "Here's what I want: {steps_sentence}",
    "Please do the following: {steps_sentence}",
    "I'd like a specific sequence: {steps_sentence}",
    "Follow this plan: {steps_sentence}",
]


# ==========================================================================
# Formatting helpers
# ==========================================================================

def _format_obj_list(labels: List[str]) -> str:
    """'a, b, and c' style formatting."""
    if len(labels) == 1:
        return labels[0]
    elif len(labels) == 2:
        return f"{labels[0]} and {labels[1]}"
    else:
        return ", ".join(labels[:-1]) + f", and {labels[-1]}"


def _obj_with_id(obj: Dict) -> str:
    """Format an object mention with its ID: 'the coffee machine (id: coffee_machine_0)'."""
    return f"the {obj['label']} (id: {obj['id']})"


def _format_obj_list_with_ids(objects: List[Dict]) -> str:
    """'the X (id: x_0), the Y (id: y_1), and the Z (id: z_2)' style."""
    mentions = [_obj_with_id(o) for o in objects]
    return _format_obj_list(mentions)


# ==========================================================================
# Template-based prompt generation
# ==========================================================================

def generate_prompts_template(
    objects: List[Dict],
    n_high: int = 2,
    n_medium: int = 2,
    n_low: int = 2,
    seed: Optional[int] = None,
    include_id_levels: bool = True,
) -> Dict[str, List[str]]:
    """
    Generate prompts using templates (no LLM API call needed).

    When include_id_levels is True, also generates 'medium_id' and 'low_id'
    prompts that include explicit object IDs.
    """
    rng = random.Random(seed)
    prompts: Dict[str, List[str]] = {
        "high": [], "medium": [], "low": [],
    }
    if include_id_levels:
        prompts["medium_id"] = []
        prompts["low_id"] = []

    # --- High-level ---
    chosen_high = rng.sample(HIGH_LEVEL_TEMPLATES, min(n_high, len(HIGH_LEVEL_TEMPLATES)))
    prompts["high"] = chosen_high

    # --- Medium-level (no IDs) ---
    for _ in range(n_medium):
        selected = select_objects(objects, min_count=3, max_count=6,
                                  seed=rng.randint(0, 2**31), deduplicate_labels=True)
        if len(selected) < 3:
            selected = objects[:3] if len(objects) >= 3 else objects
        labels = [o["label"] for o in selected]
        template = rng.choice(MEDIUM_LEVEL_TEMPLATES)
        prompt = template.format(
            obj_list=_format_obj_list(labels),
            first=labels[0],
            middle=(_format_obj_list(labels[1:-1]) if len(labels) > 2
                    else labels[1] if len(labels) > 1 else labels[0]),
            last=labels[-1],
        )
        prompts["medium"].append(prompt)

    # --- Medium-ID-level ---
    if include_id_levels:
        for _ in range(n_medium):
            selected = select_objects(objects, min_count=3, max_count=6,
                                      seed=rng.randint(0, 2**31),
                                      deduplicate_labels=False)
            if len(selected) < 3:
                selected = objects[:3] if len(objects) >= 3 else objects
            template = rng.choice(MEDIUM_ID_TEMPLATES)
            mentions = [_obj_with_id(o) for o in selected]
            prompt = template.format(
                obj_list=_format_obj_list(mentions),
                first=_obj_with_id(selected[0]),
                middle=(_format_obj_list([_obj_with_id(o) for o in selected[1:-1]])
                        if len(selected) > 2
                        else _obj_with_id(selected[1]) if len(selected) > 1
                        else _obj_with_id(selected[0])),
                last=_obj_with_id(selected[-1]),
            )
            prompts["medium_id"].append(prompt)

    # --- Low-level (no IDs, placement-aware + flag-aware + diverse) ---
    for _ in range(n_low):
        selected = select_objects(objects, min_count=3, max_count=5,
                                  seed=rng.randint(0, 2**31),
                                  deduplicate_labels=True,
                                  skip_ceiling=True)
        if len(selected) < 3:
            # Fallback without ceiling filter if not enough objects
            selected = select_objects(objects, min_count=3, max_count=5,
                                      seed=rng.randint(0, 2**31),
                                      deduplicate_labels=True,
                                      skip_ceiling=False)
        if len(selected) < 3:
            selected = objects[:3] if len(objects) >= 3 else objects

        steps = []
        used_movement_types: set = set()
        for obj in selected:
            placement = obj.get("placement", "freestanding")
            against_wall = obj.get("against_wall", False)
            attached_to_ceiling = obj.get("attached_to_ceiling", False)
            allowed = get_allowed_movements(placement, against_wall, attached_to_ceiling)
            _, movement_phrase = pick_diverse_movement(allowed, used_movement_types, rng)
            steps.append(f"{movement_phrase} the {obj['label']}")

        steps_sentence = _join_steps(steps)
        template = rng.choice(LOW_LEVEL_TEMPLATES)
        prompts["low"].append(template.format(steps_sentence=steps_sentence))

    # --- Low-ID-level (with IDs, placement-aware + flag-aware + diverse) ---
    if include_id_levels:
        for _ in range(n_low):
            selected = select_objects(objects, min_count=3, max_count=5,
                                      seed=rng.randint(0, 2**31),
                                      deduplicate_labels=True,
                                      skip_ceiling=True)
            if len(selected) < 3:
                selected = select_objects(objects, min_count=3, max_count=5,
                                          seed=rng.randint(0, 2**31),
                                          deduplicate_labels=True,
                                          skip_ceiling=False)
            if len(selected) < 3:
                selected = objects[:3] if len(objects) >= 3 else objects

            steps = []
            used_movement_types: set = set()
            for obj in selected:
                placement = obj.get("placement", "freestanding")
                against_wall = obj.get("against_wall", False)
                attached_to_ceiling = obj.get("attached_to_ceiling", False)
                allowed = get_allowed_movements(placement, against_wall, attached_to_ceiling)
                _, movement_phrase = pick_diverse_movement(allowed, used_movement_types, rng)
                steps.append(f"{movement_phrase} {_obj_with_id(obj)}")

            steps_sentence = _join_steps(steps)
            template = rng.choice(LOW_LEVEL_TEMPLATES)
            prompts["low_id"].append(template.format(steps_sentence=steps_sentence))

    return prompts


def _join_steps(steps: List[str]) -> str:
    """Join movement steps with transitional language."""
    if len(steps) == 1:
        return steps[0].capitalize() + "."
    parts = []
    for i, step in enumerate(steps):
        if i == 0:
            parts.append("First, " + step)
        elif i == len(steps) - 1:
            parts.append("and finally " + step)
        else:
            parts.append("then " + step)
    return ", ".join(parts) + "."


# ==========================================================================
# LLM-based prompt generation (higher quality, needs API)
# ==========================================================================

LLM_PROMPT_GEN_SYSTEM = """You are a benchmark prompt generator for a 3D camera trajectory system.

Given a list of objects in a 3D indoor scene (each with an inferred placement, boolean
flags `against_wall` and `attached_to_ceiling`, and a unique ID), generate camera
trajectory prompts at FIVE levels:

1. **high**: A natural, abstract request with NO object names and NO camera movement terms.
   Describe a mood, purpose, or scenario.

2. **medium**: Mentions at least 3 specific objects by label (NO IDs) and does NOT
   mention any camera movements.

3. **medium_id**: Like medium, but each object mention includes its unique ID in parentheses,
   e.g. "the coffee machine (id: coffee_machine_0)".  This removes ambiguity when multiple
   objects share the same label.

4. **low**: Mentions at least 3 specific objects AND explicit camera movements.
   Camera movements: orbit_full, orbit_half, orbit_quarter, pan_left, pan_right,
   move_in, move_out, crane, tilt_up, tilt_down, static, zoom_in_out, zoom_out_in.

5. **low_id**: Like low, but each object mention includes its unique ID in parentheses.

CRITICAL PLACEMENT / FLAG RULES for low and low_id prompts:
   - against_wall=true: NEVER use orbit_full. The camera cannot go behind the wall.
   - attached_to_ceiling=true: NEVER use crane. The camera cannot rise above the ceiling.
   - If BOTH against_wall=true AND attached_to_ceiling=true: NEVER use orbit_full AND
     NEVER use crane.
   - Legacy placement shortcuts (for reference):
       placement="wall": NEVER orbit_full or orbit_half.
       placement="ceiling": NEVER crane.
       placement="floor_wall": NEVER orbit_full.
       placement="freestanding": All movements allowed.

RULES:
- Each prompt must be a single natural-sounding English sentence or short paragraph.
- high: ZERO object names, ZERO camera terms.
- medium / medium_id: ≥3 objects, ZERO camera terms.
- low / low_id: ≥3 objects + explicit camera movement for each, RESPECTING placement
  and flag constraints.
- Vary sentence structure and style across prompts.
- DIVERSITY in low / low_id prompts:
  - Do NOT select two objects of the same class (e.g., no two chairs, no two tables).
  - Do NOT use the same movement type twice in one prompt. Movement types are grouped:
    orbit (orbit_full/orbit_half/orbit_quarter), pan (pan_left/pan_right),
    dolly (move_in/move_out), tilt (tilt_up/tilt_down), zoom (zoom_in_out/zoom_out_in),
    crane, static. Each group may appear at most once per prompt.
  - AVOID selecting objects with attached_to_ceiling=true. The 3D reconstruction quality
    near ceilings is poor. Only include ceiling objects if there are not enough
    non-ceiling objects in the scene.

Respond ONLY with valid JSON:
{
  "high": ["prompt1", "prompt2"],
  "medium": ["prompt1", "prompt2"],
  "medium_id": ["prompt1", "prompt2"],
  "low": ["prompt1", "prompt2"],
  "low_id": ["prompt1", "prompt2"]
}
"""


def generate_prompts_llm(
    objects: List[Dict],
    api_key: str,
    model: str = "gpt-4o-mini",
    n_per_level: int = 2,
    temperature: float = 0.8,
) -> Dict[str, List[str]]:
    """Use an LLM to generate diverse, natural prompts.  Falls back to templates."""
    client = OpenAI(api_key=api_key)

    obj_descriptions = [
        f"- {o['label']} (id: {o['id']}, placement: {o.get('placement', 'freestanding')}, "
        f"against_wall: {o.get('against_wall', False)}, "
        f"attached_to_ceiling: {o.get('attached_to_ceiling', False)})"
        for o in objects
    ]
    user_msg = (
        f"Scene objects:\n" + "\n".join(obj_descriptions) + "\n\n"
        f"Generate {n_per_level} prompts per level (high, medium, medium_id, low, low_id).\n"
        f"Remember: for low/low_id prompts, respect the flag constraints strictly:\n"
        f"  - against_wall=true → NEVER orbit_full\n"
        f"  - attached_to_ceiling=true → NEVER crane\n"
        f"  - AVOID selecting attached_to_ceiling=true objects (poor 3D quality near ceilings)\n"
        f"  - Do NOT repeat the same object class (e.g. no two chairs)\n"
        f"  - Do NOT repeat the same movement type (orbit/pan/dolly/tilt/zoom/crane/static)\n"
        f"For medium_id and low_id prompts, include (id: <object_id>) after each object mention."
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": LLM_PROMPT_GEN_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=temperature,
        )
        text = response.choices[0].message.content
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            result = json.loads(text[start:end])
            for level in ("high", "medium", "medium_id", "low", "low_id"):
                if level not in result or not isinstance(result[level], list):
                    raise ValueError(f"Missing or invalid '{level}' in response")
            return result
    except Exception as e:
        print(f"  ⚠ LLM prompt generation failed ({e}), falling back to templates")

    return generate_prompts_template(objects, n_per_level, n_per_level, n_per_level,
                                     include_id_levels=True)


# ==========================================================================
# Scene-level entry points
# ==========================================================================

def generate_prompts_for_scene(
    scene_path: str,
    fmt: str = "scene_graph",
    api_key: Optional[str] = None,
    use_llm: bool = False,
    n_per_level: int = 2,
    seed: Optional[int] = None,
    include_id_levels: bool = True,
) -> Dict[str, List[str]]:
    """
    Generate benchmark prompts for a single scene.

    Args:
        scene_path:  Path to the scene's labels.json or scene_graph.json.
        fmt:         "scene_graph" or "legacy".
        api_key:     OpenAI API key (required if use_llm=True).
        use_llm:     If True, use LLM for prompt generation; else templates.
        n_per_level: Number of prompts per difficulty level.
        seed:        Random seed for template-based generation.
        include_id_levels: Generate medium_id and low_id levels.

    Returns:
        {"high": [...], "medium": [...], "medium_id": [...],
         "low": [...], "low_id": [...]}
    """
    objects = load_scene_objects(scene_path, fmt=fmt)

    if len(objects) < 3:
        print(f"  ⚠ Scene has only {len(objects)} usable objects — prompts may be limited")

    if use_llm and api_key:
        return generate_prompts_llm(objects, api_key, n_per_level=n_per_level)
    else:
        return generate_prompts_template(
            objects,
            n_high=n_per_level,
            n_medium=n_per_level,
            n_low=n_per_level,
            seed=seed,
            include_id_levels=include_id_levels,
        )


def discover_scenes_legacy(data_root: str, max_scenes: int = 200) -> List[str]:
    """Discover scene IDs under data_root/compressed/ (legacy InteriorGS)."""
    compressed_dir = Path(data_root) / "compressed"
    if not compressed_dir.exists():
        raise FileNotFoundError(f"Compressed directory not found: {compressed_dir}")

    scene_ids = []
    for scene_dir in sorted(compressed_dir.iterdir()):
        if not scene_dir.is_dir():
            continue
        labels_file = scene_dir / "labels.json"
        if labels_file.exists():
            scene_ids.append(scene_dir.name)
        if len(scene_ids) >= max_scenes:
            break
    return scene_ids


def discover_scenes_scene_graph(
    data_root: str,
    sg_subdir: str = "dslr/sg",
    max_scenes: int = 200,
) -> List[str]:
    """
    Discover scene IDs under data_root/<scene_id>/<sg_subdir>/.

    Looks for any .json file in the scene-graph subdirectory.
    """
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"Data root not found: {root}")

    scene_ids = []
    for scene_dir in sorted(root.iterdir()):
        if not scene_dir.is_dir():
            continue
        sg_dir = scene_dir / sg_subdir
        if sg_dir.exists():
            json_files = list(sg_dir.glob("*.json"))
            if json_files:
                scene_ids.append(scene_dir.name)
        if len(scene_ids) >= max_scenes:
            break
    return scene_ids


def discover_scenes(
    data_root: str,
    fmt: str = "scene_graph",
    sg_subdir: str = "dslr/sg",
    max_scenes: int = 200,
) -> List[str]:
    """Unified scene discovery."""
    if fmt == "scene_graph":
        return discover_scenes_scene_graph(data_root, sg_subdir, max_scenes)
    else:
        return discover_scenes_legacy(data_root, max_scenes)


def get_scene_graph_path(data_root: str, scene_id: str, sg_subdir: str = "dslr/sg") -> str:
    """Find the scene-graph JSON file for a given scene."""
    sg_dir = Path(data_root) / scene_id / sg_subdir
    json_files = sorted(sg_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No scene-graph JSON found in {sg_dir}")
    return str(json_files[0])


def generate_all_prompts(
    data_root: str,
    fmt: str = "scene_graph",
    sg_subdir: str = "dslr/sg",
    max_scenes: int = 200,
    api_key: Optional[str] = None,
    use_llm: bool = False,
    n_per_level: int = 2,
    seed: int = 42,
    include_id_levels: bool = True,
) -> Dict[str, Dict[str, List[str]]]:
    """
    Generate prompts for all discovered scenes.

    Returns:
        {
            "scene_id_1": {"high": [...], "medium": [...], "medium_id": [...],
                           "low": [...], "low_id": [...]},
            ...
        }
    """
    scene_ids = discover_scenes(data_root, fmt=fmt, sg_subdir=sg_subdir,
                                max_scenes=max_scenes)
    print(f"Discovered {len(scene_ids)} scenes (max={max_scenes})")

    all_prompts = {}
    for i, scene_id in enumerate(scene_ids):
        print(f"[{i+1}/{len(scene_ids)}] {scene_id} ... ", end="", flush=True)

        try:
            if fmt == "scene_graph":
                scene_path = get_scene_graph_path(data_root, scene_id, sg_subdir)
            else:
                scene_path = str(Path(data_root) / "compressed" / scene_id / "labels.json")

            prompts = generate_prompts_for_scene(
                scene_path,
                fmt=fmt,
                api_key=api_key,
                use_llm=use_llm,
                n_per_level=n_per_level,
                seed=seed + i,
                include_id_levels=include_id_levels,
            )
            total = sum(len(v) for v in prompts.values())
            print(f"✓ ({total} prompts)")
            all_prompts[scene_id] = prompts
        except Exception as e:
            print(f"✗ ({e})")
            all_prompts[scene_id] = {
                "high": [], "medium": [], "medium_id": [],
                "low": [], "low_id": [], "error": str(e),
            }

    return all_prompts


# ==========================================================================
# CLI
# ==========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate benchmark prompts for GSCinema/TrajScene scenes",
    )
    parser.add_argument(
        "--data_root", type=str, required=True,
        help="Path to the data root (InteriorGS or ScanNet++ gsplat scenes)",
    )
    parser.add_argument(
        "--format", type=str, default="scene_graph", choices=["scene_graph", "legacy"],
        help="Scene data format: 'scene_graph' (ScanNet++) or 'legacy' (InteriorGS)",
    )
    parser.add_argument(
        "--sg_subdir", type=str, default="dslr/sg",
        help="Subdirectory under each scene containing scene-graph JSON (scene_graph format)",
    )
    parser.add_argument(
        "--output", type=str, default="benchmark_prompts.json",
        help="Output JSON file path",
    )
    parser.add_argument("--max_scenes", type=int, default=200)
    parser.add_argument("--n_per_level", type=int, default=2)
    parser.add_argument("--use_llm", action="store_true")
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no_id_levels", action="store_true",
        help="Skip generating medium_id and low_id levels",
    )
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")

    all_prompts = generate_all_prompts(
        data_root=args.data_root,
        fmt=args.format,
        sg_subdir=args.sg_subdir,
        max_scenes=args.max_scenes,
        api_key=api_key,
        use_llm=args.use_llm,
        n_per_level=args.n_per_level,
        seed=args.seed,
        include_id_levels=not args.no_id_levels,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_prompts, f, indent=2)

    total_prompts = sum(
        sum(len(v) for k, v in scene.items() if k != "error")
        for scene in all_prompts.values()
    )
    print(f"\n✓ Saved {total_prompts} prompts for {len(all_prompts)} scenes → {output_path}")


if __name__ == "__main__":
    import os
    main()