"""
Camera Trajectory Dialog System
==================================

Uses the 1-3-1 pattern:
  1. Initial Anchor (starting point)
  2. [Loop] Object-Level → Anchor → Transitional
  3. Final Object-Level (examine last object)

Pattern formula: For N objects, total steps = 3N - 1

Simplified movement vocabulary:
  Object-level: orbit_full, orbit_half, orbit_quarter, pan_left, pan_right,
                move_in, move_out, zoom_in_out, zoom_out_in, static
  Transitional: arc with angle parameter

NEW: The LLM now outputs per-object `viewing_preferences` that guide the
     Anchor Determinator's elevation, distance, and framing choices.
NEW: zoom_in_out / zoom_out_in — optical zoom (focal-length change) while
     the camera stays stationary.  The zoom returns to original by the end
     so the trajectory remains continuous.
NEW: `placement` field in viewing_preferences — the LLM infers whether each
     object is freestanding, wall-mounted, ceiling-mounted, or floor-against-wall,
     and restricts trajectory choices accordingly (e.g., no orbit_full for wall
     objects, no crane for ceiling objects).
NEW: Scene graph JSON now provides `against_wall` and `attached_to_ceiling`
     metadata per object, which is passed to the LLM to help it determine
     placement constraints more accurately.
"""

import os
import json
from openai import OpenAI

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.user_prompt_translator.validate_response import validate_and_clean_response, print_full_validation_report

MODEL_NAME = 'gpt-4.1'

# ==============================================================================
# Prompt Templates
# ==============================================================================

OBJECT_LEVEL_LIST = """
OBJECT-LEVEL TRAJECTORIES (examine current object):
  - orbit_full: Full 360° orbit around the object
  - orbit_half: 180° orbit, front-to-back or side-to-side
  - orbit_quarter: 90° orbit, adjacent perspective
  - pan_left: Rotate view leftward to scan surroundings
  - pan_right: Rotate view rightward to scan surroundings
  - move_in: Dolly the camera closer to the object (zoom in effect), keeping gaze fixed on it
  - move_out: Dolly the camera away from the object (zoom out / reveal effect), keeping gaze fixed on it
  - zoom_in_out: Optical zoom — camera stays still, focal length increases (narrowing FOV for a close-up feel) then returns to normal. Use for inspecting surface details without moving the camera.
  - zoom_out_in: Optical zoom — camera stays still, focal length decreases (widening FOV for a fish-eye/context feel) then returns to normal. Use for a dramatic wide-angle reveal while staying in place.
  - crane: Camera orbits the object while rising to a top-down overhead view (bird's eye reveal). Ends looking straight down (~90° pitch)
  - tilt_up: Stationary camera tilts upward (e.g., looking from ground level up to ceiling or sky)
  - tilt_down: Stationary camera tilts downward (e.g., looking from ceiling down to floor or object)
  - static: Hold camera position and gaze at the object without moving
"""

TRANSITIONAL_LIST = """
TRANSITIONAL TRAJECTORIES (move from current anchor to next anchor):
  - arc: Curved path connecting two anchors.
    The start and end points are fixed (determined by anchors).
    Parameter 'angle' (in degrees) controls how much the path curves:
      - angle=0: Straight line between anchors
      - angle=30: Gentle curve
      - angle=60: Moderate curve
      - angle=90: Wide sweeping curve
      - Negative values curve in the opposite direction
    Choose the angle to create smooth, collision-free transitions.
    Positive = curve left/up, Negative = curve right/down.
"""

VIEWING_PREFERENCE_INSTRUCTIONS = """
## VIEWING PREFERENCES

For each object in your sequence, you MUST specify viewing preferences that
tell the camera system how to best frame that object.

**elevation** — how high the camera should look from:
  - "low": Near eye-level, ~10-20° above horizontal. Best for tall objects
    whose information is on vertical surfaces (cabinets, wardrobes, bookshelves,
    paintings, doors, TVs, monitors, mirrors, refrigerators, windows)
  - "medium": Moderate angle, ~25-35°. Good default for most furniture
  - "high": Steep downward look, ~40-55°. Best for objects whose interesting
    content is on horizontal/top surfaces (tables, desks, sofas, beds, rugs,
    bathtubs, counters, sinks, nightstands, benches, ottomans)
  - "overhead": Near top-down, ~60-80°. For flat layouts (rugs, floor plans,
    place settings on a table)

**distance** — how far the camera should be:
  - "close": Tight framing, details visible. For small objects or detail inspection
  - "medium": Object fills ~60% of frame. Default for most furniture
  - "far": Wide framing, object in context. For large objects or room overview

**placement** — where the object physically sits in the room. Use the provided
`against_wall` and `attached_to_ceiling` metadata to determine this accurately.
This DIRECTLY constrains which trajectories are allowed:
  - "freestanding": Accessible from all sides — against_wall=false AND attached_to_ceiling=false.
    Tables, sofas, chairs in the middle of the room, rugs, kitchen islands, sculptures.
    → All trajectories are allowed.
  - "wall": Mounted on or flush against a wall — against_wall=true AND the object is
    vertically mounted (paintings, TVs, monitors, mirrors, windows, doors, curtains,
    wall clocks, shelves, picture frames).
    → FORBIDDEN: orbit_full, orbit_half (camera cannot go behind the wall).
    → USE INSTEAD: orbit_quarter, pan_left, pan_right, move_in, move_out,
      zoom_in_out, static, tilt_up, tilt_down.
  - "ceiling": Hanging from the ceiling — attached_to_ceiling=true.
    Chandeliers, ceiling fans, ceiling lights, pendant lamps, smoke detectors.
    → FORBIDDEN: crane (camera cannot rise above the ceiling).
    → USE INSTEAD: tilt_up, static, zoom_in_out.
  - "floor_wall": On the floor but pushed against a wall — against_wall=true AND the
    object sits on the floor (cabinets, refrigerators, desks against a wall, console
    tables, nightstands against a wall, kitchen counters, sinks).
    → FORBIDDEN: orbit_full (camera cannot go behind the wall).
    → ALLOWED BUT CAUTION: orbit_half (only if the half-orbit stays on the
      accessible side). Prefer orbit_quarter, pan, move_in, zoom_in_out.

Choose based on:
  1. The `against_wall` and `attached_to_ceiling` flags provided in the scene data
  2. WHERE the object's interesting/informative surface is (top vs front vs all-around)
  3. WHAT the user wants to see (detail inspection → close+high, overview → far+medium)
  4. The object's physical size (small objects need closer camera, large ones need more distance)

If the user's intent overrides the default (e.g., "look at the sofa cushion pattern"
→ high elevation even though sofa might normally be medium), follow the user's intent.
But NEVER override placement constraints — you cannot orbit behind a wall or crane
above a ceiling regardless of user intent.
"""

SYSTEM_PROMPT = '''
You are a camera trajectory planner for 3D scenes. Convert natural language into structured camera movements.

## TOOLS
- **get_anchor**: Find viewpoint for an object
- **infer_AtomTraj**: Generate camera trajectory
- **traj_compose**: Combine trajectory segments

## MOVEMENT TYPES
{object_level}
{transitional}

{viewing_preferences}

## TRAJECTORY PATTERN (1-3-1)

Follow this structure strictly:

```
1. Call Anchor Determinator with '<first_object>' (id: X)     ← INITIAL ANCHOR (once)

[LOOP for each transition:]
  2. Call AtomTraj with '<object_level_traj>' (object-level)    ← Examine current object
  3. Call Anchor Determinator with '<next_object>' (id: Y)    ← Next destination  
  4. Call AtomTraj with '<transitional_traj>' (transitional)    ← Move to next

N. Call AtomTraj with '<object_level_traj>' (object-level)      ← FINAL: Examine last object
```

### Pattern with 3 objects (A → B → C):
```
1. Anchor: A (id: 1)           ← Start at A
2. AtomTraj: object-level        ← Examine A
3. Anchor: B (id: 2)           ← Destination B
4. AtomTraj: transitional        ← Move A→B
5. AtomTraj: object-level        ← Examine B
6. Anchor: C (id: 3)           ← Destination C
7. AtomTraj: transitional        ← Move B→C
8. AtomTraj: object-level        ← Examine C (FINAL)
```

Formula: For N objects, steps = 3N - 1 (excluding traj_compose and render)

## RULES
1. Start with exactly ONE Anchor call (starting point)
2. Loop pattern: Object-Level → Anchor → Transitional
3. End with exactly ONE Object-Level call (examine final object)
4. ALWAYS include object IDs from scene data (these are string IDs like "table_0", "sofa_1")
5. ALWAYS label each AtomTraj as (object-level) or (transitional)
6. For 'arc' transitions, choose the angle based on spatial relationships — use larger angles when objects are close together or when obstacles lie on the direct path
7. For high-level requests (e.g., "tour"), select important objects logically
8. Use 'static' when the user implies pausing, resting, or lingering on an object without camera movement
9. Use 'move_in' when the user implies zooming in, getting closer, inspecting details, or focusing on an object
10. Use 'move_out' when the user implies pulling back, revealing context, getting an overview, or stepping away from an object
11. Set "user_specified_order" to true ONLY when the user explicitly names objects in a specific sequence (e.g., "start at the door, then the sofa, then the window"). Set it to false for open-ended requests like "give me a tour" or "show me the room" where YOU choose the order
12. Use 'crane' when the user implies rising above, bird's eye view, overhead reveal, ascending to look down, or "fly up and look down"
13. Use 'tilt_up' when the user implies looking upward, gazing at the ceiling, or scanning from ground to sky while staying in place
14. Use 'tilt_down' when the user implies looking downward, scanning from sky to ground, or inspecting the floor while staying in place
15. ALWAYS include "viewing_preferences" for every object in your sequence. Think about what surface of each object is most informative and what the user wants to see.
16. Use 'zoom_in_out' when the user implies optically zooming in to inspect fine details or textures without moving the camera (e.g., "zoom in on the fabric pattern", "get a closer look at the inscription"). Prefer this over move_in when physical camera movement is undesirable or the user explicitly says "zoom".
17. Use 'zoom_out_in' when the user implies a dramatic wide-angle reveal or fish-eye effect while staying in place (e.g., "show me a wide-angle view", "dramatic wide reveal without moving"). This is rare — use only when the user explicitly requests a wide/fish-eye stationary effect.
18. NEVER use 'orbit_full' or 'orbit_half' for objects with placement "wall" or "floor_wall". The camera CANNOT go behind a wall. Use orbit_quarter, pan_left, pan_right, move_in, move_out, zoom_in_out, or static instead. Common wall objects: paintings, TVs, mirrors, bookshelves, wardrobes, cabinets, windows, doors, radiators, wall clocks, shelves, curtains.
19. NEVER use 'crane' for objects with placement "ceiling". The camera CANNOT rise above the ceiling. Use tilt_up, static, or zoom_in_out instead. Common ceiling objects: chandeliers, ceiling fans, ceiling lights, pendant lamps, smoke detectors.
20. When choosing orbit size, ALWAYS consider the object's placement first. Only use orbit_full for freestanding objects that have open space on all sides. For ANY object near a wall, prefer orbit_quarter or smaller movements.
21. ALWAYS set the "placement" field in viewing_preferences for every object. Use the `against_wall` and `attached_to_ceiling` flags from the scene data to determine placement accurately.

## RESPONSE FORMAT
```json
{{{{
  "observation": "<single-line summary of user request>",
  "reasoning": "<single-line trajectory planning logic>",
  "user_specified_order": <true if the user explicitly stated the visitation sequence, false if you chose the order>,
  "object_sequence": ["obj1_id", "obj2_id", "obj3_id"],
  "viewing_preferences": {{{{
    "obj1_id": {{{{"elevation": "<low|medium|high|overhead>", "distance": "<close|medium|far>", "placement": "<freestanding|wall|ceiling|floor_wall>"}}}},
    "obj2_id": {{{{"elevation": "<low|medium|high|overhead>", "distance": "<close|medium|far>", "placement": "<freestanding|wall|ceiling|floor_wall>"}}}},
    "obj3_id": {{{{"elevation": "<low|medium|high|overhead>", "distance": "<close|medium|far>", "placement": "<freestanding|wall|ceiling|floor_wall>"}}}}
  }}}},
  "atomic_trajectories": "<numbered steps following 1-3-1 pattern>"
}}}}
```

All strings must be on single lines with no newline characters.
'''

EXAMPLES = '''
## EXAMPLES

### Example 1: Room Walkthrough (4 objects)
User: "Give me a tour of this room"
```json
{
  "observation": "User requests comprehensive room tour.",
  "reasoning": "Start at door (wall), visit sofa (freestanding), table (freestanding), bookshelf (wall). Door and bookshelf are against walls so use pan/orbit_quarter, not orbit_full/half.",
  "user_specified_order": false,
  "object_sequence": ["door_0", "sofa_0", "table_0", "bookshelf_0"],
  "viewing_preferences": {
    "door_0": {"elevation": "low", "distance": "medium", "placement": "wall"},
    "sofa_0": {"elevation": "high", "distance": "medium", "placement": "freestanding"},
    "table_0": {"elevation": "high", "distance": "medium", "placement": "freestanding"},
    "bookshelf_0": {"elevation": "low", "distance": "medium", "placement": "wall"}
  },
  "atomic_trajectories": "1. Call Anchor Determinator with 'door_0' (id: door_0). 2. Call AtomTraj with 'pan_right' (object-level). 3. Call Anchor Determinator with 'sofa_0' (id: sofa_0). 4. Call AtomTraj with 'arc', angle=30 (transitional). 5. Call AtomTraj with 'orbit_quarter' (object-level). 6. Call Anchor Determinator with 'table_0' (id: table_0). 7. Call AtomTraj with 'arc', angle=-45 (transitional). 8. Call AtomTraj with 'orbit_half' (object-level). 9. Call Anchor Determinator with 'bookshelf_0' (id: bookshelf_0). 10. Call AtomTraj with 'arc', angle=60 (transitional). 11. Call AtomTraj with 'pan_left' (object-level). 12. Call traj_compose. 13. Render video."
}
```

### Example 2: Specific Path (4 objects)
User: "A glance around the living room swept over the reception and the table, finally resting at the wardrobe."
```json
{
  "observation": "User describes path: living room glance → reception → table → wardrobe (rest).",
  "reasoning": "Use pan for 'glance', orbits for 'swept over', static for 'resting' since user implies lingering without movement. Reception is floor_wall and wardrobe is wall-mounted so use orbit_quarter instead of orbit_half. Arc angles chosen to avoid direct-line collisions between nearby furniture.",
  "user_specified_order": true,
  "object_sequence": ["living_room_center_0", "reception_0", "table_0", "wardrobe_0"],
  "viewing_preferences": {
    "living_room_center_0": {"elevation": "medium", "distance": "far", "placement": "freestanding"},
    "reception_0": {"elevation": "low", "distance": "medium", "placement": "floor_wall"},
    "table_0": {"elevation": "high", "distance": "medium", "placement": "freestanding"},
    "wardrobe_0": {"elevation": "low", "distance": "medium", "placement": "wall"}
  },
  "atomic_trajectories": "1. Call Anchor Determinator with 'living_room_center_0' (id: living_room_center_0). 2. Call AtomTraj with 'pan_right' (object-level). 3. Call Anchor Determinator with 'reception_0' (id: reception_0). 4. Call AtomTraj with 'arc', angle=0 (transitional). 5. Call AtomTraj with 'orbit_quarter' (object-level). 6. Call Anchor Determinator with 'table_0' (id: table_0). 7. Call AtomTraj with 'arc', angle=-30 (transitional). 8. Call AtomTraj with 'orbit_half' (object-level). 9. Call Anchor Determinator with 'wardrobe_0' (id: wardrobe_0). 10. Call AtomTraj with 'arc', angle=45 (transitional). 11. Call AtomTraj with 'static' (object-level). 12. Call traj_compose. 13. Render video."
}
```

### Example 3: Two Objects
User: "Show me the sofa then the window"
```json
{
  "observation": "User wants to see sofa then window.",
  "reasoning": "Two-object sequence: sofa is freestanding so orbit_quarter from above to see cushions, arc up to window which is wall-mounted so use pan_left instead of orbit.",
  "user_specified_order": true,
  "object_sequence": ["sofa_0", "window_0"],
  "viewing_preferences": {
    "sofa_0": {"elevation": "high", "distance": "medium", "placement": "freestanding"},
    "window_0": {"elevation": "low", "distance": "medium", "placement": "wall"}
  },
  "atomic_trajectories": "1. Call Anchor Determinator with 'sofa_0' (id: sofa_0). 2. Call AtomTraj with 'orbit_quarter' (object-level). 3. Call Anchor Determinator with 'window_0' (id: window_0). 4. Call AtomTraj with 'arc', angle=60 (transitional). 5. Call AtomTraj with 'pan_left' (object-level). 6. Call traj_compose. 7. Render video."
}
```

### Example 4: Single Object
User: "Give me a detailed view of the sculpture"
```json
{
  "observation": "User wants detailed view of single object.",
  "reasoning": "Single object: sculpture is freestanding so full orbit examination is safe, no transitions needed.",
  "user_specified_order": true,
  "object_sequence": ["sculpture_0"],
  "viewing_preferences": {
    "sculpture_0": {"elevation": "medium", "distance": "close", "placement": "freestanding"}
  },
  "atomic_trajectories": "1. Call Anchor Determinator with 'sculpture_0' (id: sculpture_0). 2. Call AtomTraj with 'orbit_full' (object-level). 3. Call traj_compose. 4. Render video."
}
```

### Example 5: Airbnb Walkthrough (6 objects)
User: "Can you give an incoming Airbnb guest a detailed walkthrough of this house?"
```json
{
  "observation": "User requests detailed Airbnb walkthrough for guest.",
  "reasoning": "Comprehensive tour starting from entrance. Door and window are wall-mounted → pan only. Reception is floor_wall and wardrobe is wall → orbit_quarter max. Table and sofa are freestanding → orbit_half OK. Arc angles navigate around furniture smoothly.",
  "user_specified_order": false,
  "object_sequence": ["door_0", "reception_0", "table_0", "sofa_0", "wardrobe_0", "window_0"],
  "viewing_preferences": {
    "door_0": {"elevation": "low", "distance": "medium", "placement": "wall"},
    "reception_0": {"elevation": "low", "distance": "medium", "placement": "floor_wall"},
    "table_0": {"elevation": "high", "distance": "medium", "placement": "freestanding"},
    "sofa_0": {"elevation": "high", "distance": "far", "placement": "freestanding"},
    "wardrobe_0": {"elevation": "low", "distance": "medium", "placement": "wall"},
    "window_0": {"elevation": "low", "distance": "medium", "placement": "wall"}
  },
  "atomic_trajectories": "1. Call Anchor Determinator with 'door_0' (id: door_0). 2. Call AtomTraj with 'pan_right' (object-level). 3. Call Anchor Determinator with 'reception_0' (id: reception_0). 4. Call AtomTraj with 'arc', angle=0 (transitional). 5. Call AtomTraj with 'orbit_quarter' (object-level). 6. Call Anchor Determinator with 'table_0' (id: table_0). 7. Call AtomTraj with 'arc', angle=-30 (transitional). 8. Call AtomTraj with 'orbit_half' (object-level). 9. Call Anchor Determinator with 'sofa_0' (id: sofa_0). 10. Call AtomTraj with 'arc', angle=45 (transitional). 11. Call AtomTraj with 'orbit_half' (object-level). 12. Call Anchor Determinator with 'wardrobe_0' (id: wardrobe_0). 13. Call AtomTraj with 'arc', angle=0 (transitional). 14. Call AtomTraj with 'orbit_quarter' (object-level). 15. Call Anchor Determinator with 'window_0' (id: window_0). 16. Call AtomTraj with 'arc', angle=60 (transitional). 17. Call AtomTraj with 'static' (object-level). 18. Call traj_compose. 19. Render video."
}
```

### Example 6: Pause and Observe
User: "Fly to the painting and just stay there looking at it"
```json
{
  "observation": "User wants to move to the painting and hold a static view.",
  "reasoning": "Single object with emphasis on stillness: use static to hold camera at the anchor without any movement. Painting is wall-mounted so static is the perfect choice.",
  "user_specified_order": true,
  "object_sequence": ["painting_0"],
  "viewing_preferences": {
    "painting_0": {"elevation": "low", "distance": "medium", "placement": "wall"}
  },
  "atomic_trajectories": "1. Call Anchor Determinator with 'painting_0' (id: painting_0). 2. Call AtomTraj with 'static' (object-level). 3. Call traj_compose. 4. Render video."
}
```

### Example 7: Close-up Inspection (move_in)
User: "Zoom into the clock on the wall to see the details, then pull back and look around the room"
```json
{
  "observation": "User wants to zoom into clock details, then pull back for room context.",
  "reasoning": "Clock is wall-mounted → use move_in (allowed for wall objects) for detail inspection, NOT orbit. Then transition to room center (freestanding) for pan overview.",
  "user_specified_order": true,
  "object_sequence": ["clock_0", "room_center_0"],
  "viewing_preferences": {
    "clock_0": {"elevation": "low", "distance": "close", "placement": "wall"},
    "room_center_0": {"elevation": "medium", "distance": "far", "placement": "freestanding"}
  },
  "atomic_trajectories": "1. Call Anchor Determinator with 'clock_0' (id: clock_0). 2. Call AtomTraj with 'move_in' (object-level). 3. Call Anchor Determinator with 'room_center_0' (id: room_center_0). 4. Call AtomTraj with 'arc', angle=30 (transitional). 5. Call AtomTraj with 'pan_right' (object-level). 6. Call traj_compose. 7. Render video."
}
```

### Example 8: Dramatic Reveal (move_out)
User: "Start close to the fireplace then slowly reveal the whole living area"
```json
{
  "observation": "User wants a dramatic reveal starting close to the fireplace and pulling back.",
  "reasoning": "Fireplace is wall-mounted → move_out is allowed (camera pulls back from wall into open space), capturing the cinematic pull-back the user described.",
  "user_specified_order": true,
  "object_sequence": ["fireplace_0"],
  "viewing_preferences": {
    "fireplace_0": {"elevation": "low", "distance": "close", "placement": "wall"}
  },
  "atomic_trajectories": "1. Call Anchor Determinator with 'fireplace_0' (id: fireplace_0). 2. Call AtomTraj with 'move_out' (object-level). 3. Call traj_compose. 4. Render video."
}
```
'''

USER_PROMPT_TEMPLATE = '''
USER REQUEST: "{user_words}"

SCENE OBJECTS (use these IDs in your response):
{objects_json}

ROOMS:
{rooms_json}

INSTRUCTIONS:
1. Analyze the request and determine object sequence
2. Follow the 1-3-1 pattern strictly:
   - Start: ONE Anchor call
   - Loop: Object-Level → Anchor → Transitional
   - End: ONE Object-Level call
3. Include object labels AND IDs in every Anchor call (IDs are string keys like "table_0", "sofa_1")
4. Label every AtomTraj as (object-level) or (transitional)
5. For object-level: choose from orbit_full, orbit_half, orbit_quarter, pan_left, pan_right, move_in, move_out, zoom_in_out, zoom_out_in, crane, tilt_up, tilt_down, static
6. For transitional: use 'arc' with an angle parameter in degrees (e.g., angle=30)
7. Use 'static' when the user implies holding, resting, pausing, or lingering at an object
8. Use 'move_in' when the user implies physically moving closer, inspecting details by approaching
9. Use 'move_out' when the user implies physically pulling back, revealing context by retreating
10. Use 'zoom_in_out' when the user implies optically zooming in to see fine details/textures without moving the camera (the zoom returns to normal automatically)
11. Use 'zoom_out_in' when the user implies a dramatic stationary wide-angle effect (the zoom returns to normal automatically)
12. ALWAYS include "viewing_preferences" with "elevation", "distance", AND "placement" for EVERY object in your sequence
13. CRITICAL PLACEMENT RULES — use the `against_wall` and `attached_to_ceiling` flags from the scene data:
    - "wall" objects (against_wall=true, vertically mounted: paintings, TVs, mirrors, windows, doors, curtains): NEVER orbit_full or orbit_half
    - "ceiling" objects (attached_to_ceiling=true): NEVER crane
    - "floor_wall" objects (against_wall=true, on floor: cabinets, refrigerators, kitchen counters, desks against wall): NEVER orbit_full
    - "freestanding" objects (against_wall=false, attached_to_ceiling=false): all trajectories allowed

Generate your response in the specified JSON format.
'''


# ==============================================================================
# Helper Functions
# ==============================================================================

def build_system_prompt() -> str:
    """Assemble the complete system prompt"""
    return SYSTEM_PROMPT.format(
        object_level=OBJECT_LEVEL_LIST,
        transitional=TRANSITIONAL_LIST,
        viewing_preferences=VIEWING_PREFERENCE_INSTRUCTIONS,
    ) + "\n" + EXAMPLES


def build_user_prompt(user_words: str, objects_summary: list, rooms: dict = None) -> str:
    """Build user prompt with scene context"""
    rooms_json = json.dumps(rooms, indent=2) if rooms else "{}"
    return USER_PROMPT_TEMPLATE.format(
        user_words=user_words,
        objects_json=json.dumps(objects_summary, indent=2),
        rooms_json=rooms_json,
    )


def get_prompts_for_api_call(user_words: str, objects_summary: list, rooms: dict = None) -> tuple:
    """
    Returns (system_prompt, user_prompt) ready for OpenAI API call.

    Usage:
        system_prompt, user_prompt = get_prompts_for_api_call(user_input, objects, rooms)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
        )
    """
    return build_system_prompt(), build_user_prompt(user_words, objects_summary, rooms)


def get_all_objects_summary_from_scene_graph(scene_json: dict) -> list:
    """
    Extract all objects with their positions from the new scene graph JSON format.

    New format:
        {
          "objects": {
              "table_0": {
                  "obb": [cx, cy, cz, sx, sy, sz, qx, qy, qz, qw],
                  "against_wall": true,
                  "attached_to_ceiling": false
              },
              ...
          },
          "rooms": { "living_room_0": ["table_0", ...], ... }
        }

    Returns list of dicts with:
        - id: string key (e.g., "table_0")
        - label: class name (e.g., "table")
        - center: [cx, cy, cz]
        - size: [sx, sy, sz]
        - against_wall: bool
        - attached_to_ceiling: bool
        - room: room name if available
    """
    objects = scene_json.get("objects", {})
    rooms = scene_json.get("rooms", {})

    # Build reverse mapping: object_id → room_name
    obj_to_room = {}
    for room_name, obj_ids in rooms.items():
        for obj_id in obj_ids:
            obj_to_room[obj_id] = room_name

    summary = []
    for obj_id, obj_data in objects.items():
        obb = obj_data["obb"]
        cx, cy, cz = obb[0], obb[1], obb[2]
        sx, sy, sz = obb[3], obb[4], obb[5]

        # Derive class label from the key (strip trailing _N)
        label = obj_id.rsplit("_", 1)[0]

        summary.append({
            "id": obj_id,
            "label": label,
            "center": [round(cx, 3), round(cy, 3), round(cz, 3)],
            "size": [round(sx, 3), round(sy, 3), round(sz, 3)],
            "against_wall": obj_data.get("against_wall", False),
            "attached_to_ceiling": obj_data.get("attached_to_ceiling", False),
            "room": obj_to_room.get(obj_id),
        })
    return summary


# Keep the old function as a fallback for legacy labels.json files
def get_all_objects_summary(objects: list) -> list:
    """Extract all objects with their positions from legacy scene labels (labels.json)."""
    summary = []
    for obj in objects:
        label = obj["label"]
        ins_id = obj["ins_id"]

        if "bounding_box" in obj and obj["bounding_box"]:
            bbox = obj["bounding_box"]
            center_x = sum(p["x"] for p in bbox) / 8
            center_y = sum(p["y"] for p in bbox) / 8
            center_z = sum(p["z"] for p in bbox) / 8
            summary.append({
                "id": ins_id,
                "label": label,
                "center": [round(center_x, 2), round(center_y, 2), round(center_z, 2)]
            })
    return summary


# ==============================================================================
# Constants for validation
# ==============================================================================

VALID_OBJECT_LEVEL = {
    "orbit_full", "orbit_half", "orbit_quarter",
    "pan_left", "pan_right",
    "move_in", "move_out",
    "zoom_in_out", "zoom_out_in",
    "crane", "tilt_up", "tilt_down",
    "static",
}
VALID_TRANSITIONAL = {"arc"}
VALID_ELEVATIONS = {"low", "medium", "high", "overhead"}
VALID_DISTANCES = {"close", "medium", "far"}
VALID_PLACEMENTS = {"freestanding", "wall", "ceiling", "floor_wall"}


# ==============================================================================
# Validation
# ==============================================================================

def validate_viewing_preferences(parsed: dict) -> dict:
    """
    Validate the viewing_preferences block from the LLM response.

    Returns a dict with 'valid', 'errors', 'warnings', and the
    cleaned 'viewing_preferences' (with defaults filled in).
    """
    result = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "viewing_preferences": {},
    }

    prefs = parsed.get("viewing_preferences")
    obj_sequence = parsed.get("object_sequence", [])

    if prefs is None:
        result["warnings"].append(
            "No viewing_preferences found — will use geometry-based defaults"
        )
        return result

    if not isinstance(prefs, dict):
        result["warnings"].append(
            f"viewing_preferences is not a dict (got {type(prefs).__name__}) — ignoring"
        )
        return result

    for obj_name in obj_sequence:
        if obj_name not in prefs:
            result["warnings"].append(
                f"Missing viewing_preferences for '{obj_name}' — will use defaults"
            )
            continue

        obj_prefs = prefs[obj_name]
        cleaned = {}

        # Validate elevation
        elev = obj_prefs.get("elevation", "medium")
        if elev not in VALID_ELEVATIONS:
            result["warnings"].append(
                f"Invalid elevation '{elev}' for '{obj_name}' — defaulting to 'medium'"
            )
            elev = "medium"
        cleaned["elevation"] = elev

        # Validate distance
        dist = obj_prefs.get("distance", "medium")
        if dist not in VALID_DISTANCES:
            result["warnings"].append(
                f"Invalid distance '{dist}' for '{obj_name}' — defaulting to 'medium'"
            )
            dist = "medium"
        cleaned["distance"] = dist

        # Validate placement
        placement = obj_prefs.get("placement", "freestanding")
        if placement not in VALID_PLACEMENTS:
            result["warnings"].append(
                f"Invalid placement '{placement}' for '{obj_name}' — defaulting to 'freestanding'"
            )
            placement = "freestanding"
        cleaned["placement"] = placement

        result["viewing_preferences"][obj_name] = cleaned

    return result


def validate_1_3_1_pattern(atomic_trajectories: str) -> dict:
    """
    Validate that the atomic_trajectories follows the 1-3-1 pattern.

    Expected:
    - Starts with Anchor
    - Loop: ObjectLevel → Anchor → Transitional
    - Ends with ObjectLevel
    """
    import re

    result = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "pattern": [],
        "step_count": 0,
        "object_count": 0
    }

    # Parse steps
    steps = atomic_trajectories.split(".")
    steps = [s.strip() for s in steps if s.strip()]

    for step in steps:
        step_lower = step.lower()

        # Skip compose and render steps
        if "traj_compose" in step_lower or "render" in step_lower:
            continue

        result["step_count"] += 1

        if "anchor" in step_lower:
            result["pattern"].append("A")
            result["object_count"] += 1
        elif "atomtraj" in step_lower:
            if "object-level" in step_lower:
                result["pattern"].append("O")
                # Validate movement name
                for name in VALID_OBJECT_LEVEL:
                    if name in step_lower:
                        break
                else:
                    result["warnings"].append(f"Unknown object-level movement: {step[:60]}...")
            elif "transitional" in step_lower:
                result["pattern"].append("T")
                # Validate 'arc', angle=N format
                if not re.search(r"'arc'", step_lower) and "arc" not in step_lower:
                    result["warnings"].append(f"Transitional should use 'arc' with angle: {step[:60]}...")
            else:
                result["pattern"].append("G")
                result["warnings"].append(f"AtomTraj not labeled: {step[:50]}...")

    pattern_str = "".join(result["pattern"])

    # Validate pattern
    if not pattern_str:
        result["errors"].append("No valid steps found")
        result["valid"] = False
        return result

    # Rule 1: Must start with Anchor
    if not pattern_str.startswith("A"):
        result["errors"].append("Pattern must start with Anchor")
        result["valid"] = False

    # Rule 2: Must end with Object-Level
    if not pattern_str.endswith("O") and not pattern_str.endswith("G"):
        result["errors"].append("Pattern must end with Object-Level AtomTraj")
        result["valid"] = False

    # Rule 3: Check for consecutive same types (except allowed patterns)
    for i in range(1, len(pattern_str)):
        curr, prev = pattern_str[i], pattern_str[i-1]

        if curr == "A" and prev == "A":
            result["errors"].append(f"Consecutive Anchors at position {i}")
            result["valid"] = False

        if curr == "T" and prev == "T":
            result["errors"].append(f"Consecutive Transitionals at position {i}")
            result["valid"] = False

    # Rule 4: After transitional must come object-level (in the loop)
    for i in range(len(pattern_str) - 1):
        if pattern_str[i] == "T" and pattern_str[i+1] not in ["O", "G"]:
            result["errors"].append(f"Transitional at {i} not followed by Object-Level")
            result["valid"] = False

    # Check expected step count: 3N - 1 for N objects
    expected_steps = 3 * result["object_count"] - 1
    actual_steps = len([p for p in pattern_str if p in "AOT"])

    if actual_steps != expected_steps and result["object_count"] > 0:
        result["warnings"].append(
            f"Expected {expected_steps} steps for {result['object_count']} objects, got {actual_steps}"
        )

    return result


def parse_response(response_text: str) -> dict:
    """Parse the LLM response JSON"""
    try:
        start = response_text.find("{")
        end = response_text.rfind("}") + 1
        if start >= 0 and end > start:
            json_str = response_text[start:end]
            return json.loads(json_str)
    except json.JSONDecodeError:
        pass
    return None


# ==============================================================================
# Main Dialog Function
# ==============================================================================

def run_camera_dialog(
    user_words: str,
    objects_summary: list,
    api_key: str,
    rooms: dict = None,
    model: str = MODEL_NAME,
    temperature: float = 0.1,
    verbose: bool = True
) -> dict:
    """
    Run the camera trajectory dialog system.

    Args:
        user_words: User's natural language request
        objects_summary: List of objects in the scene
        api_key: OpenAI API key
        rooms: Optional room→object mapping
        model: Model to use (default: gpt-4.1)
        temperature: Sampling temperature (default: 0.1)
        verbose: Print debug information

    Returns:
        dict with keys: response, parsed, validation, pref_validation, success
    """
    client = OpenAI(api_key=api_key)

    system_prompt, user_prompt = get_prompts_for_api_call(user_words, objects_summary, rooms)

    if verbose:
        print("=" * 60)
        print("CAMERA TRAJECTORY DIALOG")
        print("=" * 60)
        print(f"User request: {user_words}")
        print(f"Objects in scene: {len(objects_summary)}")
        print("-" * 60)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
        )

        response_text = response.choices[0].message.content
        parsed = parse_response(response_text)

        validation = None
        pref_validation = None
        if parsed:
            if "atomic_trajectories" in parsed:
                validation = validate_1_3_1_pattern(parsed["atomic_trajectories"])
            pref_validation = validate_viewing_preferences(parsed)

        if verbose:
            print("\nLLM RESPONSE:")
            print("-" * 60)
            print(response_text)

            if validation:
                print("\n" + "-" * 60)
                print("TRAJECTORY VALIDATION:")
                print(f"  Valid: {validation['valid']}")
                print(f"  Pattern: {' → '.join(validation['pattern'])}")
                print(f"  Objects: {validation['object_count']}")
                if validation['errors']:
                    print(f"  Errors: {validation['errors']}")
                if validation['warnings']:
                    print(f"  Warnings: {validation['warnings']}")

            if pref_validation:
                print("\nVIEWING PREFERENCES VALIDATION:")
                print(f"  Valid: {pref_validation['valid']}")
                if pref_validation["viewing_preferences"]:
                    for obj, p in pref_validation["viewing_preferences"].items():
                        print(f"    {obj}: elevation={p['elevation']}, distance={p['distance']}, placement={p['placement']}")
                if pref_validation['warnings']:
                    print(f"  Warnings: {pref_validation['warnings']}")

        return {
            "success": True,
            "response": response_text,
            "parsed": parsed,
            "validation": validation,
            "pref_validation": pref_validation,
        }

    except Exception as e:
        if verbose:
            print(f"Error: {e}")
        return {
            "success": False,
            "error": str(e),
            "response": None,
            "parsed": None,
            "validation": None,
            "pref_validation": None,
        }


def run_from_scene_graph(
    user_words: str,
    scene_graph_path: str,
    api_key: str,
    **kwargs
) -> tuple:
    """
    Run dialog with scene loaded from the new scene graph JSON file.

    Args:
        user_words: User's natural language request
        scene_graph_path: Path to the scene graph JSON file (new format)
        api_key: OpenAI API key

    Returns:
        (dialog_result, objects_summary)
    """
    with open(scene_graph_path, "r") as f:
        scene_json = json.load(f)

    objects_summary = get_all_objects_summary_from_scene_graph(scene_json)
    rooms = scene_json.get("rooms", {})

    return run_camera_dialog(
        user_words, objects_summary, api_key, rooms=rooms, **kwargs
    ), objects_summary


# Keep old function as fallback
def run_from_scene_file(
    user_words: str,
    labels_path: str,
    api_key: str,
    **kwargs
) -> tuple:
    """
    Run dialog with scene loaded from legacy labels.json file.
    """
    with open(labels_path, "r") as f:
        object_labels = json.load(f)

    objects_summary = get_all_objects_summary(object_labels)

    return run_camera_dialog(user_words, objects_summary, api_key, **kwargs), object_labels


# ==============================================================================
# Example Usage
# ==============================================================================

if __name__ == "__main__":
    test_prompts = [
        "Can you give an incoming Airbnb guest a detailed walkthrough of this house? List at least 8 objects, and use zoom_in_out or zoom_out_in at least once",
        "A glance around the living room swept over the console and the central table, finally resting in front of the wardrobe.",
        "Start from the door, move to the sofa, then end at the window.",
        "Show me a 360 view of the coffee table, then move to examine the cabinet.",
        "Zoom into the painting on the wall to see the brushwork, then pull back to see the whole room.",
        "Show me a 360 view of the central table (id: table_0), then zoom in the wardrobe, and finally dolly in to the cabinet.",
    ]

    user_words = test_prompts[0]

    # --- New format: scene graph JSON ---
    scene_graph_path = os.environ.get(
        "CINEMATRAJ_SCENE_GRAPH",
        "data/ScanNetpp/scenes/09c1414f1b/dslr/sg/09c1414f1b-simple.json",
    )

    api_key = os.environ.get("OPENAI_API_KEY")

    llm_result, objects_summary = run_from_scene_graph(
        user_words=user_words,
        scene_graph_path=scene_graph_path,
        api_key=api_key
    )

    parsed_response = llm_result['parsed']

    result = validate_and_clean_response(
        parsed_response,
        objects_summary,
        auto_correct=True,
        delete_nonexistent=True
    )

    print_full_validation_report(result)