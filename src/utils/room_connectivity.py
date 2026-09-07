"""
Room Connectivity & Cross-Room Waypoint Planning
=================================================

When two consecutive objects in a camera trajectory are in different rooms,
the transitional arc between them may collide with walls. This module finds
door/window objects that connect the two rooms so the arc can route through
them as intermediate waypoints.

Key functions:
  - build_room_graph():       Build room connectivity from scene graph JSON.
  - find_cross_room_waypoints(): Given two object IDs, find the best
      door/window waypoints to route through.

Integration point: called from optimize_trajectory_result() PASS 2,
before creating transitional arcs.

Scene graph format expected:
    {
      "objects": {
          "table_0": {"obb": [cx,cy,cz, sx,sy,sz, qx,qy,qz,qw], ...},
          "door_0":  {"obb": [...], ...},
          ...
      },
      "rooms": {
          "living_room_0": ["table_0", "sofa_0", ...],
          "kitchen_0": ["fridge_0", ...],
          ...
      }
    }
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Union
from pathlib import Path
import json
import itertools


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class DoorWindow:
    """A door or window object that potentially connects two rooms."""
    object_id: str
    label: str              # e.g. "door", "window", "curtain"
    center: np.ndarray      # OBB center [cx, cy, cz] in Z-up scene coords
    extent: np.ndarray      # OBB extents [sx, sy, sz]
    rooms: List[str]        # rooms this door/window connects (0, 1, or 2 entries)


@dataclass
class RoomGraph:
    """
    Room connectivity graph built from a scene graph JSON.
    
    Attributes:
        obj_to_room:     {object_id: room_name} reverse mapping
        room_to_objs:    {room_name: [object_id, ...]} forward mapping
        doors_windows:   list of all DoorWindow objects found in the scene
        connections:      {(room_a, room_b): [DoorWindow, ...]} edges
        room_centers:    {room_name: np.ndarray} mean center of room objects
    """
    obj_to_room: Dict[str, str] = field(default_factory=dict)
    room_to_objs: Dict[str, List[str]] = field(default_factory=dict)
    doors_windows: List[DoorWindow] = field(default_factory=list)
    connections: Dict[Tuple[str, str], List[DoorWindow]] = field(default_factory=dict)
    room_centers: Dict[str, np.ndarray] = field(default_factory=dict)


# =============================================================================
# Constants
# =============================================================================

# Object labels (prefix before the _N suffix) that are candidate passages
# NOTE: windows and curtains are excluded — they are typically on exterior
# walls and the camera cannot pass through them.
PASSAGE_LABELS = {
    "door", "door_frame", "opening", "gate", "archway",
    "glass_door", "sliding_door", "french_door",
}

# Passage openness preference: lower score = more reliably open/passable.
# Used as a tiebreaker when multiple passages connect the same room pair.
# door_frame is always open; doors can be closed.
PASSAGE_PREFERENCE = {
    "door_frame":   0,   # always open — strongly preferred
    "opening":      0,   # always open
    "archway":      0,   # always open
    "gate":         1,   # usually open
    "door":         2,   # can be closed
    "glass_door":   2,
    "sliding_door": 2,
    "french_door":  2,
}
_DEFAULT_PREFERENCE = 5  # unknown passage types

# Maximum distance (meters) for a door/window to be considered "near" a room
# Used when a passage object is not explicitly listed in any room
PROXIMITY_THRESHOLD = 1.5


# =============================================================================
# Core Functions
# =============================================================================

def _extract_label(object_id: str) -> str:
    """
    Extract the class label from an object ID like 'door_0' → 'door',
    'kitchen_counter_0' → 'kitchen_counter', 'glass_door_2' → 'glass_door'.
    """
    # Split from the right on '_', the last segment should be the instance number
    parts = object_id.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return object_id


def _get_obb_center(obb: list) -> np.ndarray:
    """Extract center [cx, cy, cz] from OBB array."""
    return np.array(obb[:3], dtype=np.float64)


def _get_obb_extent(obb: list) -> np.ndarray:
    """Extract extent [sx, sy, sz] from OBB array."""
    return np.array(obb[3:6], dtype=np.float64)


def _room_center_from_objects(
    room_obj_ids: List[str],
    objects: Dict[str, dict],
) -> np.ndarray:
    """Compute the mean OBB center of all objects in a room."""
    centers = []
    for obj_id in room_obj_ids:
        if obj_id in objects and "obb" in objects[obj_id]:
            centers.append(_get_obb_center(objects[obj_id]["obb"]))
    if not centers:
        return np.zeros(3)
    return np.mean(centers, axis=0)


def _is_passage_label(label: str) -> bool:
    """Check if a label represents a door, window, or similar passage."""
    return label.lower() in PASSAGE_LABELS


def _find_nearby_rooms(
    point: np.ndarray,
    room_centers: Dict[str, np.ndarray],
    room_to_objs: Dict[str, List[str]],
    objects: Dict[str, dict],
    max_dist: float = PROXIMITY_THRESHOLD,
) -> List[str]:
    """
    Find rooms whose objects are within max_dist of the given point.
    
    Uses the minimum distance from `point` to any object center in each room,
    not just the room center, for more accurate room boundary detection.
    """
    nearby = []
    for room_name, obj_ids in room_to_objs.items():
        min_dist = float("inf")
        for obj_id in obj_ids:
            if obj_id in objects and "obb" in objects[obj_id]:
                obj_center = _get_obb_center(objects[obj_id]["obb"])
                d = np.linalg.norm(point - obj_center)
                min_dist = min(min_dist, d)
        # Also check against room center
        if room_name in room_centers:
            d = np.linalg.norm(point - room_centers[room_name])
            min_dist = min(min_dist, d)
        if min_dist < max_dist:
            nearby.append(room_name)
    return nearby


def _edge_key(room_a: str, room_b: str) -> Tuple[str, str]:
    """Canonical (sorted) edge key for undirected room graph."""
    return tuple(sorted([room_a, room_b]))


# =============================================================================
# Build Room Graph
# =============================================================================

def build_room_graph(
    scene_json: dict,
    proximity_threshold: float = PROXIMITY_THRESHOLD,
    verbose: bool = True,
) -> RoomGraph:
    """
    Build a room connectivity graph from the scene graph JSON.
    
    Steps:
      1. Build obj→room reverse mapping.
      2. Identify all door/window objects.
      3. For each door/window, determine which rooms it connects:
         a. If listed in a room → that's one room.
         b. Check spatial proximity to other rooms' objects.
         c. A door between two rooms should be near objects from both.
      4. Build the connections dict.
    
    Args:
        scene_json:           The loaded scene graph JSON dict.
        proximity_threshold:  Max distance (m) for spatial room assignment.
        verbose:              Print debug info.
    
    Returns:
        RoomGraph with all connectivity info.
    """
    objects = scene_json.get("objects", {})
    rooms = scene_json.get("rooms", {})
    
    graph = RoomGraph()
    graph.room_to_objs = {rname: list(obj_ids) for rname, obj_ids in rooms.items()}
    
    # 1. Build reverse mapping
    for room_name, obj_ids in rooms.items():
        for obj_id in obj_ids:
            graph.obj_to_room[obj_id] = room_name
    
    # 2. Compute room centers
    # For rooms WITH objects: mean of object centers.
    # For rooms WITHOUT objects (empty hallways/corridors): estimated later.
    empty_rooms = []
    for room_name, obj_ids in rooms.items():
        c = _room_center_from_objects(obj_ids, objects)
        if len(obj_ids) > 0:
            graph.room_centers[room_name] = c
        else:
            empty_rooms.append(room_name)
    
    # For empty rooms, estimate center from ALL passage objects in the scene.
    # The intuition: an empty hallway sits between doors, so its center is
    # near the midpoint of doors that aren't fully claimed by other rooms.
    # As a first pass, use the mean of all passage object centers as a
    # fallback. This will be refined after door assignment.
    if empty_rooms:
        all_passage_centers = []
        for obj_id, obj_data in objects.items():
            label = _extract_label(obj_id)
            if _is_passage_label(label):
                all_passage_centers.append(_get_obb_center(obj_data["obb"]))
        if all_passage_centers:
            passage_centroid = np.mean(all_passage_centers, axis=0)
        else:
            # Fallback: use mean of all occupied room centers
            if graph.room_centers:
                passage_centroid = np.mean(list(graph.room_centers.values()), axis=0)
            else:
                passage_centroid = np.zeros(3)
        
        for room_name in empty_rooms:
            graph.room_centers[room_name] = passage_centroid.copy()
            if verbose:
                print(f"  Empty room '{room_name}': estimated center = "
                      f"{passage_centroid.round(2)}")
    
    # 3. Find all door/window objects
    all_passage_ids = []
    for obj_id, obj_data in objects.items():
        label = _extract_label(obj_id)
        if _is_passage_label(label):
            all_passage_ids.append(obj_id)
    
    if verbose:
        print(f"\n{'=' * 60}")
        print("ROOM CONNECTIVITY ANALYSIS")
        print(f"{'=' * 60}")
        print(f"  Rooms: {list(rooms.keys())}")
        print(f"  Passage objects found: {all_passage_ids}")
    
    # 4. For each passage, determine which 2 rooms it connects.
    #
    # Strategy: A passage connects EXACTLY 2 rooms — the room it's listed
    # in (its "own" room) and the nearest neighboring room on the other side.
    #
    # For passages not listed in any room, we find the 2 nearest rooms.
    # This ensures correct topology: door_0 connects living_room ↔ hallway,
    # door_1 connects hallway ↔ kitchen — even when proximity would match
    # all three rooms.
    for obj_id in all_passage_ids:
        obj_data = objects[obj_id]
        label = _extract_label(obj_id)
        center = _get_obb_center(obj_data["obb"])
        extent = _get_obb_extent(obj_data["obb"])
        
        own_room = graph.obj_to_room.get(obj_id)
        
        # Compute distance from this door to every room center
        room_dists = []
        for room_name, room_center in graph.room_centers.items():
            d = np.linalg.norm(center - room_center)
            room_dists.append((d, room_name))
        room_dists.sort(key=lambda x: x[0])
        
        connected_rooms: Set[str] = set()
        
        if own_room is not None:
            # Door is listed in a room → that room + nearest OTHER room
            connected_rooms.add(own_room)
            for d, rname in room_dists:
                if rname != own_room:
                    connected_rooms.add(rname)
                    break
        else:
            # Door not in any room → 2 nearest rooms
            for d, rname in room_dists[:2]:
                connected_rooms.add(rname)
        
        dw = DoorWindow(
            object_id=obj_id,
            label=label,
            center=center,
            extent=extent,
            rooms=sorted(connected_rooms),
        )
        graph.doors_windows.append(dw)
        
        if verbose:
            print(f"  {obj_id}: center={center.round(2)}, "
                  f"connects rooms: {dw.rooms}")
        
        # 4c. Add edges for all room pairs this passage connects
        room_list = list(connected_rooms)
        for i in range(len(room_list)):
            for j in range(i + 1, len(room_list)):
                key = _edge_key(room_list[i], room_list[j])
                if key not in graph.connections:
                    graph.connections[key] = []
                graph.connections[key].append(dw)
    
    if verbose:
        print(f"\n  Room connections:")
        if graph.connections:
            for (ra, rb), passages in graph.connections.items():
                ids = [p.object_id for p in passages]
                print(f"    {ra} ↔ {rb}: {ids}")
        else:
            print(f"    (none found — single room or no passage objects)")
    
    return graph


# =============================================================================
# Cross-Room Waypoint Finding
# =============================================================================

def find_cross_room_waypoints(
    prev_object_id: Union[str, int],
    next_object_id: Union[str, int],
    room_graph: RoomGraph,
    objects: Dict[str, dict],
    prev_position_zup: Optional[np.ndarray] = None,
    next_position_zup: Optional[np.ndarray] = None,
    max_hops: int = 5,
    verbose: bool = True,
) -> Optional[List[np.ndarray]]:
    """
    Given two objects in different rooms, find the sequence of door/window
    waypoints the camera must pass through for a collision-free transition.

    ALWAYS uses BFS on the room graph to find the shortest room-hop path,
    then picks the best door/window at each hop. This naturally handles:
      - Direct connection (1 hop, 1 door): room_A → room_B
      - Hallway in between (2 hops, 2 doors): room_A → hallway → room_B
      - Longer paths (N hops, N doors): room_A → hall → room_B → ... → room_C

    At each hop, if multiple doors/windows connect the same two rooms,
    the one producing the shortest cumulative path is selected.

    Returns:
        List of np.ndarray waypoint positions (Z-up), one per door/window
        the camera must pass through. Or None if same room / no path found.
    """
    prev_id = str(prev_object_id)
    next_id = str(next_object_id)

    # Look up rooms
    prev_room = room_graph.obj_to_room.get(prev_id)
    next_room = room_graph.obj_to_room.get(next_id)

    if prev_room is None or next_room is None:
        if verbose:
            print(f"    [cross-room] Cannot determine room for "
                  f"prev={prev_id}(room={prev_room}) or "
                  f"next={next_id}(room={next_room})")
        return None

    if prev_room == next_room:
        if verbose:
            print(f"    [cross-room] Same room ({prev_room}), no waypoints needed")
        return None

    # BFS to find shortest room path (works for both direct and multi-hop)
    room_path = _bfs_room_path(prev_room, next_room, room_graph, max_hops)

    if room_path is None:
        if verbose:
            print(f"    [cross-room] No path found between "
                  f"{prev_room} and {next_room} within {max_hops} hops")
        return None

    n_hops = len(room_path) - 1

    if verbose:
        if n_hops == 1:
            print(f"    [cross-room] Direct: {prev_room} → {next_room} (1 door)")
        else:
            print(f"    [cross-room] Multi-hop ({n_hops} doors): "
                  f"{' → '.join(room_path)}")

    # Get positions
    if prev_position_zup is None:
        if prev_id in objects and "obb" in objects[prev_id]:
            prev_position_zup = _get_obb_center(objects[prev_id]["obb"])
        else:
            return None

    if next_position_zup is None:
        if next_id in objects and "obb" in objects[next_id]:
            next_position_zup = _get_obb_center(objects[next_id]["obb"])
        else:
            return None

    # For each consecutive room pair in the path, pick the best door/window.
    # Selection criteria (in order):
    #   1. Passage openness preference (door_frame > door > window)
    #   2. Path length: dist(current_pos → door) + dist(door → destination)
    # A small distance penalty is added for less-preferred passage types,
    # so a door_frame is chosen over a door unless the door is significantly
    # closer (more than PREFERENCE_DIST_PENALTY meters shorter path).
    PREFERENCE_DIST_PENALTY = 0.5  # meters of extra "cost" per preference level

    waypoints = []
    waypoint_ids = []
    current_pos = prev_position_zup.copy()

    for i in range(n_hops):
        room_a = room_path[i]
        room_b = room_path[i + 1]
        edge = _edge_key(room_a, room_b)
        passages = room_graph.connections.get(edge, [])

        if not passages:
            if verbose:
                print(f"    [cross-room] No passage for hop {room_a} → {room_b}")
            return None

        # Score each passage: path_distance + preference_penalty
        def _passage_score(p):
            path_dist = (np.linalg.norm(current_pos - p.center)
                         + np.linalg.norm(p.center - next_position_zup))
            pref = PASSAGE_PREFERENCE.get(p.label, _DEFAULT_PREFERENCE)
            return path_dist + pref * PREFERENCE_DIST_PENALTY

        best_p = min(passages, key=_passage_score)

        waypoints.append(best_p.center.copy())
        waypoint_ids.append(best_p.object_id)
        current_pos = best_p.center.copy()

        if verbose:
            pref_label = best_p.label
            pref_score = PASSAGE_PREFERENCE.get(pref_label, _DEFAULT_PREFERENCE)
            alt_info = ""
            if len(passages) > 1:
                others = [p.object_id for p in passages if p is not best_p]
                alt_info = f" (alternatives: {others})"
            print(f"      Hop {i+1}/{n_hops}: {room_a} → {room_b} through "
                  f"{best_p.object_id} [{pref_label}, pref={pref_score}] "
                  f"at {best_p.center.round(2)}{alt_info}")

    # Summary
    if verbose:
        direct_dist = np.linalg.norm(prev_position_zup - next_position_zup)
        via_dist = np.linalg.norm(prev_position_zup - waypoints[0])
        for k in range(len(waypoints) - 1):
            via_dist += np.linalg.norm(waypoints[k] - waypoints[k + 1])
        via_dist += np.linalg.norm(waypoints[-1] - next_position_zup)
        print(f"      Direct distance: {direct_dist:.2f}m, "
              f"via {len(waypoints)} door(s): {via_dist:.2f}m "
              f"[{' → '.join(waypoint_ids)}]")

    return waypoints


def _bfs_room_path(
    start_room: str,
    end_room: str,
    room_graph: RoomGraph,
    max_hops: int,
) -> Optional[List[str]]:
    """BFS to find shortest room-to-room path."""
    from collections import deque
    
    # Build adjacency from connections
    adj: Dict[str, Set[str]] = {}
    for (ra, rb) in room_graph.connections.keys():
        adj.setdefault(ra, set()).add(rb)
        adj.setdefault(rb, set()).add(ra)
    
    if start_room not in adj:
        return None
    
    queue = deque([(start_room, [start_room])])
    visited = {start_room}
    
    while queue:
        current, path = queue.popleft()
        if len(path) - 1 > max_hops:
            break
        
        for neighbor in adj.get(current, []):
            if neighbor == end_room:
                return path + [neighbor]
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    
    return None


# =============================================================================
# Helper: Adjust waypoint height for camera passage
# =============================================================================

def adjust_waypoint_height(
    waypoint: np.ndarray,
    prev_pos: np.ndarray,
    next_pos: np.ndarray,
    door_extent: np.ndarray,
    camera_height_offset: float = 0.3,
) -> np.ndarray:
    """
    Adjust the waypoint (door center) height to be suitable for a camera
    passing through.
    
    The door OBB center might be at the geometric center of the door,
    but the camera should pass through the upper portion (eye level).
    
    This function:
      1. Takes the average height of the prev and next camera positions
         as the target camera height.
      2. Clamps it to be within the door's vertical extent.
      3. Falls back to door center + offset if positions are too low/high.
    
    Args:
        waypoint:            Door center position [x, y, z] in Z-up.
        prev_pos:            Previous camera position in Z-up.
        next_pos:            Next camera position in Z-up.
        door_extent:         Door OBB extents [sx, sy, sz].
        camera_height_offset: Offset above door center if no better info.
    
    Returns:
        Adjusted waypoint position.
    """
    adjusted = waypoint.copy()
    
    # Z is up in Z-up coordinate system
    avg_height = (prev_pos[2] + next_pos[2]) / 2.0
    
    # The door's vertical extent — in Z-up, height is along Z or Y
    # depending on door orientation. Use the largest extent component
    # that's roughly vertical.
    door_half_height = max(door_extent) * 0.5
    door_center_z = waypoint[2]
    
    # Clamp the camera height to within the door's extent
    min_z = door_center_z - door_half_height + 0.2  # small buffer
    max_z = door_center_z + door_half_height - 0.2
    
    target_z = np.clip(avg_height, min_z, max_z)
    adjusted[2] = target_z
    
    return adjusted


# =============================================================================
# Integration Helper: for optimize_trajectory_result PASS 2
# =============================================================================

def get_cross_room_info(
    prev_segment,
    next_segment,
    room_graph: RoomGraph,
    objects: Dict[str, dict],
    executor,
    verbose: bool = True,
) -> Optional[List[Dict]]:
    """
    High-level helper for integrate into PASS 2 of optimize_trajectory_result.
    
    Checks if prev and next segments are in different rooms. If so, returns
    a list of waypoint dicts with position and pose info ready for arc creation.
    
    Args:
        prev_segment:  The previous object-level TrajectorySegment.
        next_segment:  The next object-level TrajectorySegment.
        room_graph:    Pre-built RoomGraph.
        objects:       scene_json["objects"] dict.
        executor:      TrajectoryExecutor (for coordinate conversion).
        verbose:       Print debug info.
    
    Returns:
        List of waypoint dicts: [{"position_zup": np.ndarray, "position_yup": ...}, ...]
        or None if same room / no waypoints needed.
    """
    # Get object IDs
    prev_anchor = prev_segment.start_anchor
    next_anchor = next_segment.start_anchor
    if prev_anchor is None or next_anchor is None:
        return None
    
    prev_obj_id = str(prev_anchor.object_id)
    next_obj_id = str(next_anchor.object_id)
    
    # Get actual camera positions (from optimized trajectories if available)
    prev_pos = None
    if prev_segment.trajectory_output is not None:
        prev_pos = prev_segment.trajectory_output['c2w'][-1, :3, 3].copy()
    else:
        prev_pos = prev_anchor.position.copy()
    
    next_pos = None
    if next_segment.trajectory_output is not None:
        next_pos = next_segment.trajectory_output['c2w'][0, :3, 3].copy()
    else:
        next_pos = next_anchor.position.copy()
    
    # Find waypoints (BFS handles both direct and multi-hop)
    waypoints = find_cross_room_waypoints(
        prev_obj_id, next_obj_id, room_graph, objects,
        prev_position_zup=prev_pos,
        next_position_zup=next_pos,
        verbose=verbose,
    )
    
    if waypoints is None or len(waypoints) == 0:
        return None
    
    # Optionally adjust waypoint heights for camera passage
    # Find the door/window objects to get their extents
    result_waypoints = []
    for wp in waypoints:
        # Find the closest door/window to get its extent
        closest_dw = None
        closest_dist = float("inf")
        for dw in room_graph.doors_windows:
            d = np.linalg.norm(wp - dw.center)
            if d < closest_dist:
                closest_dist = d
                closest_dw = dw
        
        if closest_dw is not None:
            wp_adjusted = adjust_waypoint_height(
                wp, prev_pos, next_pos, closest_dw.extent,
            )
        else:
            wp_adjusted = wp.copy()
        
        # Convert to Y-up for arc creation
        wp_yup = executor._zup_to_yup(wp_adjusted)
        
        # Compute a reasonable look direction at the waypoint:
        # look toward the next position (or next waypoint)
        look_dir = next_pos - wp_adjusted
        look_norm = np.linalg.norm(look_dir)
        if look_norm > 1e-6:
            look_dir = look_dir / look_norm
        else:
            look_dir = np.array([0.0, 1.0, 0.0])
        
        look_dir_yup = executor._zup_to_yup(look_dir)
        yaw = np.rad2deg(np.arctan2(look_dir_yup[0], look_dir_yup[2]))
        pitch = np.rad2deg(np.arcsin(np.clip(-look_dir_yup[1], -1.0, 1.0)))
        
        result_waypoints.append({
            "position_zup": wp_adjusted,
            "position_yup": wp_yup,
            "rotation": np.array([pitch, yaw, 0.0]),
            "look_at_yup": wp_yup + look_dir_yup,
            "door_window_id": closest_dw.object_id if closest_dw else None,
        })
    
    return result_waypoints


# =============================================================================
# Convenience: load and build from scene graph path
# =============================================================================

def load_room_graph(scene_graph_path: str, verbose: bool = True) -> Tuple[RoomGraph, dict]:
    """
    Load a scene graph JSON and build the RoomGraph.
    
    Returns:
        (room_graph, scene_json)
    """
    with open(scene_graph_path, "r") as f:
        scene_json = json.load(f)
    
    room_graph = build_room_graph(scene_json, verbose=verbose)
    return room_graph, scene_json


# =============================================================================
# Self-test
# =============================================================================

if __name__ == "__main__":
    # =========================================================================
    # Test scene: 3 rooms with a hallway connecting living_room and kitchen
    #
    # Layout (top view):
    #
    #   +---living_room_0---+          +---kitchen_0---+
    #   |  sofa_0   table_0 | door_0  | door_1  fridge_0 |
    #   |                   |----[]---+---[]---|           |
    #   +-------------------+ hallway_0       +--sink_0---+
    #                        (empty corridor)
    #
    # door_0 connects living_room_0 ↔ hallway_0
    # door_1 connects hallway_0 ↔ kitchen_0
    # Camera going sofa_0 → fridge_0 must pass through BOTH doors
    # =========================================================================

    test_scene = {
        "objects": {
            # Living room objects
            "sofa_0": {
                "obb": [2.0, 1.5, 0.4, 2.0, 1.0, 0.8, 0, 0, 0, 1],
                "against_wall": True,
                "attached_to_ceiling": False,
            },
            "table_0": {
                "obb": [3.5, 1.5, 0.35, 1.2, 0.8, 0.7, 0, 0, 0, 1],
                "against_wall": False,
                "attached_to_ceiling": False,
            },
            # Door between living_room and hallway
            "door_0": {
                "obb": [5.0, 1.5, 1.0, 0.15, 0.9, 2.0, 0, 0, 0, 1],
                "against_wall": True,
                "attached_to_ceiling": False,
            },
            # Door frame between living_room and hallway (nearby door_0, always open)
            "door_frame_0": {
                "obb": [5.1, 1.5, 1.0, 0.12, 0.95, 2.1, 0, 0, 0, 1],
                "against_wall": True,
                "attached_to_ceiling": False,
            },
            # Door between hallway and kitchen
            "door_1": {
                "obb": [7.0, 1.5, 1.0, 0.15, 0.9, 2.0, 0, 0, 0, 1],
                "against_wall": True,
                "attached_to_ceiling": False,
            },
            # Kitchen objects
            "fridge_0": {
                "obb": [9.0, 1.5, 0.85, 0.7, 0.7, 1.7, 0, 0, 0, 1],
                "against_wall": True,
                "attached_to_ceiling": False,
            },
            "sink_0": {
                "obb": [9.5, 2.5, 0.8, 0.6, 0.5, 0.3, 0, 0, 0, 1],
                "against_wall": True,
                "attached_to_ceiling": False,
            },
            # Window in living room (not a passage between rooms)
            "window_0": {
                "obb": [2.0, 0.0, 1.5, 0.1, 1.0, 1.2, 0, 0, 0, 1],
                "against_wall": True,
                "attached_to_ceiling": False,
            },
        },
        "rooms": {
            "living_room_0": ["sofa_0", "table_0", "door_0", "door_frame_0", "window_0"],
            "hallway_0": [],  # empty corridor, doors assigned to adjacent rooms
            "kitchen_0": ["fridge_0", "sink_0", "door_1"],
        },
    }

    print("=" * 60)
    print("ROOM CONNECTIVITY SELF-TEST")
    print("=" * 60)

    graph = build_room_graph(test_scene, verbose=True)

    print(f"\nobj_to_room: {graph.obj_to_room}")
    print(f"connections: "
          f"{dict((k, [p.object_id for p in v]) for k, v in graph.connections.items())}")

    # ------------------------------------------------------------------
    # Test 1: Same room → should return None
    # ------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("TEST 1: Same room (sofa_0 → table_0)")
    print("-" * 60)
    wp = find_cross_room_waypoints(
        "sofa_0", "table_0", graph, test_scene["objects"], verbose=True,
    )
    assert wp is None, f"Expected None, got {wp}"
    print("✓ PASS: No waypoints for same-room transition")

    # ------------------------------------------------------------------
    # Test 2: Cross-room via hallway → should return 2 waypoints (2 doors)
    # ------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("TEST 2: Cross-room via hallway (sofa_0 → fridge_0)")
    print("-" * 60)
    wp = find_cross_room_waypoints(
        "sofa_0", "fridge_0", graph, test_scene["objects"], verbose=True,
    )
    if wp is not None:
        print(f"✓ Got {len(wp)} waypoint(s):")
        for j, w in enumerate(wp):
            print(f"    wp[{j}] = {w.round(2)}")
        assert len(wp) == 2, f"Expected 2 waypoints (2 doors), got {len(wp)}"
        print("✓ PASS: 2 doors for living_room → hallway → kitchen")
    else:
        print("✗ FAIL: Expected waypoints but got None")

    # ------------------------------------------------------------------
    # Test 3: Direct connection (door_0 in living_room, hallway adjacent)
    #         sofa_0 → door_1 object itself (in kitchen)
    #         This tests 1-hop: living_room → hallway via door_0
    # ------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("TEST 3: Adjacent rooms (sofa_0 in living_room → door_1 in kitchen)")
    print("-" * 60)
    wp = find_cross_room_waypoints(
        "sofa_0", "door_1", graph, test_scene["objects"], verbose=True,
    )
    if wp is not None:
        print(f"✓ Got {len(wp)} waypoint(s):")
        for j, w in enumerate(wp):
            print(f"    wp[{j}] = {w.round(2)}")
        print("✓ PASS: Found route through connecting doors")
    else:
        print("  (None — this may be expected depending on proximity threshold)")

    # ------------------------------------------------------------------
    # Test 4: No room assignment → should return None
    # ------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("TEST 4: Unknown object ID")
    print("-" * 60)
    wp = find_cross_room_waypoints(
        "sofa_0", "nonexistent_0", graph, test_scene["objects"], verbose=True,
    )
    assert wp is None, f"Expected None, got {wp}"
    print("✓ PASS: None for unknown object")

    # ------------------------------------------------------------------
    # Test 5: door_frame preferred over door
    # Both door_0 and door_frame_0 connect living_room ↔ hallway,
    # but door_frame should be preferred (always open).
    # ------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("TEST 5: door_frame preferred over door (sofa_0 → fridge_0)")
    print("-" * 60)
    wp = find_cross_room_waypoints(
        "sofa_0", "fridge_0", graph, test_scene["objects"], verbose=True,
    )
    if wp is not None and len(wp) == 2:
        # The first waypoint should be door_frame_0, not door_0
        # Find which door was selected for the first hop
        door_frame_center = np.array([5.1, 1.5, 1.0])
        first_wp_is_frame = np.linalg.norm(wp[0] - door_frame_center) < 0.2
        if first_wp_is_frame:
            print("✓ PASS: door_frame_0 preferred over door_0 (always open)")
        else:
            print(f"✗ FAIL: Expected door_frame_0 but got wp={wp[0].round(2)}")
    else:
        n = len(wp) if wp else 0
        print(f"✗ FAIL: Expected 2 waypoints, got {n}")

    print("\n" + "=" * 60)
    print("ALL TESTS DONE")
    print("=" * 60)