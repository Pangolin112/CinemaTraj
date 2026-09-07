"""
Camera Trajectory Dialog - Object Validation Module
====================================================

Validates that object IDs and names in LLM responses match the provided scene data.

Handles two cases:
1. User mentions objects not in the scene → Report as non-existent
2. LLM assigns wrong IDs to valid object names → Auto-correct with valid IDs

Supports both:
- New scene graph format: string IDs like "table_0", "sofa_1"
- Legacy labels.json format: integer IDs like 1, 23, 45
"""

import re
import json
from typing import Optional, Union
from difflib import SequenceMatcher


def similarity_ratio(a: str, b: str) -> float:
    """Calculate string similarity ratio between two strings."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


# Common furniture/object synonyms
OBJECT_SYNONYMS = {
    "couch": ["sofa", "settee", "loveseat"],
    "sofa": ["couch", "settee", "loveseat"],
    "table": ["desk"],
    "desk": ["table"],
    "wardrobe": ["closet", "armoire", "cabinet"],
    "closet": ["wardrobe", "armoire", "cabinet"],
    "cabinet": ["wardrobe", "closet", "cupboard"],
    "tv": ["television", "monitor", "screen"],
    "television": ["tv", "monitor", "screen"],
    "lamp": ["light", "lighting"],
    "light": ["lamp", "lighting"],
    "chair": ["seat", "armchair"],
    "armchair": ["chair", "seat"],
    "bed": ["mattress"],
    "refrigerator": ["fridge"],
    "fridge": ["refrigerator"],
    "entrance": ["door", "entry", "doorway"],
    "door": ["entrance", "entry", "doorway"],
    "window": ["glass", "pane"],
    "rug": ["carpet", "mat"],
    "carpet": ["rug", "mat"],
    "painting": ["picture", "artwork", "art"],
    "picture": ["painting", "artwork", "photo"],
    "plant": ["flower", "greenery"],
    "flower": ["plant"],
    "kitchen_counter": ["counter", "countertop", "kitchen counter"],
    "counter": ["kitchen_counter", "countertop"],
    "trash_can": ["bin", "waste basket", "garbage"],
    "obstacle": [],
    "footwear": ["shoes", "boots", "slippers"],
}


def find_best_matching_object(
    query_label: str, 
    objects_summary: list, 
    threshold: float = 0.6,
    _normalized: bool = False
) -> Optional[dict]:
    """
    Find the best matching object from the scene for a given label.
    
    Args:
        query_label: The object label to search for
        objects_summary: List of objects (supports various key formats)
        threshold: Minimum similarity ratio to consider a match
        _normalized: Internal flag - if True, skip normalization
    
    Returns:
        Best matching object dict or None if no good match found
    """
    # Normalize objects if not already done
    if not _normalized:
        objects_summary = normalize_objects_summary(objects_summary)
    
    query_clean = query_label.lower().strip().replace("_", " ").replace("-", " ")
    query_words = set(query_clean.split())
    
    best_match = None
    best_score = 0
    
    for obj in objects_summary:
        obj_label = obj["label"].lower().strip().replace("_", " ").replace("-", " ")
        obj_id_str = str(obj["id"]).lower().strip().replace("_", " ").replace("-", " ")
        obj_words = set(obj_label.split())
        
        # Exact match on label
        if query_clean == obj_label:
            return obj
        
        # Exact match on full ID (e.g., query "table_0" matches id "table_0")
        query_as_id = query_label.lower().strip()
        if query_as_id == str(obj["id"]).lower().strip():
            return obj
        
        # Check if query is contained in object label/id or vice versa
        if query_clean in obj_label or obj_label in query_clean:
            score = 0.9  # High score for containment
            if score > best_score:
                best_score = score
                best_match = obj
            continue
        
        if query_clean in obj_id_str or obj_id_str in query_clean:
            score = 0.88
            if score > best_score:
                best_score = score
                best_match = obj
            continue
        
        # Check for word overlap
        common_words = query_words & obj_words
        if common_words:
            score = 0.85
            if score > best_score:
                best_score = score
                best_match = obj
            continue
        
        # Check synonyms
        synonym_match = False
        for query_word in query_words:
            if query_word in OBJECT_SYNONYMS:
                for synonym in OBJECT_SYNONYMS[query_word]:
                    if synonym in obj_label or synonym in obj_words:
                        synonym_match = True
                        break
            if synonym_match:
                break
        
        if synonym_match:
            score = 0.8  # Good score for synonym match
            if score > best_score:
                best_score = score
                best_match = obj
            continue
        
        # Fuzzy matching
        score = similarity_ratio(query_clean, obj_label)
        if score > best_score and score >= threshold:
            best_score = score
            best_match = obj
    
    return best_match


def parse_anchor_calls(atomic_trajectories: str) -> list:
    """
    Extract all Anchor Determinator calls from the atomic_trajectories string.
    
    Returns list of dicts with:
        - step_number: The step number in the sequence
        - raw_text: The original text of the step
        - object_name: Extracted object name/label
        - object_id: Extracted object ID (string or int)
    """
    anchor_calls = []
    
    # Split by step numbers or periods
    steps = re.split(r'\d+\.', atomic_trajectories)
    steps = [s.strip() for s in steps if s.strip()]
    
    # Also try splitting by periods if the above doesn't work well
    if len(steps) <= 1:
        steps = atomic_trajectories.split(".")
        steps = [s.strip() for s in steps if s.strip()]
    
    for i, step in enumerate(steps, 1):
        step_lower = step.lower()
        
        if "anchor" in step_lower:
            anchor_info = {
                "step_number": i,
                "raw_text": step,
                "object_name": None,
                "object_id": None
            }
            
            # Extract object name - look for patterns like:
            # "with 'object_name'" or "with \"object_name\"" or "with 'object name' (id: X)"
            name_patterns = [
                r"with\s+['\"]([^'\"]+)['\"]",  # 'name' or "name"
                r"with\s+(\w+(?:\s+\w+)*)\s*\(",  # name (id:
                r"determinator\s+(?:with\s+)?['\"]?(\w+(?:[\s_-]\w+)*)['\"]?",  # Anchor Determinator with name
            ]
            
            for pattern in name_patterns:
                match = re.search(pattern, step, re.IGNORECASE)
                if match:
                    anchor_info["object_name"] = match.group(1).strip()
                    break
            
            # Extract object ID - now supports both string IDs and integer IDs
            # String ID patterns: (id: table_0), (id: sofa_1)
            # Integer ID patterns: (id: 42), (id:42)
            id_patterns = [
                r"\(id:\s*([^\)]+)\)",       # (id: anything_until_closing_paren)
                r"\(id\s*:\s*([^\)]+)\)",    # (id : anything)
                r"id:\s*(\S+)",              # id: something
                r"id\s*=\s*(\S+)",           # id=something
            ]
            
            for pattern in id_patterns:
                match = re.search(pattern, step, re.IGNORECASE)
                if match:
                    raw_id = match.group(1).strip().rstrip(".,;)")
                    # Try to parse as int for legacy format, otherwise keep as string
                    try:
                        anchor_info["object_id"] = int(raw_id)
                    except ValueError:
                        anchor_info["object_id"] = raw_id
                    break
            
            anchor_calls.append(anchor_info)
    
    return anchor_calls


def get_object_id(obj: dict) -> Optional[Union[int, str]]:
    """
    Get the object ID from an object dict, handling different key names and types.
    
    Returns:
        int or str ID if valid, None if ID is empty or invalid
    """
    obj_id = None
    if "id" in obj:
        obj_id = obj["id"]
    elif "ins_id" in obj:
        obj_id = obj["ins_id"]
    elif "instance_id" in obj:
        obj_id = obj["instance_id"]
    elif "object_id" in obj:
        obj_id = obj["object_id"]
    else:
        return None  # No ID key found
    
    # Handle None or empty values
    if obj_id is None or obj_id == "":
        return None
    
    # Keep strings as-is (new format), convert numeric strings to int (legacy)
    if isinstance(obj_id, str):
        # If it looks like a pure integer, convert it
        try:
            return int(obj_id)
        except ValueError:
            return obj_id  # String ID like "table_0"
    
    return obj_id


def get_object_label(obj: dict) -> str:
    """Get the object label from an object dict, handling different key names."""
    if "label" in obj:
        return obj["label"]
    elif "name" in obj:
        return obj["name"]
    elif "class" in obj:
        return obj["class"]
    elif "category" in obj:
        return obj["category"]
    else:
        raise KeyError(f"Cannot find label key in object. Available keys: {list(obj.keys())}")


def normalize_objects_summary(objects_summary: list) -> list:
    """
    Normalize objects_summary to have consistent 'id' and 'label' keys.
    
    This handles different input formats like:
    - {"id": "table_0", "label": "table", ...}   (new scene graph format)
    - {"id": 1, "label": "chair", ...}            (legacy integer IDs)
    - {"ins_id": 1, "label": "chair", ...}
    - {"instance_id": 1, "name": "chair", ...}
    
    Objects with missing or invalid IDs are skipped.
    """
    normalized = []
    skipped_count = 0
    
    for obj in objects_summary:
        obj_id = get_object_id(obj)
        
        # Skip objects with invalid/empty IDs
        if obj_id is None:
            skipped_count += 1
            continue
        
        try:
            obj_label = get_object_label(obj)
        except KeyError:
            skipped_count += 1
            continue
            
        normalized.append({
            "id": obj_id,
            "label": obj_label,
            "center": obj.get("center"),
            "_original": obj  # Keep reference to original
        })
    
    if skipped_count > 0:
        print(f"Note: Skipped {skipped_count} objects with missing/invalid IDs or labels")
    
    return normalized


def _id_matches(given_id, obj_id) -> bool:
    """Check if two IDs match, handling string/int comparison."""
    if given_id is None or obj_id is None:
        return False
    # Direct equality
    if given_id == obj_id:
        return True
    # String comparison as fallback
    return str(given_id).strip().lower() == str(obj_id).strip().lower()


def validate_object_references(
    atomic_trajectories: str,
    objects_summary: list,
    auto_correct: bool = True
) -> dict:
    """
    Validate all object references in the atomic_trajectories.
    
    Args:
        atomic_trajectories: The trajectory string from LLM response
        objects_summary: List of valid objects. Supports various key formats.
        auto_correct: Whether to attempt auto-correction of wrong IDs
    
    Returns:
        dict with validation results
    """
    result = {
        "valid": True,
        "anchor_calls": [],
        "errors": [],
        "warnings": [],
        "corrections": [],
        "non_existent_objects": [],
        "corrected_trajectories": atomic_trajectories
    }
    
    # Normalize objects to have consistent keys
    normalized_objects = normalize_objects_summary(objects_summary)
    
    # Build lookup dictionaries — support both string and int IDs
    id_to_obj = {}
    for obj in normalized_objects:
        id_to_obj[obj["id"]] = obj
        # Also index by string representation for cross-format matching
        id_to_obj[str(obj["id"])] = obj
    
    label_to_obj = {obj["label"].lower(): obj for obj in normalized_objects}
    
    # Parse anchor calls
    anchor_calls = parse_anchor_calls(atomic_trajectories)
    result["anchor_calls"] = anchor_calls
    
    corrected_text = atomic_trajectories
    
    for anchor in anchor_calls:
        step_num = anchor["step_number"]
        obj_name = anchor["object_name"]
        obj_id = anchor["object_id"]
        
        if obj_name is None and obj_id is None:
            result["errors"].append({
                "step": step_num,
                "type": "parse_error",
                "message": f"Could not parse object name or ID from: {anchor['raw_text'][:80]}..."
            })
            result["valid"] = False
            continue
        
        # Case 1: Check if ID exists in scene (handle both string and int)
        id_valid = False
        if obj_id is not None:
            # Try direct lookup, then string-based lookup
            if obj_id in id_to_obj:
                id_valid = True
            elif str(obj_id) in id_to_obj:
                id_valid = True
                obj_id = str(obj_id)  # Normalize to the working key
        
        # Case 2: Check if name exists in scene (exact or fuzzy match)
        name_match = None
        if obj_name:
            # Try exact match on label first
            name_lower = obj_name.lower().strip()
            if name_lower in label_to_obj:
                name_match = label_to_obj[name_lower]
            else:
                # Try matching by ID string (e.g., name="table_0" matches id="table_0")
                if name_lower in id_to_obj:
                    name_match = id_to_obj[name_lower]
                elif obj_name in id_to_obj:
                    name_match = id_to_obj[obj_name]
                else:
                    # Try fuzzy match (use normalized objects)
                    name_match = find_best_matching_object(obj_name, normalized_objects, _normalized=True)
        
        # Validation logic
        if id_valid and name_match:
            # Both ID and name exist - check if they refer to compatible objects
            expected_obj = id_to_obj.get(obj_id) or id_to_obj.get(str(obj_id))
            
            expected_label_lower = expected_obj["label"].lower().strip().replace("_", " ").replace("-", " ")
            name_match_label_lower = name_match["label"].lower().strip().replace("_", " ").replace("-", " ")
            query_name_lower = obj_name.lower().strip().replace("_", " ").replace("-", " ")
            
            labels_compatible = (
                _id_matches(expected_obj["id"], name_match["id"]) or
                expected_label_lower == name_match_label_lower or
                query_name_lower in expected_label_lower or
                expected_label_lower in query_name_lower or
                bool(set(expected_label_lower.split()) & set(name_match_label_lower.split()))
            )
            
            if not labels_compatible:
                result["warnings"].append({
                    "step": step_num,
                    "type": "id_name_mismatch",
                    "message": f"ID {obj_id} refers to '{expected_obj['label']}' but name '{obj_name}' matches '{name_match['label']}' (id: {name_match['id']})",
                    "given_id": obj_id,
                    "given_name": obj_name,
                    "id_refers_to": expected_obj["label"],
                    "name_matches": name_match["label"],
                    "correct_id": name_match["id"]
                })
                
                if auto_correct:
                    old_pattern = f"(id: {obj_id})"
                    new_pattern = f"(id: {name_match['id']})"
                    corrected_text = corrected_text.replace(
                        anchor["raw_text"],
                        anchor["raw_text"].replace(old_pattern, new_pattern)
                    )
                    result["corrections"].append({
                        "step": step_num,
                        "object_name": obj_name,
                        "old_id": obj_id,
                        "new_id": name_match["id"],
                        "matched_label": name_match["label"]
                    })
                    
        elif id_valid and not name_match:
            expected_obj = id_to_obj.get(obj_id) or id_to_obj.get(str(obj_id))
            result["warnings"].append({
                "step": step_num,
                "type": "name_not_found",
                "message": f"Name '{obj_name}' not found in scene, but ID {obj_id} is valid (refers to '{expected_obj['label']}')",
                "given_name": obj_name,
                "given_id": obj_id,
                "id_refers_to": expected_obj["label"]
            })
            
        elif not id_valid and name_match:
            result["errors"].append({
                "step": step_num,
                "type": "wrong_id",
                "message": f"Object '{obj_name}' found as '{name_match['label']}' (id: {name_match['id']}), but given ID {obj_id} is invalid",
                "given_name": obj_name,
                "given_id": obj_id,
                "correct_id": name_match["id"],
                "correct_label": name_match["label"]
            })
            result["valid"] = False
            
            if auto_correct:
                if obj_id is not None:
                    old_pattern = f"(id: {obj_id})"
                    new_pattern = f"(id: {name_match['id']})"
                    corrected_text = corrected_text.replace(
                        anchor["raw_text"],
                        anchor["raw_text"].replace(old_pattern, new_pattern)
                    )
                result["corrections"].append({
                    "step": step_num,
                    "object_name": obj_name,
                    "old_id": obj_id,
                    "new_id": name_match["id"],
                    "matched_label": name_match["label"]
                })
                
        else:
            # Neither ID nor name found in scene
            result["non_existent_objects"].append({
                "step": step_num,
                "type": "object_not_found",
                "message": f"Object '{obj_name}' (id: {obj_id}) not found in scene",
                "given_name": obj_name,
                "given_id": obj_id
            })
            result["errors"].append({
                "step": step_num,
                "type": "object_not_found",
                "message": f"Object '{obj_name}' with ID {obj_id} does not exist in the scene",
                "given_name": obj_name,
                "given_id": obj_id
            })
            result["valid"] = False
    
    result["corrected_trajectories"] = corrected_text
    
    # Update valid flag based on corrections
    if auto_correct and result["corrections"]:
        uncorrectable_errors = [e for e in result["errors"] if e["type"] == "object_not_found"]
        result["valid"] = len(uncorrectable_errors) == 0
    
    return result


def validate_and_correct_response(
    parsed_response: dict,
    objects_summary: list,
    auto_correct: bool = True
) -> dict:
    """
    Validate and optionally correct a parsed LLM response.
    """
    if not parsed_response or "atomic_trajectories" not in parsed_response:
        return {
            "valid": False,
            "error": "No atomic_trajectories found in response",
            "validation": None,
            "corrected_response": None
        }
    
    validation = validate_object_references(
        parsed_response["atomic_trajectories"],
        objects_summary,
        auto_correct=auto_correct
    )
    
    corrected_response = None
    if auto_correct:
        corrected_response = parsed_response.copy()
        corrected_response["atomic_trajectories"] = validation["corrected_trajectories"]
    
    return {
        "valid": validation["valid"],
        "validation": validation,
        "corrected_response": corrected_response,
        "original_response": parsed_response
    }


def print_validation_report(validation_result: dict) -> None:
    """Print a formatted validation report."""
    print("\n" + "=" * 60)
    print("OBJECT REFERENCE VALIDATION REPORT")
    print("=" * 60)
    
    val = validation_result.get("validation", {})
    
    print(f"\nOverall Valid: {validation_result.get('valid', False)}")
    print(f"Anchor Calls Found: {len(val.get('anchor_calls', []))}")
    
    non_existent = val.get("non_existent_objects", [])
    if non_existent:
        print(f"\n❌ NON-EXISTENT OBJECTS ({len(non_existent)}):")
        for item in non_existent:
            print(f"   Step {item['step']}: '{item['given_name']}' (id: {item['given_id']})")
            print(f"      → This object is not in the scene data")
    
    errors = val.get("errors", [])
    if errors:
        print(f"\n❌ ERRORS ({len(errors)}):")
        for err in errors:
            print(f"   Step {err['step']} [{err['type']}]: {err['message']}")
    
    warnings = val.get("warnings", [])
    if warnings:
        print(f"\n⚠️  WARNINGS ({len(warnings)}):")
        for warn in warnings:
            print(f"   Step {warn['step']} [{warn['type']}]: {warn['message']}")
    
    corrections = val.get("corrections", [])
    if corrections:
        print(f"\n✅ AUTO-CORRECTIONS ({len(corrections)}):")
        for corr in corrections:
            print(f"   Step {corr['step']}: '{corr['object_name']}'")
            print(f"      → ID {corr['old_id']} → {corr['new_id']} (matched: '{corr['matched_label']}')")
    
    print(f"\n📍 ANCHOR CALLS PARSED:")
    for anchor in val.get("anchor_calls", []):
        print(f"   Step {anchor['step_number']}: name='{anchor['object_name']}', id={anchor['object_id']}")
    
    print("\n" + "=" * 60)


# ==============================================================================
# Trajectory Deletion for Non-existent Objects
# ==============================================================================

def parse_trajectory_steps(atomic_trajectories: str) -> list:
    """
    Parse atomic_trajectories into a list of step dictionaries.
    """
    steps = []
    
    # Split by step numbers
    pattern = r'(\d+)\.\s*'
    parts = re.split(pattern, atomic_trajectories)
    
    # parts will be: ['', '1', 'step1 text', '2', 'step2 text', ...]
    i = 1
    while i < len(parts) - 1:
        step_num = int(parts[i])
        step_text = parts[i + 1].strip().rstrip('.')
        
        step_info = {
            "step_number": step_num,
            "raw_text": step_text,
            "type": "unknown",
            "object_name": None,
            "object_id": None
        }
        
        step_lower = step_text.lower()
        
        if "anchor" in step_lower:
            step_info["type"] = "anchor"
            name_match = re.search(r"with\s+['\"]([^'\"]+)['\"]", step_text, re.IGNORECASE)
            if name_match:
                step_info["object_name"] = name_match.group(1).strip()
            # Support both string and int IDs
            id_match = re.search(r"\(id:\s*([^\)]+)\)", step_text, re.IGNORECASE)
            if id_match:
                raw_id = id_match.group(1).strip()
                try:
                    step_info["object_id"] = int(raw_id)
                except ValueError:
                    step_info["object_id"] = raw_id
                
        elif "atomtraj" in step_lower:
            if "object-level" in step_lower:
                step_info["type"] = "object_level"
            elif "transitional" in step_lower:
                step_info["type"] = "transitional"
            else:
                step_info["type"] = "AtomTraj_unknown"
                
        elif "traj_compose" in step_lower or "compose" in step_lower:
            step_info["type"] = "compose"
            
        elif "render" in step_lower:
            step_info["type"] = "render"
        
        steps.append(step_info)
        i += 2
    
    return steps


def delete_nonexistent_object_trajectories(
    atomic_trajectories: str,
    objects_summary: list,
    verbose: bool = False
) -> dict:
    """
    Delete trajectory steps for objects that don't exist in the scene.
    """
    result = {
        "success": True,
        "deleted_objects": [],
        "remaining_objects": [],
        "cleaned_trajectories": "",
        "original_object_count": 0,
        "final_object_count": 0,
        "deletion_details": []
    }
    
    normalized_objects = normalize_objects_summary(objects_summary)
    
    # Build lookup — support string and int IDs
    id_to_obj = {}
    for obj in normalized_objects:
        id_to_obj[obj["id"]] = obj
        id_to_obj[str(obj["id"])] = obj
    label_to_obj = {obj["label"].lower(): obj for obj in normalized_objects}
    
    steps = parse_trajectory_steps(atomic_trajectories)
    
    if verbose:
        print("\nParsed steps:")
        for s in steps:
            print(f"  {s['step_number']}: {s['type']} - {s.get('object_name', '')} (id: {s.get('object_id', '')})")
    
    # Find all anchor steps and check validity
    anchor_indices = []
    for i, step in enumerate(steps):
        if step["type"] == "anchor":
            obj_name = step["object_name"]
            obj_id = step["object_id"]
            
            # Check if object exists
            id_valid = False
            if obj_id is not None:
                id_valid = obj_id in id_to_obj or str(obj_id) in id_to_obj
            
            name_match = None
            if obj_name:
                name_lower = obj_name.lower().strip()
                if name_lower in label_to_obj:
                    name_match = label_to_obj[name_lower]
                elif name_lower in id_to_obj:
                    name_match = id_to_obj[name_lower]
                elif obj_name in id_to_obj:
                    name_match = id_to_obj[obj_name]
                else:
                    name_match = find_best_matching_object(obj_name, normalized_objects, _normalized=True)
            
            is_valid = id_valid or name_match is not None
            
            anchor_indices.append({
                "index": i,
                "step": step,
                "is_valid": is_valid,
                "matched_object": name_match if name_match else (id_to_obj.get(obj_id) or id_to_obj.get(str(obj_id)) if id_valid else None)
            })
    
    result["original_object_count"] = len(anchor_indices)
    
    if verbose:
        print(f"\nFound {len(anchor_indices)} anchor calls")
        for ai in anchor_indices:
            status = "✓" if ai["is_valid"] else "✗"
            print(f"  {status} Index {ai['index']}: {ai['step']['object_name']} (id: {ai['step']['object_id']})")
    
    # Identify steps to delete
    steps_to_delete = set()
    
    for anchor_pos, anchor_info in enumerate(anchor_indices):
        if anchor_info["is_valid"]:
            result["remaining_objects"].append(anchor_info["step"]["object_name"])
            continue
        
        anchor_idx = anchor_info["index"]
        obj_name = anchor_info["step"]["object_name"]
        obj_id = anchor_info["step"]["object_id"]
        
        deletion_info = {
            "object_name": obj_name,
            "object_id": obj_id,
            "position": anchor_pos + 1,
            "deleted_steps": []
        }
        
        # Always delete the anchor itself
        steps_to_delete.add(anchor_idx)
        deletion_info["deleted_steps"].append(f"Anchor '{obj_name}'")
        
        if anchor_pos == 0:
            # FIRST object: Delete A1, O1, and T1 (the transitional AFTER A2)
            if anchor_idx + 1 < len(steps) and steps[anchor_idx + 1]["type"] == "object_level":
                steps_to_delete.add(anchor_idx + 1)
                deletion_info["deleted_steps"].append("Object-level (examine)")
            
            if anchor_pos + 1 < len(anchor_indices):
                next_anchor_idx = anchor_indices[anchor_pos + 1]["index"]
                if next_anchor_idx + 1 < len(steps) and steps[next_anchor_idx + 1]["type"] == "transitional":
                    steps_to_delete.add(next_anchor_idx + 1)
                    deletion_info["deleted_steps"].append("Transitional (move to next)")
                
        else:
            # NON-FIRST object: Delete T_{i-1}, A_i, O_i
            if anchor_idx + 1 < len(steps) and steps[anchor_idx + 1]["type"] == "transitional":
                steps_to_delete.add(anchor_idx + 1)
                deletion_info["deleted_steps"].append("Transitional (move from previous)")
            
            if anchor_idx + 2 < len(steps) and steps[anchor_idx + 2]["type"] == "object_level":
                steps_to_delete.add(anchor_idx + 2)
                deletion_info["deleted_steps"].append("Object-level (examine)")
        
        result["deleted_objects"].append(deletion_info)
        result["deletion_details"].append(deletion_info)
    
    if verbose:
        print(f"\nSteps to delete: {sorted(steps_to_delete)}")
    
    # Build cleaned trajectory
    remaining_steps = []
    for i, step in enumerate(steps):
        if i not in steps_to_delete:
            remaining_steps.append(step)
    
    # Renumber steps
    cleaned_lines = []
    for new_num, step in enumerate(remaining_steps, 1):
        cleaned_lines.append(f"{new_num}. {step['raw_text']}.")
    
    result["cleaned_trajectories"] = " ".join(cleaned_lines)
    result["final_object_count"] = len([s for s in remaining_steps if s["type"] == "anchor"])
    
    if result["final_object_count"] == 0:
        result["success"] = False
    
    return result


def validate_and_clean_response(
    parsed_response: dict,
    objects_summary: list,
    auto_correct: bool = True,
    delete_nonexistent: bool = True,
    verbose: bool = False
) -> dict:
    """
    Full validation pipeline: validate, correct IDs, and delete non-existent objects.
    """
    if not parsed_response or "atomic_trajectories" not in parsed_response:
        return {
            "valid": False,
            "error": "No atomic_trajectories found in response",
            "validation": None,
            "corrected_response": None,
            "cleaned_response": None
        }
    
    # Step 1: Validate and auto-correct IDs
    validation = validate_object_references(
        parsed_response["atomic_trajectories"],
        objects_summary,
        auto_correct=auto_correct
    )
    
    corrected_response = parsed_response.copy()
    corrected_response["atomic_trajectories"] = validation["corrected_trajectories"]
    
    # Step 2: Delete non-existent object trajectories
    deletion_result = None
    cleaned_response = None
    
    if delete_nonexistent and validation["non_existent_objects"]:
        deletion_result = delete_nonexistent_object_trajectories(
            corrected_response["atomic_trajectories"],
            objects_summary,
            verbose=verbose
        )
        
        if deletion_result["success"]:
            cleaned_response = corrected_response.copy()
            cleaned_response["atomic_trajectories"] = deletion_result["cleaned_trajectories"]
            if "object_sequence" in cleaned_response:
                cleaned_response["object_sequence"] = deletion_result["remaining_objects"]
    elif not validation["non_existent_objects"]:
        cleaned_response = corrected_response
        deletion_result = {
            "success": True,
            "deleted_objects": [],
            "remaining_objects": [a["object_name"] for a in validation["anchor_calls"]],
            "cleaned_trajectories": corrected_response["atomic_trajectories"],
            "original_object_count": len(validation["anchor_calls"]),
            "final_object_count": len(validation["anchor_calls"])
        }
    
    return {
        "valid": validation["valid"] or (deletion_result and deletion_result["success"]),
        "validation": validation,
        "deletion": deletion_result,
        "original_response": parsed_response,
        "corrected_response": corrected_response,
        "cleaned_response": cleaned_response,
        "final_trajectories": cleaned_response["atomic_trajectories"] if cleaned_response else None
    }


def print_full_validation_report(result: dict) -> None:
    """Print a comprehensive validation and cleaning report."""
    print("\n" + "=" * 70)
    print("OBJECT REFERENCE VALIDATION & CLEANING REPORT")
    print("=" * 70)
    
    val = result.get("validation", {})
    deletion = result.get("deletion", {})
    
    print(f"\nOverall Valid: {result.get('valid', False)}")
    
    print("\n" + "-" * 40)
    print("STEP 1: ID VALIDATION & CORRECTION")
    print("-" * 40)
    
    print(f"Anchor Calls Found: {len(val.get('anchor_calls', []))}")
    
    errors = val.get("errors", [])
    if errors:
        print(f"\n❌ ID ERRORS ({len(errors)}):")
        for err in errors:
            print(f"   Step {err['step']} [{err['type']}]: {err['message']}")
    
    corrections = val.get("corrections", [])
    if corrections:
        print(f"\n✅ AUTO-CORRECTIONS ({len(corrections)}):")
        for corr in corrections:
            print(f"   '{corr['object_name']}': ID {corr['old_id']} → {corr['new_id']}")
    
    warnings = val.get("warnings", [])
    if warnings:
        print(f"\n⚠️  WARNINGS ({len(warnings)}):")
        for warn in warnings:
            print(f"   {warn['message']}")
    
    if deletion:
        print("\n" + "-" * 40)
        print("STEP 2: NON-EXISTENT OBJECT REMOVAL")
        print("-" * 40)
        
        non_existent = val.get("non_existent_objects", [])
        if non_existent:
            print(f"\n🗑️  NON-EXISTENT OBJECTS ({len(non_existent)}):")
            for item in non_existent:
                print(f"   '{item['given_name']}' (id: {item['given_id']}) - NOT IN SCENE")
        
        deleted = deletion.get("deleted_objects", [])
        if deleted:
            print(f"\n🧹 DELETED TRAJECTORIES:")
            for d in deleted:
                print(f"   Object #{d['position']}: '{d['object_name']}'")
                for step in d["deleted_steps"]:
                    print(f"      - {step}")
        
        print(f"\n📊 OBJECT COUNT: {deletion.get('original_object_count', 0)} → {deletion.get('final_object_count', 0)}")
        
        remaining = deletion.get("remaining_objects", [])
        if remaining:
            print(f"📍 REMAINING OBJECTS: {remaining}")
    
    print("\n" + "-" * 40)
    print("FINAL RESULT")
    print("-" * 40)
    
    final = result.get("final_trajectories")
    cleaned_response = result.get("cleaned_response")
    if final:
        print(f"\n{final}")
        print(f"\n{cleaned_response}")
    else:
        print("\n❌ No valid trajectory could be generated")
    
    print("\n" + "=" * 70)