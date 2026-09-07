"""
LLM Agent Module
=================
Uses GPT-4 (or other LLMs) as an agent to parse camera operation instructions,
reason about the request, and generate a plan with tool calls.

Reference: Section 3.3 of Liu et al. (2024)
"""

import json
import re
from typing import Dict, List, Optional


# The system prompt from the ChatCam paper (provided in the uploaded file)
CHATCAM_SYSTEM_PROMPT = """You are a dialog agent that helps users to operate cameras in 3D scenes using dialog. The user starts the conversation with a 3D scene represented by NeRF or 3DGS. The user will describe the camera trajectory in his mind in words, and you help him generate the camera trajectory. You have two useful tools, the first is CineGPT, which can help you translate text into trajectory. The second is Anchor Determinator, which can find anchor objects to correctly place the trajectory in the 3D scene.

Please act according to the following instructions: INSTRUCTIONS: 1. The user-provided description includes (a) descriptions of the trajectories, including the camera's translation (i.e., "pan forward"), rotation (i.e., "turn left"), and camera parameters (i.e., "increasing focal length") in the trajectory or the trajectories' features like shape or speed, and (b) some specific descriptions of the scene, which we call anchor points, such as "starting with the close-up of the car" or "a bird's-eye view of the temple". 2. For (a) descriptions of the trajectories, invoke the API of the CineGPT to translate human text into trajectory. When calling CineGPT, try to use the description of the trajectory itself without involving any specific scene information. i. To summon CineGPT, the command is termed "infer_cinegpt". Its arguments are: "traj_description": "<traj_description>". ii. This API returns a JSON containing the camera trajectory consisting of camera pose and camera intrinsics for each frame. 3. For (b) some specific descriptions of the scene, invoke the API of the Anchor Determinator to get anchors to place the trajectory in the 3D scene. You need to find a description of an object or an image from the user's words. Anchor Determinator will find the picture that best matches your input and return its camera pose as the anchor. i. To call Anchor Determinator, the command is termed "get_anchor". Its arguments are: "anchor_description": "<anchor_description>". ii. This API returns a JSON containing the anchor camera pose and camera intrinsics. 4. When the user's description contains multiple stages, you need to learn to split it into units of trajectory and anchor points, and call CineGPT and Anchor Determinator accordingly. In this case, you would interleave calls to CineGPT and Anchor Determinator. 5. Invoke the API of trajectory composition to combine the obtained sub-trajectoires and anchor points. i. To call trajectory composition, the command is termed "traj_compose". Its arguments are: " compose ": "<list_of_traj_anchor>". ii. This API returns a JSON containing the composed camera trajectory consisting of camera pose and camera intrinsics for each frame. iii. When encountering illegal input, this API will raise an error. 6. Your generated plan should follow these steps. i. Call CineGPT with trajectory descriptions (one or more times). ii. Call Anchor Determinator with anchor descriptions (one or more times). iii. Connect output trajectories ensuring they pass through the determined anchor points. iv. Ask the user for further clarification if needed. v. Render a video using the generated camera trajectory.

RESPONSE TEMPLATE: { "observation": "observation", "reasoning": "reasoning", "plan": "a numbered list of steps to take that conveys the long-term plan" }

In your response JSON, it is generated in the order of "observation", "reasoning", and "plan". Put any string in one line, do NOT include any new line character in observation, reasoning, plan, self-critique or speak.

Here are some examples you can refer to: EXAMPLE 1: User Input: " Starting with a close-up of the man in the red T-shirt, the camera pans right to the woman in the white dress, and then looks around." Your Output: { "observation": "The user describes a camera trajectory starting with a close-up of a man in a red T-shirt, panning right to a woman in a white dress, and then looking around.", "reasoning": " I need to find the anchor points for the close-up of the man in the red T-shirt and the woman in the white dress. Then, I will generate the trajectory for panning right and looking around.", "plan": " 1. Call Anchor Determinator with 'close-up of the man in the red T-shirt'. 2. Call Anchor Determinator with 'the woman in the white dress'. 3. Call CineGPT with 'pan right'. 4. Call CineGPT with 'look around'. 5. Connect the output trajectories, ensuring they pass through the anchor points. 6. Ask the user for further clarification if needed. 7. Render a video using the generated camera trajectory." }

EXAMPLE 2: User Input: "Take a dolly zoom shot of the woman's face." Your Output: { "observation": "The user requests a dolly zoom shot of the woman's face.", "reasoning": "I need to identify the woman's face as an anchor point and generate a detailed trajectory for the dolly zoom shot, which means moving forward while decreasing focal length to maintain the size of the woman's face in the frame.", "plan": "1. Call Anchor Determinator with 'the woman's face'. 2. Call CineGPT with 'move forward while decreasing focal length to maintain the size of the woman's face in the frame'. 3. Connect the output trajectories, ensuring they pass through the anchor points. 4. Ask the user for further clarification if needed. 5. Render a video using the generated camera trajectory." }

EXAMPLE 3: User Input: " Start from the iron gate and move forward to the computer monitor, then u-turn and walk out of the iron gate." Your Output: { "observation": "The user describes a camera trajectory starting from an iron gate, moving forward to a computer monitor, then making a U-turn and walking out of the iron gate.", "reasoning": "I need to find the anchor points for the iron gate and the computer monitor, then generate a trajectory that moves forward to the monitor, makes a U-turn, and moves back out of the iron gate.", "plan": "1. Call Anchor Determinator with 'the iron gate'. 2. Call Anchor Determinator with 'the computer monitor'. 3. Call CineGPT with 'move forward'. 4. Call CineGPT with 'U-turn and move backward'. 5. Connect the output trajectories, ensuring they pass through the anchor points. 6. Ask the user for further clarification if needed. 7. Render a video using the generated camera trajectory." }"""


class ChatCamAgent:
    """
    LLM Agent that parses camera operation instructions and generates
    execution plans with tool calls to CineGPT and Anchor Determinator.
    
    Uses GPT-4.1 by default, with fallback to local parsing for offline use.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4.1",
        system_prompt: str = CHATCAM_SYSTEM_PROMPT,
    ):
        self.model = model
        self.system_prompt = system_prompt
        self.api_key = api_key
        self.client = None

        if api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=api_key)
                print(f"    Using {model} as LLM agent")
            except ImportError:
                print("    Warning: openai package not found. Using local parser fallback.")
        else:
            print("    No API key provided. Using local instruction parser.")

    def parse_instruction(
        self,
        instruction: str,
        conversation_history: List[dict] = None,
    ) -> Dict:
        """
        Parse a camera operation instruction into an execution plan.
        
        Returns:
            Dict with keys:
                - observation: Summary of the user's request
                - reasoning: Agent's reasoning about how to handle it
                - plan_steps: List of human-readable plan steps
                - tool_calls: List of tool call dicts with 'tool' and 'args'
        """
        if self.client:
            return self._parse_with_llm(instruction, conversation_history)
        else:
            return self._parse_locally(instruction)

    def _parse_with_llm(
        self,
        instruction: str,
        conversation_history: List[dict] = None,
    ) -> Dict:
        """Parse instruction using the LLM API."""
        messages = [{"role": "system", "content": self.system_prompt}]

        # Add conversation history for multi-turn
        if conversation_history:
            for msg in conversation_history:
                messages.append(msg)

        messages.append({"role": "user", "content": instruction})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,
                max_tokens=2048,
            )

            response_text = response.choices[0].message.content.strip()

            # Parse the JSON response
            parsed = self._extract_json(response_text)
            if parsed:
                plan_text = parsed.get("plan", "")
                tool_calls = self._extract_tool_calls(plan_text)
                return {
                    "observation": parsed.get("observation", ""),
                    "reasoning": parsed.get("reasoning", ""),
                    "plan_steps": self._extract_plan_steps(plan_text),
                    "tool_calls": tool_calls,
                    "raw_response": response_text,
                }

        except Exception as e:
            print(f"    LLM API error: {e}")

        # Fallback
        return self._parse_locally(instruction)

    def _parse_locally(self, instruction: str) -> Dict:
        """
        Local fallback parser that extracts trajectory descriptions
        and anchor points from the instruction without an LLM.
        
        Uses heuristic rules to identify:
        - Anchor points: references to specific objects/locations
        - Trajectory descriptions: camera movement descriptions
        """
        instruction_lower = instruction.lower()
        tool_calls = []
        plan_steps = []

        # Split instruction into segments
        segments = self._split_instruction(instruction)

        # Classify each segment as anchor or trajectory
        anchor_keywords = [
            "starting from", "start from", "starting with", "close-up of",
            "close up of", "facing", "from the", "of the", "at the",
            "above the", "behind the", "in front of", "next to",
            "bird's-eye", "bird's eye", "aerial view", "top-down",
            "directly above", "looking at",
        ]

        trajectory_keywords = [
            "pan", "tilt", "zoom", "dolly", "orbit", "rotate", "turn",
            "move forward", "move backward", "pull back", "push forward",
            "sweep", "glide", "roll", "track", "crane", "ascend", "descend",
            "u-turn", "s-shaped", "circular", "look around",
        ]

        step_num = 1
        for seg in segments:
            seg_lower = seg.lower().strip()

            is_anchor = any(kw in seg_lower for kw in anchor_keywords)
            is_trajectory = any(kw in seg_lower for kw in trajectory_keywords)

            if is_anchor and is_trajectory:
                # Contains both: extract anchor and trajectory separately
                anchor_desc = self._extract_anchor_from_segment(seg)
                traj_desc = self._extract_trajectory_from_segment(seg)

                if anchor_desc:
                    tool_calls.append({
                        "tool": "get_anchor",
                        "args": {"anchor_description": anchor_desc},
                    })
                    plan_steps.append(
                        f"{step_num}. Call Anchor Determinator with '{anchor_desc}'."
                    )
                    step_num += 1

                if traj_desc:
                    tool_calls.append({
                        "tool": "infer_cinegpt",
                        "args": {"traj_description": traj_desc},
                    })
                    plan_steps.append(
                        f"{step_num}. Call CineGPT with '{traj_desc}'."
                    )
                    step_num += 1

            elif is_anchor:
                anchor_desc = self._clean_anchor_description(seg)
                tool_calls.append({
                    "tool": "get_anchor",
                    "args": {"anchor_description": anchor_desc},
                })
                plan_steps.append(
                    f"{step_num}. Call Anchor Determinator with '{anchor_desc}'."
                )
                step_num += 1

            elif is_trajectory:
                traj_desc = self._clean_trajectory_description(seg)
                tool_calls.append({
                    "tool": "infer_cinegpt",
                    "args": {"traj_description": traj_desc},
                })
                plan_steps.append(
                    f"{step_num}. Call CineGPT with '{traj_desc}'."
                )
                step_num += 1

            else:
                # Ambiguous: treat as anchor if it references an object
                if self._has_object_reference(seg):
                    tool_calls.append({
                        "tool": "get_anchor",
                        "args": {"anchor_description": seg.strip()},
                    })
                    plan_steps.append(
                        f"{step_num}. Call Anchor Determinator with '{seg.strip()}'."
                    )
                else:
                    tool_calls.append({
                        "tool": "infer_cinegpt",
                        "args": {"traj_description": seg.strip()},
                    })
                    plan_steps.append(
                        f"{step_num}. Call CineGPT with '{seg.strip()}'."
                    )
                step_num += 1

        # Add composition step
        plan_steps.append(
            f"{step_num}. Connect output trajectories ensuring they pass through anchor points."
        )
        plan_steps.append(
            f"{step_num + 1}. Render a video using the generated camera trajectory."
        )

        # Add compose tool call
        tool_calls.append({
            "tool": "traj_compose",
            "args": {"compose": [r["args"] for r in tool_calls]},
        })

        return {
            "observation": f"The user describes a camera trajectory: {instruction}",
            "reasoning": f"Identified {sum(1 for t in tool_calls if t['tool'] == 'get_anchor')} "
                        f"anchor points and {sum(1 for t in tool_calls if t['tool'] == 'infer_cinegpt')} "
                        f"trajectory segments.",
            "plan_steps": plan_steps,
            "tool_calls": tool_calls,
        }

    # -----------------------------------------------------------------------
    # Helper methods
    # -----------------------------------------------------------------------

    def _split_instruction(self, instruction: str) -> List[str]:
        """Split compound instruction into segments."""
        text = instruction
        for delimiter in ["and then", "then", "next", "followed by", "finally"]:
            text = re.sub(rf'\b{delimiter}\b', ',', text, flags=re.IGNORECASE)
        segments = re.split(r'[,;.]', text)
        return [s.strip() for s in segments if s.strip() and len(s.strip()) > 3]

    def _extract_anchor_from_segment(self, segment: str) -> Optional[str]:
        """Extract object/anchor description from a mixed segment."""
        # Look for "of the X", "from the X", "at the X" patterns
        patterns = [
            r"(?:of|from|at|above|behind|facing)\s+(?:the\s+)?(.+?)(?:\s*,|\s*$)",
            r"close[- ]up\s+(?:of\s+)?(?:the\s+)?(.+?)(?:\s*,|\s*$)",
            r"starting\s+(?:from|with)\s+(?:the\s+)?(.+?)(?:\s*,|\s*$)",
        ]
        for pat in patterns:
            m = re.search(pat, segment, re.IGNORECASE)
            if m:
                return m.group(1).strip()
        return segment.strip()

    def _extract_trajectory_from_segment(self, segment: str) -> Optional[str]:
        """Extract camera movement description from a mixed segment."""
        traj_patterns = [
            r"(pan\s+\w+)", r"(zoom\s+\w+)", r"(dolly\s+\w+)",
            r"(orbit\s+\w+)", r"(tilt\s+\w+)", r"(move\s+\w+)",
            r"(pull\s+\w+)", r"(push\s+\w+)", r"(sweep\s+\w+)",
            r"(rotate\s+\w+)", r"(turn\s+\w+)", r"(look\s+around)",
        ]
        for pat in traj_patterns:
            m = re.search(pat, segment, re.IGNORECASE)
            if m:
                return m.group(1).strip()
        return None

    def _clean_anchor_description(self, segment: str) -> str:
        """Clean an anchor segment for CLIP matching."""
        # Remove trajectory keywords
        cleaned = re.sub(
            r'\b(starting from|start from|starting with|begin at)\b',
            '', segment, flags=re.IGNORECASE
        ).strip()
        return cleaned if cleaned else segment.strip()

    def _clean_trajectory_description(self, segment: str) -> str:
        """Clean a trajectory segment for CineGPT."""
        # Remove scene-specific object references for pure trajectory
        # CineGPT works best with abstract trajectory descriptions
        return segment.strip()

    def _has_object_reference(self, segment: str) -> bool:
        """Check if a segment references a specific object."""
        # Simple heuristic: contains a noun that could be an object
        object_indicators = [
            "the", "a ", "an ", "this", "that",
        ]
        seg_lower = segment.lower()
        return any(ind in seg_lower for ind in object_indicators)

    def _extract_json(self, text: str) -> Optional[dict]:
        """Extract JSON from LLM response text."""
        # Try to find JSON block
        json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # Try to parse the whole response
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Manual extraction
        result = {}
        for key in ["observation", "reasoning", "plan"]:
            pattern = rf'"{key}"\s*:\s*"([^"]*)"'
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                result[key] = m.group(1)

        return result if result else None

    def _extract_plan_steps(self, plan_text: str) -> List[str]:
        """Extract numbered steps from plan text."""
        steps = re.findall(r'\d+\.\s*(.+?)(?=\d+\.|$)', plan_text)
        if steps:
            return [f"{i+1}. {s.strip()}" for i, s in enumerate(steps)]
        return [plan_text]

    def _extract_tool_calls(self, plan_text: str) -> List[dict]:
        """Extract tool calls from the plan text."""
        tool_calls = []

        # Find Anchor Determinator calls
        anchor_matches = re.findall(
            r"(?:Call|call)\s+Anchor\s+Determinator\s+with\s+['\"](.+?)['\"]",
            plan_text, re.IGNORECASE
        )
        for desc in anchor_matches:
            tool_calls.append({
                "tool": "get_anchor",
                "args": {"anchor_description": desc},
            })

        # Find CineGPT calls
        cinegpt_matches = re.findall(
            r"(?:Call|call)\s+CineGPT\s+with\s+['\"](.+?)['\"]",
            plan_text, re.IGNORECASE
        )
        for desc in cinegpt_matches:
            tool_calls.append({
                "tool": "infer_cinegpt",
                "args": {"traj_description": desc},
            })

        return tool_calls
