"""Routes grounding prompts + response parsing to per-family adapters.

Supported families:
    - qwen_vl : existing default; Qwen 3 VL Ref format, 0-1000 normalized bbox
    - ui_tars : ByteDance UI-TARS (Pixels mode); emits `Thought:\\nAction: click(point='<x,y>')`
    - cogagent: Zhipu CogAgent; emits `Grounded Operation: [[x1,y1,x2,y2]] ...`

All parsers return (pixel_x, pixel_y, action_type, target_desc) or None.
"""

import re
import logging
from typing import Optional, Tuple, List, Dict, Any, Callable

logger = logging.getLogger(__name__)


ParsedGrounding = Dict[str, Any]  # {x, y, action_type, target}


def build_grounding_prompt(
    family: str,
    img_width: int,
    img_height: int,
    scene: str,
    recent_actions: List[Dict[str, Any]],
    task_context: str = "",
    qwen_prompt_builder: Optional[Callable] = None,
) -> str:
    """Construct System-1-style grounding prompt for the chosen family.

    qwen_prompt_builder lets the endless_winter adapter pass in its
    scene-aware prompt builder to avoid a circular import.
    """
    family = (family or "qwen_vl").lower()
    if family == "qwen_vl":
        assert qwen_prompt_builder is not None, "qwen_vl needs a prompt builder"
        base = qwen_prompt_builder(img_width, img_height, scene, recent_actions=recent_actions)
        return _inject_task_context(base, task_context)

    if family == "ui_tars":
        return _build_ui_tars_prompt(img_width, img_height, scene, recent_actions, task_context)

    if family == "cogagent":
        return _build_cogagent_prompt(img_width, img_height, scene, recent_actions, task_context)

    raise ValueError(f"unknown grounding family: {family}")


def parse_grounding_response(
    family: str,
    response: str,
    img_width: int,
    img_height: int,
    qwen_parser: Optional[Callable] = None,
) -> Optional[ParsedGrounding]:
    family = (family or "qwen_vl").lower()
    if family == "qwen_vl":
        assert qwen_parser is not None, "qwen_vl needs a parser"
        coords = qwen_parser(response, img_width, img_height)
        if coords is None:
            return None
        action_type = _sniff_action_type(response)
        target = _sniff_qwen_target(response)
        return {"x": coords[0], "y": coords[1], "action_type": action_type, "target": target}

    if family == "ui_tars":
        return _parse_ui_tars(response, img_width, img_height)

    if family == "cogagent":
        return _parse_cogagent(response, img_width, img_height)

    raise ValueError(f"unknown grounding family: {family}")


def _inject_task_context(prompt: str, task_context: str) -> str:
    if not task_context:
        return prompt
    return f"{task_context}\n\n{prompt}"


def _build_ui_tars_prompt(img_width, img_height, scene, recent_actions, task_context) -> str:
    hist = "\n".join(
        f"- {a.get('target','?')} @({a.get('x','?')},{a.get('y','?')}) [{a.get('outcome','?')}]"
        for a in recent_actions
    ) or "(none)"
    ctx = f"{task_context}\n" if task_context else ""
    return f"""{ctx}You are a GUI automation agent for the mobile strategy game "Endless Winter".
Screenshot resolution: {img_width}x{img_height} pixels. Current scene: {scene}.

Recent actions (most recent last):
{hist}

Choose the single most important UI element to interact with this step.

Output format (strict):
Thought: <one-line reasoning in Chinese or English>
Action: click(point='<x>,<y>')

Use pixel coordinates in [0,{img_width}] for x and [0,{img_height}] for y.
For drag use: Action: drag(start_point='<x1>,<y1>', end_point='<x2>,<y2>')
For long press use: Action: long_press(point='<x>,<y>', duration=<seconds>)
If nothing actionable, respond: Action: wait()
"""


def _parse_ui_tars(response: str, img_width: int, img_height: int) -> Optional[ParsedGrounding]:
    action_match = re.search(r"Action:\s*(\w+)\s*\((.*)\)", response, re.DOTALL)
    if not action_match:
        return None
    action_type = action_match.group(1).lower()
    args = action_match.group(2)

    if action_type == "wait":
        return None

    if action_type in ("click", "tap", "long_press"):
        m = re.search(r"point\s*=\s*['\"]?\s*(\d+)\s*,\s*(\d+)", args)
        if not m:
            return None
        x, y = int(m.group(1)), int(m.group(2))
        x = max(0, min(x, img_width - 1))
        y = max(0, min(y, img_height - 1))
        mapped = "long_press" if action_type == "long_press" else "click"
        return {"x": x, "y": y, "action_type": mapped, "target": _sniff_ui_tars_thought(response)}

    if action_type == "drag":
        m = re.search(
            r"start_point\s*=\s*['\"]?\s*(\d+)\s*,\s*(\d+).*?end_point\s*=\s*['\"]?\s*(\d+)\s*,\s*(\d+)",
            args,
            re.DOTALL,
        )
        if not m:
            return None
        x1, y1, x2, y2 = map(int, m.groups())
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        return {
            "x": cx, "y": cy, "action_type": "drag",
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "target": _sniff_ui_tars_thought(response),
        }

    return None


def _build_cogagent_prompt(img_width, img_height, scene, recent_actions, task_context) -> str:
    ctx = f"{task_context}\n\n" if task_context else ""
    return f"""{ctx}Task: Identify the single most important clickable UI element in this screenshot of the mobile game "Endless Winter" (scene={scene}, size={img_width}x{img_height}).

Respond with:
Thought: <reasoning>
Grounded Operation: click [[x1,y1,x2,y2]] — where coords are 0-1000 normalized.
If no clear target, respond: Grounded Operation: wait
"""


def _parse_cogagent(response: str, img_width: int, img_height: int) -> Optional[ParsedGrounding]:
    if "wait" in response.lower() and "[[" not in response:
        return None
    m = re.search(r"\[\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]\]", response)
    if not m:
        return None
    x1, y1, x2, y2 = map(int, m.groups())
    cx_norm = (x1 + x2) / 2.0 / 1000.0
    cy_norm = (y1 + y2) / 2.0 / 1000.0
    x = int(cx_norm * img_width)
    y = int(cy_norm * img_height)
    action_type = _sniff_action_type(response)
    return {"x": x, "y": y, "action_type": action_type, "target": _sniff_cogagent_thought(response)}


def _sniff_action_type(response: str) -> str:
    lowered = response.lower()
    if "long_press" in lowered or "long press" in lowered:
        return "long_press"
    if "drag" in lowered:
        return "drag"
    return "click"


def _sniff_qwen_target(response: str) -> str:
    m = re.search(r"目标\s*[=:：]\s*([^\n]+)", response)
    if m:
        return m.group(1).strip()[:80]
    return "UI element"


def _sniff_ui_tars_thought(response: str) -> str:
    m = re.search(r"Thought:\s*(.+?)(?=\n|Action:)", response, re.DOTALL)
    if m:
        return m.group(1).strip()[:80]
    return "UI element (UI-TARS)"


def _sniff_cogagent_thought(response: str) -> str:
    m = re.search(r"Thought:\s*(.+?)(?=\n|Grounded)", response, re.DOTALL)
    if m:
        return m.group(1).strip()[:80]
    return "UI element (CogAgent)"
