"""Prompts for Endless Winter game analysis (System 1 + System 2)."""

import re
import json
from typing import Optional, Tuple, List, Dict, Any


def build_system1_prompt(img_width: int, img_height: int) -> str:
    """Build System 1 (Qwen Ref) prompt for fast UI element detection.

    Returns a prompt that uses Qwen's Ref (REC) capability to return
    bounding box of the most important clickable UI element.

    Args:
        img_width: Screenshot width
        img_height: Screenshot height

    Returns:
        Prompt string for Qwen
    """
    return f"""你是一个游戏自动化助手。这是策略游戏《无尽冬日》的截图，分辨率 {img_width}x{img_height}。

【任务】找出最重要的可点击 UI 元素，优先级如下：
1. 弹窗关闭按钮（X 按钮或"关闭"）
2. 红点通知（新消息提示）
3. "跳过"/"跳过剧情" 按钮
4. "确认" 按钮
5. 其他蓝色/绿色操作按钮

【返回格式】
使用 Ref 模式返回找到的元素的精确位置（必须包含坐标框）。

例如：找到了 <|object_ref_start|>关闭按钮<|object_ref_end|><|box_start|>(100,50,150,80)<|box_end|>

坐标格式：<|box_start|>(xmin,ymin,xmax,ymax)<|box_end|>
- xmin, ymin, xmax, ymax 是 0-1000 的归一化坐标
- x 轴从左到右，y 轴从上到下
- 如果看不到任何可点击元素，返回：无可点击元素

【重要】
- 必须返回坐标框，即使不确定也要给出最佳估计
- 只需返回一个元素（最重要的那个）
- 不要返回除了坐标框外的其他内容"""


def build_system2_prompt(img_width: int, img_height: int, action_history: List[Dict[str, Any]]) -> str:
    """Build System 2 (Gemini) prompt for strategic decision making.

    When System 1 fails repeatedly, escalate to Gemini for complex reasoning.

    Args:
        img_width: Screenshot width
        img_height: Screenshot height
        action_history: List of last 5 actions: [{"action": "click", "x": int, "y": int, "target": str, "time": float}]

    Returns:
        Prompt string for Gemini
    """
    action_str = "无" if not action_history else "\n".join(
        f"  {i+1}. {a.get('target', 'unknown')} @ ({a.get('x', 0)}, {a.get('y', 0)})"
        for i, a in enumerate(action_history[-5:])
    )

    return f"""你是《无尽冬日》游戏的战略助手。这是一个策略/建造游戏，玩家需要：
- 收集资源（木材、石材、粮食、钢材）
- 升级建筑和科技
- 参与战斗
- 应对暴风雪威胁

【当前状态】
分辨率：{img_width}x{img_height}
最近 5 步操作（快速模式已尝试但失败）：
{action_str}

【任务】
分析这张截图，识别当前最重要的可交互元素。可能的情况：
1. 有弹窗对话 → 返回关闭/跳过按钮坐标
2. 有红点通知 → 返回红点坐标
3. 有可升级建筑 → 返回升级按钮坐标
4. 有资源不足警告 → 返回收集资源按钮坐标

【返回格式】
必须是有效的 JSON，包含以下字段：
{{
  "action": "click",
  "x": <0 到 {img_width} 之间的整数>,
  "y": <0 到 {img_height} 之间的整数>,
  "target": "<点击目标的描述，如'关闭按钮'或'红点通知'>",
  "reasoning": "<简短解释为什么选择这个目标>",
  "confidence": <0.0 到 1.0 的浮点数>
}}

如果没有明显的可点击元素，返回：
{{
  "action": "skip",
  "reasoning": "当前无明确可操作目标",
  "confidence": 0.5
}}

【约束】
- 坐标必须在有效范围内
- confidence >= 0.5 表示可信
- 只返回 JSON，不要其他文字"""


def parse_qwen_ref_bbox(response: str, img_width: int, img_height: int) -> Optional[Tuple[int, int]]:
    """Parse Qwen Ref bounding box format and return center pixel coordinates.

    Supports multiple Qwen format variations:
    - <|box_start|>(xmin,ymin,xmax,ymax)<|box_end|> (standard)
    - <|im_start|>(xmin,ymin,xmax,ymax) (LM Studio variant)
    - (xmin,ymin,xmax,ymax) (fallback)

    Coordinates can be:
    - 0-1000 (normalized)
    - 0-image_size (pixel coordinates)

    Args:
        response: Qwen's response string
        img_width: Screenshot width
        img_height: Screenshot height

    Returns:
        Tuple (center_x, center_y) in pixel coordinates, or None if parsing failed
    """
    # Try multiple patterns (allow optional whitespace around commas)
    patterns = [
        # Standard Qwen format
        r'<\|box_start\|>\((\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\)',
        # LM Studio variant with <|im_start|>
        r'<\|im_start\|>\((\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\)',
        # Fallback: any (xmin,ymin,xmax,ymax) pattern
        r'\((\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\)',
    ]

    match = None
    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            break

    if not match:
        return None

    try:
        # Qwen VL models use (xmin, ymin, xmax, ymax) format with 0-1000 normalized coordinates
        xmin, ymin, xmax, ymax = map(int, match.groups())

        # Qwen VL ALWAYS returns 0-1000 normalized coords - no pixel mode
        center_x_norm = (xmin + xmax) / 2.0 / 1000.0
        center_y_norm = (ymin + ymax) / 2.0 / 1000.0
        center_x_px = int(center_x_norm * img_width)
        center_y_px = int(center_y_norm * img_height)

        # Clamp to valid range
        center_x_px = max(0, min(center_x_px, img_width - 1))
        center_y_px = max(0, min(center_y_px, img_height - 1))

        return (center_x_px, center_y_px)

    except (ValueError, AttributeError):
        return None


def parse_gemini_json(response: str) -> Optional[Dict[str, Any]]:
    """Parse Gemini JSON response.

    Args:
        response: Gemini's response string

    Returns:
        Parsed dict with action, x, y, target, reasoning, confidence, or None if parsing failed
    """
    # Try to extract JSON from response
    try:
        # Look for { ... }
        start = response.find('{')
        end = response.rfind('}')
        if start >= 0 and end > start:
            json_str = response[start:end+1]
            data = json.loads(json_str)
            return data
    except (json.JSONDecodeError, ValueError):
        pass

    return None


def build_scene_classification_prompt() -> str:
    """Build prompt for scene classification.

    Returns:
        Prompt to identify current game scene
    """
    return """请识别这是《无尽冬日》游戏的哪个场景，只返回以下单词之一：

main_city（主城画面，有大熔炉、建筑、红点）
wilderness（野外地图，有格子地图、探索区域）
battle（战斗画面，有攻击/技能按钮）
loading（加载中，有进度条或转场动画）
dialog（有弹出对话框或操作提示）
ad（有广告、充值礼包等弹窗）
unknown（无法判断）

【重要】只返回一个单词，不要其他内容。"""


def build_system1_prompt_for_scene(
    img_width: int,
    img_height: int,
    scene: str = "unknown",
    recent_actions: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Build System 1 prompt adapted to current scene.

    Args:
        img_width: Screenshot width
        img_height: Screenshot height
        scene: Current scene type from SceneClassifier
        recent_actions: Recent action history [{target, outcome}] for short-term memory

    Returns:
        Scene-specific System 1 prompt
    """
    # Scene-specific priorities
    scene_priorities = {
        "dialog": "【最高优先级】弹窗关闭按钮(X)、确认按钮、跳过按钮",
        "ad": "【最高优先级】广告关闭按钮(X)，通常在顶部角落",
        "main_city": "红点通知（新消息）、建筑升级感叹号、资源收集按钮、任务按钮",
        "wilderness": "移动确认按钮、地图拖拽指示、探索按钮、手指引导",
        "battle": "攻击按钮、技能按钮、自动战斗开关",
        "loading": "【等待】当前场景加载中，暂时无操作",
        "unknown": "任何明显的可点击 UI 元素、手指引导、按钮",
    }

    priority = scene_priorities.get(scene, scene_priorities["unknown"])

    # Short-term memory: summarize recent actions + outcomes + coords
    memory_block = ""
    if recent_actions:
        failed_entries = [a for a in recent_actions if a.get("outcome") == "no_change"]
        lines = []
        for a in recent_actions:
            oc = a.get("outcome", "unknown")
            mark = "✗画面未变" if oc == "no_change" else ("✓已变化" if oc == "changed" else "…")
            lines.append(f"  - {a.get('target', '?')} @({a.get('x','?')},{a.get('y','?')})  [{mark}]")

        # Group failed points by target name to surface "same-name different-location" guidance
        from collections import defaultdict
        by_target = defaultdict(list)
        for a in failed_entries:
            by_target[a["target"]].append((a.get("x", "?"), a.get("y", "?")))

        rules = []
        for t, pts in by_target.items():
            if len(pts) >= 1:
                pts_str = ", ".join(f"({x},{y})" for x, y in pts)
                if len(pts) >= 2:
                    rules.append(
                        f"- '{t}' 已在 {pts_str} 点击均无效。若画面仍存在同名按钮，**必须选择位置不同于上述坐标的那个**；如果只此一处，改选完全不同的元素（关闭/返回/其他建筑）。"
                    )
                else:
                    rules.append(
                        f"- '{t}' 在 {pts_str} 点过无变化。若画面存在多个同名按钮，请选**别的位置**的那个；否则改选其他元素。"
                    )

        warn = ""
        if rules:
            warn = "\n⚠️ **避免重复错误操作的规则**：\n" + "\n".join(rules)

        memory_block = f"""
【最近操作记录（短期记忆）】
{chr(10).join(lines)}{warn}
"""

    return f"""你是一个游戏自动化助手。这是策略游戏《无尽冬日》的截图，分辨率 {img_width}x{img_height}。
{memory_block}

【⚠️ 主城画面禁区（极其重要）】
主城画面（可见雪地、帐篷、建筑群、底部锁图标栏）**不存在**返回箭头或关闭按钮。请不要把以下元素误识别为导航按钮：
- 左上角的**圆形头像**（含人物肖像，常带等级/称号）→ 是玩家档案入口，不是"返回箭头"，**禁止点击**
- 右下角靠上的**方形蓝色信封图标**（x≈850-950, y≈700-900 归一化）→ 是邮件按钮，不是"地图/野外"，**禁止点击**
- 右下角最底的"野外"按钮 → 会跳转野外地图，偏离主流程，**非必要不点**
判断依据：如果画面没有半透明遮罩层、没有弹出面板、背景是地图/建筑，那就是主城，这里没有任何返回/关闭按钮。

【最高优先级 A — 关闭/返回按钮（仅在覆盖层画面）】
仅当画面是弹窗、通知面板、设置页、建筑详情页等**明显的覆盖层**（有白色/深色面板盖住主画面），画面边角才会有关闭/返回按钮，**必须优先点击**：
- 右上角 ✕ 或 X 图标（归一化坐标常在 x>850, y<150）
- 左上角 ← 返回箭头（x<150, y<150，形状是**箭头**不是圆形头像）
- 底部中央"关闭"或"取消"按钮
返回关闭/返回按钮本身的框，不是面板里的内容。**再次提醒：主城画面没有这类按钮。**

【最高优先级 B — 卡通手指/手掌图标】
若画面没有关闭/返回按钮，但出现独立的**卡通手指/手掌图标**（白色肤色、动画风格、不是地图上的角色），
它本身就是可点击的教学按钮，**点击手指图标的中心**即可推进教学。

手指图标特征：
- 白色/米色的卡通手掌，通常有明显的手指和拇指轮廓
- 独立漂浮在画面上，不属于地图场景
- 常出现在底部角落（左下或右下），有时伴随光圈
- 位置大多在 y=1200~1400 之间（图像下部）

**重要：如果看到手指图标，必须返回手指图标本身的框，不是它"指向"的地方！**

底部任务条（如"点燃火堆"、"建造xxx"）只是文字说明，不必去点那条文字。
若画面上没有手指图标，再考虑地图物体/其他按钮。

【升级/建造按钮 — 红色数字表示资源不足】
建筑升级/建造详情页常见蓝色"升级"或"建造"按钮，按钮下方/右侧会显示资源需求（图标+数字，如"18/30"）。
- 数字是**白色**：资源充足，点击后直接开始升级
- 数字是**红色/橙红色**：资源不足，但**仍然应该点击该升级按钮** —— 游戏会自动弹出资源获取引导（去指定建筑收集/生产对应资源），沿着引导继续即可推进主流程
- **禁止因为看到红色数字就放弃升级按钮**；也不要自行去点其他建筑凑资源，让游戏的引导带路
示例：看到蓝色"升级"按钮 + 红色 "18/30"，target="升级按钮"，坐标就是升级按钮本身的框。

【次优先级】{priority}

【返回格式】输出两行：
第一行：目标=xxx（简要描述要点击的元素）
第二行：(xmin,ymin,xmax,ymax)

坐标说明：
- 0-1000 的归一化值
- 格式：(xmin,ymin,xmax,ymax)
- x 从左到右，y 从上到下

示例1（左下角手指图标）：
目标=左下角手指图标
(40,830,180,920)

示例2（手指指向右侧建造按钮，无动词任务）：
目标=建造按钮
(650,480,820,560)

如果找不到可点击元素，只返回：无可点击元素"""


def build_system1_drag_prompt(img_width: int, img_height: int) -> str:
    """Build System 1 prompt for drag operations (wilderness scene).

    Args:
        img_width: Screenshot width
        img_height: Screenshot height

    Returns:
        Prompt for DRAG operation
    """
    return f"""你是一个游戏自动化助手。这是《无尽冬日》野外地图的截图，分辨率 {img_width}x{img_height}。

【任务】识别地图拖拽操作的【起点】和【终点】坐标。

游戏需要向上/下拖拽地图来探索。请找出：
1. 拖拽的起点（当前可见的地图区域中心）
2. 拖拽的终点（向上拖拽则终点更高，向下拖拽则终点更低）

【返回格式】
返回两个坐标框：起点和终点。例如：

起点：<|object_ref_start|>拖拽起点<|object_ref_end|><|box_start|>(400,300,500,400)<|box_end|>
终点：<|object_ref_start|>拖拽终点<|object_ref_end|><|box_start|>(200,300,300,400)<|box_end|>

坐标都是 0-1000 归一化。
如果无法判断拖拽方向，优先选择向上拖拽。"""


def build_system2_prompt_with_drag(img_width: int, img_height: int, action_history: List[Dict[str, Any]]) -> str:
    """Build System 2 prompt that supports DRAG and LONG_PRESS actions.

    Args:
        img_width: Screenshot width
        img_height: Screenshot height
        action_history: List of last 5 actions

    Returns:
        Enhanced System 2 prompt
    """
    action_str = "无" if not action_history else "\n".join(
        f"  {i+1}. {a.get('target', 'unknown')} @ ({a.get('x', 0)}, {a.get('y', 0)})"
        for i, a in enumerate(action_history[-5:])
    )

    action_str = "无" if not action_history else "\n".join(
        f"  {i+1}. {a.get('target', 'unknown')} @ ({a.get('x', 0)}, {a.get('y', 0)})"
        for i, a in enumerate(action_history[-5:])
    )

    return f"""你是《无尽冬日》游戏的战略助手。

【当前状态】
图像实际分辨率：{img_width} 像素宽 × {img_height} 像素高
返回的 x,y 必须是**像素坐标**，不是 0-1000 归一化值。
- x ∈ [0, {img_width}]，0=最左，{img_width}=最右
- y ∈ [0, {img_height}]，0=顶部，{img_height}=底部

最近 5 步操作：
{action_str}

【强制规则 — 看到关闭/返回按钮就点它，不做其他选择】
以下按钮只要画面上出现，**你就必须返回它的坐标，不允许选择任何其他目标**（包括 tab 切换、"一键已读"、"领取"、"前往"等）：
- 右上角 ✕ / X 关闭图标（典型位置：x > {int(0.85 * img_width)}, y < {int(0.15 * img_height)}）
- 左上角 ← 返回箭头（典型位置：x < {int(0.15 * img_width)}, y < {int(0.15 * img_height)}）
- "关闭"、"返回"、"取消" 文字按钮

理由：这是全局规则 —— 当前目标是回到主城界面继续游戏主流程，邮件/通知/弹窗都不是主流程，必须先脱离。
不要尝试探索面板里的 tab 或内容；不要点"一键已读"期望它自动关闭面板。直接点 X 或 ← 即可。
只有在画面上**完全看不到**关闭/返回按钮时，才考虑其他元素。

【可执行的操作及其使用场景】
1. click - 点击按钮、UI 元素（大部分情况）
2. drag - 拖拽地图边缘进行移动、平移视图；或在通知面板内上下滑动查看更多内容
3. long_press - 长按加速升级、长按冲锋（见到升级按钮或冲锋指示时）

【坐标提示】
- 顶部关闭 X / 返回箭头 → y < {int(0.15 * img_height)}
- 底部操作栏按钮（如"一键已读"、"前往"、"领取"）→ y > {int(0.9 * img_height)}
- 如果最近 3 步都是同一目标（说明点击无效），**立刻切换**到关闭/返回按钮或 DRAG

【Y 坐标锚点（无尽冬日常见 UI 区段，按 {img_height} 像素图）】
以下是典型详情面板（弹出设施/建筑升级卡片）的 Y 分布，务必据此校正你的 y：
- 顶部状态栏（资源/时间）：      y ∈ [0, {int(0.125 * img_height)}]
- 画面中段（主视口）：            y ∈ [{int(0.125 * img_height)}, {int(0.34 * img_height)}]
- 面板标题 + 进度条（如"厨房 1级 62%"）：y ∈ [{int(0.34 * img_height)}, {int(0.45 * img_height)}]
- **主操作按钮（升级 / 确认 / 领取 / 前往 / 一键XX）**：y ∈ [{int(0.45 * img_height)}, {int(0.56 * img_height)}]
- 按钮文字/资源消耗标签：        y ∈ [{int(0.56 * img_height)}, {int(0.70 * img_height)}]
- 分类 tab（属性/家具/幸存者）：  y ∈ [{int(0.70 * img_height)}, {int(0.84 * img_height)}]
- 底部导航栏：                    y ∈ [{int(0.90 * img_height)}, {img_height}]

**关键纠偏：** 模型常把"升级/确认"这类主操作按钮的 y 报**偏小 ~100 像素**，
误落在上方的进度条行上。如果你要点的是升级/确认/领取/前往：
- 先定位画面上那一**行的文字**（如看到"升级 10.4万/10"），
- 那一行的 y 通常在 [{int(0.45 * img_height)}, {int(0.56 * img_height)}]；
- 不要返回该行上方（进度条 y ≈ {int(0.38 * img_height)}-{int(0.44 * img_height)}）的 y。

【任务】识别最重要的可交互元素和最适合的操作类型。
**重要：返回 y 前再自查一遍：这个 y 在图像上究竟对应什么元素？
若对应的是进度条或标签行而不是按钮本身，必须把 y 往下调到按钮所在行。**

【返回格式】
根据操作类型返回相应 JSON：

点击：
{{
  "action": "click",
  "x": <0 到 {img_width} 的整数>,
  "y": <0 到 {img_height} 的整数>,
  "target": "点击目标描述",
  "reasoning": "简短解释",
  "confidence": <0.0 到 1.0 的浮点数>
}}

拖拽：
{{
  "action": "drag",
  "x1": <起点 X>,
  "y1": <起点 Y>,
  "x2": <终点 X>,
  "y2": <终点 Y>,
  "duration": <0.3 到 1.0 的浮点数>,
  "target": "拖拽方向描述",
  "reasoning": "为什么需要拖拽",
  "confidence": <浮点数>
}}

长按：
{{
  "action": "long_press",
  "x": <X 坐标>,
  "y": <Y 坐标>,
  "duration": <1.0 到 3.0 的浮点数>,
  "target": "长按目标",
  "reasoning": "为什么要长按",
  "confidence": <浮点数>
}}

如果没有明显操作，返回：
{{
  "action": "skip",
  "reasoning": "当前无明确可操作目标",
  "confidence": 0.5
}}

【约束】
- 坐标必须在有效范围内
- confidence >= 0.5 表示可信
- 只返回 JSON，不要其他文字"""
