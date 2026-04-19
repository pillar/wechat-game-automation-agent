"""Task tree planner with subgoal state machine.

Loads a hierarchical task plan from YAML; exposes the currently active leaf
task so the game adapter can bias prompts and verify post-conditions.
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any, Union

import yaml

from core.completion_checks import resolve as _resolve_check

logger = logging.getLogger(__name__)


STATUS_PENDING = "pending"
STATUS_ACTIVE = "active"
STATUS_DONE = "done"
STATUS_FAILED = "failed"


@dataclass
class Task:
    name: str
    description: str = ""
    expected_scene: Optional[str] = None
    success_keywords: List[str] = field(default_factory=list)
    failure_keywords: List[str] = field(default_factory=list)
    max_failures: int = 5
    # B: 验收靠"稳定"而非"关键字命中"
    stability_rounds: int = 3
    # A: done 后立即重置为 pending（popup/task_bar 会不断再次触发）
    repeat_until_stable: bool = False
    # C: 显式完成判据（yaml: 字符串或 {type, ...kwargs} dict），优先于 B
    completion_check: Optional[Any] = None
    children: List["Task"] = field(default_factory=list)
    status: str = STATUS_PENDING
    fail_count: int = 0
    stable_count: int = 0
    started_at: float = 0.0

    @property
    def is_leaf(self) -> bool:
        return not self.children


def _task_from_dict(d: Dict[str, Any]) -> Task:
    t = Task(
        name=d["name"],
        description=d.get("description", ""),
        expected_scene=d.get("expected_scene"),
        success_keywords=list(d.get("success_keywords", [])),
        failure_keywords=list(d.get("failure_keywords", [])),
        max_failures=int(d.get("max_failures", 5)),
        stability_rounds=int(d.get("stability_rounds", 3)),
        repeat_until_stable=bool(d.get("repeat_until_stable", False)),
        completion_check=d.get("completion_check"),
    )
    for child in d.get("children", []) or []:
        t.children.append(_task_from_dict(child))
    return t


class Planner:
    """DFS task-tree scheduler.

    Selects the first pending leaf task; biases prompts via `active_task_context()`.
    Marks tasks done/failed based on post-action verification.
    """

    def __init__(self, plan_path: Optional[str] = None):
        self.root: Optional[Task] = None
        if plan_path:
            self.load(plan_path)

    def load(self, plan_path: str) -> None:
        p = Path(plan_path)
        if not p.exists():
            logger.info(f"[PLAN] no plan file at {p}, planner idle")
            self.root = None
            return
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        root_dict = data.get("task") or data
        self.root = _task_from_dict(root_dict)
        logger.info(f"[PLAN] loaded: {self.root.name} with {self._count_leaves(self.root)} leaves")

    def active_task(self) -> Optional[Task]:
        if self.root is None:
            return None
        return self._first_pending_leaf(self.root)

    def active_task_context(self) -> str:
        t = self.active_task()
        if t is None:
            return ""
        parts = [f"【当前子目标】{t.name}"]
        if t.description:
            parts.append(f"描述：{t.description}")
        if t.expected_scene:
            parts.append(f"预期场景：{t.expected_scene}")
        if t.success_keywords:
            parts.append(f"成功标志（若目标描述/按钮文字包含这些则优先选择）：{', '.join(t.success_keywords)}")
        return "\n".join(parts)

    def mark_success(self, task: Optional[Task] = None) -> None:
        t = task or self.active_task()
        if t is None:
            return
        t.status = STATUS_DONE
        logger.info(f"[PLAN] done: {t.name} (stable_count={t.stable_count})")
        # A: repeat_until_stable → 立即重置为 pending，popup/task_bar 下一轮可再次进入
        if t.repeat_until_stable:
            t.status = STATUS_PENDING
            t.stable_count = 0
            t.fail_count = 0
            logger.info(f"[PLAN] re-arm: {t.name} (repeat_until_stable)")

    def mark_failure(self, task: Optional[Task] = None, reason: str = "") -> None:
        t = task or self.active_task()
        if t is None:
            return
        t.fail_count += 1
        t.stable_count = 0  # 任何失败都打断稳定计数
        if t.fail_count >= t.max_failures:
            t.status = STATUS_FAILED
            logger.warning(f"[PLAN] failed: {t.name} ({reason}) after {t.fail_count} attempts")

    def tick(self, ctx: Any, idle: bool) -> None:
        """Unified per-round tick. C completion_check (if set) overrides B stability.

        Args:
            ctx: core.completion_checks.CheckContext — has screenshot/scene/active_task/memory/qwen
            idle: True if this round produced no action (skip / no_change / no actionable target)
        """
        t = self.active_task()
        if t is None:
            return
        # C: explicit check wins when present
        if t.completion_check is not None:
            fn = _resolve_check(t.completion_check)
            if fn is not None:
                try:
                    if fn(ctx):
                        logger.info(f"[PLAN] completion_check OK for {t.name}")
                        self.mark_success(t)
                        return
                except NotImplementedError as e:
                    logger.debug(f"[PLAN] completion_check not implemented for {t.name}: {e}")
                except Exception as e:
                    logger.warning(f"[PLAN] completion_check error on {t.name}: {e}")
        # B: stability fallback
        scene = getattr(ctx, "scene", None)
        scene_matches = (t.expected_scene is None or scene == t.expected_scene)
        self.tick_stability(scene_matches=scene_matches, idle=idle)

    def tick_stability(self, scene_matches: bool, idle: bool) -> None:
        """B: 稳定性验收 —— active 叶任务在 expected_scene 下连续 N 轮 idle 则视为完成。

        Args:
            scene_matches: 当前 scene 是否等于 active.expected_scene
            idle: 本轮是否"没做有效事"（未执行动作 / ChangeDetector no_change / 无可点击元素）
        """
        t = self.active_task()
        if t is None:
            return
        if scene_matches and idle:
            t.stable_count += 1
            logger.debug(f"[PLAN] stability {t.name}: {t.stable_count}/{t.stability_rounds}")
            if t.stable_count >= t.stability_rounds:
                self.mark_success(t)
        else:
            if t.stable_count > 0:
                logger.debug(f"[PLAN] stability reset: {t.name} (scene_ok={scene_matches} idle={idle})")
            t.stable_count = 0

    def reset(self) -> None:
        if self.root is None:
            return
        self._reset_recursive(self.root)

    def summary(self) -> Dict[str, int]:
        if self.root is None:
            return {"leaves": 0, "done": 0, "failed": 0, "pending": 0}
        leaves = self._collect_leaves(self.root)
        return {
            "leaves": len(leaves),
            "done": sum(1 for t in leaves if t.status == STATUS_DONE),
            "failed": sum(1 for t in leaves if t.status == STATUS_FAILED),
            "pending": sum(1 for t in leaves if t.status in (STATUS_PENDING, STATUS_ACTIVE)),
        }

    def _first_pending_leaf(self, node: Task) -> Optional[Task]:
        if node.is_leaf:
            if node.status in (STATUS_PENDING, STATUS_ACTIVE):
                if node.status == STATUS_PENDING:
                    node.status = STATUS_ACTIVE
                    node.started_at = time.time()
                return node
            return None
        for child in node.children:
            found = self._first_pending_leaf(child)
            if found is not None:
                return found
        return None

    def _collect_leaves(self, node: Task) -> List[Task]:
        if node.is_leaf:
            return [node]
        out = []
        for c in node.children:
            out.extend(self._collect_leaves(c))
        return out

    def _count_leaves(self, node: Task) -> int:
        return len(self._collect_leaves(node))

    def _reset_recursive(self, node: Task) -> None:
        node.status = STATUS_PENDING
        node.fail_count = 0
        node.stable_count = 0
        node.started_at = 0.0
        for c in node.children:
            self._reset_recursive(c)
