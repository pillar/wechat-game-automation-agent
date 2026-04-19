"""Post-condition verifier for actions.

Each action declares an expected outcome (scene change, visual change); after
the next screenshot, verifier compares and reports mismatches so the planner
can mark failure and the memory_store can blacklist the coord.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class Expectation:
    action: Dict[str, Any]
    expected_scene: Optional[str] = None
    expected_change: bool = True
    scene_before: Optional[str] = None


@dataclass
class VerifyResult:
    ok: bool
    reason: str = ""
    scene_after: Optional[str] = None
    changed: bool = False
    action: Optional[Dict[str, Any]] = None
    scene_before: Optional[str] = None
    scene_matched: bool = False
    expected_scene: Optional[str] = None


class PostConditionVerifier:
    """Stateful verifier: record expectation before acting, verify after.

    Usage:
        verifier.record(action, expected_scene="main_city", scene_before="dialog")
        # ... execute action, wait a round, capture new screenshot ...
        result = verifier.verify(new_scene, change_detector_says_changed)
        if not result.ok: planner.mark_failure(); memory_store.add_blacklist(...)
    """

    def __init__(self):
        self._pending: Optional[Expectation] = None
        self._consecutive_failures = 0
        # Lifetime counters for honest stats reporting.
        self.total_verified = 0         # verify() calls with a pending expectation
        self.total_changed = 0          # subset where changed=True
        self.total_scene_matched = 0    # subset where scene_after matched expected
        self.total_no_change = 0        # subset where expected change but got none

    def record(
        self,
        action: Dict[str, Any],
        expected_scene: Optional[str] = None,
        scene_before: Optional[str] = None,
        expected_change: bool = True,
    ) -> None:
        self._pending = Expectation(
            action=action,
            expected_scene=expected_scene,
            expected_change=expected_change,
            scene_before=scene_before,
        )

    def has_pending(self) -> bool:
        return self._pending is not None

    def pending_action(self) -> Optional[Dict[str, Any]]:
        return self._pending.action if self._pending else None

    def verify(self, scene_after: Optional[str], changed: bool) -> VerifyResult:
        if self._pending is None:
            return VerifyResult(ok=True, reason="no_pending")
        exp = self._pending
        self._pending = None

        self.total_verified += 1
        if changed:
            self.total_changed += 1

        if exp.expected_change and not changed:
            self._consecutive_failures += 1
            self.total_no_change += 1
            return VerifyResult(
                ok=False,
                reason="no_visual_change",
                scene_after=scene_after,
                changed=False,
                action=exp.action,
                scene_before=exp.scene_before,
                scene_matched=False,
                expected_scene=exp.expected_scene,
            )

        if exp.expected_scene is not None and scene_after is not None:
            if scene_after == exp.expected_scene:
                self._consecutive_failures = 0
                self.total_scene_matched += 1
                return VerifyResult(
                    ok=True,
                    reason="scene_match",
                    scene_after=scene_after,
                    changed=changed,
                    action=exp.action,
                    scene_before=exp.scene_before,
                    scene_matched=True,
                    expected_scene=exp.expected_scene,
                )
            self._consecutive_failures = 0
            return VerifyResult(
                ok=True,
                reason="scene_no_match_but_changed",
                scene_after=scene_after,
                changed=changed,
                action=exp.action,
                scene_before=exp.scene_before,
                scene_matched=False,
                expected_scene=exp.expected_scene,
            )

        self._consecutive_failures = 0
        return VerifyResult(
            ok=True,
            reason="ok",
            scene_after=scene_after,
            changed=changed,
            action=exp.action,
            scene_before=exp.scene_before,
            scene_matched=False,
            expected_scene=exp.expected_scene,
        )

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    def reset(self) -> None:
        self._pending = None
        self._consecutive_failures = 0
