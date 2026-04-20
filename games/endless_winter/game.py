"""Endless Winter (无尽冬日) game automation with dual-system architecture."""

import os
import time
import logging
import json
from typing import Dict, Any, Optional, Tuple
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from core.base_game import BaseGame
from core.screen import ScreenCapture
from core.input_controller import InputController
from core.ai_client import LocalVisionClient, GeminiVisionClient
from core.memory_store import MemoryStore
from core.planner import Planner
from core.verifier import PostConditionVerifier
from core.trajectory_logger import TrajectoryLogger
from core.model_router import build_grounding_prompt, parse_grounding_response
from core.completion_checks import CheckContext
from core.research import ResearchStore
from .change_detector import ChangeDetector, compute_mse
from .scene_classifier import SceneClassifier
from .stuck_monitor import StuckMonitor
from .prompts import (
    build_system1_prompt,
    build_system1_prompt_for_scene,
    build_system2_prompt,
    build_system2_prompt_with_drag,
    parse_qwen_ref_bbox,
    parse_gemini_json,
)

logger = logging.getLogger(__name__)


class EndlessWinterGame(BaseGame):
    """Endless Winter game automation with System 1 (Qwen) + System 2 (Gemini)."""

    def __init__(self, config: Dict[str, Any], ai_client=None):
        """Initialize Endless Winter game.

        Args:
            config: Game configuration dict with system1, system2, timing, vision, change_detection
            ai_client: AI client (used as System 2 / Gemini). System 1 is instantiated locally.
        """
        self.config = config
        self.input_controller = InputController()
        self.screen_capture = ScreenCapture()

        # Extract config sections
        self._vision = config.get("vision", {})
        self._s1_cfg = config.get("system1", {})
        self._s2_cfg = config.get("system2", {})
        self._timing = config.get("timing", {})
        self._cd_cfg = config.get("change_detection", {})

        # game_loop reads these attributes directly (lines 171-176)
        self.top_offset = self._vision.get("top_offset", 88)
        self.bottom_offset = self._vision.get("bottom_offset", 0)
        self.resize_width = self._vision.get("resize_width", 540)
        self.use_resize = True

        # Dual-system clients
        # System 2: prefer Gemini cloud API if GEMINI_API_KEY is set (smarter for fallback);
        # fall back to whatever main.py passed in (could be another Qwen instance).
        gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if gemini_key:
            from core.ai_client import GeminiVisionClient
            self._gemini_client = GeminiVisionClient(
                api_key=gemini_key,
                model=os.environ.get("GEMINI_MODEL", "gemini-flash-latest"),
            )
            logger.info("[S2] Using Gemini cloud API (from GEMINI_API_KEY)")
        else:
            self._gemini_client = ai_client
            logger.info("[S2] No GEMINI_API_KEY; reusing main.py ai_client for S2")

        # System 1: Qwen (instantiated locally) - fast UI element detection
        local_api_url = os.environ.get("LOCAL_API_URL", "http://192.168.1.156:1234")
        infer_cfg = config.get("inference", {})
        self._qwen_client = LocalVisionClient(
            api_base=local_api_url,
            model="qwen/qwen3-vl-8b",
            image_format=infer_cfg.get("image_format", "webp"),  # ⑤ WebP压缩
            image_quality=infer_cfg.get("image_quality", 70),
        )

        # Change detection
        cd_enabled = self._cd_cfg.get("enabled", True)
        if cd_enabled:
            self._change_detector = ChangeDetector(
                threshold=self._cd_cfg.get("threshold", 30),
                roi_skip_top=self._cd_cfg.get("roi_skip_top", 0.1),
            )
        else:
            self._change_detector = None

        # Dual-system state
        self._s1_fail_count = 0
        self._fail_threshold = self._s1_cfg.get("fail_threshold", 3)
        self._action_history = deque(maxlen=5)
        self._last_parsed_action: Optional[Dict[str, Any]] = None
        self._s2_last_call_time = 0.0
        self._s2_min_interval = self._s2_cfg.get("min_interval_s", 5.0)

        # Window info cache
        self._window_info: Optional[Dict[str, Any]] = None

        # ===== 5项改进集成 =====

        # ① Scene state machine
        scene_cfg = config.get("scene_classifier", {})
        if scene_cfg.get("enabled", True):
            self._scene_classifier = SceneClassifier(
                self._qwen_client,
                classify_interval=scene_cfg.get("classify_interval_s", 5.0),
            )
        else:
            self._scene_classifier = None

        # ④ Stuck monitor (死循环检测)
        stuck_cfg = config.get("stuck_monitor", {})
        if stuck_cfg.get("enabled", True):
            self._stuck_monitor = StuckMonitor(
                click_threshold=stuck_cfg.get("click_threshold", 5),
                grid_size=stuck_cfg.get("grid_size", 20),
            )
        else:
            self._stuck_monitor = None

        # ② Instruction freshness validation（时效性校验）
        infer_cfg = config.get("inference", {})
        self._last_screenshot_time = 0.0
        self._last_screenshot_gray: Optional[np.ndarray] = None
        self._stale_threshold = infer_cfg.get("stale_threshold_s", 1.5)
        self._stale_mse_threshold = infer_cfg.get("stale_mse_threshold", 500)

        # Track consecutive NO_CHANGE detections (force S2 if too many)
        self._no_change_count = 0
        self._no_change_force_s2_threshold = 5  # Force System 2 after 5 consecutive unchanged frames

        # Performance optimization: scene-level action caching
        self._scene_action_cache = {}  # Cache: scene → last_action
        self._cache_ttl = 2.0  # Cache valid for 2 seconds
        self._last_cache_time = 0.0

        # Force System 2 flag: when StuckMonitor triggers, bypass S1 next round
        self._force_system2 = False

        # Stuck-zone blacklist: reject S1 coords near recently stuck grid for N rounds
        self._stuck_blacklist: list = []  # [(gx, gy, remaining_rounds), ...]
        self._blacklist_rounds = 5
        self._blacklist_radius = 60  # pixels in screenshot space

        # Stuck coord-refinement: first trigger asks the model to re-locate instead
        # of blacklisting. Second trigger at the same grid cell falls back to blacklist.
        self._stuck_attempts: Dict[Tuple[int, int], int] = {}
        self._stuck_hint: Optional[Dict[str, Any]] = None

        # Target-description stuck detection: same target without screen progress
        self._recent_targets = deque(maxlen=3)
        self._same_target_threshold = 3

        # ===== 高阶能力：planner / memory / verifier / learning / model_router =====
        planner_cfg = config.get("planner", {}) or {}
        if planner_cfg.get("enabled", False):
            self._planner = Planner(planner_cfg.get("plan_path", "config/plans/endless_winter.yaml"))
        else:
            self._planner = None

        memory_cfg = config.get("memory", {}) or {}
        if memory_cfg.get("enabled", False):
            self._memory_store = MemoryStore(
                db_path=memory_cfg.get("db_path"),
                skill_min_success=int(memory_cfg.get("skill_min_success", 1)),
            )
        else:
            self._memory_store = None

        verifier_cfg = config.get("verifier", {}) or {}
        self._verifier = PostConditionVerifier() if verifier_cfg.get("enabled", False) else None
        self._persistent_bl_ttl = float(verifier_cfg.get("blacklist_ttl_s", 600))

        learning_cfg = config.get("learning", {}) or {}
        if learning_cfg.get("enabled", False):
            self._trajectory = TrajectoryLogger(out_dir=learning_cfg.get("out_dir"))
        else:
            self._trajectory = None

        self._model_family = (self._s1_cfg.get("model_family") or "qwen_vl").lower()
        self._current_scene: str = "unknown"
        self._round_prompt: Optional[str] = None
        self._round_response: Optional[str] = None
        self._round_system: Optional[str] = None

        # Research: load cached game guide hints (injected into S1 prompt)
        research_cfg = config.get("research", {}) or {}
        self._research_hints: str = ""
        if research_cfg.get("enabled", False):
            store = ResearchStore("endless_winter")
            budget = int(research_cfg.get("prompt_budget_chars", 600))
            self._research_hints = store.load_for_prompt(max_chars=budget)
            if self._research_hints:
                logger.info(f"[RESEARCH] loaded {len(self._research_hints)} chars of hints")
            else:
                logger.info("[RESEARCH] enabled but no cache; run scripts/fetch_research.py")

        logger.info(
            f"EndlessWinterGame initialized: system1={self._s1_cfg.get('enabled', True)}, "
            f"system2={self._s2_cfg.get('enabled', True)}, "
            f"change_detection={cd_enabled}"
        )

    def build_prompt(self, screenshot: Image) -> str:
        """Build System 1 prompt. Required by BaseGame but not used (overridden by analyze_with_retry)."""
        return build_system1_prompt(screenshot.width, screenshot.height)

    def parse_ai_response(self, response: str, screenshot=None) -> Dict[str, Any]:
        """Parse AI response. Consumes cached _last_parsed_action from analyze_with_retry."""
        # Handle sentinel strings returned by analyze_with_retry
        if response in ("NO_CHANGE", "SKIP", "S1_OK", "S2_OK"):
            if response == "NO_CHANGE":
                return {"action": "skip", "reason": "frame_unchanged"}
            if response == "SKIP":
                return {"action": "skip", "reason": "both_systems_failed"}
            # S1_OK and S2_OK: consume cached action
            if self._last_parsed_action is not None:
                action = self._last_parsed_action
                self._last_parsed_action = None
                return action

        # Fallback: if somehow called without cached action
        return {"action": "skip", "reason": "no_cached_action"}

    def execute_action(self, action: Dict[str, Any], screenshot=None) -> bool:
        """Execute action (click/drag/long_press).

        ② Action freshness validation happens here.
        """
        action_type = action.get("action")

        if action_type == "skip":
            return False

        # ② 时效性校验：检查指令是否过期
        if not self._action_is_fresh(screenshot):
            logger.warning("[STALE] Action expired, discarding")
            return False

        # ③ 多模态指令支持
        if action_type == "click":
            return self._do_click(action, screenshot)
        elif action_type == "drag":
            return self._do_drag(action, screenshot)
        elif action_type == "long_press":
            return self._do_long_press(action, screenshot)

        return False

    def analyze_with_retry(self, screenshot: Image, max_retries: int = 1) -> str:
        """Dual-system inference with scene awareness.

        Flow:
        1. Change detection (no change → NO_CHANGE, only if previous attempt succeeded)
        2. Scene classification → prompt adaptation
        3. Screenshot timestamp + gray copy (for ② freshness validation)
        4. System 1 (Qwen) → System 2 (Gemini) escalation
        """
        img_width = screenshot.width
        img_height = screenshot.height

        # Tick blacklist: decrement remaining rounds, drop expired entries
        self._tick_blacklist()
        if self._memory_store is not None:
            self._memory_store.purge_expired_blacklist()

        # Step 1: Change detection gate (skip only if last action was executed successfully)
        # For static UI states (Endless Winter), always retry if last action was skip/failed
        # But if too many consecutive NO_CHANGE, force analysis to break the loop
        if (self._change_detector is not None and self._last_parsed_action is not None and
            self._no_change_count < self._no_change_force_s2_threshold):
            last_action = self._last_parsed_action.get("action")
            # Only use cached result if last action was executed (click/drag/long_press)
            if last_action not in ("skip",):
                changed = self._change_detector.has_changed(screenshot)
                # Tag the previous action's outcome in history for S1 context
                if len(self._action_history) > 0 and "outcome" not in self._action_history[-1]:
                    self._action_history[-1]["outcome"] = "changed" if changed else "no_change"
                if not changed:
                    # Consume verifier expectation with no-change signal before short-circuit
                    self._consume_verification(scene_after=None, changed=False)
                    self._tick_planner(idle=True, screenshot=screenshot)
                    self._no_change_count += 1
                    logger.debug(f"⏩ Frame unchanged ({self._no_change_count}/{self._no_change_force_s2_threshold}), skipping inference")
                    return "NO_CHANGE"

        # If we reach here, reset NO_CHANGE counter (frame changed or force-analyzing)
        if self._no_change_count > 0:
            logger.debug(f"[FORCE] Breaking NO_CHANGE loop after {self._no_change_count} frames")
        self._no_change_count = 0

        # ① Step 2: Scene classification (for prompt adaptation)
        scene = "unknown"
        if self._scene_classifier is not None:
            scene = self._scene_classifier.get_scene(screenshot)
            logger.debug(f"[SCENE] current={scene}")
        self._current_scene = scene

        # Verify previous action's expectation (frame changed → we can judge now)
        self._consume_verification(scene_after=scene, changed=True)

        # Performance optimization: Check scene-level cache
        now = time.time()
        if scene in self._scene_action_cache and (now - self._last_cache_time) < self._cache_ttl:
            cached_action = self._scene_action_cache[scene]
            logger.debug(f"[CACHE] Using cached action for scene {scene}")
            self._last_parsed_action = cached_action
            # Reset NO_CHANGE counter since we have a valid action
            self._no_change_count = 0
            cached_type = cached_action.get("action")
            self._tick_planner(idle=(cached_type == "skip"), screenshot=screenshot)
            if cached_type == "skip":
                return "SKIP"
            elif cached_type == "click":
                return "S1_OK"
            else:
                return "S2_OK"

        # ② Step 3: Record screenshot timestamp & grayscale (for freshness validation)
        self._last_screenshot_time = time.time()
        try:
            gray = screenshot.convert("L")
            self._last_screenshot_gray = np.array(gray)
        except Exception:
            self._last_screenshot_gray = None

        # Step 4: System 1 (Qwen Ref mode, with scene-specific prompt)
        # Skip S1 if StuckMonitor forced S2 (S1 coords were inaccurate)
        if self._s1_cfg.get("enabled", True) and not self._force_system2:
            action = self._run_system1(screenshot, scene)
            if action is not None:
                # Note: target-repeat stuck detection moved to _do_click so
                # S1 + S2 share the same logic.
                self._last_parsed_action = action
                self._s1_fail_count = 0  # Reset failure counter
                # Performance: cache successful action
                now = time.time()
                self._last_cache_time = now
                self._scene_action_cache[scene] = action
                self._record_expectation(action, scene)
                self._tick_planner(idle=False, screenshot=screenshot)
                logger.debug(f"[S1] Success: {action}")
                return "S1_OK"

            self._s1_fail_count += 1
            logger.debug(f"[S1] Failed (count={self._s1_fail_count}/{self._fail_threshold})")

            # Check if we should escalate to System 2
            if self._s1_fail_count < self._fail_threshold:
                self._tick_planner(idle=True, screenshot=screenshot)
                return "SKIP"
        elif self._force_system2:
            logger.info("[FORCE_S2] Bypassing System 1 due to stuck detection")

        # Step 5: System 2 (Gemini JSON mode, supports drag) - escalation path
        if self._s2_cfg.get("enabled", True) and self._gemini_client is not None:
            now = time.time()
            if now - self._s2_last_call_time >= self._s2_min_interval:
                action = self._run_system2(screenshot)
                self._s2_last_call_time = now
                if action is not None:
                    self._last_parsed_action = action
                    self._s1_fail_count = 0  # Reset failure counter
                    self._force_system2 = False  # Reset force flag
                    # Performance: cache successful action
                    self._last_cache_time = now
                    self._scene_action_cache[scene] = action
                    self._record_expectation(action, scene)
                    self._tick_planner(idle=False, screenshot=screenshot)
                    logger.debug(f"[S2] Success: {action}")
                    return "S2_OK"
                logger.debug("[S2] Failed to find action")
            else:
                logger.debug(f"[S2] Rate limited (wait {self._s2_min_interval}s)")

        # Cache SKIP result
        if self._last_parsed_action is not None:
            now = time.time()
            self._last_cache_time = now
            self._scene_action_cache[scene] = self._last_parsed_action

        self._tick_planner(idle=True, screenshot=screenshot)
        return "SKIP"

    def _run_system1(self, screenshot: Image, scene: str = "unknown") -> Optional[Dict[str, Any]]:
        """Run System 1 (Qwen Ref mode) with scene-specific prompts.

        Args:
            screenshot: Current screenshot
            scene: Current scene type from SceneClassifier

        Returns:
            Action dict with action="click"|"drag"|"long_press", or None on failure
        """
        # Route to non-qwen family if configured (UI-TARS / CogAgent)
        if self._model_family != "qwen_vl":
            return self._run_system1_routed(screenshot, scene)

        try:
            # ① Use scene-specific prompt
            if scene == "loading":
                # Loading scene: no action
                logger.debug("[S1] Scene is loading, skipping")
                return None

            recent_actions = [
                {
                    "target": a.get("target", "unknown"),
                    "x": a.get("x", 0),
                    "y": a.get("y", 0),
                    "outcome": a.get("outcome", "unknown"),
                }
                for a in list(self._action_history)[-3:]
            ]
            prompt = build_system1_prompt_for_scene(
                screenshot.width, screenshot.height, scene, recent_actions=recent_actions,
            )
            # Planner-driven subgoal biasing + research hints (research first, planner closer to prompt body)
            preface_parts = []
            if self._research_hints:
                preface_parts.append(f"【游戏攻略参考】\n{self._research_hints}")
            if self._planner is not None:
                ctx = self._planner.active_task_context()
                if ctx:
                    preface_parts.append(ctx)
            if preface_parts:
                prompt = "\n\n".join(preface_parts) + "\n\n" + prompt
            # Use adaptive retries: more retries on consecutive failures
            max_retries = 1 if self._s1_fail_count < 2 else 0  # Skip retry on repeated failures
            response = self._qwen_client.analyze(screenshot, prompt, max_retries=max_retries)
            self._round_prompt = prompt
            self._round_response = response
            self._round_system = "s1"

            # Log raw response for debugging
            logger.debug(f"[S1] Raw response: {response[:300]}")

            # Try to parse Qwen Ref bounding box
            coords = parse_qwen_ref_bbox(response, screenshot.width, screenshot.height)
            if coords is None:
                logger.warning(f"[S1] Failed to parse bbox. Response was: {response[:500]}")
                # Check if Qwen said "无可点击元素" (no clickable elements found)
                if "无可点击元素" in response:
                    logger.debug("[S1] Qwen returned: no clickable elements found")
                return None

            x, y = coords

            # Reject if coords fall within blacklisted stuck zones
            if self._is_blacklisted(x, y):
                logger.warning(f"[S1] Coords ({x},{y}) in blacklisted stuck zone, rejecting")
                return None

            # Extract target description from response (e.g. "目标=xxx")
            target_desc = "UI element (Qwen Ref)"
            import re as _re
            desc_match = _re.search(r'目标\s*[=:：]\s*([^\n]+)', response)
            if desc_match:
                target_desc = desc_match.group(1).strip()[:80]
                logger.info(f"[S1] Target: {target_desc}")

            # Parse action type from response (CLICK/DRAG/LONG_PRESS)
            action_type = "click"  # Default
            if "LONG_PRESS" in response:
                action_type = "long_press"
            elif "DRAG" in response:
                action_type = "drag"
            elif "CLICK" in response:
                action_type = "click"

            logger.info(f"[S1] Detected {action_type} at ({x}, {y})")

            # Handle different action types
            if action_type == "click":
                return {
                    "action": "click",
                    "x": x,
                    "y": y,
                    "target": target_desc,
                    "confidence": 0.85,
                    "reasoning": "System 1 detected interactive element",
                }

            elif action_type == "long_press":
                return {
                    "action": "long_press",
                    "x": x,
                    "y": y,
                    "duration": 1.5,  # Default 1.5 seconds
                    "target": "Long-press target (Qwen Ref)",
                    "confidence": 0.8,
                    "reasoning": "System 1 detected long-press target",
                }

            elif action_type == "drag":
                # For drag, use detected point as center and create a default drag motion
                # (upward drag by default, or infer from scene context)
                return {
                    "action": "drag",
                    "x1": x,
                    "y1": y,
                    "x2": x,  # Vertical drag (same x)
                    "y2": max(0, y - 150),  # Drag upward 150 pixels
                    "duration": 0.6,
                    "target": "Map drag (Qwen Ref)",
                    "confidence": 0.8,
                    "reasoning": "System 1 detected drag area",
                }

            return None

        except Exception as e:
            logger.debug(f"[S1] Exception: {e}")
            return None

    def _run_system2(self, screenshot: Image) -> Optional[Dict[str, Any]]:
        """Run System 2 (Gemini JSON mode) with ③ DRAG/LONG_PRESS support.

        Returns:
            Action dict from Gemini JSON response, or None on failure
        """
        try:
            # Serialize action history for context
            history = [
                {
                    "target": a.get("target", "unknown"),
                    "x": a.get("x", 0),
                    "y": a.get("y", 0),
                }
                for a in list(self._action_history)
            ]

            # ③ Use prompt that supports drag/long_press
            prompt = build_system2_prompt_with_drag(screenshot.width, screenshot.height, history)
            if self._stuck_hint is not None:
                hint = self._stuck_hint
                same_target_coords = [
                    (int(a["x"]), int(a["y"]))
                    for a in list(self._action_history)
                    if a.get("target") == hint["target"]
                    and isinstance(a.get("x"), (int, float))
                    and isinstance(a.get("y"), (int, float))
                ]
                cur = (int(hint["x"]), int(hint["y"]))
                if cur not in same_target_coords:
                    same_target_coords.append(cur)
                radius = self._blacklist_radius
                min_offset = radius + 20
                coords_str = "\n".join(f"  - ({cx}, {cy})" for cx, cy in same_target_coords)
                prompt = (
                    f"【上轮同一目标连续点击失败 — 必须换位置】\n"
                    f"已在以下像素反复尝试点击 \"{hint['target']}\" 均未生效"
                    f"（系统已屏蔽半径 {radius}px 范围）：\n"
                    f"{coords_str}\n"
                    f"规则：新坐标若落在任一上述点的 {radius}px 范围内会被系统**直接拒绝**。\n"
                    f"- 若你依然认为目标是 \"{hint['target']}\"，"
                    f"新的 x 或 y 必须距上述任一点 ≥ {min_offset} 像素；否则请换目标。\n"
                    f"- 若画面里根本没有真正的 \"{hint['target']}\"，"
                    f"请改为点击其它可操作元素（右上角 X、左上角 ←、其它按钮）。\n\n"
                    + prompt
                )
                logger.info(
                    f"[STUCK] injecting coord-refinement hint: '{hint['target']}' "
                    f"blocked at {same_target_coords}"
                )
                self._stuck_hint = None
            response = self._gemini_client.analyze(screenshot, prompt, max_retries=1)
            self._round_prompt = prompt
            self._round_response = response
            self._round_system = "s2"

            # Try to parse Gemini JSON response
            action = parse_gemini_json(response)
            if action is None:
                logger.debug(f"[S2] Failed to parse JSON: {response[:200]}")
                return None

            action_type = action.get("action")
            if action_type not in ["click", "drag", "long_press", "skip"]:
                logger.debug(f"[S2] Unknown action type: {action_type}")
                return None

            if action_type == "skip":
                logger.debug("[S2] Skipping based on Gemini decision")
                return None

            # ③ Handle drag action
            if action_type == "drag":
                x1 = action.get("x1", 0)
                y1 = action.get("y1", 0)
                x2 = action.get("x2", 0)
                y2 = action.get("y2", 0)
                if not (0 <= x1 <= screenshot.width and 0 <= y1 <= screenshot.height and
                       0 <= x2 <= screenshot.width and 0 <= y2 <= screenshot.height):
                    logger.debug(f"[S2] Drag coordinates out of range")
                    return None
                logger.info(f"[S2] Drag action: ({x1},{y1}) → ({x2},{y2})")
                return {
                    "action": "drag",
                    "x1": int(x1), "y1": int(y1),
                    "x2": int(x2), "y2": int(y2),
                    "duration": float(action.get("duration", 0.5)),
                    "target": action.get("target", "drag_operation"),
                    "confidence": float(action.get("confidence", 0.7)),
                    "reasoning": action.get("reasoning", ""),
                }

            # ③ Handle long_press action
            if action_type == "long_press":
                x = action.get("x", 0)
                y = action.get("y", 0)
                if not (0 <= x <= screenshot.width and 0 <= y <= screenshot.height):
                    logger.debug(f"[S2] Long_press coordinates out of range")
                    return None
                logger.info(f"[S2] Long_press action at ({x}, {y})")
                return {
                    "action": "long_press",
                    "x": int(x),
                    "y": int(y),
                    "duration": float(action.get("duration", 2.0)),
                    "target": action.get("target", "long_press_element"),
                    "confidence": float(action.get("confidence", 0.7)),
                    "reasoning": action.get("reasoning", ""),
                }

            # ③ Handle click action
            x = action.get("x", 0)
            y = action.get("y", 0)
            confidence = action.get("confidence", 0.7)
            target = action.get("target", "strategic_element")

            # Only snap for the specific bottom-bar mail-panel buttons that
            # Gemini reliably mis-locates. '前往' / '立即' etc. appear in many
            # non-bottom contexts (e.g. inline prerequisite links) — do NOT snap them.
            bottom_bar_kw = ("一键已读", "删除所有已读")
            if any(k in target for k in bottom_bar_kw) and y < screenshot.height * 0.85:
                new_y = int(screenshot.height * 0.955)
                logger.info(f"[S2-SNAP] bottom-bar target '{target}', y {y} → {new_y}")
                y = new_y

            # Close/back buttons are in top corners; snap if Gemini put them mid-screen.
            close_kw = ("关闭", "取消", "返回", "X 按钮", "X按钮", "✕")
            if any(k in target for k in close_kw):
                in_top_band = y < screenshot.height * 0.15
                if not in_top_band:
                    # Snap to top-right corner (most common close-X position)
                    new_x = int(screenshot.width * 0.93)
                    new_y = int(screenshot.height * 0.07)
                    logger.info(f"[S2-SNAP] close-btn target '{target}', ({x},{y}) → ({new_x},{new_y})")
                    x, y = new_x, new_y

            # Validate coordinates
            if not (0 <= x <= screenshot.width and 0 <= y <= screenshot.height):
                logger.debug(f"[S2] Coordinates out of range: ({x}, {y})")
                return None

            logger.info(f"[S2] Strategic decision at ({x}, {y}): {target}")

            return {
                "action": "click",
                "x": int(x),
                "y": int(y),
                "target": target,
                "confidence": float(confidence),
                "reasoning": action.get("reasoning", "System 2 strategic analysis"),
            }

        except Exception as e:
            logger.debug(f"[S2] Exception: {e}")
            return None

    def _tick_planner(self, idle: bool, screenshot: Optional[Image] = None) -> None:
        """Unified planner tick: C completion_check (if set) preempts B stability."""
        if self._planner is None:
            return
        active = self._planner.active_task()
        if active is None:
            return
        ctx = CheckContext(
            screenshot=screenshot,
            scene=self._current_scene,
            active_task=active,
            memory_store=self._memory_store,
            qwen_client=self._qwen_client,
        )
        self._planner.tick(ctx, idle=idle)

    def _record_expectation(self, action: Dict[str, Any], scene: str) -> None:
        """Record post-condition expectation for the next verify cycle."""
        if self._verifier is None:
            return
        expected_scene: Optional[str] = None
        if self._planner is not None:
            active = self._planner.active_task()
            if active is not None:
                expected_scene = active.expected_scene
        self._verifier.record(
            action=action,
            expected_scene=expected_scene,
            scene_before=scene,
            expected_change=True,
        )

    def _consume_verification(self, scene_after: Optional[str], changed: bool) -> None:
        """Verify pending expectation and fan out to memory / planner / trajectory."""
        if self._stuck_monitor is not None:
            self._stuck_monitor.record_verify(changed)
        if self._verifier is None or not self._verifier.has_pending():
            return
        result = self._verifier.verify(scene_after, changed)
        action = result.action or {}
        scene = result.scene_before or self._current_scene or "unknown"
        target = action.get("target", "") or ""
        x = action.get("x")
        y = action.get("y")
        if result.reason == "no_visual_change":
            outcome = "no_change"
        elif result.ok and result.scene_matched:
            outcome = "success"
        elif result.ok and result.changed and result.expected_scene and not result.scene_matched:
            outcome = "scene_drift"
        elif result.ok and result.changed:
            outcome = "changed"
        elif result.ok:
            outcome = "ok"
        else:
            outcome = "rejected"

        if self._memory_store is not None:
            self._memory_store.record_action(
                scene=scene,
                target=target,
                action_type=action.get("action", "click"),
                x=int(x) if isinstance(x, (int, float)) else None,
                y=int(y) if isinstance(y, (int, float)) else None,
                outcome=outcome,
                notes=result.reason,
            )
            if outcome in ("no_change", "rejected") and x is not None and y is not None:
                self._memory_store.add_blacklist(
                    scene=scene,
                    x=int(x),
                    y=int(y),
                    radius=self._blacklist_radius,
                    reason=result.reason,
                    ttl_s=self._persistent_bl_ttl,
                )

        # B: 不再以"关键字命中"即宣告成功。成功靠 tick_stability 的"连续稳定"判定。
        # 这里只处理显式失败。
        if self._planner is not None:
            active = self._planner.active_task()
            if active is not None and outcome in ("no_change", "rejected", "scene_drift"):
                self._planner.mark_failure(active, reason=result.reason)

        if self._trajectory is not None:
            self._trajectory.log(
                scene=scene,
                model_family=self._model_family,
                prompt=self._round_prompt or "",
                response=self._round_response or "",
                parsed_action=action,
                executed=result.ok,
                outcome=outcome,
                verify_reason=result.reason,
            )
        # Emit verify event to the dashboard (no-op when disabled)
        try:
            from utils.dashboard_bus import bus as _dash_bus
            if _dash_bus.is_enabled():
                _dash_bus.emit("verify", {
                    "round": getattr(self, "_current_round", None),
                    "ok": bool(result.ok),
                    "reason": result.reason,
                    "scene_matched": bool(getattr(result, "scene_matched", False)),
                    "scene_after": scene,
                    "changed": bool(changed),
                    "outcome": outcome,
                })
        except Exception:
            pass

        self._round_prompt = None
        self._round_response = None
        self._round_system = None

    def _run_system1_routed(self, screenshot: Image, scene: str) -> Optional[Dict[str, Any]]:
        """System 1 via pluggable family (ui_tars / cogagent / qwen_vl via router)."""
        try:
            if scene == "loading":
                return None
            recent_actions = [
                {
                    "target": a.get("target", "unknown"),
                    "x": a.get("x", 0),
                    "y": a.get("y", 0),
                    "outcome": a.get("outcome", "unknown"),
                }
                for a in list(self._action_history)[-3:]
            ]
            preface = []
            if self._research_hints:
                preface.append(f"【游戏攻略参考】\n{self._research_hints}")
            if self._planner is not None:
                pc = self._planner.active_task_context()
                if pc:
                    preface.append(pc)
            task_ctx = "\n\n".join(preface)
            prompt = build_grounding_prompt(
                family=self._model_family,
                img_width=screenshot.width,
                img_height=screenshot.height,
                scene=scene,
                recent_actions=recent_actions,
                task_context=task_ctx,
                qwen_prompt_builder=build_system1_prompt_for_scene,
            )
            max_retries = 1 if self._s1_fail_count < 2 else 0
            response = self._qwen_client.analyze(screenshot, prompt, max_retries=max_retries)
            self._round_prompt = prompt
            self._round_response = response
            self._round_system = "s1"
            parsed = parse_grounding_response(
                family=self._model_family,
                response=response,
                img_width=screenshot.width,
                img_height=screenshot.height,
                qwen_parser=parse_qwen_ref_bbox,
            )
            if parsed is None:
                return None
            x, y = parsed["x"], parsed["y"]
            if self._is_blacklisted(x, y):
                logger.warning(f"[S1-{self._model_family}] ({x},{y}) blacklisted, rejecting")
                return None
            action_type = parsed.get("action_type", "click")
            target = parsed.get("target", f"UI element ({self._model_family})")
            if action_type == "click":
                return {
                    "action": "click", "x": x, "y": y, "target": target,
                    "confidence": 0.8, "reasoning": f"System 1 [{self._model_family}]",
                }
            if action_type == "long_press":
                return {
                    "action": "long_press", "x": x, "y": y, "duration": 1.5,
                    "target": target, "confidence": 0.8,
                    "reasoning": f"System 1 [{self._model_family}]",
                }
            if action_type == "drag":
                return {
                    "action": "drag",
                    "x1": parsed.get("x1", x), "y1": parsed.get("y1", y),
                    "x2": parsed.get("x2", x), "y2": parsed.get("y2", max(0, y - 150)),
                    "duration": 0.6, "target": target, "confidence": 0.8,
                    "reasoning": f"System 1 [{self._model_family}]",
                }
            return None
        except Exception as e:
            logger.debug(f"[S1-{self._model_family}] exception: {e}")
            return None

    def _do_click(self, action: Dict[str, Any], screenshot: Optional[Image] = None) -> bool:
        """Execute click action with snap-to-UI and coordinate translation.

        ④ Integrates StuckMonitor.

        Args:
            action: Action dict with x, y coordinates (in screenshot space)
            screenshot: Screenshot for snap-to-UI refinement

        Returns:
            True if click executed, False otherwise
        """
        try:
            x, y = action.get("x", 0), action.get("y", 0)
            target = action.get("target", "unknown")

            # Target-repeat stuck detection — fires for BOTH S1 and S2 paths.
            # Historical bug: _recent_targets was only tracked in _run_system1, so
            # R9/R11/R15/R19 all pointing "建造按钮" via S2 never triggered, even
            # though the game clearly wasn't progressing. Now tracked here so any
            # click path participates. changed=true is an unreliable "progressed"
            # signal (animations/popups cause noise); same target N times is not.
            if target and target != "unknown":
                self._recent_targets.append(target)
            recent_targets_list = list(self._recent_targets)
            if (len(recent_targets_list) >= self._same_target_threshold
                    and len(set(recent_targets_list)) == 1):
                logger.warning(
                    f"[TARGET_STUCK] '{target}' repeated "
                    f"{len(recent_targets_list)}x without progress — "
                    f"requesting re-location, temp-blacklisting ({x},{y})"
                )
                self._stuck_hint = {"target": target, "x": int(x), "y": int(y)}
                self._stuck_blacklist.append((int(x), int(y), self._blacklist_rounds))
                self._recent_targets.clear()
                self._force_system2 = True
                self._scene_action_cache.clear()
                return False

            # Main-city zone guard: AI often misidentifies the round avatar (top-left)
            # as "返回箭头" and the mail envelope (bottom-right) as "地图图标/野外".
            # Both are at accurate pixels for what AI *thinks* they are, so CV-SNAP
            # can't fix it — reject based on scene + position instead.
            if screenshot is not None and self._scene_classifier is not None:
                try:
                    scene = self._scene_classifier.get_scene(screenshot)
                except Exception:
                    scene = "unknown"
                if scene == "main_city":
                    w_img, h_img = screenshot.size
                    if x < w_img * 0.22 and y < h_img * 0.12:
                        logger.warning(f"[GUARD] 主城左上角是头像，拒绝点 '{target}' ({x},{y})")
                        return False
                    if x > w_img * 0.83 and h_img * 0.80 < y < h_img * 0.92:
                        logger.warning(f"[GUARD] 主城右下角是邮件，拒绝点 '{target}' ({x},{y})")
                        return False
                # Persistent blacklist from prior-session failures
                if self._memory_store is not None:
                    reason = self._memory_store.is_blacklisted(scene, x, y)
                    if reason is not None:
                        logger.warning(f"[MEM-BL] '{target}' ({x},{y}) in persistent blacklist: {reason}")
                        return False

            # In-memory stuck-zone blacklist (set by StuckMonitor in previous rounds).
            # Must run for every action path — S2 results skip _run_system1's own check.
            if self._is_blacklisted(x, y):
                logger.warning(f"[STUCK-BL] '{target}' ({x},{y}) in in-memory stuck blacklist, rejecting")
                return False

            # Snap to nearest colored-button blob if screenshot provided.
            if screenshot is not None:
                orig = (x, y)

                # Generic OCR override: extract Chinese keywords from `target`,
                # match any OCR-detected text on screen, use its bbox center as
                # the click coord. Removes the need for a hand-maintained button
                # whitelist — any text button the VLM labels ends up covered.
                hit = self._ocr_snap(screenshot, target, x, y)
                if hit is not None:
                    new_x, new_y = int(hit["cx"]), int(hit["cy"])
                    logger.info(
                        f"[OCR-SNAP] '{target}' {orig} → ({new_x},{new_y}) "
                        f"matched='{hit['text']}' conf={hit['confidence']:.2f} "
                        f"dist={hit.get('dist_to_hint', 0):.0f}"
                    )
                    x, y = new_x, new_y
                    action["ocr_override"] = True

                did_ocr_override = action.get("ocr_override", False)

                # '前往' is a narrow blue button always on the right side of a prerequisite
                # row. AI typically returns the row/section center with y 40-70px off —
                # use a dedicated wider-y-band rightmost-button search instead.
                if did_ocr_override:
                    pass  # OCR coord is authoritative; no further snap
                elif "前往" in target:
                    snapped = self._snap_to_qianwang(screenshot, x, y)
                    if snapped is not None:
                        x, y = snapped
                        logger.info(f"[CV-SNAP-前往] '{target}' {orig} → ({x},{y})")
                elif ("任务" in target or "任务栏" in target or "任务提示" in target) and "邮件" not in target:
                    snapped = self._snap_to_task_bar(screenshot, x, y)
                    if snapped is not None:
                        x, y = snapped
                        logger.info(f"[CV-SNAP-任务栏] '{target}' {orig} → ({x},{y})")
                else:
                    # 默认信 AI 坐标:OCR 都没命中说明要么是图标无文字,要么 OCR 漏检,
                    # CV-SNAP 历史上常把 Y 推错方向(525→504),收益 < 风险,改为 opt-in。
                    # 只在 target 明确指向 icon/图标 时才做颜色 blob snap。
                    use_cv_snap = any(k in target for k in ("图标", "头像", "箭头", "图案", "icon"))
                    if not use_cv_snap:
                        logger.debug(f"[CV-SNAP-SKIP] '{target}' trust AI coord {orig}")
                    else:
                        snapped = self._snap_to_nearest_button(screenshot, x, y, radius=40)
                        if snapped is not None:
                            x, y = snapped
                            if (x, y) != orig:
                                logger.info(f"[CV-SNAP] '{target}' {orig} → ({x},{y}) (button blob, r=40)")

            # Get fresh window info (window may have moved since last click)
            self._window_info = self.screen_capture.find_wechat_window()
            if self._window_info is None:
                logger.error("Could not find WeChat window")
                return False

            screen_x, screen_y = self._to_screen_coords(x, y, screenshot)
            # Publish post-snap screenshot-space coords so the dashboard can overlay
            # the actual click location (vs the model's original target).
            action["executed_x"] = int(x)
            action["executed_y"] = int(y)
            logger.info(f"💬 Clicking {target} at screen ({screen_x}, {screen_y})")

            # Ensure WeChat is frontmost so the click reaches it (background apps swallow clicks)
            self.screen_capture.activate_wechat()
            time.sleep(0.05)

            # Execute click
            self.input_controller.click(int(screen_x), int(screen_y), delay=self._timing.get("click_delay", 0.1))

            # Record in action history
            self._action_history.append({
                "target": target,
                "x": x,
                "y": y,
                "time": time.time(),
            })

            # ④ Record in StuckMonitor and check for stuck state
            if self._stuck_monitor is not None:
                self._stuck_monitor.record_click(x, y)
                if self._stuck_monitor.is_stuck():
                    stuck_pos = self._stuck_monitor.last_stuck_position()
                    # Force System 2 next round (S1 coords were likely inaccurate)
                    self._force_system2 = True
                    self._scene_action_cache.clear()  # Clear stale cache

                    attempts = 0
                    if stuck_pos is not None:
                        attempts = self._stuck_attempts.get(stuck_pos, 0) + 1
                        self._stuck_attempts[stuck_pos] = attempts

                    # First trigger: don't blacklist — ask the model to re-locate
                    # the target on the next round via an injected hint.
                    if attempts <= 1:
                        self._stuck_hint = {
                            "target": target,
                            "x": int(x),
                            "y": int(y),
                        }
                        logger.info(
                            f"[STUCK] coord refinement requested for '{target}' at {stuck_pos} "
                            f"(attempt {attempts}) — next round will re-locate"
                        )
                        # Don't run recover() here — the game isn't actually stuck yet,
                        # we just want a better coord next round.
                        return False

                    # Second+ trigger at same cell: true stuck, blacklist and recover.
                    self._stuck_monitor.recover(
                        self.input_controller,
                        self._scene_classifier,
                        window_info=self._window_info,
                        top_offset=self.top_offset,
                    )
                    self._stuck_blacklist.append((stuck_pos[0], stuck_pos[1], self._blacklist_rounds))
                    logger.info(
                        f"[STUCK] Blacklisted zone {stuck_pos} for {self._blacklist_rounds} rounds "
                        f"(attempt {attempts} exceeded refinement budget)"
                    )
                    return False  # Skip this action, recovery occurred

            return True

        except Exception as e:
            logger.error(f"Failed to execute click: {e}")
            return False

    def _find_button_below(self, screenshot: Image, x: int, y: int, search_h: int = 300) -> Optional[Tuple[int, int]]:
        """Find the largest saturated/bright UI button region at/below (x,y).

        Buttons in Endless Winter tutorials have distinct colors (cyan, green, blue).
        Scan a vertical strip centered on x, from y downward, find the largest
        high-saturation contiguous region, return its centroid.

        Returns:
            (cx, cy) in screenshot space, or None if no button-like region found
        """
        try:
            img = np.array(screenshot)
            h, w = img.shape[:2]

            # Vertical search strip: full width, from y to y+search_h
            y_end = min(h, y + search_h)
            y_start = max(0, y - 30)
            if y_end <= y_start:
                return None

            strip = img[y_start:y_end]

            # Convert to HSV; look for high-saturation regions (buttons vs map/terrain)
            hsv = cv2.cvtColor(strip, cv2.COLOR_RGB2HSV)
            saturation = hsv[:, :, 1]
            value = hsv[:, :, 2]
            # Bright AND saturated mask (tutorial buttons)
            mask = ((saturation > 80) & (value > 120)).astype(np.uint8) * 255

            # Morphological close to merge button fragments
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None

            # Pick largest contour with min area (avoid noise)
            best = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(best)
            if area < 300:
                return None

            M = cv2.moments(best)
            if M["m00"] == 0:
                return None
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"]) + y_start

            return (cx, cy)

        except Exception as e:
            logger.debug(f"Button detection failed: {e}")
            return None

    def _snap_to_qianwang(
        self, screenshot: Image, x: int, y: int, y_band: int = 80,
    ) -> Optional[Tuple[int, int]]:
        """Find the '前往' button: small blue rectangle on the right side of a row.

        Searches the horizontal band [y - y_band, y + y_band] across the full
        image width, keeps blue blobs with area 300-1500 and aspect 2.0-5.0
        (typical 前往 shape ~66x16). Picks the rightmost qualifying blob.
        Returns centroid or None.
        """
        try:
            arr = np.array(screenshot)
            if arr.shape[-1] == 4:
                arr = arr[:, :, :3]
            h, w = arr.shape[:2]
            y1, y2 = max(0, y - y_band), min(h, y + y_band)
            if y1 >= y2:
                return None
            strip = arr[y1:y2]
            hsv = cv2.cvtColor(strip, cv2.COLOR_RGB2HSV)
            hue = hsv[:, :, 0]
            sat = hsv[:, :, 1]
            val = hsv[:, :, 2]
            blue = ((hue >= 95) & (hue <= 130) & (sat >= 90) & (val >= 110)).astype(np.uint8) * 255
            nlabels, _, stats, centroids = cv2.connectedComponentsWithStats(blue, 8)
            if nlabels <= 1:
                return None
            best = None
            best_x = -1
            for i in range(1, nlabels):
                area = int(stats[i, cv2.CC_STAT_AREA])
                if area < 300 or area > 1500:
                    continue
                bw = int(stats[i, cv2.CC_STAT_WIDTH])
                bh = int(stats[i, cv2.CC_STAT_HEIGHT])
                if bh == 0:
                    continue
                ar = bw / bh
                if ar < 2.0 or ar > 5.0:
                    continue
                cx_i, cy_i = centroids[i]
                # Must be on right half of image (where 前往 always lives)
                if cx_i < w * 0.55:
                    continue
                if cx_i > best_x:
                    best_x = cx_i
                    best = (int(cx_i), int(cy_i) + y1, area, bw, bh)
            if best is None:
                return None
            bx, by, area, bw, bh = best
            logger.debug(f"[前往] blob area={area} {bw}x{bh} at ({bx},{by})")
            return (bx, by)
        except Exception as e:
            logger.debug(f"[前往-snap] failed: {e}")
            return None

    def _snap_to_task_bar(
        self, screenshot: Image, x: int, y: int,
    ) -> Optional[Tuple[int, int]]:
        """Find the bottom-left task bar: wide blue rectangle with task text.

        Qwen often frames the entire bottom row (task bar + mail icon) and returns
        the center, landing near the right-side mail icon. The task bar is a much
        wider blob (width > 80, aspect > 1.8) on the left half of the image,
        while the mail icon is ~35x35 on the right.

        Searches y ∈ [0.75*h, h], picks the widest blue blob with center x < 0.7*w.
        """
        try:
            arr = np.array(screenshot)
            if arr.shape[-1] == 4:
                arr = arr[:, :, :3]
            h, w = arr.shape[:2]
            y1 = int(h * 0.78)
            y2 = int(h * 0.87)
            if y1 >= y2:
                return None
            strip = arr[y1:y2]
            hsv = cv2.cvtColor(strip, cv2.COLOR_RGB2HSV)
            hue = hsv[:, :, 0]
            sat = hsv[:, :, 1]
            val = hsv[:, :, 2]
            blue = ((hue >= 95) & (hue <= 130) & (sat >= 90) & (val >= 90)).astype(np.uint8) * 255
            nlabels, _, stats, centroids = cv2.connectedComponentsWithStats(blue, 8)
            if nlabels <= 1:
                return None
            best = None
            best_w = -1
            for i in range(1, nlabels):
                area = int(stats[i, cv2.CC_STAT_AREA])
                if area < 1500 or area > 12000:
                    continue
                bw = int(stats[i, cv2.CC_STAT_WIDTH])
                bh = int(stats[i, cv2.CC_STAT_HEIGHT])
                if bh == 0:
                    continue
                ar = bw / bh
                if ar < 1.8:
                    continue
                cx_i, cy_i = centroids[i]
                if cx_i >= w * 0.7:
                    continue
                if bw > best_w:
                    best_w = bw
                    best = (int(cx_i), int(cy_i) + y1, area, bw, bh)
            if best is None:
                return None
            bx, by, area, bw, bh = best
            logger.debug(f"[任务栏] blob area={area} {bw}x{bh} at ({bx},{by})")
            return (bx, by)
        except Exception as e:
            logger.debug(f"[任务栏-snap] failed: {e}")
            return None

    # Non-keyword stopwords stripped from `target` before OCR matching.
    _OCR_STOPWORDS = (
        "按钮", "图标", "标识", "入口", "选项", "区域", "的", "和", "与",
        "点击", "可点击", "界面", "元素", "菜单", "面板",
    )

    def _ocr_snap(
        self, screenshot: Image, target: str, hint_x: int, hint_y: int,
        min_confidence: float = 0.3, max_dist: int = 400,
    ) -> Optional[Dict[str, Any]]:
        """Generic OCR coord override: extract keywords from `target` and match
        against on-screen text. Returns best OCR hit or None.

        No hand-maintained whitelist — any text button the VLM names gets a shot
        at being OCR-corrected. Strips common UI suffixes like "按钮/图标/的" so
        "捕猎工位升级按钮" → tries ["捕猎工位升级", "捕猎工位", "升级"] in turn.
        """
        if not target:
            return None
        try:
            from core.text_finder import find_nearest
        except Exception as e:
            logger.debug(f"[OCR-SNAP] import failed: {e}")
            return None

        import re as _re
        # Strip punctuation/quotes that the VLM likes adding around target names.
        clean = _re.sub(r'[\"“”‘’\'（）()\[\]【】<>《》]', '', target).strip()
        for sw in self._OCR_STOPWORDS:
            clean = clean.replace(sw, " ")
        # Candidates: full cleaned string, then each ≥2-char segment.
        cands = []
        if clean.strip():
            cands.append(clean.strip())
        for seg in _re.split(r'[\s，、,/]+', clean):
            seg = seg.strip()
            if len(seg) >= 2 and seg not in cands:
                cands.append(seg)
        if not cands:
            return None

        for kw in cands:
            hit = find_nearest(
                screenshot, keywords=[kw], hint_x=hint_x, hint_y=hint_y,
                min_confidence=min_confidence, max_dist=max_dist,
            )
            if hit is not None:
                return hit
        logger.debug(f"[OCR-SNAP] '{target}' no match for candidates {cands}")
        return None

    def _snap_to_nearest_button(
        self, screenshot: Image, x: int, y: int, radius: int = 40,
        min_area: int = 400, max_area: int = 7000, max_snap_dist: int = 25,
    ) -> Optional[Tuple[int, int]]:
        """Conservative snap — only corrects small AI coord errors.

        Searches high-saturation blobs within `radius`, keeps only button-shaped
        candidates (area 400-3000, aspect 1.2-5.0), requires centroid within
        `max_snap_dist` pixels of AI point. Returns None if no confident match —
        better to trust AI than snap to the wrong blob.
        """
        try:
            arr = np.array(screenshot)
            if arr.shape[-1] == 4:
                arr = arr[:, :, :3]
            h, w = arr.shape[:2]

            x1, x2 = max(0, x - radius), min(w, x + radius)
            y1, y2 = max(0, y - radius), min(h, y + radius)
            if x1 >= x2 or y1 >= y2:
                return None

            crop = arr[y1:y2, x1:x2]
            hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
            mask = ((hsv[:, :, 1] >= 80) & (hsv[:, :, 2] >= 110)).astype(np.uint8) * 255

            nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
            if nlabels <= 1:
                return None

            cx_local, cy_local = x - x1, y - y1
            best = None
            best_d = float("inf")
            for i in range(1, nlabels):
                area = int(stats[i, cv2.CC_STAT_AREA])
                if area < min_area or area > max_area:
                    continue
                bw = int(stats[i, cv2.CC_STAT_WIDTH])
                bh = int(stats[i, cv2.CC_STAT_HEIGHT])
                if bh == 0:
                    continue
                ar = bw / bh
                if ar < 1.2 or ar > 5.0:
                    continue
                cx_i, cy_i = centroids[i]
                d = ((cx_i - cx_local) ** 2 + (cy_i - cy_local) ** 2) ** 0.5
                if d > max_snap_dist:
                    continue
                if d < best_d:
                    best_d = d
                    best = (int(cx_i), int(cy_i), area)

            if best is None:
                return None
            bx, by, area = best
            logger.debug(f"[CV-SNAP] button area={area} dist={best_d:.1f}")
            return (x1 + bx, y1 + by)
        except Exception as e:
            logger.debug(f"[CV-SNAP] failed: {e}")
            return None

    def _snap_to_ui_boundary(self, screenshot: Image, x: int, y: int, radius: int = 10) -> Tuple[int, int]:
        """Snap (x, y) to nearest UI edge using Canny edge detection.

        Args:
            screenshot: PIL Image
            x: Target x coordinate
            y: Target y coordinate
            radius: Search radius in pixels

        Returns:
            Snapped (x, y) coordinates, or original if no edge found
        """
        try:
            # Crop region around target
            img_array = np.array(screenshot)
            h, w = img_array.shape[:2]

            x1 = max(0, x - radius)
            x2 = min(w, x + radius)
            y1 = max(0, y - radius)
            y2 = min(h, y + radius)

            if x1 >= x2 or y1 >= y2:
                return (x, y)

            crop = img_array[y1:y2, x1:x2]

            # Convert to grayscale and apply Canny edge detection
            if len(crop.shape) == 3:
                gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
            else:
                gray = crop

            edges = cv2.Canny(gray, 50, 150)

            # Find non-zero pixels (edges) and compute centroid
            edge_points = np.argwhere(edges > 0)
            if len(edge_points) == 0:
                # No edges found, return original
                return (x, y)

            # Compute centroid of edge pixels
            centroid_y = int(np.mean(edge_points[:, 0]))
            centroid_x = int(np.mean(edge_points[:, 1]))

            # Convert back to original image space
            snapped_x = x1 + centroid_x
            snapped_y = y1 + centroid_y

            # Clamp to valid range
            snapped_x = max(0, min(snapped_x, w - 1))
            snapped_y = max(0, min(snapped_y, h - 1))

            return (snapped_x, snapped_y)

        except Exception as e:
            logger.debug(f"Snap-to-UI failed: {e}")
            return (x, y)

    def _to_screen_coords(self, x: int, y: int, screenshot=None) -> Tuple[float, float]:
        """Convert screenshot-space coordinates to screen-space.

        Args:
            x: X coordinate in screenshot (may be in resized space)
            y: Y coordinate in screenshot (may be in resized space)
            screenshot: Screenshot object carrying resize_scale metadata

        Returns:
            (screen_x, screen_y) in display coordinates
        """
        if self._window_info is None:
            return (float(x), float(y))

        # Apply resize_scale if screenshot was resized (both up/downscale)
        orig_x, orig_y = x, y
        has_scale = hasattr(screenshot, 'resize_scale') if screenshot is not None else False
        scale_val = getattr(screenshot, 'resize_scale', 1.0) if screenshot is not None else 1.0

        if screenshot is not None and hasattr(screenshot, 'resize_scale'):
            scale = screenshot.resize_scale
            if scale != 1.0:
                x = int(x * scale)
                y = int(y * scale)

        # Get window position
        win_x = self._window_info.get("x", 0)
        win_y = self._window_info.get("y", 0)

        # Adjust for screenshot offset (top_offset is the gap from window top to game area)
        screen_x = win_x + x
        screen_y = win_y + self.top_offset + y

        logger.debug(f"[COORDS] in=({orig_x},{orig_y}) scale={scale_val} has_scale={has_scale} "
                     f"scaled=({x},{y}) win=({win_x},{win_y}) top={self.top_offset} "
                     f"→ screen=({screen_x},{screen_y})")

        return (screen_x, screen_y)

    def _is_blacklisted(self, x: int, y: int) -> bool:
        """Check if (x,y) is within any active blacklisted stuck zone."""
        for bx, by, _ in self._stuck_blacklist:
            if abs(x - bx) <= self._blacklist_radius and abs(y - by) <= self._blacklist_radius:
                return True
        return False

    def _tick_blacklist(self) -> None:
        """Decrement remaining rounds for each blacklist entry, drop expired."""
        if not self._stuck_blacklist:
            return
        new_list = []
        for bx, by, remaining in self._stuck_blacklist:
            if remaining > 1:
                new_list.append((bx, by, remaining - 1))
            else:
                logger.debug(f"[BLACKLIST] Expired zone ({bx},{by})")
        self._stuck_blacklist = new_list

    def _action_is_fresh(self, current_screenshot: Optional[Image]) -> bool:
        """② Check if action is fresh (not stale based on time and frame change).

        Args:
            current_screenshot: Current screenshot for MSE comparison

        Returns:
            True if action is fresh, False if stale
        """
        # Time check: instruction too old?
        age = time.time() - self._last_screenshot_time
        if age > self._stale_threshold:
            logger.debug(f"[STALE] Action {age:.1f}s old (threshold {self._stale_threshold}s)")
            return False

        # MSE check: screen changed too much?
        if self._last_screenshot_gray is not None and current_screenshot is not None:
            try:
                current_gray = np.array(current_screenshot.convert("L"))
                if current_gray.shape == self._last_screenshot_gray.shape:
                    mse = compute_mse(current_gray, self._last_screenshot_gray)
                    if mse > self._stale_mse_threshold:
                        logger.debug(f"[STALE] MSE {mse:.1f} > {self._stale_mse_threshold}, screen changed")
                        return False
            except Exception as e:
                logger.debug(f"[STALE] MSE check failed: {e}")

        return True

    def _do_drag(self, action: Dict[str, Any], screenshot: Optional[Image] = None) -> bool:
        """③ Execute drag action.

        Args:
            action: Action dict with x1, y1, x2, y2, duration
            screenshot: Current screenshot

        Returns:
            True if executed, False otherwise
        """
        try:
            x1, y1 = action.get("x1", 0), action.get("y1", 0)
            x2, y2 = action.get("x2", 0), action.get("y2", 0)
            duration = action.get("duration", 0.5)
            target = action.get("target", "drag")

            self._window_info = self.screen_capture.find_wechat_window()
            if self._window_info is None:
                logger.error("Could not find WeChat window")
                return False

            screen_x1, screen_y1 = self._to_screen_coords(x1, y1, screenshot)
            screen_x2, screen_y2 = self._to_screen_coords(x2, y2, screenshot)
            action["executed_x1"] = int(x1); action["executed_y1"] = int(y1)
            action["executed_x2"] = int(x2); action["executed_y2"] = int(y2)

            logger.info(f"🎯 Dragging {target}: ({screen_x1:.0f},{screen_y1:.0f}) → ({screen_x2:.0f},{screen_y2:.0f})")
            self.screen_capture.activate_wechat()
            time.sleep(0.05)
            self.input_controller.drag(screen_x1, screen_y1, screen_x2, screen_y2, duration)

            # Record in history
            self._action_history.append({
                "target": target,
                "x": (x1 + x2) / 2,
                "y": (y1 + y2) / 2,
                "time": time.time(),
            })

            return True

        except Exception as e:
            logger.error(f"Failed to execute drag: {e}")
            return False

    def _do_long_press(self, action: Dict[str, Any], screenshot: Optional[Image] = None) -> bool:
        """③ Execute long_press action.

        Args:
            action: Action dict with x, y, duration
            screenshot: Current screenshot

        Returns:
            True if executed, False otherwise
        """
        try:
            x, y = action.get("x", 0), action.get("y", 0)
            duration = action.get("duration", 2.0)
            target = action.get("target", "long_press")

            # Snap to UI if needed
            if screenshot is not None:
                snapped_x, snapped_y = self._snap_to_ui_boundary(screenshot, x, y)
                x, y = snapped_x, snapped_y

            self._window_info = self.screen_capture.find_wechat_window()
            if self._window_info is None:
                logger.error("Could not find WeChat window")
                return False

            screen_x, screen_y = self._to_screen_coords(x, y, screenshot)
            action["executed_x"] = int(x)
            action["executed_y"] = int(y)
            logger.info(f"📌 Long-pressing {target} at ({screen_x:.0f},{screen_y:.0f}) for {duration}s")
            self.screen_capture.activate_wechat()
            time.sleep(0.05)
            self.input_controller.press_and_hold(screen_x, screen_y, duration)

            # Record in history
            self._action_history.append({
                "target": target,
                "x": x,
                "y": y,
                "time": time.time(),
            })

            return True

        except Exception as e:
            logger.error(f"Failed to execute long_press: {e}")
            return False

    def is_game_over(self, screenshot: Image) -> bool:
        """Check if game is over. Endless Winter doesn't have game over."""
        return False

    def on_game_over(self) -> bool:
        """Handle game over. Endless Winter doesn't end, so always return True to continue."""
        logger.debug("on_game_over called, but Endless Winter doesn't end. Continuing...")
        return True

    def get_game_name(self) -> str:
        """Return game name."""
        return "endless_winter"
