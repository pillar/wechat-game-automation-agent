import os
import time
import logging
from collections import defaultdict
from .base_game import BaseGame
from .ai_client import GeminiVisionClient
from .screen import ScreenCapture
from .input_controller import InputController

try:
    from utils.dashboard_bus import bus as _dash_bus

    def _dash_emit(event_type, payload):
        if _dash_bus.is_enabled():
            _dash_bus.emit(event_type, payload)
except Exception:  # pragma: no cover - bus is optional
    def _dash_emit(event_type, payload):
        pass

logger = logging.getLogger(__name__)


class PerformanceStats:
    """Track performance metrics for diagnostics."""

    def __init__(self):
        self.frame_times = []
        self.screenshot_times = []
        self.ai_times = []
        self.parse_times = []
        self.execute_times = []
        self.confidence_scores = []
        self.error_types = defaultdict(int)
        self.action_types = defaultdict(int)

    def add_frame(self, total_time: float, ss_time: float, ai_time: float, parse_time: float, exec_time: float):
        """Record frame timing."""
        self.frame_times.append(total_time)
        self.screenshot_times.append(ss_time)
        self.ai_times.append(ai_time)
        self.parse_times.append(parse_time)
        self.execute_times.append(exec_time)

    def add_confidence(self, confidence: float):
        """Record AI confidence."""
        self.confidence_scores.append(confidence)

    def add_error(self, error_type: str):
        """Record error type."""
        self.error_types[error_type] += 1

    def add_action(self, action_type: str):
        """Record action type."""
        self.action_types[action_type] += 1

    def get_summary(self) -> str:
        """Get performance summary."""
        if not self.frame_times:
            return "No data"

        avg_frame = sum(self.frame_times) / len(self.frame_times)
        avg_ss = sum(self.screenshot_times) / len(self.screenshot_times)
        avg_ai = sum(self.ai_times) / len(self.ai_times) if self.ai_times else 0
        avg_confidence = sum(self.confidence_scores) / len(self.confidence_scores) if self.confidence_scores else 0

        summary = f"""
╔═══════════════════════════════════════════════════════════╗
║              🎯 Performance Summary                        ║
╠═══════════════════════════════════════════════════════════╣
║ Total Frames:     {len(self.frame_times):3d}                              ║
║ Avg Frame Time:   {avg_frame:6.1f}ms                            ║
║   - Screenshot:   {avg_ss:6.1f}ms                            ║
║   - AI Analysis:  {avg_ai:6.1f}ms                            ║
║ Avg Confidence:   {avg_confidence:6.2f}                            ║
║ Action Types:     {dict(self.action_types)}"""
        if self.error_types:
            summary += f"\n║ Error Types:      {dict(self.error_types)}"
        summary += "\n╚═══════════════════════════════════════════════════════════╝"
        return summary


class GameLoop:
    """Main game automation loop with performance optimizations."""

    def __init__(
        self,
        game: BaseGame,
        ai_client: GeminiVisionClient,
        loop_interval: float = 1.5,
        max_rounds: int = 200,
        dry_run: bool = False,
    ):
        self.game = game
        self.ai_client = ai_client
        self.loop_interval = loop_interval
        self.max_rounds = max_rounds
        self.dry_run = dry_run
        self.round_count = 0
        self.screen_capture = ScreenCapture()
        self.input_controller = InputController()

        self.consecutive_failures = 0
        self.max_consecutive_failures = 3
        self.consecutive_skip_actions = 0
        self.max_consecutive_skips = 5

        self.total_jumps_executed = 0
        self.total_jumps_skipped = 0
        self.total_game_overs = 0
        # Rounds where we skipped analysis entirely to let the game render after
        # the previous action (skip_next_screenshot path). These aren't "skipped
        # actions" — they're intentional idle waits.
        self.total_post_action_skips = 0
        self.perf_stats = PerformanceStats()

        self.invalid_screenshots = 0
        self.loading_detections = 0

        self.skip_next_screenshot = False

        logger.info(
            f"Initialized GameLoop for {game.get_game_name()} "
            f"(max_rounds={max_rounds}, dry_run={dry_run})"
        )

    def run(self) -> None:
        """Run the game loop.

        Main loop that captures screen, analyzes with AI, and executes actions.
        Gracefully handles Ctrl+C interruption.
        """
        logger.info(f"Starting game loop for {self.game.get_game_name()}")

        try:
            while self.round_count < self.max_rounds:
                self.round_count += 1
                frame_start_time = time.time()
                logger.info(f"Round {self.round_count}/{self.max_rounds}")
                _dash_emit("round_start", {"round": self.round_count, "max": self.max_rounds})
                try:
                    setattr(self.game, "_current_round", self.round_count)
                except Exception:
                    pass

                if self.skip_next_screenshot:
                    logger.debug(f"⏩ Skipping screenshot analysis, waiting {self.loop_interval}s for game to render...")
                    self.skip_next_screenshot = False
                    self.total_post_action_skips += 1
                    time.sleep(self.loop_interval)
                    continue

                self.game.on_round_start()

                # Capture screenshot once per round (optimize: pass to execute_action)
                # Get capture parameters from game config if available
                resize_width = None
                if hasattr(self.game, 'use_resize') and self.game.use_resize:
                    resize_width = getattr(self.game, 'resize_width', None)

                top_offset = getattr(self.game, 'top_offset', 88)
                bottom_offset = getattr(self.game, 'bottom_offset', 0)

                ss_start = time.time()
                screenshot = self.screen_capture.capture_game_area(
                    top_offset=top_offset,
                    bottom_offset=bottom_offset,
                    resize_width=resize_width
                )
                ss_time = time.time() - ss_start

                if screenshot is None:
                    logger.error("Failed to capture screenshot")
                    self.consecutive_failures += 1
                    if self.consecutive_failures >= self.max_consecutive_failures:
                        logger.error("Too many failures, stopping")
                        break
                    continue
                else:
                    self.consecutive_failures = 0  # Reset on success

                # Quality check: validate screenshot content
                is_valid, reason = self.screen_capture.is_screenshot_valid(screenshot)
                if not is_valid:
                    logger.warning(f"Invalid screenshot: {reason}")
                    self.invalid_screenshots += 1
                    self.perf_stats.add_error("invalid_screenshot")
                    if self.invalid_screenshots > 5:
                        logger.warning("Too many invalid screenshots, might be loading. Adding extra wait...")
                        time.sleep(1.0)
                        self.invalid_screenshots = 0
                    continue

                # [新增] 识别当前游戏场景
                if hasattr(self.game, 'identify_scene'):
                    scene = self.game.identify_scene(screenshot)
                    logger.debug(f"Scene: {scene}")

                    # 根据场景采取不同行动
                    if scene == "game_over":
                        logger.info("📊 游戏结束界面检测，准备重启游戏")
                        # 立即重启，不继续其他逻辑
                        if self.game.on_game_over():
                            logger.info("游戏已重启")
                            continue
                        else:
                            logger.error("无法重启游戏，停止运行")
                            break

                    elif scene == "loading":
                        logger.info("⏳ 游戏加载中，等待...")
                        time.sleep(1.0)
                        continue

                    elif scene == "menu":
                        logger.info("📱 检测到菜单界面，点击开始")
                        # TODO: 实现菜单点击逻辑
                        continue

                    elif scene == "error":
                        logger.error("❌ 检测到异常状态，重启游戏")
                        if self.game.on_game_over():
                            continue
                        else:
                            break

                    # scene == "playing": 继续正常的跳跃逻辑

                if self.round_count > 5 and self.total_jumps_executed > 3 and self.game.is_game_over(screenshot):
                    logger.info("Game over detected")
                    self.total_game_overs += 1
                    self.consecutive_skip_actions = 0

                    if self.game.on_game_over():
                        logger.info("Game restarted, continuing...")
                        continue
                    else:
                        logger.info("Could not restart game, stopping")
                        break

                # Analyze with Gemini (with smart retry on low confidence)
                logger.debug("Analyzing with Gemini Vision API...")
                ai_start = time.time()
                try:
                    # Use smart retry if game supports it
                    if hasattr(self.game, 'analyze_with_retry'):
                        response = self.game.analyze_with_retry(screenshot, max_retries=1)
                    else:
                        prompt = self.game.build_prompt(screenshot)
                        response = self.ai_client.analyze(screenshot, prompt)
                    ai_time = time.time() - ai_start
                    logger.debug(f"Gemini response ({ai_time:.1f}s): {response[:300]}...")

                    # Save screenshot and AI response for debugging
                    game_name = getattr(self.game, 'config', {}).get('name', 'unknown')
                    ss_path = self.screen_capture.save_debug_screenshot(screenshot, game_name, self.round_count, response)
                    # Dashboard emits (no-op when bus disabled)
                    try:
                        rel = os.path.relpath(ss_path) if ss_path else None
                    except Exception:
                        rel = None
                    active_task = None
                    try:
                        if getattr(self.game, "_planner", None) is not None:
                            t = self.game._planner.active_task()
                            if t is not None:
                                active_task = getattr(t, "name", None)
                    except Exception:
                        pass
                    _dash_emit("screenshot_captured", {
                        "round": self.round_count,
                        "path": rel,
                        "width": screenshot.width,
                        "height": screenshot.height,
                        "scene": getattr(self.game, "_current_scene", None),
                        "task": active_task,
                    })
                    _dash_emit("infer", {
                        "round": self.round_count,
                        "system": getattr(self.game, "_round_system", None) or "?",
                        "prompt": getattr(self.game, "_round_prompt", None) or "",
                        "response": getattr(self.game, "_round_response", None) or response,
                        "latency_ms": int(ai_time * 1000),
                        "image_w": screenshot.width,
                        "image_h": screenshot.height,
                    })
                except Exception as e:
                    logger.error(f"AI analysis failed: {e}")
                    self.consecutive_failures += 1
                    if self.consecutive_failures >= self.max_consecutive_failures:
                        logger.error("Too many AI failures, stopping")
                        break
                    continue

                # Parse response
                parse_start = time.time()
                try:
                    action = self.game.parse_ai_response(response, screenshot=screenshot)
                    parse_time = time.time() - parse_start
                    logger.info(f"Parsed action: {action}")

                    # Record stats
                    if isinstance(action, dict):
                        self.perf_stats.add_confidence(action.get("confidence", 0.0))
                        self.perf_stats.add_action(action.get("action", "unknown"))
                        if error := action.get("error"):
                            self.perf_stats.add_error(error)
                except Exception as e:
                    logger.error(f"Failed to parse AI response: {e}")
                    parse_time = time.time() - parse_start
                    self.consecutive_failures += 1
                    if self.consecutive_failures >= self.max_consecutive_failures:
                        logger.error("Too many parse failures, stopping")
                        break
                    continue

                # Execute action (optimize: pass screenshot to avoid re-capture)
                exec_start = time.time()
                if not self.dry_run:
                    try:
                        # Pass screenshot to avoid duplicate capture in execute_action
                        success = self.game.execute_action(action, screenshot=screenshot)
                        exec_time = time.time() - exec_start
                        _dash_emit("action", {
                            "round": self.round_count,
                            "success": bool(success),
                            **({
                                "target": action.get("target"),
                                "x": action.get("x"),
                                "y": action.get("y"),
                                "x1": action.get("x1"),
                                "y1": action.get("y1"),
                                "x2": action.get("x2"),
                                "y2": action.get("y2"),
                                # Post-snap, pre-screen-translation coords — where the
                                # click actually landed in screenshot space.
                                "executed_x": action.get("executed_x"),
                                "executed_y": action.get("executed_y"),
                                "executed_x1": action.get("executed_x1"),
                                "executed_y1": action.get("executed_y1"),
                                "executed_x2": action.get("executed_x2"),
                                "executed_y2": action.get("executed_y2"),
                                "type": action.get("action"),
                                "confidence": action.get("confidence"),
                                "reasoning": action.get("reasoning"),
                                "ocr_override": bool(action.get("ocr_override", False)),
                            } if isinstance(action, dict) else {}),
                        })

                        if success:
                            self.total_jumps_executed += 1
                            self.consecutive_skip_actions = 0
                            logger.debug("Action executed successfully")
                            self.consecutive_failures = 0
                            self.skip_next_screenshot = True
                        else:
                            # Action was skipped (low confidence, error, etc.)
                            self.total_jumps_skipped += 1
                            self.consecutive_skip_actions += 1

                            if self.consecutive_skip_actions >= self.max_consecutive_skips:
                                logger.warning(
                                    f"Skipped {self.consecutive_skip_actions} actions in a row, "
                                    f"likely detection issue. Restarting game..."
                                )
                                if self.game.on_game_over():
                                    self.consecutive_skip_actions = 0
                                    continue
                                else:
                                    logger.error("Could not restart game")
                                    break
                    except Exception as e:
                        logger.error(f"Failed to execute action: {e}", exc_info=True)
                        exec_time = time.time() - exec_start
                        self.consecutive_failures += 1
                        if self.consecutive_failures >= self.max_consecutive_failures:
                            logger.error("Too many execution failures, stopping")
                            break
                else:
                    logger.info("[DRY RUN] Would execute action: " + str(action))
                    success = True
                    exec_time = 0

                # Record frame timing
                frame_time = time.time() - frame_start_time
                self.perf_stats.add_frame(frame_time, ss_time, ai_time, parse_time, exec_time)

                # Hook: post-round
                self.game.on_round_end(action, success if not self.dry_run else True)

                # Wait before next round
                if self.round_count < self.max_rounds:
                    logger.debug(f"Waiting {self.loop_interval}s before next round...")
                    time.sleep(self.loop_interval)

        except KeyboardInterrupt:
            logger.info("Game loop interrupted by user")
        except Exception as e:
            logger.error(f"Game loop error: {e}", exc_info=True)
        finally:
            self._print_stats()

    def _print_stats(self) -> None:
        """Print final statistics."""
        analyzed = self.round_count - self.total_post_action_skips
        logger.info(
            f"Rounds: {self.round_count} total "
            f"({analyzed} analyzed, {self.total_post_action_skips} post-action idle); "
            f"clicks emitted: {self.total_jumps_executed}, "
            f"no-op rounds: {self.total_jumps_skipped}, "
            f"game overs: {self.total_game_overs}"
        )

        # Print detailed performance report
        logger.info(self.perf_stats.get_summary())

        # Print quality metrics
        if self.invalid_screenshots > 0:
            logger.info(f"Invalid screenshots detected: {self.invalid_screenshots}")
        if self.loading_detections > 0:
            logger.info(f"Game loading states detected: {self.loading_detections}")

        # Click emission rate (NOT success — just "did we send a click this round")
        if analyzed > 0:
            emit_rate = (self.total_jumps_executed / analyzed) * 100
            logger.info(
                f"Click emission rate: {emit_rate:.1f}% "
                f"({self.total_jumps_executed}/{analyzed} analyzed rounds)"
            )

        # Real verified outcomes — measured by the post-condition verifier,
        # which compares scene_after/changed to the expectation we recorded
        # before acting. This is closer to "did we actually point the game
        # somewhere new" than click emission rate.
        verifier = getattr(self.game, "_verifier", None)
        if verifier is not None and getattr(verifier, "total_verified", 0) > 0:
            v_total = verifier.total_verified
            v_changed = verifier.total_changed
            v_scene = verifier.total_scene_matched
            v_nochg = verifier.total_no_change
            changed_rate = (v_changed / v_total) * 100 if v_total else 0
            logger.info(
                f"Verified outcomes: {v_total} actions verified — "
                f"changed: {v_changed} ({changed_rate:.1f}%), "
                f"scene-matched: {v_scene}, "
                f"no-op (click landed blank): {v_nochg}"
            )
