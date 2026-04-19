"""Stuck detection and self-healing mechanism."""

import time
import logging
from collections import deque

logger = logging.getLogger(__name__)


class StuckMonitor:
    """Detect infinite loops (repeated clicks at same location) and auto-recover."""

    def __init__(self, click_threshold: int = 5, grid_size: int = 20):
        """Initialize stuck monitor.

        Args:
            click_threshold: Number of clicks to buffer before checking stuck
            grid_size: Quantize click positions to this grid (in pixels)
        """
        # Each entry: (gx, gy, changed_after) where changed_after is None until verify arrives
        self._recent_clicks = deque(maxlen=click_threshold)
        self._threshold = click_threshold
        self._grid_size = grid_size

    def record_click(self, x: float, y: float) -> None:
        """Record a click position.

        Args:
            x: Click X coordinate
            y: Click Y coordinate
        """
        # Quantize to grid to allow small variations
        gx = round(x / self._grid_size) * self._grid_size
        gy = round(y / self._grid_size) * self._grid_size
        self._recent_clicks.append((gx, gy, None))

    def record_verify(self, changed: bool) -> None:
        """Attach the post-click screen-change outcome to the latest recorded click.

        Called from the verifier after the next frame arrives. Effective clicks
        (changed=True) are not considered evidence of being stuck, so repeat
        clicks on a sequential UI (e.g. 下一个) don't trip the detector.
        """
        if not self._recent_clicks:
            return
        gx, gy, _ = self._recent_clicks[-1]
        self._recent_clicks[-1] = (gx, gy, bool(changed))

    def is_stuck(self) -> bool:
        """Stuck = same position N times AND at least N-1 verified no-change.

        Leaves the last entry free (verify for the just-made click hasn't
        arrived yet) but requires every other entry to be `changed_after is
        False` — pending (None) or True both disqualify.
        """
        if len(self._recent_clicks) < self._threshold:
            return False

        positions = [(gx, gy) for gx, gy, _ in self._recent_clicks]
        unique_positions = len(set(positions))
        # Exclude the latest entry from the change check (verify not yet in).
        prior = list(self._recent_clicks)[:-1]
        prior_all_no_change = all(c is False for _, _, c in prior)

        if unique_positions == 1 and prior_all_no_change:
            logger.warning(f"[STUCK] {self._threshold} clicks at {positions[0]} with no visual change")
            return True
        if unique_positions == 2 and self._threshold >= 3 and prior_all_no_change:
            logger.warning(f"[STUCK] A-B cycle between {sorted(set(positions))} with no visual change")
            return True
        return False

    def recover(self, input_controller, scene_classifier=None, window_info=None, top_offset: int = 88) -> None:
        """Attempt recovery from stuck state.

        Strategy:
            1. Press Escape (cheapest, often closes panels)
            2. If window_info given, click top-right corner (likely X close button)
               and top-left corner (likely back arrow) as fallbacks

        Args:
            input_controller: InputController to execute recovery actions
            scene_classifier: Optional SceneClassifier to reset state
            window_info: Window info dict with x, y, width, height
            top_offset: Top offset from window top to game area
        """
        logger.warning("[STUCK] Recovery: Escape + top-corner close/back probes")
        try:
            input_controller.key_press('escape')
            time.sleep(0.3)

            if window_info is not None:
                wx = window_info.get('x', 0)
                wy = window_info.get('y', 0)
                ww = window_info.get('width', 0)
                wh = window_info.get('height', 0)
                game_top = wy + top_offset
                game_h = wh - top_offset

                # Top-right X close probe (x ≈ 93%, y ≈ 5% of game area)
                close_x = int(wx + ww * 0.93)
                close_y = int(game_top + game_h * 0.05)
                logger.info(f"[STUCK] Probe close-X at screen ({close_x}, {close_y})")
                input_controller.click(close_x, close_y)
                time.sleep(0.3)

                # Top-left back arrow probe (x ≈ 7%, y ≈ 5%)
                back_x = int(wx + ww * 0.07)
                back_y = int(game_top + game_h * 0.05)
                logger.info(f"[STUCK] Probe back-arrow at screen ({back_x}, {back_y})")
                input_controller.click(back_x, back_y)
                time.sleep(0.3)
        except Exception as e:
            logger.error(f"[STUCK] Recovery failed: {e}")

        self.reset()

        if scene_classifier:
            scene_classifier.force_reclassify()

    def last_stuck_position(self):
        """Return the grid-quantized (gx, gy) of the most recent click, or None."""
        if len(self._recent_clicks) == 0:
            return None
        gx, gy, _ = self._recent_clicks[-1]
        return (gx, gy)

    def reset(self) -> None:
        """Clear click history."""
        self._recent_clicks.clear()
