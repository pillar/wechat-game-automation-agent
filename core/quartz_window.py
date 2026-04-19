"""High-performance window detection using Quartz (macOS)."""

import logging
from typing import Optional, Dict
from Quartz import CGWindowListCopyWindowInfo, kCGWindowListOptionOnScreenOnly
from Quartz import kCGWindowNumber, kCGWindowBounds, kCGWindowOwnerName

logger = logging.getLogger(__name__)


class QuartzWindowDetector:
    """Detect windows using Quartz - faster and more accurate than AppleScript."""

    @staticmethod
    def find_window_by_title(window_title: str) -> Optional[Dict]:
        """Find window by title using Quartz.

        Args:
            window_title: Window title to search for

        Returns:
            Dict with x, y, width, height or None if not found
        """
        try:
            # Get list of on-screen windows
            window_list = CGWindowListCopyWindowInfo(
                kCGWindowListOptionOnScreenOnly, 0
            )

            if not window_list:
                logger.debug("No windows found")
                return None

            # Search for window with matching title
            # Prefer reasonable-sized window (not full-screen, not tiny)
            best_window = None
            best_score = -1

            for window_info in window_list:
                owner_name = window_info.get(kCGWindowOwnerName, "")
                bounds = window_info.get(kCGWindowBounds)

                if not bounds:
                    continue

                # Check if this is a WeChat window (supports both English and Chinese names)
                if owner_name in ("WeChat", "微信"):
                    x = int(bounds["X"])
                    y = int(bounds["Y"])
                    width = int(bounds["Width"])
                    height = int(bounds["Height"])

                    if width > 0 and height > 0:
                        # Prefer windows with reasonable aspect ratio (not full-screen)
                        # WeChat game window should be portrait-ish (width < height * 1.5)
                        # and not super wide (width < 600)
                        aspect_ratio = width / height

                        # Scoring: prefer windows that are not full-screen
                        # and have reasonable mobile-game dimensions
                        if width < 600 and 0.3 < aspect_ratio < 1.5:
                            # This looks like a real game window
                            score = width * height  # Just use area for score
                        else:
                            # Not ideal, but might be fallback
                            score = (width * height) * 0.1

                        if score > best_score:
                            best_score = score
                            best_window = {
                                "x": x,
                                "y": y,
                                "width": width,
                                "height": height,
                                "scale": 1.0,  # Physical pixels, no scaling needed
                            }

            if best_window:
                logger.debug(
                    f"Found WeChat window: x={best_window['x']}, y={best_window['y']}, "
                    f"w={best_window['width']}, h={best_window['height']} (score={best_score:.0f})"
                )
                return best_window

            logger.debug(f"No window found with owner name 'WeChat' or '微信'")
            return None

        except Exception as e:
            logger.error(f"Quartz window detection failed: {e}")
            return None

    @staticmethod
    def find_window_by_pid(pid: int) -> Optional[Dict]:
        """Find window by process ID using Quartz.

        Args:
            pid: Process ID

        Returns:
            Dict with x, y, width, height or None if not found
        """
        try:
            window_list = CGWindowListCopyWindowInfo(
                kCGWindowListOptionOnScreenOnly, 0
            )

            if not window_list:
                return None

            for window_info in window_list:
                window_pid = window_info.get("kCGWindowOwnerPID", 0)

                if window_pid == pid:
                    bounds = window_info.get(kCGWindowBounds)
                    if bounds:
                        x = int(bounds["X"])
                        y = int(bounds["Y"])
                        width = int(bounds["Width"])
                        height = int(bounds["Height"])

                        if width > 0 and height > 0:
                            return {
                                "x": x,
                                "y": y,
                                "width": width,
                                "height": height,
                                "scale": 1.0,
                            }

            return None

        except Exception as e:
            logger.error(f"Quartz PID lookup failed: {e}")
            return None

    @staticmethod
    def list_all_windows() -> list:
        """List all on-screen windows for debugging.

        Returns:
            List of window info dicts
        """
        try:
            window_list = CGWindowListCopyWindowInfo(
                kCGWindowListOptionOnScreenOnly, 0
            )

            windows = []
            for window_info in window_list:
                owner_name = window_info.get(kCGWindowOwnerName, "Unknown")
                bounds = window_info.get(kCGWindowBounds)

                if bounds:
                    windows.append(
                        {
                            "name": owner_name,
                            "x": int(bounds["X"]),
                            "y": int(bounds["Y"]),
                            "width": int(bounds["Width"]),
                            "height": int(bounds["Height"]),
                        }
                    )

            return windows

        except Exception as e:
            logger.error(f"Failed to list windows: {e}")
            return []
