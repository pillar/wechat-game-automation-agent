import subprocess
import time
import logging

logger = logging.getLogger(__name__)

# Use cliclick for actual input (pyautogui events are silently dropped by WeChat on macOS).
# Install: brew install cliclick
_CLICLICK = "cliclick"


def _run(args: list) -> None:
    subprocess.run([_CLICLICK, *args], check=True)


class InputController:
    """Handle mouse and keyboard input on macOS via cliclick."""

    @staticmethod
    def press_and_hold(x: float, y: float, duration: float) -> None:
        logger.debug(f"Press and hold at ({x}, {y}) for {duration}s")
        try:
            start = time.time()
            _run([f"dd:{int(x)},{int(y)}"])
            elapsed = time.time() - start
            remaining = max(0.0, duration - elapsed)
            time.sleep(remaining)
            _run([f"du:{int(x)},{int(y)}"])
        except Exception as e:
            logger.error(f"Failed to press and hold: {e}")
            raise

    @staticmethod
    def click(x: float, y: float, delay: float = 0.1) -> None:
        logger.debug(f"Click at ({x}, {y}) delay={delay}")
        try:
            if delay <= 0.0:
                _run([f"c:{int(x)},{int(y)}"])
            else:
                _run([f"dd:{int(x)},{int(y)}"])
                time.sleep(delay)
                _run([f"du:{int(x)},{int(y)}"])
        except Exception as e:
            logger.error(f"Failed to click: {e}")
            raise

    @staticmethod
    def drag(
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        duration: float = 0.5
    ) -> None:
        logger.debug(f"Drag from ({x1}, {y1}) to ({x2}, {y2}) in {duration}s")
        try:
            _run([f"dd:{int(x1)},{int(y1)}"])
            steps = max(4, int(duration / 0.05))
            for i in range(1, steps + 1):
                progress = i / steps
                cx = x1 + (x2 - x1) * progress
                cy = y1 + (y2 - y1) * progress
                _run([f"dm:{int(cx)},{int(cy)}"])
                time.sleep(duration / steps)
            _run([f"du:{int(x2)},{int(y2)}"])
        except Exception as e:
            logger.error(f"Failed to drag: {e}")
            raise

    @staticmethod
    def key_press(key: str) -> None:
        logger.debug(f"Press key: {key}")
        key_map = {
            "escape": "esc",
            "return": "return",
            "space": "space",
            "enter": "return",
            "tab": "tab",
        }
        k = key_map.get(key.lower(), key)
        try:
            _run([f"kp:{k}"])
        except Exception as e:
            logger.error(f"Failed to press key: {e}")
            raise

    @staticmethod
    def move_to(x: float, y: float, duration: float = 0.5) -> None:
        logger.debug(f"Move to ({x}, {y}) in {duration}s")
        try:
            _run([f"m:{int(x)},{int(y)}"])
        except Exception as e:
            logger.error(f"Failed to move mouse: {e}")
            raise
