from typing import Optional, Dict, Tuple
from PIL import ImageGrab, ImageEnhance, Image as PILImage
from PIL.Image import Image
import logging
import numpy as np
from io import BytesIO
import os
from datetime import datetime

from .quartz_window import QuartzWindowDetector
from Quartz import (
    CGDisplayPixelsWide, CGDisplayPixelsHigh, CGDisplayBounds, CGMainDisplayID,
    CGWindowListCopyWindowInfo, kCGWindowListOptionOnScreenOnly,
    kCGWindowBounds, kCGWindowNumber, kCGWindowOwnerName,
    CGWindowListCreateImage, kCGWindowListOptionIncludingWindow,
    kCGWindowImageBoundsIgnoreFraming, CGRectNull
)
from AppKit import NSImage, NSWorkspace, NSApplicationActivateIgnoringOtherApps

logger = logging.getLogger(__name__)


class ScreenCapture:
    """Handle screenshot capture and window detection for macOS using Quartz."""

    @classmethod
    def activate_wechat(cls) -> bool:
        """Bring WeChat to the foreground so clicks register.

        Returns True if activated, False otherwise.
        """
        try:
            workspace = NSWorkspace.sharedWorkspace()
            for app in workspace.runningApplications():
                name = app.localizedName()
                if name in ("WeChat", "微信"):
                    app.activateWithOptions_(NSApplicationActivateIgnoringOtherApps)
                    return True
            return False
        except Exception as e:
            logger.warning(f"Failed to activate WeChat: {e}")
            return False

    @classmethod
    def find_wechat_window(cls) -> Optional[Dict]:
        """Find WeChat window position and size using Quartz.

        Returns:
            Dict with keys: x, y, width, height, scale, window_id
            Returns None if window not found
        """
        try:
            window_list = CGWindowListCopyWindowInfo(
                kCGWindowListOptionOnScreenOnly, 0
            )

            if not window_list:
                logger.debug("No windows found")
                return None

            best_window = None
            best_score = -1

            for window_info in window_list:
                owner_name = window_info.get(kCGWindowOwnerName, "")
                bounds = window_info.get(kCGWindowBounds)

                if not bounds or owner_name not in ("WeChat", "微信"):
                    continue

                x = int(bounds["X"])
                y = int(bounds["Y"])
                width = int(bounds["Width"])
                height = int(bounds["Height"])

                if width > 0 and height > 0:
                    aspect_ratio = width / height
                    if width < 600 and 0.3 < aspect_ratio < 1.5:
                        score = width * height
                    else:
                        score = (width * height) * 0.1

                    if score > best_score:
                        best_score = score
                        window_id = window_info.get(kCGWindowNumber, 0)
                        best_window = {
                            "x": x,
                            "y": y,
                            "width": width,
                            "height": height,
                            "scale": 1.0,
                            "window_id": window_id,
                        }

            if best_window:
                logger.debug(
                    f"Found WeChat window: id={best_window['window_id']}, "
                    f"pos=({best_window['x']},{best_window['y']}), "
                    f"size={best_window['width']}x{best_window['height']}"
                )
                return best_window

            logger.debug("No suitable WeChat window found")
            return None

        except Exception as e:
            logger.error(f"Failed to find WeChat window: {e}")
            return None

    @classmethod
    def _get_screen_scale(cls) -> float:
        """Get screen scale factor for Retina displays using Quartz.

        Returns:
            Scale factor (usually 1.0 or 2.0)
        """
        try:
            main_display = CGMainDisplayID()
            pixels_wide = CGDisplayPixelsWide(main_display)
            bounds = CGDisplayBounds(main_display)
            logical_width = bounds.size.width

            if logical_width > 0:
                scale = pixels_wide / logical_width
                logger.debug(f"Screen scale: {scale:.1f}x")
                return scale
        except Exception as e:
            logger.warning(f"Failed to detect screen scale: {e}")

        logger.debug("Using default screen scale: 1.0x")
        return 1.0

    @classmethod
    def capture_full_screen(cls) -> Image:
        """Capture the full screen.

        Returns:
            PIL Image of the full screen
        """
        return ImageGrab.grab()

    @classmethod
    def _capture_window_direct(cls, window_info: Dict, top_offset: int, bottom_offset: int) -> Optional[Image]:
        """Capture window content directly using Quartz (ignores other windows).

        Args:
            window_info: Window info dict with x, y, window_id, width, height
            top_offset: Pixels to skip from top
            bottom_offset: Pixels to skip from bottom

        Returns:
            PIL Image or None if failed
        """
        try:
            window_id = window_info.get('window_id', 0)
            if not window_id:
                return None

            x = window_info['x']
            y = window_info['y']
            width = window_info['width']
            height = window_info['height']

            # Bounds in screen coordinates
            from Quartz import CGRectMake
            bounds = CGRectMake(x, y + top_offset, width, height - top_offset - bottom_offset)

            logger.debug(f"Capturing window {window_id} from bounds: x={x} y={y+top_offset} w={width} h={height-top_offset-bottom_offset}")

            # Capture just this window (ignoring other windows on top)
            cg_image = CGWindowListCreateImage(
                bounds,
                kCGWindowListOptionIncludingWindow,
                window_id,
                0  # No special flags
            )

            if not cg_image:
                logger.warning(f"CGWindowListCreateImage failed for window {window_id}")
                return None

            # Convert CGImage → NSImage → TIFF → PIL
            ns_image = NSImage.alloc().initWithCGImage_(cg_image)
            if not ns_image:
                logger.warning("NSImage init failed")
                return None

            tiff_repr = ns_image.TIFFRepresentation()
            if not tiff_repr:
                logger.warning("TIFFRepresentation failed")
                return None

            # Convert TIFF to PIL Image
            pil_image = PILImage.open(BytesIO(tiff_repr))
            return pil_image.convert('RGB')

        except Exception as e:
            logger.warning(f"Direct window capture failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    @classmethod
    def capture_game_area(
        cls,
        top_offset: int = 88,
        bottom_offset: int = 0,
        resize_width: Optional[int] = None
    ) -> Optional[Image]:
        """Capture just the game area within the WeChat window.

        Uses direct window capture (Quartz) to avoid being blocked by overlapping windows.

        Args:
            top_offset: Pixels to skip from top (title bar height)
            bottom_offset: Pixels to skip from bottom
            resize_width: Optional width to resize screenshot to

        Returns:
            PIL Image of the game area, or None if window not found
        """
        window_info = cls.find_wechat_window()
        if not window_info:
            logger.error("Could not find WeChat window")
            return None

        # Try direct window capture first (most reliable)
        screenshot = cls._capture_window_direct(window_info, top_offset, bottom_offset)

        if screenshot is None:
            logger.warning("Direct window capture failed, cannot use fallback (would be blocked by overlapping windows)")
            return None

        original_size = screenshot.size
        logger.info(f"📸 Captured from window: {original_size}")

        # Optional: resize to target width (both up and downscale supported)
        if resize_width and screenshot.width != resize_width:
            aspect_ratio = screenshot.height / screenshot.width
            resize_height = int(resize_width * aspect_ratio)
            screenshot = screenshot.resize((resize_width, resize_height), PILImage.LANCZOS)

            screenshot.original_size = original_size
            screenshot.resize_scale = original_size[0] / resize_width

            direction = "↑ upscaled" if resize_width > original_size[0] else "↓ downscaled"
            logger.debug(
                f"Resized ({direction}): {original_size} → {screenshot.size} (scale {screenshot.resize_scale:.2f}x)"
            )
        else:
            screenshot.original_size = screenshot.size

        return screenshot

    @classmethod
    def save_debug_screenshot(cls, screenshot: Image, game_name: str, round_number: int, ai_response: Optional[str] = None) -> str:
        """Save screenshot and AI response to debug directory.

        Args:
            screenshot: PIL Image to save
            game_name: Name of the game (e.g., 'endless_winter')
            round_number: Round/level number
            ai_response: Optional AI response text to save alongside

        Returns:
            Path to saved screenshot
        """
        try:
            # Create directory structure: debug/screenshots/{game_name}/
            debug_dir = os.path.join(os.getcwd(), 'debug', 'screenshots', game_name)
            os.makedirs(debug_dir, exist_ok=True)

            # Save screenshot with round number
            timestamp = datetime.now().strftime('%H%M%S')
            ss_filename = f"round_{round_number:03d}_{timestamp}.png"
            ss_path = os.path.join(debug_dir, ss_filename)
            screenshot.save(ss_path)
            logger.debug(f"Saved screenshot: {ss_path}")

            # Save AI response if provided
            if ai_response:
                resp_filename = f"round_{round_number:03d}_{timestamp}.txt"
                resp_path = os.path.join(debug_dir, resp_filename)
                with open(resp_path, 'w', encoding='utf-8') as f:
                    f.write(ai_response)
                logger.debug(f"Saved AI response: {resp_path}")

            return ss_path

        except Exception as e:
            logger.warning(f"Failed to save debug screenshot: {e}")
            return ""

    @classmethod
    def is_screenshot_valid(cls, screenshot: Image, min_color_variance: float = 5.0) -> tuple[bool, str]:
        """Check if screenshot contains valid game content (not black screen, loading, etc).

        Args:
            screenshot: PIL Image to validate
            min_color_variance: Minimum standard deviation of pixel values

        Returns:
            Tuple of (is_valid, reason)
        """
        try:
            # Convert to numpy array for analysis
            img_array = np.array(screenshot)

            # Check 1: Brightness (black screen = loading or error)
            mean_brightness = np.mean(img_array)
            if mean_brightness < 20:
                return False, "Mostly black (likely loading or error)"

            # Check 2: Color variance (single color = loading screen)
            color_std = np.std(img_array)
            if color_std < min_color_variance:
                return False, f"Low color variance (std={color_std:.1f})"

            # Check 3: Unique colors (too few = loading screen)
            unique_colors = len(np.unique(img_array.reshape(-1, 3), axis=0))
            if unique_colors < 20:
                return False, f"Too few colors ({unique_colors})"

            return True, "Valid"

        except Exception as e:
            logger.warning(f"Screenshot validation failed: {e}")
            return False, f"Validation error: {e}"

    @classmethod
    def enhance_image(
        cls,
        screenshot: Image,
        contrast: float = 1.5,
        brightness: float = 1.1
    ) -> Image:
        """Enhance image for better AI analysis (higher contrast, adjusted brightness).

        Args:
            screenshot: PIL Image to enhance
            contrast: Contrast factor (> 1.0 increases contrast)
            brightness: Brightness factor (> 1.0 increases brightness)

        Returns:
            Enhanced PIL Image
        """
        try:
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(screenshot)
            enhanced = enhancer.enhance(contrast)

            # Enhance brightness
            enhancer = ImageEnhance.Brightness(enhanced)
            enhanced = enhancer.enhance(brightness)

            logger.debug(f"Enhanced image: contrast={contrast}, brightness={brightness}")
            return enhanced

        except Exception as e:
            logger.warning(f"Image enhancement failed: {e}")
            return screenshot
