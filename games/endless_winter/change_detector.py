"""Change detection for reducing unnecessary API calls."""

from typing import Optional
import numpy as np
import cv2
from PIL import Image


def compute_mse(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute Mean Squared Error between two arrays.

    Args:
        img1: First image as numpy array
        img2: Second image as numpy array

    Returns:
        MSE value, or inf if shapes don't match
    """
    if img1.shape != img2.shape:
        return float('inf')

    diff = img1.astype(np.float32) - img2.astype(np.float32)
    mse = float(np.mean(diff ** 2))
    return mse


class ChangeDetector:
    """Detect meaningful changes between consecutive frames."""

    def __init__(self, threshold: int = 30, roi_skip_top: float = 0.1):
        """Initialize change detector.

        Args:
            threshold: Pixel difference threshold (0-255 range mean)
            roi_skip_top: Fraction of top area to exclude (status bar)
        """
        self._prev_frame: Optional[np.ndarray] = None
        self.threshold = threshold
        self.roi_skip_top = roi_skip_top

    def has_changed(self, screenshot: Image) -> bool:
        """Check if frame has meaningfully changed vs previous.

        Args:
            screenshot: PIL Image to analyze

        Returns:
            True if changed, False if static
        """
        # Convert to grayscale
        img_array = np.array(screenshot)
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array

        # Apply gaussian blur to reduce single-pixel noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Exclude top roi_skip_top fraction (status bar, clock, etc.)
        h = blurred.shape[0]
        roi_start = int(h * self.roi_skip_top)
        roi = blurred[roi_start:]

        # First call: no previous frame
        if self._prev_frame is None:
            self._prev_frame = roi.copy()
            return True

        # Handle case where prev_frame and roi have different heights
        # (shouldn't happen, but be safe)
        if roi.shape[0] != self._prev_frame.shape[0]:
            self._prev_frame = roi.copy()
            return True

        # Compute mean absolute difference
        diff = cv2.absdiff(roi, self._prev_frame)
        mean_diff = float(diff.mean())

        # Update previous frame for next call
        self._prev_frame = roi.copy()

        # Return True if significant change detected
        changed = mean_diff > self.threshold
        return changed
