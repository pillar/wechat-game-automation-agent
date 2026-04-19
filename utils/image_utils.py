from PIL import Image, ImageDraw
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def save_screenshot(image: Image, name: str = "") -> Path:
    """Save screenshot to debug directory.

    Args:
        image: PIL Image to save
        name: Optional name prefix

    Returns:
        Path to saved image
    """
    debug_dir = Path(__file__).parent.parent / "debug" / "screenshots"
    debug_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    filename = f"{timestamp}_{name}.png" if name else f"{timestamp}.png"
    filepath = debug_dir / filename

    image.save(filepath)
    logger.debug(f"Saved screenshot: {filepath}")

    return filepath


def draw_crosshair(image: Image, x: float, y: float, color: str = "red", size: int = 20) -> Image:
    """Draw a crosshair on the image.

    Args:
        image: PIL Image
        x: X coordinate
        y: Y coordinate
        color: Color name
        size: Size of the crosshair

    Returns:
        Modified PIL Image
    """
    draw = ImageDraw.Draw(image)

    # Draw crosshair lines
    draw.line([(x - size, y), (x + size, y)], fill=color, width=2)
    draw.line([(x, y - size), (x, y + size)], fill=color, width=2)

    return image


def resize_image(image: Image, width: int, height: int = None) -> Image:
    """Resize image to specified width, maintaining aspect ratio.

    Args:
        image: PIL Image
        width: Target width
        height: Target height (if None, maintain aspect ratio)

    Returns:
        Resized PIL Image
    """
    if height is None:
        # Calculate height to maintain aspect ratio
        aspect_ratio = image.height / image.width
        height = int(width * aspect_ratio)

    return image.resize((width, height), Image.Resampling.LANCZOS)
