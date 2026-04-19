import logging
import sys
from pathlib import Path

# Get project root
PROJECT_ROOT = Path(__file__).parent.parent


def setup_logger(log_level: str = "INFO") -> None:
    """Setup logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # Create formatter
    formatter = logging.Formatter(log_format, datefmt=date_format)

    # Setup console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)

    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(console_handler)

    # Also setup file logging
    log_dir = PROJECT_ROOT / "debug"
    log_dir.mkdir(exist_ok=True)

    file_handler = logging.FileHandler(log_dir / "autoplay.log", encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    logging.info(f"Logger initialized with level: {log_level}")
