import yaml
import os
from pathlib import Path
from typing import Dict, Any
from pydantic import BaseModel, Field

# Get project root
PROJECT_ROOT = Path(__file__).parent.parent


class GeminiConfig(BaseModel):
    api_key: str = Field(default="")
    model: str = Field(default="gemini-2.0-flash")
    timeout: int = Field(default=30)

    class Config:
        extra = "allow"


class WeCharConfig(BaseModel):
    window_title: str = Field(default="微信")
    game_area: Dict[str, int] = Field(
        default={
            "top_offset": 88,
            "bottom_offset": 0,
        }
    )

    class Config:
        extra = "allow"


class LoopConfig(BaseModel):
    interval: float = Field(default=1.5)
    max_rounds: int = Field(default=200)

    class Config:
        extra = "allow"


class LoggingConfig(BaseModel):
    level: str = Field(default="INFO")
    save_screenshots: bool = Field(default=True)

    class Config:
        extra = "allow"


class GlobalConfig(BaseModel):
    gemini: GeminiConfig
    wechat: WeCharConfig
    loop: LoopConfig
    logging: LoggingConfig

    class Config:
        extra = "allow"


class GameConfig(BaseModel):
    name: str
    display_name: str = ""
    vision: Dict[str, Any] = Field(default={})
    calibration: Dict[str, Any] = Field(default={})
    timing: Dict[str, Any] = Field(default={})

    class Config:
        extra = "allow"


def load_global_config() -> GlobalConfig:
    """Load global configuration from settings.yaml.

    Returns:
        GlobalConfig object
    """
    config_path = PROJECT_ROOT / "config" / "settings.yaml"

    if not config_path.exists():
        # Create default config
        _create_default_config(config_path)

    with open(config_path, "r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f) or {}

    # Override API key from environment
    if "gemini" not in raw_config:
        raw_config["gemini"] = {}
    if "GEMINI_API_KEY" in os.environ:
        raw_config["gemini"]["api_key"] = os.environ["GEMINI_API_KEY"]

    return GlobalConfig(**raw_config)


def load_game_config(game_name: str) -> GameConfig:
    """Load game-specific configuration.

    Args:
        game_name: Name of the game (e.g., "endless_winter")

    Returns:
        GameConfig object
    """
    config_path = PROJECT_ROOT / "config" / "games" / f"{game_name}.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Game config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f) or {}

    return GameConfig(**raw_config)


def _create_default_config(config_path: Path) -> None:
    """Create default settings.yaml file.

    Args:
        config_path: Path to settings.yaml
    """
    config_path.parent.mkdir(parents=True, exist_ok=True)

    default_config = {
        "gemini": {
            "api_key": "",  # Will be read from GEMINI_API_KEY env var
            "model": "gemini-2.0-flash",
            "timeout": 30,
        },
        "wechat": {
            "window_title": "微信",
            "game_area": {
                "top_offset": 88,
                "bottom_offset": 0,
            },
        },
        "loop": {
            "interval": 1.5,
            "max_rounds": 200,
        },
        "logging": {
            "level": "INFO",
            "save_screenshots": True,
        },
    }

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(default_config, f, allow_unicode=True)

    print(f"Created default config: {config_path}")
