"""Scene classification for Endless Winter game state management."""

import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class SceneType:
    """Scene constants."""
    MAIN_CITY = "main_city"
    WILDERNESS = "wilderness"
    BATTLE = "battle"
    LOADING = "loading"
    DIALOG = "dialog"
    AD = "ad"
    UNKNOWN = "unknown"


SCENE_DESCRIPTIONS = {
    SceneType.MAIN_CITY: "主城（大熔炉、建筑群）",
    SceneType.WILDERNESS: "野外地图（格子地图、探索）",
    SceneType.BATTLE: "战斗（攻击/技能按钮）",
    SceneType.LOADING: "加载/转场",
    SceneType.DIALOG: "弹窗/对话框/操作提示",
    SceneType.AD: "广告/充值礼包弹窗",
    SceneType.UNKNOWN: "无法判断",
}


class SceneClassifier:
    """Classify game scenes to adapt AI prompts and strategies."""

    def __init__(self, qwen_client, classify_interval: float = 5.0):
        """Initialize scene classifier.

        Args:
            qwen_client: LocalVisionClient for classification
            classify_interval: Seconds between reclassifications (avoid every frame)
        """
        self._client = qwen_client
        self._current_scene = SceneType.UNKNOWN
        self._last_classified = 0.0
        self._classify_interval = classify_interval

    def get_scene(self, screenshot) -> str:
        """Get current scene, using cache if recent.

        Args:
            screenshot: PIL Image to analyze

        Returns:
            Scene type string (one of SceneType constants)
        """
        now = time.time()
        if now - self._last_classified < self._classify_interval:
            # Use cached result
            return self._current_scene

        # Reclassify
        self._current_scene = self._classify(screenshot)
        self._last_classified = now
        logger.debug(f"[SCENE] Reclassified: {self._current_scene}")
        return self._current_scene

    def force_reclassify(self):
        """Force reclassification on next get_scene call."""
        self._last_classified = 0.0

    def _classify(self, screenshot) -> str:
        """Internal: classify scene using Qwen.

        Args:
            screenshot: PIL Image

        Returns:
            Scene type
        """
        try:
            from .prompts import build_scene_classification_prompt

            prompt = build_scene_classification_prompt()
            response = self._client.analyze(screenshot, prompt, max_retries=0)

            # Try exact match in response
            response_lower = response.strip().lower()
            for scene_key in [SceneType.MAIN_CITY, SceneType.WILDERNESS, SceneType.BATTLE,
                             SceneType.LOADING, SceneType.DIALOG, SceneType.AD]:
                if scene_key in response_lower:
                    logger.debug(f"[SCENE] Qwen returned: {scene_key}")
                    return scene_key

            # Fallback: keyword matching
            keywords = {
                SceneType.DIALOG: ["弹窗", "关闭", "对话", "弹出", "确认", "跳过"],
                SceneType.AD: ["广告", "充值", "礼包", "限时"],
                SceneType.WILDERNESS: ["野外", "地图", "探索", "移动"],
                SceneType.BATTLE: ["战斗", "攻击", "技能", "战场"],
                SceneType.LOADING: ["加载", "loading", "转场", "进度"],
                SceneType.MAIN_CITY: ["主城", "城市", "大熔炉", "建筑"],
            }

            for scene, kws in keywords.items():
                if any(kw in response for kw in kws):
                    logger.debug(f"[SCENE] Keyword match: {scene}")
                    return scene

            logger.debug(f"[SCENE] No match, keeping {self._current_scene}")
            return self._current_scene

        except Exception as e:
            logger.debug(f"[SCENE] Classification failed: {e}")
            return self._current_scene or SceneType.UNKNOWN
