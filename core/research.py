"""Game research notes — cached guide summaries scraped via Gemini grounding.

Stores per-game markdown under `data/research/<game>.md` with a timestamp
frontmatter. At game init the adapter loads and truncates this file into a
short hint block that biases System 1 prompts.

Refresh is explicit via `scripts/fetch_research.py`, not per-round — the
content is reference material, not live state.
"""

import os
import re
import time
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)

_DEFAULT_ROOT = Path("data") / "research"
_FM = "---"


class ResearchStore:
    def __init__(self, game: str, root: Optional[Path] = None):
        self.game = game
        self.root = Path(root or _DEFAULT_ROOT)
        self.path = self.root / f"{game}.md"

    def exists(self) -> bool:
        return self.path.exists()

    def load(self) -> Optional[dict]:
        if not self.path.exists():
            return None
        text = self.path.read_text(encoding="utf-8")
        meta, body = _split_frontmatter(text)
        return {"meta": meta, "body": body}

    def age_days(self) -> Optional[float]:
        data = self.load()
        if data is None:
            return None
        ts = data.get("meta", {}).get("fetched_at")
        try:
            return (time.time() - float(ts)) / 86400.0 if ts else None
        except Exception:
            return None

    def is_stale(self, refresh_days: int) -> bool:
        age = self.age_days()
        return age is None or age > refresh_days

    def save(self, body: str, queries: List[str], source: str) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        meta = {
            "game": self.game,
            "fetched_at": time.time(),
            "source": source,
            "queries": queries,
        }
        text = _render_frontmatter(meta) + body.rstrip() + "\n"
        self.path.write_text(text, encoding="utf-8")
        logger.info(f"[RESEARCH] wrote {self.path} ({len(body)} chars)")

    def load_for_prompt(self, max_chars: int) -> str:
        """Return body trimmed to max_chars, URLs stripped. Empty string if absent."""
        data = self.load()
        if data is None:
            return ""
        return _format_for_prompt(data["body"], max_chars)


def fetch_with_gemini(
    client,
    game: str,
    queries: List[str],
) -> Optional[str]:
    """Ask Gemini (text mode) to search the web and produce a tight Chinese digest.

    Returns markdown body (without frontmatter), or None on failure. Caller
    writes via ResearchStore.save(). No grounding-tool handshake yet — the
    prompt just instructs Gemini to act as if it's summarizing web results;
    upgrading to an explicit google_search tool call is future work.
    """
    if client is None:
        logger.warning("[RESEARCH] no Gemini client; cannot fetch")
        return None
    query_block = "\n".join(f"- {q}" for q in queries)
    prompt = (
        f"请到网上搜索手机游戏《{game}》的最新玩法攻略，聚焦以下 query：\n"
        f"{query_block}\n\n"
        f"以中文 markdown 整理：3-5 个小节，每节列 3-8 条简短要点（每条 < 40 字），"
        f"侧重建筑/研究升级顺序、任务优先级、常见卡点与避坑建议。不要抄正文段落，"
        f"只提炼经验。避免投机/付费讨论。若有出处，在每节末尾标注。"
    )
    try:
        md = client.analyze_text(prompt)
    except Exception as e:
        logger.warning(f"[RESEARCH] fetch failed: {e}")
        return None
    return md.strip() if md else None


def _split_frontmatter(text: str) -> Tuple[dict, str]:
    if not text.startswith(f"{_FM}\n"):
        return {}, text
    rest = text[len(f"{_FM}\n"):]
    end = rest.find(f"\n{_FM}\n")
    if end < 0:
        return {}, text
    meta_text = rest[:end]
    body = rest[end + len(f"\n{_FM}\n"):]
    try:
        meta = yaml.safe_load(meta_text) or {}
    except Exception:
        meta = {}
    return meta, body


def _render_frontmatter(meta: dict) -> str:
    return f"{_FM}\n{yaml.safe_dump(meta, allow_unicode=True, sort_keys=False).rstrip()}\n{_FM}\n\n"


def _format_for_prompt(body: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    cleaned = _strip_urls(body)
    if len(cleaned) <= max_chars:
        return cleaned
    clipped = cleaned[:max_chars]
    tail = clipped.rfind("\n")
    return (clipped[:tail] if tail > 0 else clipped) + "\n……"


def _strip_urls(md: str) -> str:
    md = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", md)
    md = re.sub(r"https?://\S+", "", md)
    return md
