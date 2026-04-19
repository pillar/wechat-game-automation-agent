#!/usr/bin/env python3
"""Fetch game guide summary via Gemini, cache to data/research/<game>.md.

Usage:
    python3 scripts/fetch_research.py --game endless_winter
    python3 scripts/fetch_research.py --game endless_winter --force
    python3 scripts/fetch_research.py --game endless_winter --queries "无尽冬日 联盟" "无尽冬日 钻石"
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.ai_client import GeminiVisionClient
from core.research import ResearchStore, fetch_with_gemini
from utils.config_loader import load_game_config


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="endless_winter")
    ap.add_argument("--queries", nargs="*", help="override queries from yaml")
    ap.add_argument("--force", action="store_true", help="fetch even if cache is fresh")
    args = ap.parse_args()

    cfg = load_game_config(args.game)
    d = cfg.model_dump() if hasattr(cfg, "model_dump") else cfg.dict()
    rcfg = (d.get("research") or {}) if isinstance(d, dict) else {}
    queries = args.queries or rcfg.get("queries") or [
        f"{args.game} 新手攻略",
        f"{args.game} 建筑升级顺序",
        f"{args.game} 任务优先级",
    ]
    refresh_days = int(rcfg.get("refresh_days", 30))

    store = ResearchStore(args.game)
    if not args.force and not store.is_stale(refresh_days):
        age = store.age_days()
        print(f"cache fresh ({age:.1f} days < {refresh_days}); use --force to refetch")
        return 0

    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        print("GEMINI_API_KEY required (set in .env)", file=sys.stderr)
        return 1
    client = GeminiVisionClient(
        api_key=key,
        model=os.environ.get("GEMINI_MODEL", "gemini-flash-latest"),
    )
    body = fetch_with_gemini(client, args.game, queries)
    if not body:
        print("fetch failed", file=sys.stderr)
        return 1
    store.save(body=body, queries=queries, source="gemini")
    print(f"wrote {store.path} ({len(body)} chars)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
