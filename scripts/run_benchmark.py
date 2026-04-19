#!/usr/bin/env python3
"""Grounding-accuracy benchmark runner.

Loads scenarios from tests/benchmark/scenarios.yaml, runs System 1
(Qwen VL / UI-TARS / CogAgent), checks whether the predicted (x,y)
falls inside `expected_region` (± tolerance_px).

Usage:
    python3 scripts/run_benchmark.py
    python3 scripts/run_benchmark.py --family ui_tars
    python3 scripts/run_benchmark.py --scenarios tests/benchmark/scenarios.yaml
"""

import argparse
import sys
from pathlib import Path

import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.ai_client import LocalVisionClient
from core.model_router import build_grounding_prompt, parse_grounding_response
from games.endless_winter.prompts import (
    build_system1_prompt_for_scene,
    parse_qwen_ref_bbox,
)


def inside(region, x, y, tol=0):
    xmin, ymin, xmax, ymax = region
    return (xmin - tol) <= x <= (xmax + tol) and (ymin - tol) <= y <= (ymax + tol)


def run(scenarios_path: Path, family: str, api_url: str, model: str, bench_root: Path):
    data = yaml.safe_load(scenarios_path.read_text(encoding="utf-8"))
    scenarios = data.get("scenarios", [])
    if not scenarios:
        print("no scenarios")
        return 1

    client = LocalVisionClient(api_base=api_url, model=model)

    hits = 0
    total = 0
    for s in scenarios:
        sp = bench_root / s["screenshot"]
        if not sp.exists():
            print(f"SKIP {s['id']}: screenshot missing ({sp})")
            continue
        img = Image.open(sp)
        prompt = build_grounding_prompt(
            family=family,
            img_width=img.width,
            img_height=img.height,
            scene=s.get("scene", "unknown"),
            recent_actions=[],
            qwen_prompt_builder=build_system1_prompt_for_scene,
        )
        response = client.analyze(img, prompt, max_retries=0)
        parsed = parse_grounding_response(
            family=family,
            response=response,
            img_width=img.width,
            img_height=img.height,
            qwen_parser=parse_qwen_ref_bbox,
        )
        total += 1
        if parsed is None:
            print(f"MISS {s['id']}: parse failed")
            continue
        x, y = parsed["x"], parsed["y"]
        ok = inside(s["expected_region"], x, y, s.get("tolerance_px", 0))
        print(f"{'HIT ' if ok else 'MISS'} {s['id']}: predicted=({x},{y}) expected={s['expected_region']}")
        if ok:
            hits += 1

    if total == 0:
        print("no scenarios were runnable")
        return 1
    print(f"\naccuracy: {hits}/{total} = {hits/total:.1%}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", type=Path, default=Path("tests/benchmark/scenarios.yaml"))
    ap.add_argument("--family", default="qwen_vl", choices=["qwen_vl", "ui_tars", "cogagent"])
    ap.add_argument("--api-url", default="http://192.168.1.156:1234")
    ap.add_argument("--model", default="qwen/qwen3-vl-8b")
    args = ap.parse_args()

    bench_root = args.scenarios.parent
    sys.exit(run(args.scenarios, args.family, args.api_url, args.model, bench_root))


if __name__ == "__main__":
    main()
