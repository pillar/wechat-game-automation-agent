#!/usr/bin/env python3
"""Export trajectory JSONL logs to DPO preference pairs.

DPO format (OpenAI / HuggingFace TRL standard):
    {"prompt": "...", "chosen": "...", "rejected": "..."}

Pairing rule:
    Within the same scene, pair a success trajectory (outcome in {success, changed, ok})
    against a failure trajectory (outcome in {no_change, rejected, stuck, verify_failed})
    that shared a similar prompt. The "chosen" is the model response that led to success;
    "rejected" is the response that led to failure.

Usage:
    python3 scripts/export_dpo.py --input data/trajectories/ --output data/dpo_pairs.jsonl
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict


SUCCESS_OUTCOMES = {"success", "changed", "ok", "verified"}
FAILURE_OUTCOMES = {"no_change", "rejected", "stuck", "verify_failed", "stale"}


def load_records(input_path: Path):
    if input_path.is_file():
        files = [input_path]
    else:
        files = sorted(input_path.glob("*.jsonl"))
    for f in files:
        with open(f, "r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def pair_records(records):
    by_scene = defaultdict(lambda: {"success": [], "failure": []})
    for r in records:
        scene = r.get("scene") or "unknown"
        outcome = r.get("outcome") or ""
        if outcome in SUCCESS_OUTCOMES:
            by_scene[scene]["success"].append(r)
        elif outcome in FAILURE_OUTCOMES:
            by_scene[scene]["failure"].append(r)

    pairs = []
    for scene, groups in by_scene.items():
        successes = groups["success"]
        failures = groups["failure"]
        for s in successes:
            if not s.get("prompt") or not s.get("response"):
                continue
            best = None
            for f in failures:
                if f.get("prompt") == s.get("prompt"):
                    best = f
                    break
            if best is None and failures:
                best = failures[0]
            if best is None:
                continue
            pairs.append({
                "prompt": s["prompt"],
                "chosen": s["response"],
                "rejected": best["response"],
                "scene": scene,
            })
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("data/trajectories"))
    ap.add_argument("--output", type=Path, default=Path("data/dpo_pairs.jsonl"))
    args = ap.parse_args()

    if not args.input.exists():
        print(f"no input: {args.input}", file=sys.stderr)
        sys.exit(1)

    records = list(load_records(args.input))
    print(f"loaded {len(records)} records")
    pairs = pair_records(records)
    print(f"built {len(pairs)} DPO pairs")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fp:
        for p in pairs:
            fp.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
