#!/usr/bin/env python3
"""Render never-solved benchmark levels to PNGs for offline validation.

Given a bucket and a JSON list of never-solved level_ids (from the v16 TB
per_level/<lid>/total_solves == 0 telemetry), pulls each level's map_data from
the benchmark dataset pkl and renders a ground-truth PNG via the headless
renderer. A FRESH NPlayHeadless per map (render_png) so the renderer's tile-surface
cache never bleeds the previous map's geometry (see render-tile-cache-bleed).

Usage:
  render_never_solved.py --bucket CHAMBER_MAZE --ids zero_solve_ids.json \
      --out /path/out --limit 12
If --ids is omitted, renders a random sample of the bucket (any solve status).
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
from pathlib import Path

from nclone.map_generation.compose.visualize import render_png

DATASET = Path("datasets/mechanic_benchmark_v7_official/train")


def _index_bucket(bucket: str) -> dict[str, Path]:
    """level_id (pkl stem) -> pkl path, across all tiers of the bucket."""
    root = DATASET / bucket
    return {p.stem: p for p in root.rglob("*.pkl")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", default="CHAMBER_MAZE")
    ap.add_argument("--ids", default=None, help="JSON list of never-solved level_ids")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    index = _index_bucket(args.bucket)
    print(f"{args.bucket}: {len(index)} levels in dataset")

    if args.ids:
        want = set(json.loads(Path(args.ids).read_text()))
        ids = [lid for lid in index if lid in want]
        print(f"never-solved ids matching this bucket: {len(ids)}")
    else:
        ids = list(index)

    rng = random.Random(args.seed)
    rng.shuffle(ids)
    ids = ids[: args.limit]

    for lid in ids:
        with open(index[lid], "rb") as fh:
            rec = pickle.load(fh)
        tier = rec.get("tier", "T?")
        gen = rec.get("generator", "?")
        preset = rec.get("preset", "?")
        map_data = rec["map_data"]
        name = f"{args.bucket}_{tier}_{gen}_{preset}_{lid}.png"
        render_png(list(map_data), str(out / name))
        print(f"  rendered {name}")

    print(f"done -> {out}")


if __name__ == "__main__":
    main()
