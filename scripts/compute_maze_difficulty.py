#!/usr/bin/env python3
"""v17 P3: precompute a per-level dead-end "easiness" prior for the maze buckets.

For each level in the given buckets, build the free-space grid and count dead-end
cells (free cells with <=1 free 4-neighbor, inside the wall bbox) — the geometric
proxy that, in our analysis, was the SOLE separator between solved and never-solved
CHAMBER_MAZE levels (never-solved had ~2x the dead-ends). Writes

  <dataset_root>/train/maze_difficulty.json  ->  {level_id: {deadends, easiness}}

where ``easiness`` in [0, 1] is 1 - (deadends / bucket_p95_deadends), clipped — 1.0
for the lowest-branching mazes, ~0 for the most tortuous. The training-side
PhaseLearnabilityWeighter blends this (annealed by progress) so early training
favors low-dead-end mazes within each (bucket, tier) row.

Usage:
  python scripts/compute_maze_difficulty.py \
      --dataset-root datasets/mechanic_benchmark_v7_official \
      --buckets CHAMBER_MAZE [--split train]
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from nclone.value_function.free_space_distance import FreeSpaceDistanceSource


def _deadends(map_data) -> int:
    src = FreeSpaceDistanceSource(list(map_data))
    free = src._free
    solid = ~free
    sfx, sfy = np.where(solid)
    if not sfx.size:
        return 0
    x0, x1 = sfx.min() + 1, sfx.max() - 1
    y0, y1 = sfy.min() + 1, sfy.max() - 1
    if x1 <= x0 or y1 <= y0:
        return 0
    sub = free[x0:x1 + 1, y0:y1 + 1]
    nbr = np.zeros_like(sub, dtype=np.int16)
    nbr[1:, :] += sub[:-1, :]
    nbr[:-1, :] += sub[1:, :]
    nbr[:, 1:] += sub[:, :-1]
    nbr[:, :-1] += sub[:, 1:]
    return int(np.sum(sub & (nbr <= 1)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--buckets", default="CHAMBER_MAZE",
                    help="Comma-separated bucket names.")
    ap.add_argument("--split", default="train")
    args = ap.parse_args()

    root = Path(args.dataset_root).expanduser().resolve()
    buckets = [b.strip() for b in args.buckets.split(",") if b.strip()]

    # deadends per level, grouped by (bucket, tier) — the weighter row-normalizes
    # per (bucket, tier), so easiness is a within-row rank to align exactly.
    per_level: dict[str, int] = {}
    by_row: dict[tuple, list[str]] = {}
    for bucket in buckets:
        bdir = root / args.split / bucket
        pkls = sorted(bdir.rglob("*.pkl"))
        print(f"{bucket}: {len(pkls)} levels")
        for i, p in enumerate(pkls):
            try:
                rec = pickle.load(open(p, "rb"))
                de = _deadends(rec["map_data"])
                tier = int(rec.get("tier", 0))
            except Exception as exc:  # noqa: BLE001
                print(f"  skip {p.stem}: {exc}")
                continue
            per_level[p.stem] = de
            by_row.setdefault((bucket, tier), []).append(p.stem)
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(pkls)}")

    # easiness = within-(bucket, tier) rank of ASCENDING dead-ends -> [0, 1],
    # 1.0 = fewest dead-ends in its row, 0.0 = most. Rank-based so the noise-heavy
    # multi_chamber outliers can't compress the small-maze population; the spread
    # is uniform within each row (exactly where the weighter applies it).
    out: dict[str, dict] = {}
    for (bucket, tier), ids in by_row.items():
        order = sorted(ids, key=lambda i: per_level[i])  # ascending deadends
        n = len(order)
        for rank, lid in enumerate(order):
            # rank 0 (fewest) -> easiness 1.0; last -> 0.0.
            easiness = 1.0 - (rank / (n - 1)) if n > 1 else 1.0
            out[lid] = {"deadends": per_level[lid], "easiness": round(easiness, 4)}
        des = sorted(per_level[i] for i in ids)
        print(f"{bucket} T{tier}: n={n}  deadends min/med/max="
              f"{des[0]}/{des[n // 2]}/{des[-1]}")

    dest = root / args.split / "maze_difficulty.json"
    dest.write_text(json.dumps(out))
    print(f"wrote {len(out)} levels -> {dest}")


if __name__ == "__main__":
    main()
