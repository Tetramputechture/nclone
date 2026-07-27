#!/usr/bin/env python3
"""Quantify what distinguishes never-solved CHAMBER_MAZE levels from solved ones.

For each level: build the free-space d* fields the reward/obs pipeline actually
uses (FreeSpaceDistanceSource), plus a headless sim for the ninja spawn, and
compute geometry/pipeline diagnostics:

  * switch/exit d*-reachability (is the dense potential even defined at spawn?)
  * spawn->switch d* path length (px) and straight-line dist -> tortuosity
    (how much the maze forces a detour AWAY from the switch direction)
  * playable maze bounding box vs the full 1008x552 field (the "tiny maze in a
    huge empty field" spatial-resolution hypothesis: global obs channels spend
    their resolution on empty space)
  * free-cell fraction inside the maze bbox (corridor density)
  * spawn-in-open vs spawn-in-maze

Prints solved-vs-unsolved distribution deltas so the pipeline gap is data-driven.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from nclone.value_function.free_space_distance import FreeSpaceDistanceSource
from nclone.nplay_headless import NPlayHeadless

DATASET = Path("datasets/mechanic_benchmark_v7_official/train")
_CELL = 6.0  # free-space grid cell px (TILE_PIXEL_SIZE/4)
FIELD_W, FIELD_H = 1008.0, 552.0


def _spawn(map_data):
    """Ninja spawn px via a fresh headless sim (robust to entity layout)."""
    nplay = NPlayHeadless(render_mode="rgb_array", enable_animation=False,
                          enable_logging=False, enable_debug_overlay=False)
    nplay.load_map_from_map_data(list(map_data))
    nplay.reset()
    p = nplay.ninja_position()
    return float(p[0]), float(p[1])


def _sample_cost(field, x, y):
    ix, iy = int(x / _CELL), int(y / _CELL)
    if 0 <= ix < field.shape[0] and 0 <= iy < field.shape[1]:
        # nearest finite in a small ring (spawn cell may be sub-solid)
        best = np.inf
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                jx, jy = ix + dx, iy + dy
                if 0 <= jx < field.shape[0] and 0 <= jy < field.shape[1]:
                    v = field[jx, jy]
                    if np.isfinite(v) and v < best:
                        best = v
        return best
    return np.inf


def analyze(map_data):
    src = FreeSpaceDistanceSource(list(map_data))
    out = {"available": bool(src.available)}
    if not src.available or src.switch is None:
        return out
    sx, sy = src.switch
    ex, ey = src.exit
    nx, ny = _spawn(map_data)
    out["spawn"] = (nx, ny)
    out["switch"] = (sx, sy)

    d_sw = _sample_cost(src._d_switch, nx, ny)
    out["switch_reachable"] = bool(np.isfinite(d_sw))
    out["dstar_spawn_to_switch"] = float(d_sw) if np.isfinite(d_sw) else -1.0
    straight = float(((sx - nx) ** 2 + (sy - ny) ** 2) ** 0.5)
    out["straight_spawn_to_switch"] = straight
    out["tortuosity"] = (float(d_sw) / straight) if (np.isfinite(d_sw) and straight > 1) else -1.0

    free = src._free
    # SOLID maze structure: the wall cells (the actual maze). The surrounding
    # empty playfield is "free" too, so measure the SOLID bbox to size the maze.
    solid = ~free
    sfx, sfy = np.where(solid)
    if sfx.size:
        bw = (sfx.max() - sfx.min()) * _CELL
        bh = (sfy.max() - sfy.min()) * _CELL
        out["maze_wall_bbox_px"] = (float(bw), float(bh))
        out["maze_wall_frac_of_field"] = float((bw * bh) / (FIELD_W * FIELD_H))
        # Is the switch inside the wall cluster (occluded) vs out in the open?
        out["switch_in_wall_bbox"] = bool(
            sfx.min() * _CELL <= sx <= sfx.max() * _CELL
            and sfy.min() * _CELL <= sy <= sfy.max() * _CELL
        )
        out["spawn_in_wall_bbox"] = bool(
            sfx.min() * _CELL <= nx <= sfx.max() * _CELL
            and sfy.min() * _CELL <= ny <= sfy.max() * _CELL
        )
    # Dead-end density: free cells with <=1 free 4-neighbor = maze tips the agent
    # must explore and back out of. Restrict to the wall-bbox region so the huge
    # open field's border cells don't dominate.
    if sfx.size:
        x0, x1 = sfx.min(), sfx.max()
        y0, y1 = sfy.min(), sfy.max()
        sub = free[x0:x1 + 1, y0:y1 + 1]
        nbr = np.zeros_like(sub, dtype=np.int16)
        nbr[1:, :] += sub[:-1, :]
        nbr[:-1, :] += sub[1:, :]
        nbr[:, 1:] += sub[:, :-1]
        nbr[:, :-1] += sub[:, 1:]
        deadends = int(np.sum(sub & (nbr <= 1)))
        free_in_bbox = int(sub.sum())
        out["deadend_cells"] = deadends
        out["deadend_frac"] = float(deadends / max(1, free_in_bbox))
        out["free_in_wall_bbox"] = free_in_bbox
    return out


def summarize(rows, label):
    def col(k):
        return [r[k] for r in rows if k in r and isinstance(r.get(k), (int, float)) and r[k] >= 0]
    n = len(rows)
    reach = sum(1 for r in rows if r.get("switch_reachable"))
    avail = sum(1 for r in rows if r.get("available"))
    print(f"\n=== {label}  (n={n}) ===")
    print(f"  free-space available: {avail}/{n}   switch d*-reachable: {reach}/{n}")
    sw_occ = sum(1 for r in rows if r.get("switch_in_wall_bbox"))
    sp_in = sum(1 for r in rows if r.get("spawn_in_wall_bbox"))
    print(f"  switch inside maze-wall bbox: {sw_occ}/{n}   spawn inside: {sp_in}/{n}")
    for k in ("dstar_spawn_to_switch", "straight_spawn_to_switch", "tortuosity",
              "maze_wall_frac_of_field", "deadend_cells", "deadend_frac",
              "free_in_wall_bbox"):
        vals = col(k)
        if vals:
            a = np.array(vals)
            print(f"  {k:26s} median={np.median(a):8.3f}  p10={np.percentile(a,10):8.3f}  p90={np.percentile(a,90):8.3f}")
    bw = [r["maze_wall_bbox_px"][0] for r in rows if "maze_wall_bbox_px" in r]
    bh = [r["maze_wall_bbox_px"][1] for r in rows if "maze_wall_bbox_px" in r]
    if bw:
        print(f"  maze_wall_bbox_w median={np.median(bw):.0f}px  h median={np.median(bh):.0f}px "
              f"(field 1008x552)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", default="CHAMBER_MAZE")
    ap.add_argument("--zero-ids", required=True)
    ap.add_argument("--solves", default=None, help="all_level_solves.json (>0 = solved)")
    ap.add_argument("--limit", type=int, default=60)
    args = ap.parse_args()

    zero = set(json.load(open(args.zero_ids)))
    index = {p.stem: p for p in (DATASET / args.bucket).rglob("*.pkl")}
    solved_ids = None
    if args.solves:
        sd = json.load(open(args.solves))
        solved_ids = {k for k, v in sd.items() if v > 0}

    never = [lid for lid in index if lid in zero]
    if solved_ids is not None:
        solved = [lid for lid in index if lid in solved_ids]
    else:
        solved = [lid for lid in index if lid not in zero]
    import random
    random.Random(0).shuffle(never)
    random.Random(1).shuffle(solved)
    never, solved = never[: args.limit], solved[: args.limit]

    def run(ids):
        rows = []
        for lid in ids:
            rec = pickle.load(open(index[lid], "rb"))
            try:
                r = analyze(rec["map_data"])
                r["tier"] = rec["tier"]
                rows.append(r)
            except Exception as e:  # noqa: BLE001
                print(f"  skip {lid}: {e}")
        return rows

    print(f"analyzing {len(never)} never-solved + {len(solved)} solved {args.bucket} ...")
    summarize(run(never), "NEVER-SOLVED")
    summarize(run(solved), "SOLVED")


if __name__ == "__main__":
    main()
