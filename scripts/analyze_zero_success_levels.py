#!/usr/bin/env python
"""Render + diagnose why a bucket's test levels are hard (0% success).

For each sampled level: render a PNG, check solvability, and measure the
spawn->switch->exit ROUND-TRIP route length via the graph path cost (the
long-horizon driver). Prints a per-level table and a per-bucket horizon summary
so we can see *why* one bucket is 0% while another succeeds.
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle

import numpy as np


def _round_trip_cost(map_data):
    """Graph path cost spawn->switch + switch->exit (px), or None if unreachable.

    Uses the same reachability graph the env/solvability uses, so it reflects
    physics-feasible travel, not euclidean."""
    from nclone.map_generation.solvability import _build_level_data, _reachable_set
    from nclone.graph.reachability.level_data_helpers import extract_goal_positions
    from nclone.graph.reachability.node_lookup import find_closest_node_to_position
    from nclone.graph.reachability.path_distance_calculator import (
        CachedPathDistanceCalculator,
    )

    try:
        ld = _build_level_data(list(map_data))
        reachable, adjacency = _reachable_set(ld)
        if not adjacency:
            return None
        goals = extract_goal_positions(ld)
        # goals = [(world_xy, id), ...]; switch first, exit second by convention
        calc = CachedPathDistanceCalculator()
        spawn = ld.start_position
        pts = [spawn] + [g for g, _ in goals]
        total = 0.0
        for a, b in zip(pts[:-1], pts[1:]):
            d = calc.get_path_cost(
                (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), adjacency, adjacency
            )
            if not np.isfinite(d):
                return None
            total += d
        return total
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="datasets/mechanic_benchmark_v3/test")
    ap.add_argument("--buckets", default="FAR_OBJECTIVE,FLAT_RUN")
    ap.add_argument("--per-tier", type=int, default=2, help="levels per tier to render")
    ap.add_argument("--out", default="/tmp/zero_success")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    from nclone.map_generation.solvability import is_map_solvable
    from nclone.map_generation.compose.visualize import render_png
    from nclone.map_generation.map_visualizer import parse_entities_from_map_data

    summary = {}
    print(f"{'level':16s} {'bucket':14s} {'tier':4s} {'solv':5s} "
          f"{'roundtrip_px':12s} render")
    for bucket in args.buckets.split(","):
        costs = []
        for tier_dir in sorted(glob.glob(os.path.join(args.root, bucket, "T*"))):
            tier = os.path.basename(tier_dir)
            pkls = sorted(glob.glob(os.path.join(tier_dir, "*.pkl")))[: args.per_tier]
            for p in pkls:
                d = pickle.load(open(p, "rb"))
                md = list(d["map_data"])
                lid = str(d["level_id"])[:14]
                solv = is_map_solvable(md)
                cost = _round_trip_cost(md)
                if cost is not None:
                    costs.append(cost)
                png = os.path.join(args.out, f"{bucket}_{tier}_{lid}.png")
                ok = ""
                try:
                    render_png(md, png)
                    ok = os.path.basename(png)
                except Exception as exc:
                    ok = f"render-fail({exc})"
                cs = f"{cost:.0f}" if cost is not None else "UNREACHABLE"
                print(f"{lid:16s} {bucket:14s} {tier:4s} {str(solv):5s} {cs:12s} {ok}")
        summary[bucket] = costs

    print("\n=== ROUND-TRIP HORIZON BY BUCKET (graph px, the difficulty driver) ===")
    for bucket, costs in summary.items():
        if costs:
            a = np.array(costs)
            print(f"{bucket:14s} n={len(a):2d}  min={a.min():.0f}  "
                  f"median={np.median(a):.0f}  max={a.max():.0f}  mean={a.mean():.0f}")
        else:
            print(f"{bucket:14s} (no finite round-trips)")
    print(f"\nimages -> {args.out}")


if __name__ == "__main__":
    main()
