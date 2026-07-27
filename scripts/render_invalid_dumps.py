#!/usr/bin/env python
"""Re-render the JSON dumps written by ``invalid_value_capture`` into PNGs.

In-worker PNG rendering is best-effort (a second GL renderer inside a training
worker can fail), but every dump always carries the full ``map_data``, so this
re-renders them cleanly offline and prints a triage summary: for each dump it
re-runs ``is_map_solvable`` to tell you at a glance whether the captured level is
genuinely unsolvable (bad procgen) or solvable (pointing at a graph/grid logic
error in how reachability was computed for that spawn).

Usage:
  python -m scripts.render_invalid_dumps --dir $NCLONE_GRAPH_CACHE_DIR/invalid_dumps
"""
from __future__ import annotations

import argparse
import glob
import json
import os


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", required=True, help="directory of *.json invalid dumps")
    ap.add_argument("--no-render", action="store_true", help="triage only, skip PNGs")
    args = ap.parse_args()

    jsons = sorted(glob.glob(os.path.join(args.dir, "*.json")))
    if not jsons:
        print(f"no dumps in {args.dir}")
        return

    from nclone.map_generation.solvability import is_map_solvable
    render_png = None
    if not args.no_render:
        from nclone.map_generation.compose.visualize import render_png

    print(f"{len(jsons)} dumps in {args.dir}\n")
    print(f"{'dump':40s} {'reason':22s} {'solvable':9s} verdict")
    for jp in jsons:
        with open(jp) as f:
            d = json.load(f)
        md = d.get("map_data")
        reason = d.get("reason", "?")
        name = os.path.basename(jp)[:38]
        try:
            solv = is_map_solvable(list(md)) if md else None
        except Exception:
            solv = None
        verdict = (
            "BAD PROCGEN (unsolvable)" if solv is False
            else "graph/grid logic? (solvable)" if solv is True
            else "unknown"
        )
        print(f"{name:40s} {reason:22s} {str(solv):9s} {verdict}")
        if render_png is not None and md:
            png = jp[:-5] + ".png"
            try:
                render_png(list(md), png)
            except Exception as exc:
                print(f"    render failed: {exc}")
    print(f"\nPNGs written alongside each .json in {args.dir}" if render_png else "")


if __name__ == "__main__":
    main()
