#!/usr/bin/env python
"""Validity sweep over EVERY level generator: valid spawn + reachable goals.

For each ``(generator_type, preset)`` combo exposed by GENERATOR_PRESETS (all
hardcoded ``map_*`` generators AND the auto-registered ``template_*`` composable
templates), generate N levels at distinct seeds and check two invariants the RL
env depends on:

  1. SPAWN VALID  - the ninja spawn cell is not a full-solid (buried) tile.
  2. GOALS REACHABLE - exit switch AND exit door are reachable from spawn
     (``is_map_solvable``, the jump-aware grid traversability graph).

Writes a per-combo failure report and records every failing ``(combo, seed)`` so
they can be visualized. Use ``--visualize`` to also dump PNGs of the failures.

Usage:
  python -m scripts.sweep_generator_validity --seeds 8
  python -m scripts.sweep_generator_validity --seeds 8 --visualize --out /tmp/genfail
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import numpy as np

from nclone.map_generation.generator_factory import GeneratorFactory, GENERATOR_PRESETS
from nclone.map_generation.solvability import is_map_solvable
from nclone.graph.level_data import extract_start_position_from_map_data

# Full-solid / wall tile id. A spawn whose cell is a full solid tile is "buried".
FULL_SOLID_TILE = 1

_NPLAY = None


def _nplay():
    global _NPLAY
    if _NPLAY is None:
        from nclone.nplay_headless import NPlayHeadless
        _NPLAY = NPlayHeadless(
            render_mode=None, enable_animation=False, enable_logging=False,
            enable_debug_overlay=False, enable_rendering=False,
        )
    return _NPLAY


def check_one(map_data):
    """Return dict(solvable, buried, spawn_cell, spawn_tile).

    Uses the SIM's own tile view (``get_tile_data``) — the same authority
    ``is_map_solvable`` parses — so the spawn-cell lookup matches the sim's
    bordered coordinate frame exactly (key = (sx//24, sy//24)).
    """
    buried = False
    spawn_tile = None
    cell = None
    try:
        sx, sy = extract_start_position_from_map_data(list(map_data))
        nh = _nplay()
        nh.load_map_from_map_data(list(map_data))
        nh.reset()
        td = nh.get_tile_data()
        cell = (int(sx // 24), int(sy // 24))
        spawn_tile = td.get(cell)
        # missing tile (off the bordered playfield) or full-solid wall == buried.
        buried = spawn_tile is None or int(spawn_tile) == FULL_SOLID_TILE
    except Exception:
        buried = True
    try:
        solvable = bool(is_map_solvable(list(map_data)))
    except Exception:
        solvable = False
    return {
        "solvable": solvable,
        "buried": buried,
        "spawn_cell": cell,
        "spawn_tile": spawn_tile,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=8, help="levels per combo")
    ap.add_argument("--start-seed", type=int, default=20000)
    ap.add_argument("--only", default=None, help="substring filter on gen_type")
    ap.add_argument("--visualize", action="store_true")
    ap.add_argument("--out", default="/tmp/genfail")
    args = ap.parse_args()

    combos = []
    for gt in sorted(GENERATOR_PRESETS):
        if args.only and args.only not in gt:
            continue
        for preset in GENERATOR_PRESETS[gt]:
            combos.append((gt, preset))

    print(f"Sweeping {len(combos)} combos x {args.seeds} seeds "
          f"= {len(combos)*args.seeds} levels\n", flush=True)

    results = {}
    failures = []  # (gen_type, preset, seed, reason, detail)
    for gt, preset in combos:
        n_solv = n_buried = n_err = 0
        for k in range(args.seeds):
            seed = args.start_seed + k
            try:
                m = GeneratorFactory.create_from_preset(gt, preset, seed=seed)
                m.generate(seed=seed)
                md = m.map_data()
            except Exception as exc:
                n_err += 1
                failures.append((gt, preset, seed, "gen_error", f"{type(exc).__name__}: {exc}"))
                continue
            r = check_one(md)
            if r["buried"]:
                n_buried += 1
                failures.append((gt, preset, seed, "spawn_buried",
                                 f"cell={r['spawn_cell']} tile={r['spawn_tile']}"))
            if r["solvable"]:
                n_solv += 1
            else:
                failures.append((gt, preset, seed, "unsolvable",
                                 f"cell={r['spawn_cell']} tile={r['spawn_tile']}"))
        key = f"{gt}/{preset}"
        results[key] = {
            "solvable": n_solv, "buried": n_buried, "errors": n_err, "n": args.seeds,
        }
        bad = (n_solv < args.seeds) or n_buried or n_err
        flag = "  <-- FAIL" if bad else ""
        print(f"{key:46s} solv {n_solv}/{args.seeds} buried {n_buried} err {n_err}{flag}",
              flush=True)

    # Summary
    n_combos = len(results)
    clean = sum(1 for v in results.values()
                if v["solvable"] == v["n"] and not v["buried"] and not v["errors"])
    print(f"\n=== {clean}/{n_combos} combos fully clean; "
          f"{n_combos-clean} have >=1 invalid generation ===", flush=True)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "validity_report.json"), "w") as f:
        json.dump({"results": results, "failures": failures}, f, indent=1)
    print(f"report -> {os.path.join(args.out, 'validity_report.json')} "
          f"({len(failures)} failing levels)", flush=True)

    if args.visualize and failures:
        _visualize(failures, args.out)


def _visualize(failures, out_dir):
    """Dump a PNG per UNIQUE failing combo (first failing seed) for inspection."""
    from nclone.map_generation.map_visualizer import visualize_map  # best-effort
    seen = set()
    n = 0
    for gt, preset, seed, reason, detail in failures:
        if reason == "gen_error" or (gt, preset) in seen:
            continue
        seen.add((gt, preset))
        try:
            m = GeneratorFactory.create_from_preset(gt, preset, seed=seed)
            m.generate(seed=seed)
            path = os.path.join(out_dir, f"{gt}__{preset}__{reason}__s{seed}.png")
            visualize_map(m.map_data(), path)
            n += 1
        except Exception as exc:
            print(f"viz failed {gt}/{preset}: {exc}", flush=True)
    print(f"visualized {n} unique failing combos -> {out_dir}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
