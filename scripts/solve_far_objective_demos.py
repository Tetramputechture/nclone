#!/usr/bin/env python
"""Offline batch 'route-and-solve' over the FAR_OBJECTIVE training pool.

For each FAR_OBJECTIVE level (datasets/mechanic_benchmark_v3/train/FAR_OBJECTIVE/
T*/*.pkl) this runs the two-stage round-trip solver
(:func:`scripts._far_objective_solver.solve_far_objective`) to produce a
PHYSICS-VALID winning action sequence (spawn -> switch -> activate -> exit ->
WIN), VALIDATED by replaying on the native sim and confirming has_won().

Outputs, per validated win:
  * a ``<out>/<level_id>.replay`` file (``CompactReplay.to_binary``) — directly
    consumable by npp-rl's ``FiGARBCDataset`` / DAPG demo loader. The
    ``input_sequence`` is N++ input bytes (0-7) and ``success=True``.
  * a disk-cache entry (via far_objective_demo_cache), so re-runs are free and
    the same demos are reachable from the env reset path.

Results are also written to ``<out>/manifest.jsonl`` (one row per attempted
level: level_id, tier, preset, won, frames, solve_seconds).

Parallelized with a process pool; each worker solves one level (the RRTs are
single-threaded, so workers scale near-linearly until you saturate cores).

Usage:
  python -m scripts.solve_far_objective_demos --out /tmp/far_demos \
      --tiers T1,T2,T3 --num-workers 16 --limit-per-tier 50
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import time
from datetime import datetime
from typing import List, Optional

POOL_ROOT = "datasets/mechanic_benchmark_v3/train/FAR_OBJECTIVE"

# Tier-appropriate RRT budgets (Stage A switch RRT / Stage B exit RRT iters).
TIER_BUDGET = {
    "T1": dict(n_states=200, a_max_iter=80000, b_max_iter=120000),
    "T2": dict(n_states=300, a_max_iter=120000, b_max_iter=180000),
    "T3": dict(n_states=300, a_max_iter=160000, b_max_iter=240000),
    "T4": dict(n_states=400, a_max_iter=220000, b_max_iter=320000),
}


def _list_levels(pool_root: str, tiers: List[str],
                 limit_per_tier: Optional[int]):
    items = []
    for t in tiers:
        fs = sorted(glob.glob(os.path.join(pool_root, t, "*.pkl")))
        if limit_per_tier is not None:
            fs = fs[:limit_per_tier]
        items.extend((t, f) for f in fs)
    return items


def _solve_one(args):
    tier, pkl_path, out_dir, seed, prefix = args
    import logging
    logging.disable(logging.CRITICAL)  # silence worker noise
    from scripts._far_objective_solver import solve_far_objective
    from nclone.gym_environment.far_objective_demo_cache import get_or_solve

    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    map_data = list(d["map_data"])
    level_id = str(d["level_id"])
    preset = d.get("preset")
    budget = TIER_BUDGET.get(tier, TIER_BUDGET["T4"])

    t0 = time.time()
    # Route through the cache so the on-disk demo cache is populated as a
    # side-effect (env reset path can reuse it), then write the .replay too.
    demo = get_or_solve(map_data, seed=seed, **budget)
    dt = time.time() - t0

    won = bool(demo and demo.get("won"))
    frames = demo.get("frames") if won else None
    replay_path = None
    if won:
        replay_path = _write_replay(out_dir, level_id, map_data,
                                    demo["actions"], prefix)
    return {
        "level_id": level_id, "tier": tier, "preset": preset,
        "won": won, "frames": frames, "solve_seconds": round(dt, 2),
        "replay": replay_path,
    }


def _write_replay(out_dir: str, level_id: str, map_data, actions: List[int],
                  prefix: str = "") -> str:
    from nclone.replay.gameplay_recorder import CompactReplay
    from nclone.gym_environment.start_state_curriculum import ACT

    # Encode via the SOLVER's action table (ACT: 1=RIGHT, 2=LEFT), NOT
    # map_action_to_input (which assumes the env convention 1=LEFT, 2=RIGHT) —
    # the mismatch silently wrote L/R-swapped tapes that die on replay.
    # Input byte bit flags: bit0 = jump, bit1 = right, bit2 = left.
    def _to_input(a: int) -> int:
        h, j = ACT.get(int(a), (0, 0))
        return (1 if j else 0) | (2 if h > 0 else 0) | (4 if h < 0 else 0)

    input_sequence = [_to_input(a) for a in actions]
    now = datetime.now()
    # ``prefix`` (e.g. "loop_corridor_") flows into the FILENAME so
    # NPP_RL_DEMO_PRIORITY_PREFIXES can pin the pack under the DAPG replay cap
    # (V13_NEXT_STEPS A2). Empty prefix keeps the legacy far_objective naming.
    stem = f"{prefix}{level_id}" if prefix else level_id
    replay = CompactReplay(
        episode_id=f"{prefix or 'far_objective_'}{level_id}",
        map_data=bytes(int(b) & 0xFF for b in map_data),
        input_sequence=input_sequence,
        level_id=level_id,
        start_time=now,
        end_time=now,
        success=True,
    )
    path = os.path.join(out_dir, f"{stem}.replay")
    with open(path, "wb") as f:
        f.write(replay.to_binary())
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="output dir for .replay demos + manifest")
    ap.add_argument("--tiers", default="T1,T2,T3,T4", help="comma list e.g. T1,T2")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--limit-per-tier", type=int, default=None,
                    help="solve at most N levels per tier (subset run)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pool-root", default=POOL_ROOT,
                    help="benchmark bucket dir containing T*/ subdirs "
                         "(default: the FAR_OBJECTIVE train pool)")
    ap.add_argument("--prefix", default="",
                    help="replay filename prefix (e.g. loop_corridor_) so "
                         "NPP_RL_DEMO_PRIORITY_PREFIXES can pin the pack")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tiers = args.tiers.split(",")
    levels = _list_levels(args.pool_root, tiers, args.limit_per_tier)
    print(f"Solving {len(levels)} levels from {args.pool_root} over tiers "
          f"{tiers} with {args.num_workers} workers", flush=True)

    work = [(t, f, args.out, args.seed, args.prefix) for t, f in levels]
    manifest_path = os.path.join(args.out, "manifest.jsonl")
    results = []
    t_start = time.time()

    import multiprocessing
    with open(manifest_path, "w") as mf:
        if args.num_workers <= 1:
            it = map(_solve_one, work)
        else:
            pool = multiprocessing.Pool(processes=args.num_workers, maxtasksperchild=4)
            it = pool.imap_unordered(_solve_one, work)
        done = 0
        for r in it:
            results.append(r)
            mf.write(json.dumps(r) + "\n")
            mf.flush()
            done += 1
            if done % 10 == 0 or done == len(work):
                nw = sum(1 for x in results if x["won"])
                print(f"  {done}/{len(work)} done, {nw} won "
                      f"({time.time()-t_start:.0f}s)", flush=True)
        if args.num_workers > 1:
            pool.close(); pool.join()

    # Summary by tier.
    print("\n=== WIN RATE BY TIER ===", flush=True)
    by_tier = {}
    for r in results:
        by_tier.setdefault(r["tier"], []).append(r)
    for t in sorted(by_tier):
        rs = by_tier[t]
        nw = sum(1 for r in rs if r["won"])
        fr = sorted(r["frames"] for r in rs if r["won"])
        fr_s = (f"len {fr[0]}-{fr[-1]} med {fr[len(fr)//2]}") if fr else "-"
        print(f"{t}: {nw}/{len(rs)} won | demo {fr_s}", flush=True)
    total_w = sum(1 for r in results if r["won"])
    print(f"\nTOTAL: {total_w}/{len(results)} demos produced "
          f"in {time.time()-t_start:.0f}s -> {args.out}", flush=True)
    print(f"manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
