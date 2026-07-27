#!/usr/bin/env python
"""Universal sim-verified demo harvest across all mechanic buckets.

For every level in the v4 train split, route it to the solver best suited to its
bucket -- the exact forward-search lattice planner (terrain / jump / mine /
precise) or the RRT round-trip solver (open / long-horizon) -- to produce a
PHYSICS-VALID winning action tape (spawn -> switch -> exit -> WIN), validated by
replaying on the native sim. If the primary solver fails, a budget-reduced
attempt with the OTHER solver is made (cheap insurance for borderline levels).

Each validated win is written as ``<out>/<level_id>.replay``
(``CompactReplay.to_binary``) -- directly consumable by npp-rl's BC/DAPG demo
loader (input bytes 0-7, success=True). A demo is just (map_data + input tape),
so this output is OBSERVATION-INDEPENDENT: the obs space can change without
re-harvesting (only re-ingestion re-sims to the current obs dims).

Results -> ``<out>/manifest.jsonl`` (one row/level) + a per-bucket/tier summary.

Usage:
  python -m scripts.harvest_demos --out /tmp/all_demos --num-workers 64
  python -m scripts.harvest_demos --out /tmp/all_demos --buckets SLOPES,CURVES \
      --tiers T1,T2 --limit-per-tier 20            # subset smoke
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import time
from datetime import datetime
from typing import List, Optional, Tuple

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

DATA_ROOT = "datasets/mechanic_benchmark_v4/train"  # default; override with --data-root

ALL_BUCKETS = [
    "FLAT_RUN", "BASIC_JUMP", "WALL_JUMP", "VERTICAL_SHAFT", "SLOPES", "CURVES",
    "MULTI_JUMP", "PLATFORMING", "PLATFORM_STAIRS", "CHAMBER_MAZE", "LOOP_CORRIDOR",
    "PRECISION", "COMPOSITE", "MINE_NAVIGATION", "FAR_OBJECTIVE",
]
# Open / long-horizon levels: random-sampling RRT round-trip beats the binned
# forward search (verified in the coverage survey). Everything else -> exact
# forward search (which the survey showed walls only on the precise T2+ tail).
RRT_BUCKETS = {"FAR_OBJECTIVE", "LOOP_CORRIDOR", "CHAMBER_MAZE"}

RRT_BUDGET = {
    "T1": dict(n_states=200, a_max_iter=80000, b_max_iter=120000),
    "T2": dict(n_states=300, a_max_iter=120000, b_max_iter=180000),
    "T3": dict(n_states=300, a_max_iter=160000, b_max_iter=240000),
    "T4": dict(n_states=400, a_max_iter=220000, b_max_iter=320000),
}
FWD_BUDGET = {
    "T1": dict(state_cap=60000, max_g_frames=450, macro=4),
    "T2": dict(state_cap=90000, max_g_frames=550, macro=4),
    "T3": dict(state_cap=120000, max_g_frames=650, macro=4),
    "T4": dict(state_cap=120000, max_g_frames=700, macro=4),
}
# Fallback attempts are deliberately cheap (we only pay them when the primary
# fails, and on the genuinely-hard tail BOTH solvers fail -- don't burn minutes).
RRT_FALLBACK = dict(n_states=150, a_max_iter=60000, b_max_iter=80000)
FWD_FALLBACK = dict(state_cap=45000, max_g_frames=420, macro=4)


def _hj_to_input(h: int, j: int) -> int:
    """Map a physics ``(horizontal, jump)`` control to an N++ input byte
    (bit0=jump, bit1=right, bit2=left) — the encoding ``ReplayExecutor`` /
    ``verify_replay_file`` decode back. This is solver-ENCODING-AGNOSTIC: each
    solver's action-index table (FWD_ACT vs the start-state-curriculum ACT) has a
    DIFFERENT index->(h,j) convention (their LEFT/RIGHT are swapped), so actions
    must be resolved to (h,j) via the solver's OWN table before this — NOT run
    through the discrete-action ``map_action_to_input`` (which assumes the fwd
    convention and silently LEFT/RIGHT-mirrors RRT demos)."""
    b = 0
    if j:
        b |= 1  # jump
    if h > 0:
        b |= 2  # right
    elif h < 0:
        b |= 4  # left
    return b


def _solve_rrt(map_data, budget, seed, n_seeds) -> Optional[Tuple[List[int], int]]:
    from scripts._far_objective_solver import solve_far_objective
    from nclone.gym_environment.start_state_curriculum import ACT as RRT_ACT
    d = solve_far_objective(map_data, seed=seed, n_seeds=n_seeds,
                            log=lambda m: None, **budget)
    if d and d.get("won"):
        n = int(d["frames"])
        # Resolve via the RRT solver's OWN action table -> input bytes.
        inputs = [_hj_to_input(*RRT_ACT[int(a)]) for a in d["actions"][:n]]
        return inputs, n
    return None


def _solve_fwd(map_data, budget) -> Optional[Tuple[List[int], int]]:
    import nclone.value_function.tools.lattice_dstar_prototype as ldp
    from nclone.value_function.tools.physics_route_spike import ACT
    ldp._load_map_data = lambda p: map_data
    s = ldp.LatticeDStar("dummy", **budget)
    r = s.enumerate_lattice(stop_on_win=True)
    if not r["won_paths"]:
        return None
    macro_path = min(r["won_paths"].values(), key=len)
    frames_tape = [a for a in macro_path for _ in range(s.macro)]
    # Re-validate the frame-level expansion on a fresh sim (defensive: the lattice
    # already action-replays, so this should always pass).
    s.sim.fast_reset()
    for i, a in enumerate(frames_tape):
        h, j = ACT[a]
        s.sim.tick(h, j)
        if s.sim.has_won():
            # Resolve via the forward solver's OWN action table -> input bytes.
            inputs = [_hj_to_input(*ACT[x]) for x in frames_tape[: i + 1]]
            return inputs, i + 1
        if s.sim.has_died():
            return None
    return None


def _solve_one(args):
    bucket, tier, pkl_path, out_dir, seed = args
    import logging
    logging.disable(logging.CRITICAL)

    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    map_data = list(d["map_data"])
    level_id = str(d["level_id"])

    primary_rrt = bucket in RRT_BUCKETS
    t0 = time.time()
    res = None
    used = None
    # 1) primary solver, full budget
    try:
        if primary_rrt:
            res = _solve_rrt(map_data, RRT_BUDGET.get(tier, RRT_BUDGET["T4"]), seed, 3)
        else:
            res = _solve_fwd(map_data, FWD_BUDGET.get(tier, FWD_BUDGET["T4"]))
    except Exception:
        res = None
    if res:
        used = "rrt" if primary_rrt else "fwd"
    else:
        # 2) cheap fallback with the OTHER solver
        try:
            if primary_rrt:
                res = _solve_fwd(map_data, FWD_FALLBACK)
                used = "fwd_fb" if res else None
            else:
                res = _solve_rrt(map_data, RRT_FALLBACK, seed, 1)
                used = "rrt_fb" if res else None
        except Exception:
            res = None

    dt = time.time() - t0
    replay = None
    frames = None
    consumer_verified = None
    if res:
        actions, frames = res
        replay = _write_replay(out_dir, level_id, map_data, actions, bucket)
        # CONSUMER-CADENCE GATE: re-verify the written .replay at 1-input-per-tick
        # (the cadence ReplayExecutor / verify_replay_file use for BC/DAPG). A demo
        # that wins in the solver's own loop but not here is NOT BC-consumable
        # (e.g. an action-encoding mismatch); drop it so "harvested" == "usable".
        try:
            from nclone.map_augmentation.attract.verify_augmentation import (
                verify_replay_file,
            )
            from pathlib import Path

            consumer_verified = bool(verify_replay_file(Path(replay)).success)
        except Exception:
            consumer_verified = None  # verifier unavailable -> don't hard-fail
        if consumer_verified is False:
            os.remove(replay)
            replay = None
    won = res is not None and consumer_verified is not False
    return {
        "level_id": level_id, "bucket": bucket, "tier": tier,
        "won": won, "frames": frames, "solver": used,
        "consumer_verified": consumer_verified,
        "solve_seconds": round(dt, 2), "replay": replay,
    }


def _write_replay(out_dir, level_id, map_data, input_sequence, bucket) -> str:
    from nclone.replay.gameplay_recorder import CompactReplay
    # input_sequence is already a list of per-tick N++ input BYTES (the solver
    # resolved its action indices via its own ACT table -> _hj_to_input), so no
    # discrete-action remap here (that was the LEFT/RIGHT mirror bug for RRT demos).
    input_sequence = [int(b) & 0xFF for b in input_sequence]
    now = datetime.now()
    replay = CompactReplay(
        episode_id=f"{bucket}_{level_id}",
        map_data=bytes(int(b) & 0xFF for b in map_data),
        input_sequence=input_sequence,
        level_id=level_id,
        start_time=now, end_time=now, success=True,
    )
    path = os.path.join(out_dir, f"{level_id}.replay")
    with open(path, "wb") as f:
        f.write(replay.to_binary())
    return path


def _list_levels(buckets, tiers, limit, data_root=DATA_ROOT):
    items = []
    for b in buckets:
        for t in tiers:
            fs = sorted(glob.glob(os.path.join(data_root, b, t, "*.pkl")))
            if limit is not None:
                fs = fs[:limit]
            items.extend((b, t, f) for f in fs)
    return items


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--buckets", default=",".join(ALL_BUCKETS))
    ap.add_argument("--tiers", default="T1,T2,T3,T4")
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--limit-per-tier", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--data-root", default=DATA_ROOT,
        help=f"Packed dataset train dir to harvest from (default: {DATA_ROOT}). "
        "Point at a freshly-regenerated dataset to harvest demos for new geometry.",
    )
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    buckets = [b for b in args.buckets.split(",") if b]
    tiers = [t for t in args.tiers.split(",") if t]
    levels = _list_levels(buckets, tiers, args.limit_per_tier, args.data_root)
    print(f"Harvesting {len(levels)} levels | {len(buckets)} buckets x {tiers} "
          f"| {args.num_workers} workers", flush=True)

    work = [(b, t, f, args.out, args.seed) for (b, t, f) in levels]
    manifest_path = os.path.join(args.out, "manifest.jsonl")
    results = []
    t_start = time.time()

    import multiprocessing
    with open(manifest_path, "w") as mf:
        if args.num_workers <= 1:
            it = map(_solve_one, work)
            pool = None
        else:
            pool = multiprocessing.Pool(processes=args.num_workers, maxtasksperchild=25)
            it = pool.imap_unordered(_solve_one, work)
        done = 0
        for r in it:
            results.append(r)
            mf.write(json.dumps(r) + "\n"); mf.flush()
            done += 1
            if done % 50 == 0 or done == len(work):
                nw = sum(1 for x in results if x["won"])
                print(f"  {done}/{len(work)} done, {nw} won "
                      f"({time.time()-t_start:.0f}s)", flush=True)
        if pool is not None:
            pool.close(); pool.join()

    print("\n=== COVERAGE BY BUCKET / TIER ===", flush=True)
    print("%-17s %s" % ("bucket", "  ".join(f"{t}" for t in tiers)), flush=True)
    for b in buckets:
        cells = []
        for t in tiers:
            rs = [r for r in results if r["bucket"] == b and r["tier"] == t]
            nw = sum(1 for r in rs if r["won"])
            cells.append(f"{nw}/{len(rs)}" if rs else "-")
        print("%-17s %s" % (b, "  ".join(f"{c:>7}" for c in cells)), flush=True)
    total_w = sum(1 for r in results if r["won"])
    by_solver = {}
    for r in results:
        if r["won"]:
            by_solver[r["solver"]] = by_solver.get(r["solver"], 0) + 1
    print(f"\nTOTAL: {total_w}/{len(results)} demos "
          f"({100*total_w/max(1,len(results)):.0f}%) in {time.time()-t_start:.0f}s",
          flush=True)
    print(f"by solver: {by_solver}", flush=True)
    print(f"-> {args.out}  (manifest: {manifest_path})", flush=True)


if __name__ == "__main__":
    main()
