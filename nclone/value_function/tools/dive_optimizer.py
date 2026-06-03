#!/usr/bin/env python3
"""Cross-entropy (CEM) optimizer over per-frame air control to thread a tight
aerial entry — built to crack the ``squeeze`` corridor mouth the fixed-segment
sweeps (and the binned lattice / macro forward search) could not.

The squeeze mouth (≈573,556 moving right, just under the overhang tip) is
reachable only by a steep down-right dive; walking wedges at (570,544) and a
fast run-up glides into the overhang face at (566,527).  A naive distance-to-goal
reward gets stuck on those local optima (and on sitting atop the overhang right
above the switch).  So the reward is **layered** — it only rewards getting deep
into the corridor *below the tip*, then activating the switch, then winning:

    tier 0 (not threaded): R = -mouth_dist             # pull toward the mouth
    tier 1 (threaded):     R = 100 + corridor_depth     # penetration under the tip
    tier 2 (switch hit):   R = 1e4 - switch_frame       # then minimise frames
    tier 3 (won):          R = 1e5 - won_frame
    (a death subtracts a fixed penalty on top, so progress-then-die still ranks
     above no progress.)

CEM maintains a per-segment categorical over a small action set (a segment is
``macro`` held frames), samples a population of tapes, keeps the elite, and
refits the categorical with Laplace smoothing.  Default-safe: this is an offline
search tool; it does not touch the planner pipeline.

CLI::

    .venv/bin/python -m nclone.value_function.tools.dive_optimizer \
        --map nclone/test_maps/squeeze --iters 40 --pop 256 --out /tmp/squeeze_dive.npy
"""

from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np

import nclone_native as Nnative
from .physics_route_spike import _load_map_data, _goal_positions

# Action set: (h_input, jump_input).  6 discrete controls.
ACTIONS: Tuple[Tuple[int, int], ...] = (
    (-1, 0), (0, 0), (1, 0), (-1, 1), (0, 1), (1, 1),
)


@dataclass
class MouthGeom:
    """Geometry of the aerial entry to reward threading (squeeze defaults)."""

    mouth_xy: Tuple[float, float] = (573.0, 556.0)  # just under the overhang tip
    tip_y: float = 548.0                            # overhang underside tip
    corridor_y_lo: float = 550.0                    # below the tip = in the corridor
    corridor_y_hi: float = 572.0                    # above the corridor floor
    corridor_x0: float = 560.0                      # left edge of the corridor mouth


@dataclass
class RolloutResult:
    reward: float
    won: bool
    won_frame: Optional[int]
    switch_frame: Optional[int]
    died: bool
    corridor_depth: float
    mouth_dist: float
    frames: int
    mouth_vy: float = 0.0


def _rollout(sim, switch_xy, exit_xy, geom: MouthGeom, tape: List[Tuple[int, int]],
             cap: int) -> RolloutResult:
    """Replay a per-frame tape from spawn; score with the layered reward."""
    sim.fast_reset()
    swx, swy = switch_xy
    min_mouth = float("inf")
    corridor_depth = 0.0
    mouth_vy = 0.0  # max DOWNWARD velocity achieved while in the mouth region
    switch_frame = None
    f = 0
    won = died = False
    for (h, j) in tape:
        sim.tick(int(h), int(j))
        f += 1
        x, y = sim.get_ninja_pos()
        vx, vy = sim.get_ninja_velocity()
        d = math.hypot(x - geom.mouth_xy[0], y - geom.mouth_xy[1])
        if d < min_mouth:
            min_mouth = d
        # The mouth is passable only with JOINT rightward+downward speed
        # (injection sweep: vx>=2 AND vy>=3 threads); arriving grounded, or fast
        # in only one axis, wedges.  Reward vx+vy at the mouth so CEM builds the
        # full down-right punch, not just one component.
        if 566.0 <= x <= 576.0 and 538.0 <= y <= 556.0 and vx > 0.0 and vy > 0.0:
            mouth_vy = max(mouth_vy, vx + vy)
        if (x > geom.corridor_x0 and geom.corridor_y_lo <= y <= geom.corridor_y_hi):
            corridor_depth = max(corridor_depth, x - geom.corridor_x0)
        # Switch-proximity proxy (within contact radius) — cheap per-frame signal
        # that the dive reached the switch; ``has_won`` is the true terminal.
        if switch_frame is None and math.hypot(x - swx, y - swy) <= 16.0:
            switch_frame = f
        if sim.has_won():
            won = True
            break
        if sim.has_died():
            died = True
            break
        if f >= cap:
            break

    # Layered reward.
    if won:
        reward = 1e5 - f
    elif switch_frame is not None:
        reward = 1e4 - switch_frame
    elif corridor_depth > 0.0:
        reward = 100.0 + corridor_depth + 5.0 * mouth_vy
    else:
        # Reward approaching the mouth AND arriving descending (vy>0), so the
        # search builds the downward punch instead of stopping at the wedge.
        reward = -min_mouth + 8.0 * mouth_vy
    if died:
        reward -= 50.0
    return RolloutResult(reward, won, (f if won else None), switch_frame, died,
                         corridor_depth, min_mouth, f, mouth_vy)


def optimize(map_path: str, *, iters: int = 40, pop: int = 256,
             elite_frac: float = 0.125, macro: int = 3, horizon_segs: int = 120,
             cap: int = 420, seed_right_bias: float = 0.0,
             smoothing: float = 0.04, geom: Optional[MouthGeom] = None,
             verbose: bool = True) -> dict:
    """Run CEM; return the best tape + diagnostics.

    ``horizon_segs`` segments of ``macro`` held frames => max tape length
    ``horizon_segs * macro``; rollouts stop early on win/death/``cap``.
    """
    geom = geom or MouthGeom()
    map_data = _load_map_data(map_path)
    sim = Nnative.Simulator.load(map_data)
    switch_xy, exit_xy, _mines = _goal_positions(sim)

    n_act = len(ACTIONS)
    rng = np.random.default_rng(0)  # deterministic (Math.random-free environments)
    # Per-segment categorical, initialised ~uniform (optional rightward bias).
    probs = np.full((horizon_segs, n_act), 1.0 / n_act, dtype=np.float64)
    if seed_right_bias > 0.0:
        for a, (h, _j) in enumerate(ACTIONS):
            if h > 0:
                probs[:, a] += seed_right_bias
        probs /= probs.sum(axis=1, keepdims=True)

    n_elite = max(2, int(pop * elite_frac))
    best = None
    t0 = time.time()
    for it in range(iters):
        seg_samples = np.empty((pop, horizon_segs), dtype=np.int32)
        results: List[RolloutResult] = []
        for p in range(pop):
            segs = np.array(
                [rng.choice(n_act, p=probs[s]) for s in range(horizon_segs)],
                dtype=np.int32)
            seg_samples[p] = segs
            tape: List[Tuple[int, int]] = []
            for s in segs:
                tape.extend([ACTIONS[s]] * macro)
            r = _rollout(sim, switch_xy, exit_xy, geom, tape, cap)
            results.append(r)
            if best is None or r.reward > best[0].reward:
                best = (r, segs.copy())
        rewards = np.array([r.reward for r in results])
        elite_idx = np.argsort(rewards)[-n_elite:]
        # Refit categorical from elite segment choices (+ Laplace smoothing).
        new_probs = np.full((horizon_segs, n_act), smoothing, dtype=np.float64)
        for ei in elite_idx:
            for s in range(horizon_segs):
                new_probs[s, seg_samples[ei, s]] += 1.0
        new_probs /= new_probs.sum(axis=1, keepdims=True)
        probs = new_probs
        if verbose:
            br = best[0]
            tier = ("WON" if br.won else "switch" if br.switch_frame else
                    "corridor" if br.corridor_depth > 0 else "approach")
            print(f"  iter {it:3d}  best_reward={best[0].reward:11.2f} [{tier}]  "
                  f"elite_mean={rewards[elite_idx].mean():.1f}  "
                  f"corridor_depth={br.corridor_depth:.1f} mouth_dist={br.mouth_dist:.1f} "
                  f"mouth_vy={br.mouth_vy:.1f} won_f={br.won_frame} sw_f={br.switch_frame}  "
                  f"({time.time()-t0:.0f}s)")

    br, bsegs = best
    tape: List[Tuple[int, int]] = []
    for s in bsegs:
        tape.extend([ACTIONS[int(s)]] * macro)
    return {
        "won": br.won, "won_frame": br.won_frame, "switch_frame": br.switch_frame,
        "reward": br.reward, "corridor_depth": br.corridor_depth,
        "mouth_dist": br.mouth_dist, "tape": tape, "segs": bsegs.tolist(),
        "macro": macro, "switch_xy": switch_xy, "exit_xy": exit_xy,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--map", required=True)
    p.add_argument("--iters", type=int, default=40)
    p.add_argument("--pop", type=int, default=256)
    p.add_argument("--macro", type=int, default=3)
    p.add_argument("--horizon-segs", type=int, default=120)
    p.add_argument("--cap", type=int, default=420)
    p.add_argument("--elite-frac", type=float, default=0.125)
    p.add_argument("--seed-right-bias", type=float, default=0.0)
    p.add_argument("--out", default=None, help="npy path to save the best tape")
    args = p.parse_args()

    print(f"[dive_optimizer] {os.path.basename(args.map)} "
          f"iters={args.iters} pop={args.pop} macro={args.macro} "
          f"segs={args.horizon_segs} cap={args.cap}")
    res = optimize(args.map, iters=args.iters, pop=args.pop, macro=args.macro,
                   horizon_segs=args.horizon_segs, cap=args.cap,
                   elite_frac=args.elite_frac,
                   seed_right_bias=args.seed_right_bias)
    print(f"  RESULT won={res['won']} won_frame={res['won_frame']} "
          f"switch_frame={res['switch_frame']} reward={res['reward']:.1f} "
          f"corridor_depth={res['corridor_depth']:.1f} mouth_dist={res['mouth_dist']:.1f}")
    if args.out and res["won"]:
        np.save(args.out, np.array(res["tape"], dtype=np.int8))
        print(f"  saved winning tape ({len(res['tape'])} frames) -> {args.out}")
    return 0 if res["won"] else 2


if __name__ == "__main__":
    import sys
    sys.exit(main())
