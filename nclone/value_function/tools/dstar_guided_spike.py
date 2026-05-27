#!/usr/bin/env python3
"""Forward physics A* guided by the lattice ``d*`` heuristic (Plan→Search).

``PhysicsRouteSpike`` searches real-physics macro-action sequences but its
heuristic is straight-line distance / max-speed — useless inside a dense mine
maze, where the safe path winds and the straight line points through kill-zones.
This swaps in the **lattice ``d*`` cost-to-go** (phase-aware, already in frames)
as the heuristic: the search is pulled along the planner's known-safe corridor
while still pruning *real* deaths on the native sim.  It combines the macro plan
(``d*``, which knows the route exists) with frame-exact forward search (which the
open-loop executors can't do through tight mines).

Where the lattice has no nearby node (airborne over a gap) it falls back to the
geometric heuristic, so it degrades gracefully.

CLI::

    .venv/bin/python -m nclone.value_function.tools.dstar_guided_spike \
        --map nclone/test_maps/mine-very-difficult --output /tmp/win.mp4 \
        --max-expansions 150000 --max-frames 2600 --h-weight 2.0
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from typing import List, Optional

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np

from .physics_route_spike import PhysicsRouteSpike, ACT, _switch_done
from ..lattice_distance import LatticeDistanceSource


class DStarGuidedSpike(PhysicsRouteSpike):
    """PhysicsRouteSpike whose heuristic is the lattice ``d*`` cost-to-go."""

    def __init__(self, *args, lattice_source: Optional[LatticeDistanceSource] = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.lat = lattice_source

    def _h(self, x: float, y: float, switch_done: bool) -> float:
        if self.lat is not None and self.lat.available:
            c = self.lat.cost_to_go(x, y, bool(switch_done))  # phase-aware frames
            if math.isfinite(c):
                return float(c)
        return super()._h(x, y, switch_done)


def _tape_from_actions(actions: List[int], macro: int):
    """Expand a macro-action list to a per-frame ``(h, j)`` tape."""
    tape = []
    for a in actions:
        h, j = ACT[a]
        tape.extend([(h, j)] * macro)
    return tape


def plan_dstar_guided(map_path: str, *, macro: int = 6, vel_bin: float = 0.25,
                      h_weight: float = 2.0, max_expansions: int = 150000,
                      max_frames: int = 2600, w_risk: Optional[float] = None,
                      jump_tax: Optional[float] = None,
                      momentum_model: str = "legacy") -> dict:
    """Build the lattice d* source + run the d*-guided forward search."""
    src = LatticeDistanceSource(
        PhysicsRouteSpike(map_path).map_data, w_risk=w_risk, jump_tax=jump_tax,
        momentum_model=momentum_model, use_cache=True,
        name=os.path.basename(map_path.rstrip("/")),
    )
    sp = DStarGuidedSpike(
        map_path, macro=macro, vel_bin=vel_bin, h_weight=h_weight,
        max_expansions=max_expansions, max_frames=max_frames, lattice_source=src,
    )
    t0 = time.time()
    res = sp.plan()
    res["lattice_available"] = bool(src.available)
    res["wall_time_s"] = time.time() - t0
    res["macro"] = macro
    return res


def render_win_video(map_path: str, actions: List[int], macro: int,
                     output: str, fps: int = 60, *, overlay: bool = True,
                     animate: bool = False) -> None:
    """Replay the winning macro-actions on a render sim and write an mp4.

    ``overlay=True`` (default) composites the route trace + mine kill-zones + a
    debug HUD (mirroring ``route_video``).  ``overlay=False`` writes a CLEAN
    gameplay capture — the raw game render only.  ``animate=True`` enables the
    ninja limb/bone animation (cosmetic; physics unchanged).  For a clean
    gameplay demo pass ``overlay=False, animate=True``."""
    import imageio
    import nclone_native as Nnative
    from ...nplay_headless import NPlayHeadless
    from .route_video import _to_hwc, _draw_overlay
    from .physics_route_spike import _MINE_STATE_DEADLY, _goal_positions, _load_map_data

    tape = _tape_from_actions(actions, macro)
    # Deadly-mine positions for the overlay come from a native sim (the render
    # NPlayHeadless sim has a different API and no get_state).
    _sw, _ex, mines = _goal_positions(Nnative.Simulator.load(_load_map_data(map_path)))
    deadly = [(x, y) for (x, y, s) in mines if s == _MINE_STATE_DEADLY]

    nh = NPlayHeadless(render_mode="rgb_array", enable_animation=animate,
                       enable_debug_overlay=False)
    nh.load_map(map_path)

    def _clear(x, y):
        return min((math.hypot(x - mx, y - my) for (mx, my) in deadly),
                   default=float("inf"))

    # Pass 1: full trajectory for the (faint) planned-route overlay (overlay only).
    traj: list = []
    total = 0
    if overlay:
        nh.reset()
        traj = [nh.ninja_position()]
        for (h, j) in tape:
            nh.tick(h, j)
            traj.append(nh.ninja_position())
            if nh.ninja_has_won() or nh.ninja_has_died():
                break
        total = len(traj) - 1

    def _cap(fn):
        raw = _to_hwc(nh.render())
        if not overlay:
            return raw  # clean gameplay frame — no trace / kill-zones / HUD
        x, y = nh.ninja_position()
        vx, vy = nh.ninja_velocity()
        phase = "-> EXIT" if nh.exit_switch_activated() else "-> switch"
        return _draw_overlay(raw, traj, fn, (x, y), math.hypot(vx, vy),
                             _clear(x, y), fn, total, phase, deadly)

    # Pass 2: render frames.
    nh.reset()
    frames = [_cap(0)]
    won = False
    for fn, (h, j) in enumerate(tape, start=1):
        nh.tick(h, j)
        frames.append(_cap(fn))
        if nh.ninja_has_won():
            won = True
            break
        if nh.ninja_has_died():
            break
    if won:
        frames.extend([frames[-1]] * fps)  # hold final frame ~1s
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    imageio.mimsave(output, frames, fps=fps, macro_block_size=1)
    return won


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--map", required=True)
    p.add_argument("--output", default=None, help="mp4 path (written only on a win)")
    p.add_argument("--macro", type=int, default=6)
    p.add_argument("--vel-bin", type=float, default=0.25)
    p.add_argument("--h-weight", type=float, default=2.0)
    p.add_argument("--max-expansions", type=int, default=150000)
    p.add_argument("--max-frames", type=int, default=2600)
    p.add_argument("--w-risk", type=float, default=None)
    p.add_argument("--jump-tax", type=float, default=None,
                   help="additive per-jump frame cost (discourages wall-climb "
                        "spam; None => planner default DEFAULT_JUMP_TAX)")
    p.add_argument("--momentum-model", choices=["legacy", "tangential"],
                   default="legacy",
                   help="velocity representation: legacy world-vx/vy, or "
                        "tangential along-surface + curved/slope node extraction")
    p.add_argument("--fps", type=int, default=60)
    p.add_argument("--clean", action="store_true",
                   help="clean gameplay capture: ninja animation on, no route/"
                        "mine-radius/HUD overlay")
    args = p.parse_args()

    print(f"[dstar_guided_spike] {os.path.basename(args.map)}  "
          f"macro={args.macro} w={args.h_weight} "
          f"budget={args.max_expansions}exp/{args.max_frames}f")
    res = plan_dstar_guided(
        args.map, macro=args.macro, vel_bin=args.vel_bin, h_weight=args.h_weight,
        max_expansions=args.max_expansions, max_frames=args.max_frames,
        w_risk=args.w_risk, jump_tax=args.jump_tax,
        momentum_model=args.momentum_model,
    )
    print(f"  lattice_available={res['lattice_available']}  "
          f"SOLVED={res['solved']}  frames={res.get('frames')}  "
          f"expansions={res['expansions']}  death_branches={res['death_branches']}  "
          f"wall={res['wall_time_s']:.1f}s")
    if res["solved"] and args.output:
        won = render_win_video(args.map, res["actions"], res["macro"], args.output,
                               fps=args.fps, overlay=not args.clean, animate=args.clean)
        print(f"  render replay won={won}  ->  {args.output}" if won
              else f"  render replay did NOT win (non-determinism?) — no clean video")
    return 0 if res["solved"] else 2


if __name__ == "__main__":
    sys.exit(main())
