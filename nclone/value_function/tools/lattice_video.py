#!/usr/bin/env python3
"""Render an mp4 of the LATTICE + micro-search executor completing a level.

Unlike ``route_video`` (which plans with the forward spike), this renders the
production planner: backward-Dijkstra surface-graph ``d*`` over the jointly
optimal switch→exit tree → per-arc route → closed-loop + micro-search executor
(``lattice_audit.LatticeAudit.plan_tape``).  The executor emits a winning
per-frame input tape; because native↔Python physics are bit-identical, that
tape replays deterministically on the rendering sim.

Usage::

    python -m nclone.value_function.tools.lattice_video \
        --map "nclone/maps/official/SL/001 take the easy way out" \
        --output docs/stage0/official/easy_way_out.mp4
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # headless rendering (WSL/CI)

import math

import imageio

from ...nplay_headless import NPlayHeadless
from .route_video import _to_hwc, _draw_overlay
from .lattice_audit import LatticeAudit


def render_lattice_video(map_path: str = None, output: str = None, fps: int = 60,
                         overlay: bool = True, workers: int = 4,
                         use_cache: bool = True, map_data=None,
                         name: str = None) -> dict:
    """Render the lattice planner completing a level.

    Pass either ``map_path`` (a level file) or ``map_data`` (a serialized level
    int-list, e.g. from ``cdj_chain_map.build_cdj_chain_map`` — generated maps
    that are not on disk).
    """
    name = name or (os.path.basename(map_path.rstrip("/")) if map_path else "inline")
    print(f"[{name}] planning lattice d* route + micro-search executor...")
    audit = LatticeAudit(map_path, workers=workers, map_data=map_data, name=name)
    res = audit.plan_tape(use_cache=use_cache)
    tape = res.get("tape", [])
    if not tape or not res.get("won"):
        print(f"  executor did NOT win (won={res.get('won')}, "
              f"reached_switch=@{res.get('switch_frame')}) — not rendering.")
        return res
    print(f"  WON in {res['frames']}f over {res['n_arcs']} arcs "
          f"(min mine clearance {res['min_clear']:.1f}px); tape={len(tape)} frames")

    deadly = audit.deadly
    # Full trajectory via native replay (bit-identical to the render sim) for
    # the route-trace overlay.
    import nclone_native as Nnative
    sim = Nnative.Simulator.load(audit.map_data)
    full_traj = [sim.get_ninja_pos()]
    for (h, j) in tape:
        sim.tick(h, j)
        full_traj.append(sim.get_ninja_pos())
        if sim.has_won() or sim.has_died():
            break
    total = len(full_traj) - 1

    def _clear(x, y):
        return min((math.hypot(x - mx, y - my) for (mx, my) in deadly),
                   default=float("inf"))

    nh = NPlayHeadless(render_mode="rgb_array", enable_debug_overlay=False)
    if map_data is not None:
        nh.load_map_from_map_data(list(map_data))
    else:
        nh.load_map(map_path)
    nh.reset()

    def _cap(frame_no):
        raw = _to_hwc(nh.render())
        if not overlay:
            return raw
        x, y = nh.ninja_position()
        vx, vy = nh.ninja_velocity()
        phase = "-> EXIT" if nh.exit_switch_activated() else "-> switch"
        return _draw_overlay(raw, full_traj, frame_no, (x, y),
                             math.hypot(vx, vy), _clear(x, y),
                             frame_no, total, phase, deadly)

    frames = [_cap(0)]
    won = died = False
    frame_no = 0
    for (h, j) in tape:
        nh.tick(h, j)
        frame_no += 1
        frames.append(_cap(frame_no))
        if nh.ninja_has_won():
            won = True
            break
        if nh.ninja_has_died():
            died = True
            break

    # The native-sim planner is blind to drone/thwump/etc. AI (not ported to
    # Rust — only ToggleMine is "thinkable"), so a tape that wins natively can
    # die on the full Python render sim.  This is the ground-truth check: only
    # write a *completion* video when the render sim actually wins.
    if not won:
        print(f"  render sim did NOT complete (died={died}) — likely an "
              f"out-of-scope dynamic entity (drone/thwump). Not writing video.")
        return {"won": False, "died": died, "output": None,
                "route_frames": res["frames"]}

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    frames.extend([frames[-1]] * fps)  # hold final frame ~1s
    imageio.mimsave(output, frames, fps=fps, macro_block_size=1)
    print(f"  replay on render sim: WON | {len(frames)} frames "
          f"@ {fps}fps -> {output}")
    return {"won": won, "died": died, "n_frames": len(frames),
            "route_frames": res["frames"], "output": output}


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--map", help="level file path (omit when using --cdj-chain)")
    p.add_argument("--cdj-chain", action="store_true",
                   help="render the generated 2-corner CDJ chain staircase "
                        "(nclone.value_function.tools.cdj_chain_map)")
    p.add_argument("--output", required=True, help="output .mp4 path")
    p.add_argument("--fps", type=int, default=60)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--no-overlay", action="store_true")
    p.add_argument("--no-cache", action="store_true")
    args = p.parse_args()
    map_data = name = None
    if args.cdj_chain:
        from .cdj_chain_map import build_cdj_chain_map
        map_data = build_cdj_chain_map()
        name = "cdj-chain"
    elif not args.map:
        p.error("either --map or --cdj-chain is required")
    render_lattice_video(args.map, args.output, fps=args.fps,
                         overlay=not args.no_overlay, workers=args.workers,
                         use_cache=not args.no_cache, map_data=map_data, name=name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
