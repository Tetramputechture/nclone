#!/usr/bin/env python3
"""#2: Route-executor validation — native-planned tape replayed on the Python sim.

We *plan* on the native simulator (fast) but the gym env / RL agent *executes* on
the Python ``nsim`` engine. This tool checks the assumption that a native-planned
route is replayable downstream:

  1. Plan a route with the native-sim spike (``physics_route_spike``).
  2. Expand its macro-action sequence into a per-frame ``(h,j)`` input tape.
  3. Open-loop replay that tape on the Python sim (via ``NPlayHeadless``) and on a
     fresh native sim (sanity), recording both trajectories.
  4. Report: did each engine COMPLETE the level, and how far do the trajectories
     DIVERGE (native↔Python determinism). Large divergence ⇒ open-loop replay is
     insufficient and the closed-loop executor must correct online.

Usage::

    python -m nclone.value_function.tools.route_executor_check \
        --map nclone/test_maps/minefield --output docs/stage0/exec_minefield.png
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import nclone_native as Nnative
from ...nplay_headless import NPlayHeadless
from .physics_route_spike import PhysicsRouteSpike, ACT, _switch_done


def _frame_tape(actions: List[int], macro: int, n_frames: int) -> List[Tuple[int, int]]:
    """Expand macro-action ids into a per-frame (h,j) tape, capped at n_frames."""
    tape: List[Tuple[int, int]] = []
    for a in actions:
        tape.extend([ACT[a]] * macro)
    return tape[:n_frames]


def _replay_native(map_data: List[int], tape: List[Tuple[int, int]]) -> dict:
    sim = Nnative.Simulator.load(map_data)
    traj = [sim.get_ninja_pos()]
    won = died = False
    for (h, j) in tape:
        sim.tick(h, j)
        traj.append(sim.get_ninja_pos())
        if sim.has_won(): won = True; break
        if sim.has_died(): died = True; break
    return {"traj": traj, "won": won, "died": died, "frames": len(traj) - 1}


def _replay_python(map_path: str, tape: List[Tuple[int, int]]) -> dict:
    nh = NPlayHeadless(render_mode="rgb_array", enable_debug_overlay=False)
    nh.load_map(map_path)
    nh.reset()
    traj = [nh.ninja_position()]
    won = died = False
    switch_frame = None
    for f, (h, j) in enumerate(tape):
        nh.tick(h, j)
        traj.append(nh.ninja_position())
        if switch_frame is None and nh.exit_switch_activated():
            switch_frame = f + 1
        if nh.ninja_has_won(): won = True; break
        if nh.ninja_has_died(): died = True; break
    return {"traj": traj, "won": won, "died": died, "frames": len(traj) - 1,
            "switch_frame": switch_frame, "death_cause": nh.ninja_death_cause() if died else None}


def _divergence(a: List[Tuple[float, float]], b: List[Tuple[float, float]]):
    """Per-frame Euclidean divergence over the common prefix."""
    n = min(len(a), len(b))
    if n == 0:
        return np.array([]), 0.0, 0.0, None
    d = np.array([((a[i][0] - b[i][0]) ** 2 + (a[i][1] - b[i][1]) ** 2) ** 0.5 for i in range(n)])
    first = next((i for i, v in enumerate(d) if v > 1.0), None)  # >1px
    return d, float(d.max()), float(d.mean()), first


def check(map_path: str, output: Optional[str], macro: int = 6, h_weight: float = 1.25) -> dict:
    sp = PhysicsRouteSpike(map_path, macro=macro, h_weight=h_weight)
    res = sp.plan()
    if not res["solved"]:
        print(f"[{os.path.basename(map_path)}] planner found no route — nothing to execute.")
        return res
    native_traj = res["trajectory"]
    n_frames = len(native_traj) - 1
    tape = _frame_tape(res["actions"], macro, n_frames)

    nat = _replay_native(sp.map_data, tape)        # sanity (same engine as planner)
    py = _replay_python(map_path, tape)            # the real test (gym-env engine)
    d, dmax, dmean, dfirst = _divergence(nat["traj"], py["traj"])

    name = os.path.basename(map_path)
    print(f"[{name}] planned {n_frames}f | native-replay won={nat['won']} died={nat['died']} "
          f"frames={nat['frames']}")
    print(f"  PYTHON-replay won={py['won']} died={py['died']}({py['death_cause']}) "
          f"frames={py['frames']} switch_f={py['switch_frame']}")
    print(f"  native↔python divergence: max={dmax:.2f}px mean={dmean:.2f}px "
          f"first>1px @frame={dfirst}")

    _render(map_path, sp, nat, py, d, dmax, dfirst, output)
    return {"native": nat, "python": py, "max_div": dmax, "mean_div": dmean,
            "first_div": dfirst, "planned_frames": n_frames}


def _render(map_path, sp, nat, py, d, dmax, dfirst, output):
    from ..value_landscape import _load_tile_dict, _save_or_show
    from ..level_renderer import render_tiles_to_axis
    from .physics_route_spike import _MINE_KILL_PX

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(15, 11),
                                  gridspec_kw={"height_ratios": [3, 1]})
    ax.set_xlim(0, 1056); ax.set_ylim(600, 0); ax.set_aspect("equal")
    td = _load_tile_dict(map_path)
    if td is not None:
        render_tiles_to_axis(ax, td, alpha=0.4)
    for (mx, my) in sp.deadly0:
        ax.add_patch(plt.Circle((mx, my), _MINE_KILL_PX, color="red", alpha=0.18, zorder=3))
        ax.scatter([mx], [my], marker="X", c="red", s=50, zorder=4)
    nt = np.array(nat["traj"]); pt = np.array(py["traj"])
    ax.plot(nt[:, 0], nt[:, 1], "-", color="blue", lw=2.5, zorder=6, label="native plan")
    ax.plot(pt[:, 0], pt[:, 1], "--", color="orange", lw=2.0, zorder=7, label="python replay")
    if py["died"]:
        ax.scatter([pt[-1, 0]], [pt[-1, 1]], marker="X", c="darkred", s=160, zorder=8,
                   label=f"python DEATH ({py['death_cause']})")
    if dfirst is not None and dfirst < len(nt):
        ax.scatter([nt[dfirst, 0]], [nt[dfirst, 1]], marker="o", facecolors="none",
                   edgecolors="magenta", s=200, lw=2, zorder=9, label=f"divergence>1px @f{dfirst}")
    ax.scatter([sp.spawn[0]], [sp.spawn[1]], marker="o", c="black", s=70, zorder=7)
    ax.scatter([sp.switch[0]], [sp.switch[1]], marker="*", c="gold", s=220, edgecolors="k", zorder=7)
    ax.scatter([sp.exit[0]], [sp.exit[1]], marker="s", c="cyan", s=110, edgecolors="k", zorder=7)
    verdict = ("MATCH — open-loop replay completes on Python" if py["won"]
               else f"MISMATCH — python {'died' if py['died'] else 'did not finish'}")
    ax.set_title(f"Route executor check — {os.path.basename(map_path)}   |   {verdict}\n"
                 f"native↔python max div {dmax:.1f}px", fontsize=11)
    ax.legend(loc="upper right", fontsize=8)

    ax2.plot(d, color="purple")
    ax2.axhline(1.0, color="gray", ls="--", lw=1, label="1px")
    ax2.set_xlabel("frame"); ax2.set_ylabel("native↔python\ndivergence (px)")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
    plt.tight_layout()
    _save_or_show(fig, output)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--map", required=True)
    p.add_argument("--output", default=None)
    p.add_argument("--macro", type=int, default=6)
    args = p.parse_args()
    check(args.map, args.output, macro=args.macro)
    return 0


if __name__ == "__main__":
    sys.exit(main())
