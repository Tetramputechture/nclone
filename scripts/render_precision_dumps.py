#!/usr/bin/env python
"""Triage the PRECISION mine-death dumps written by ``precision_failure_capture``.

Each dump carries the full ``map_data`` plus the last-N frames of ninja telemetry
into a PRECISION-bucket mine death. This script answers the question those dumps
exist to answer -- **which failure mode dominates**: undershoot (fell short of the
far platform), overshoot / premature (sailed past the safe landing), or timing
(right arc shape, wrong jump-hold duration).

It prints:
  * a per-dump line (takeoff -> death geometry, hold frames, soft label), and
  * an AGGREGATE breakdown (label histogram + hold-frame / horizontal-travel
    distributions) -- the actual deliverable.

With ``--overlay`` it also writes ``<dump>.traj.png``: a world-coordinate
schematic of the trajectory arc against the level's mines / switch / exit, which
is the ground truth for separating undershoot from overshoot by eye.

Usage:
  python -m scripts.render_precision_dumps --dir $NCLONE_GRAPH_CACHE_DIR/precision_dumps
  python -m scripts.render_precision_dumps --dir <dir> --overlay
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter


# trajectory frame indices (see precision_failure_capture.trajectory_fields)
X, Y, VX, VY, AIR, STATE, JI, JD = range(8)


def _label(triage: dict) -> str:
    """Soft failure-mode label from the arc shape (geometry-blind first pass)."""
    if triage.get("takeoff_x") is None:
        return "no_jump_fell"  # walked off an edge into the gap, never jumped
    descending = bool(triage.get("descending_at_death"))
    hold = triage.get("jump_hold_frames", 0) or 0
    if descending and hold <= 2:
        return "short_hop_timing"  # released jump too early -> weak hop, fell short
    if descending:
        return "undershoot"  # full-ish jump but still fell short of the landing
    return "overshoot_or_midair"  # hit a mine ascending / at apex -> jumped into it


# Full bordered map in world pixels (44x24 * 24 = 1056, 25*24 = 600). The
# headless rgb_array render is 1:1 with world coords, so trajectory world (x, y)
# plot directly over ``imshow(frame, extent=[0, W, H, 0])``.
LEVEL_W_PX = 1056
LEVEL_H_PX = 600


def _render_level_frame(md):
    """Ground-truth tile render as a (600, 1056, 3) uint8 array (world-aligned
    1:1), or None if the headless renderer is unavailable in this environment."""
    try:
        from nclone.nplay_headless import NPlayHeadless
        nplay = NPlayHeadless(render_mode="rgb_array", enable_animation=False,
                              enable_logging=False, enable_debug_overlay=False)
        nplay.load_map_from_map_data(list(md))
        nplay.reset()
        return nplay.render()
    except Exception as exc:
        print(f"    tile render unavailable ({exc}); schematic only")
        return None


def _overlay(dump_path, d):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"    overlay skipped (matplotlib unavailable: {exc})")
        return
    traj = d.get("trajectory") or []
    ctx = d.get("context") or {}
    tri = d.get("triage") or {}
    fig, ax = plt.subplots(figsize=(11, 6.4))

    # Ground-truth tile geometry underneath, in world coords. extent maps image
    # col->world x and row->world y (top-left origin, y down = sim convention).
    frame = _render_level_frame(d.get("map_data") or [])
    if frame is not None:
        ax.imshow(frame, extent=[0, LEVEL_W_PX, LEVEL_H_PX, 0], zorder=0,
                  interpolation="nearest")
        ax.set_xlim(0, LEVEL_W_PX)
        ax.set_ylim(LEVEL_H_PX, 0)  # y down
    else:
        ax.invert_yaxis()  # schematic fallback keeps sim y-down

    if traj:
        gx = [f[X] for f in traj if not f[AIR]]
        gy = [f[Y] for f in traj if not f[AIR]]
        ax_ = [f[X] for f in traj if f[AIR]]
        ay = [f[Y] for f in traj if f[AIR]]
        ax.plot([f[X] for f in traj], [f[Y] for f in traj], "-", color="white",
                lw=1.4, alpha=0.8, zorder=3)
        if gx:
            ax.scatter(gx, gy, s=16, c="tab:blue", edgecolors="w", linewidths=0.4,
                       label="grounded", zorder=4)
        if ax_:
            ax.scatter(ax_, ay, s=16, c="tab:orange", edgecolors="w",
                       linewidths=0.4, label="airborne", zorder=4)
    if tri.get("takeoff_x") is not None:
        ax.scatter([tri["takeoff_x"]], [tri["takeoff_y"]], s=130, marker="^",
                   c="lime", edgecolors="k", label="takeoff", zorder=5)
    if ctx.get("x") is not None:
        ax.scatter([ctx.get("x")], [ctx.get("y")], s=170, marker="X",
                   c="red", edgecolors="k", label="death", zorder=5)
    # switch/exit are already drawn in the tile render; mark them to verify the
    # trajectory frame lines up with the rendered geometry.
    if ctx.get("switch_x") is not None:
        ax.scatter([ctx["switch_x"]], [ctx["switch_y"]], s=90, marker="*",
                   c="gold", edgecolors="k", label="switch", zorder=5)
    if ctx.get("exit_door_x") is not None:
        ax.scatter([ctx["exit_door_x"]], [ctx["exit_door_y"]], s=80, marker="D",
                   facecolors="none", edgecolors="cyan", linewidths=1.6,
                   label="exit", zorder=5)
    ax.set_aspect("equal")
    ax.set_title(f"{os.path.basename(dump_path)}  [{_label(tri)}]", fontsize=9)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.85)
    fig.tight_layout()
    out = dump_path[:-5] + ".traj.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"    overlay -> {os.path.basename(out)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", required=True, help="directory of precision_mine_*.json dumps")
    ap.add_argument("--overlay", action="store_true",
                    help="write a <dump>.traj.png trajectory schematic per dump")
    args = ap.parse_args()

    jsons = sorted(glob.glob(os.path.join(args.dir, "precision_mine_*.json")))
    if not jsons:
        print(f"no precision dumps in {args.dir}")
        return

    print(f"{len(jsons)} PRECISION mine-death dumps in {args.dir}\n")
    header = f"{'dump':28s} {'label':20s} {'hold':>4s} {'travel':>7s} {'desc':>4s}  {'level'}"
    print(header)
    print("-" * len(header))
    labels = Counter()
    holds, travels = [], []
    per_level = Counter()
    for jp in jsons:
        with open(jp) as f:
            d = json.load(f)
        tri = d.get("triage") or {}
        ctx = d.get("context") or {}
        lab = _label(tri)
        labels[lab] += 1
        hold = tri.get("jump_hold_frames")
        trav = tri.get("horizontal_travel")
        if hold is not None:
            holds.append(hold)
        if trav is not None:
            travels.append(trav)
        per_level[str(ctx.get("level_id"))] += 1
        name = os.path.basename(jp)[:26]
        print(f"{name:28s} {lab:20s} {str(hold):>4s} "
              f"{('%.1f' % trav) if trav is not None else '  -':>7s} "
              f"{('Y' if tri.get('descending_at_death') else 'N'):>4s}  {ctx.get('level_id')}")
        if args.overlay:
            _overlay(jp, d)

    print("\n=== FAILURE-MODE BREAKDOWN ===")
    total = sum(labels.values())
    for lab, n in labels.most_common():
        print(f"  {lab:22s} {n:4d}  ({100.0 * n / total:5.1f}%)")

    def _stats(xs, name):
        if not xs:
            return
        xs = sorted(xs)
        med = xs[len(xs) // 2]
        print(f"  {name:22s} min={xs[0]:.1f} med={med:.1f} max={xs[-1]:.1f} n={len(xs)}")

    print("\n=== DISTRIBUTIONS ===")
    _stats(holds, "jump_hold_frames")
    _stats(travels, "horizontal_travel")

    print("\n=== PER-LEVEL DEATH COUNT (top 10) ===")
    for lid, n in per_level.most_common(10):
        print(f"  {n:4d}  {lid}")


if __name__ == "__main__":
    main()
