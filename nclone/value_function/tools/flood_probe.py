"""Size the reachability-flood opportunity.

Runs the native arc sweep in measurement mode (``nclone_native.flood_stats``):
instead of emitting edges it counts total arc ticks vs distinct quantized
``(pre-state, input)`` transitions, at several position/velocity quantization
grains.  ``total / distinct`` is the redundancy a transposition-merging flood
could collapse at that grain — i.e. the per-level speedup ceiling for the
state-space-flood restructure (idea 1).  Coarser grains merge more (faster, less
accurate); the lattice itself quantizes positions to ~6px and velocities to bin
spacings of ~0.2-0.5, so the (qpos=6, qvel=0.5) row is roughly the
"quantization-equivalent" operating point.

Usage:
    .venv/bin/python -m nclone.value_function.tools.flood_probe \
        nclone/test_maps/simple-walk nclone/test_maps/di-required
"""
import os
import sys

from ..edge_enumeration import (
    RUN_DURATIONS, JUMP_HOLD_DURATIONS, H_INPUTS, POST_RELEASE_INPUTS,
    MAX_JUMP_DURATION, MAX_ARC_FRAMES_DEFAULT, _LAUNCH_PAD_BOOST_MAG,
)
from ..surface_graph import (
    velocity_bin_centers, num_velocity_bins, _SPATIAL_CELL_SIZE,
)
from ..surface_extraction import SURFACE_FLOOR, SURFACE_LAUNCH_PAD
from .kernel_ab import build
import nclone_native as Nnative


def build_args(map_path):
    md, g, mine_pos, goals = build(map_path)
    use_mine = mine_pos is not None and len(mine_pos) > 0

    node_pt = []; node_vb = []; node_x = []; node_y = []
    node_vx0 = []; node_vy0 = []; node_surf = []; node_edge = []
    for pt_idx, sp in enumerate(g.surface_points):
        st = int(sp.surface_type)
        centers = velocity_bin_centers(st)
        for vb in range(num_velocity_bins(st)):
            vel = float(centers[vb])
            if st == SURFACE_FLOOR:
                vx0, vy0 = vel, 0.0
            elif st == SURFACE_LAUNCH_PAD:
                vx0 = sp.normal_x * _LAUNCH_PAD_BOOST_MAG
                vy0 = sp.normal_y * _LAUNCH_PAD_BOOST_MAG
            else:
                vx0, vy0 = 0.0, vel
            node_pt.append(pt_idx); node_vb.append(vb)
            node_x.append(float(sp.x)); node_y.append(float(sp.y))
            node_vx0.append(float(vx0)); node_vy0.append(float(vy0))
            node_surf.append(st); node_edge.append(bool(sp.is_edge))

    goals_flat = []
    for (gx, gy, r2) in goals:
        goals_flat += [float(gx), float(gy), float(r2)]
    sp_x = [float(s.x) for s in g.surface_points]
    sp_y = [float(s.y) for s in g.surface_points]
    sp_type = [int(s.surface_type) for s in g.surface_points]
    fc = [float(v) for v in velocity_bin_centers(0)]
    wc = [float(v) for v in velocity_bin_centers(1)]
    sc = [float(v) for v in velocity_bin_centers(2)]
    lc = [float(v) for v in velocity_bin_centers(3)]
    n_threads = os.cpu_count() or 1

    args = (
        list(md),
        node_x, node_y, node_vx0, node_vy0, node_surf, node_edge,
        node_pt, node_vb,
        list(RUN_DURATIONS), list(JUMP_HOLD_DURATIONS),
        list(H_INPUTS), list(POST_RELEASE_INPUTS),
        int(MAX_ARC_FRAMES_DEFAULT), int(MAX_JUMP_DURATION), bool(use_mine),
        goals_flat, sp_x, sp_y, sp_type, fc, wc, sc, lc,
        float(_SPATIAL_CELL_SIZE), int(g.max_vel_bins), n_threads,
    )
    return args, g


def run(map_path, grains):
    args, g = build_args(map_path)
    print(f"\n{map_path}: nodes={g.n_nodes}")
    print(f"  {'qpos':>5} {'qvel':>5}  {'all redund.':>12}   {'COLLISION redund. (expensive ticks)':>40}")
    for (qpos, qvel) in grains:
        total, distinct, ctotal, cdistinct = Nnative.flood_stats(
            *args, float(qpos), float(qvel))
        ratio = total / max(1, distinct)
        cratio = ctotal / max(1, cdistinct)
        cfrac = 100.0 * ctotal / max(1, total)
        print(f"  {qpos:>5} {qvel:>5}  {ratio:10.2f}x   "
              f"coll_ticks={ctotal:>13,} ({cfrac:4.1f}% of all)  redund={cratio:6.2f}x")


if __name__ == "__main__":
    grains = [(0.5, 0.05), (1.0, 0.1), (3.0, 0.25), (6.0, 0.5), (12.0, 1.0)]
    for mp in sys.argv[1:]:
        run(mp, grains)
