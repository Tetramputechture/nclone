"""Same-binary A/B for the native enumeration-kernel fast paths.

Runs native ``enumerate_edges`` twice on each map — once with
``NCLONE_KERNEL_BASELINE`` set (forces the original full per-frame collision and
the unconditional deadly-mine check) and once without (the K1 open-air collision
skip + K3 deadly-mine broadphase) — and asserts the edge sets *and* air-goal
contacts are byte-identical, while reporting the wall-time speedup.

Because both runs use the same binary and the same native kernel, a byte-identical
result proves the fast paths do not alter physics (the legacy/native parity gate
in ``native_parity_check`` cannot: both its sides step the same tick functions, so
a uniform physics shift would move them together).

Usage:
    .venv/bin/python -m nclone.value_function.tools.kernel_ab \
        nclone/test_maps/simple-walk nclone/test_maps/long-jump-reachable \
        nclone/test_maps/di-required [N]

Trailing integer N overrides the best-of-N repeat count (default 2).
Exit code 0 iff every map is byte-identical.
"""
import os
import sys
import time

import numpy as np

from ..surface_extraction import extract_surfaces
from ..surface_graph import build_graph
from ..edge_enumeration import enumerate_edges
from ..physics_step_wrapper import PhysicsStepWrapper
from ...constants.physics_constants import (
    NINJA_RADIUS, EXIT_SWITCH_RADIUS, EXIT_DOOR_RADIUS,
)
from .lattice_audit import _load_map_data, _goal_positions
import nclone_native as Nnative


def build(map_path):
    md = _load_map_data(map_path)
    sim = Nnative.Simulator.load(md)
    switch, exit_, mines = _goal_positions(sim)
    all_mines = [(x, y) for (x, y, _s) in mines]
    phys = PhysicsStepWrapper(map_data=md)
    sp = extract_surfaces(phys.tile_dic, mine_positions=all_mines)
    g = build_graph(sp)
    mine_pos = np.asarray(all_mines, dtype=np.float32) if all_mines else None
    goals = [
        (switch[0], switch[1], NINJA_RADIUS + EXIT_SWITCH_RADIUS),
        (exit_[0], exit_[1], NINJA_RADIUS + EXIT_DOOR_RADIUS),
    ]
    return md, g, mine_pos, goals


def edge_key(e):
    return (int(e.src_node), int(e.dst_node), round(float(e.weight), 6),
            int(e.jump_hold_frames), int(e.h_input_during), int(e.h_input_after))


def enum_once(map_path, baseline):
    md, g, mine_pos, goals = build(map_path)
    if baseline:
        os.environ["NCLONE_KERNEL_BASELINE"] = "1"
    else:
        os.environ.pop("NCLONE_KERNEL_BASELINE", None)
    stats = {}
    t = time.time()
    edges, _ = enumerate_edges(
        g, md, collect_waypoints=False, mine_positions=mine_pos,
        goal_positions=goals, stats_out=stats, use_native=True)
    dt = time.time() - t
    keys = sorted(edge_key(e) for e in edges)
    gc = {int(gi): dict(d) for gi, d in stats.get("goal_contacts", {}).items()}
    return keys, gc, dt


def best(map_path, baseline, n):
    bt = float("inf")
    keys = gc = None
    for _ in range(n):
        keys, gc, dt = enum_once(map_path, baseline)
        bt = min(bt, dt)
    return keys, gc, bt


def run(map_path, n):
    kb, gb, tb = best(map_path, True, n)
    ko, go, to = best(map_path, False, n)
    identical = (kb == ko) and (gb == go)
    print(f"{map_path}: edges={len(ko)}  baseline={tb:.2f}s  optimized={to:.2f}s  "
          f"speedup={tb / max(to, 1e-9):.2f}x  BYTE-IDENTICAL={identical}")
    if not identical:
        sb, so = set(kb), set(ko)
        print(f"  edge diff: only-baseline={len(sb - so)} "
              f"only-optimized={len(so - sb)}  goal_contacts_same={gb == go}")
    return identical


if __name__ == "__main__":
    args = sys.argv[1:]
    n = 2
    if args and args[-1].isdigit():
        n = int(args[-1])
        args = args[:-1]
    ok = True
    for mp in args:
        ok = run(mp, n) and ok
    os.environ.pop("NCLONE_KERNEL_BASELINE", None)
    sys.exit(0 if ok else 1)
