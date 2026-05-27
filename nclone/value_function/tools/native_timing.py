"""Clean native-only enumeration timing (no legacy run, no CPU contention).

Reports best-of-N wall time for the native ``enumerate_edges`` pass on each map.
Use this to measure the native kernel in isolation (e.g. to A/B a kernel change)
rather than the ratio reported by ``native_parity_check`` — that script runs the
slow legacy pass alongside, which pollutes the native timing via contention.

Usage:
    .venv/bin/python -m nclone.value_function.tools.native_timing \
        nclone/test_maps/long-jump-reachable [N]

A trailing integer argument overrides the repeat count N (default 3).
"""
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


def main(map_path, n=3):
    md, _, mine_pos, goals = build(map_path)
    best = float("inf")
    edges_n = 0
    for _ in range(n):
        # Fresh graph each rep: enumerate_edges populates it in place.
        g = build(map_path)[1]
        t = time.time()
        e, _ = enumerate_edges(g, md, collect_waypoints=False,
                               mine_positions=mine_pos, goal_positions=goals,
                               use_native=True)
        dt = time.time() - t
        edges_n = len(e)
        best = min(best, dt)
    print(f"{map_path}: native best-of-{n} = {best:.2f}s  edges={edges_n}")


if __name__ == "__main__":
    args = sys.argv[1:]
    reps = 3
    if args and args[-1].isdigit():
        reps = int(args[-1])
        args = args[:-1]
    for mp in args:
        main(mp, reps)
