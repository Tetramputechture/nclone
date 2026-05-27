"""Byte-identical parity: legacy (workers=1) vs native ``enumerate_edges``.

Builds the surface graph exactly as ``LatticeAudit.precompute`` does, runs
``enumerate_edges`` twice (``use_native=False`` then ``True``) and asserts the
edge sets *and* the air-reachable goal_contacts are byte-identical.  This is the
correctness gate for the native Rust enumeration kernel (see
``nclone-native/src/enumerate.rs``).

Usage:
    .venv/bin/python -m nclone.value_function.tools.native_parity_check \
        nclone/test_maps/simple-walk nclone/test_maps/di-required ...

Exit code 0 iff every map PASSES.  Legacy single-process enumeration is slow
(~3-6 min/large map), so run maps in parallel background shells if iterating.
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
    """Build (map_data, graph, mine_positions, goal_positions) for a map."""
    map_data = _load_map_data(map_path)
    sim = Nnative.Simulator.load(map_data)
    switch, exit_, mines = _goal_positions(sim)
    all_mines = [(x, y) for (x, y, _s) in mines]

    phys = PhysicsStepWrapper(map_data=map_data)
    surface_points = extract_surfaces(phys.tile_dic, mine_positions=all_mines)
    graph = build_graph(surface_points)
    mine_pos = np.asarray(all_mines, dtype=np.float32) if all_mines else None
    goal_positions = [
        (switch[0], switch[1], NINJA_RADIUS + EXIT_SWITCH_RADIUS),
        (exit_[0], exit_[1], NINJA_RADIUS + EXIT_DOOR_RADIUS),
    ]
    return map_data, graph, mine_pos, goal_positions


def edge_key(e):
    return (int(e.src_node), int(e.dst_node), round(float(e.weight), 6),
            int(e.jump_hold_frames),
            int(e.h_input_during),
            int(e.h_input_after))


def run(map_path):
    map_data, graph, mine_pos, goals = build(map_path)
    print(f"map={map_path} nodes={graph.n_nodes} "
          f"mines={0 if mine_pos is None else len(mine_pos)}")

    # legacy single-process
    g1 = build(map_path)[1]
    s_leg = {}
    t0 = time.time()
    e_leg, _ = enumerate_edges(
        g1, map_data, num_workers=1, collect_waypoints=False,
        mine_positions=mine_pos, stats_out=s_leg, goal_positions=goals,
        use_native=False)
    t_leg = time.time() - t0

    # native
    g2 = build(map_path)[1]
    s_nat = {}
    t0 = time.time()
    e_nat, _ = enumerate_edges(
        g2, map_data, collect_waypoints=False,
        mine_positions=mine_pos, stats_out=s_nat, goal_positions=goals,
        use_native=True)
    t_nat = time.time() - t0

    k_leg = sorted(edge_key(e) for e in e_leg)
    k_nat = sorted(edge_key(e) for e in e_nat)
    identical = (k_leg == k_nat)
    print(f"  legacy edges={len(e_leg)} ({t_leg:.1f}s)  "
          f"native edges={len(e_nat)} ({t_nat:.1f}s)  "
          f"speedup={t_leg / max(t_nat, 1e-9):.1f}x")
    print(f"  EDGE SET BYTE-IDENTICAL: {identical}")
    if not identical:
        sl, sn = set(k_leg), set(k_nat)
        only_leg = sl - sn
        only_nat = sn - sl
        print(f"  only-legacy={len(only_leg)} only-native={len(only_nat)}")
        for k in list(only_leg)[:5]:
            print("   L", k)
        for k in list(only_nat)[:5]:
            print("   N", k)

    gc_leg = s_leg.get("goal_contacts", {})
    gc_nat = s_nat.get("goal_contacts", {})
    gc_ok = True
    for gi in sorted(set(gc_leg) | set(gc_nat)):
        a = dict(gc_leg.get(gi, {}))
        b = dict(gc_nat.get(gi, {}))
        same = (a == b)
        gc_ok = gc_ok and same
        print(f"  goal_contacts[{gi}]: legacy={len(a)} native={len(b)} "
              f"identical={same}")
    print(f"  RESULT: {'PASS' if identical and gc_ok else 'FAIL'}")
    return identical and gc_ok


if __name__ == "__main__":
    ok = True
    for mp in sys.argv[1:]:
        ok = run(mp) and ok
    sys.exit(0 if ok else 1)
