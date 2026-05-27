"""Grain-vs-d* accuracy study for the collision-merge idea.

Builds the exact d* cost-to-go lattice, then rebuilds it with collision-merge
*snapping* at several quantization grains (ninja pos/vel snapped to a grid after
each collision tick, forcing the same grain-G equivalence classes a collision
cache would create), and reports how much the d* potential degrades:

  * reach%   — fraction of exact-reachable nodes still reachable
  * lost     — nodes reachable in exact but not at grain G
  * |Δ| stats over nodes finite in both (cost-to-go error, in frames)
  * spawn d* — the single most important value: cost-to-go from spawn
               (≈ optimal route length); exact vs grain G

This is the accuracy half of the collision-cache decision (the speed ceiling is
the collision-tick redundancy from flood_probe).  Snapping is a *proxy* for the
cache (post-state snap, conservative on position), so degradation here is an
upper bound on a response-applying cache's error.

Usage:
    .venv/bin/python -m nclone.value_function.tools.grain_dstar_study \
        nclone/test_maps/simple-walk nclone/test_maps/di-required
"""
import sys

import numpy as np

from ..surface_extraction import extract_surfaces
from ..surface_graph import (
    build_graph, _build_reverse_adjacency, compute_two_phase_cost_to_go,
)
from ..edge_enumeration import enumerate_edges, populate_graph_edges
from ..route_extraction import find_nearest_start_node
from ..physics_step_wrapper import PhysicsStepWrapper
from ...constants.physics_constants import (
    NINJA_RADIUS, EXIT_SWITCH_RADIUS, EXIT_DOOR_RADIUS,
)
from .lattice_audit import LatticeAudit


def compute_dstar(audit, snap_qpos, snap_qvel):
    """Replicates LatticeAudit.precompute's d* steps but with collision-merge
    snapping in enumeration.  Trick edges (added identically in both exact and
    snapped runs, from the exact sim) keep trick-gated switches reachable."""
    md = audit.map_data
    switch, exit_ = audit.switch, audit.exit
    all_mines = audit.all_mines
    phys = PhysicsStepWrapper(map_data=md)
    sp = extract_surfaces(phys.tile_dic, mine_positions=all_mines)
    g = build_graph(sp)
    mine_pos = (np.asarray(all_mines, dtype=np.float32)
                if (all_mines and audit.mine_inflation) else None)
    goals = [
        (switch[0], switch[1], NINJA_RADIUS + EXIT_SWITCH_RADIUS),
        (exit_[0], exit_[1], NINJA_RADIUS + EXIT_DOOR_RADIUS),
    ]
    stats = {}
    edges, _ = enumerate_edges(
        g, md, num_workers=audit.workers, collect_waypoints=False,
        mine_positions=mine_pos, goal_positions=goals, stats_out=stats,
        use_native=audit.use_native, snap_qpos=snap_qpos, snap_qvel=snap_qvel)
    populate_graph_edges(g, edges)
    gc = stats.get("goal_contacts", {})
    sw_air = dict(gc.get(0, {}))
    ex_air = dict(gc.get(1, {}))
    tsw, tex = audit._append_cdj_trick_edges(g)
    for n, f in tsw.items():
        sw_air[n] = min(sw_air.get(n, f), f)
    for n, f in tex.items():
        ex_air[n] = min(ex_air.get(n, f), f)
    rev_adj = _build_reverse_adjacency(g)
    d_comb = compute_two_phase_cost_to_go(
        g, switch[0], switch[1], exit_[0], exit_[1], rev_adj=rev_adj,
        switch_air_contacts=sw_air, exit_air_contacts=ex_air)[0]
    d_comb = np.asarray(d_comb, dtype=np.float64)
    sx, sy = audit.spawn
    spawn_node = find_nearest_start_node(g, sx, sy, 0.0, 0.0, 1, d_comb)
    spawn_d = float(d_comb[spawn_node]) if spawn_node is not None else float("inf")
    return d_comb, len(edges), spawn_d


def run(map_path, grains):
    audit = LatticeAudit(map_path=map_path, use_native=True)
    d_exact, n_edges, spawn_exact = compute_dstar(audit, 0.0, 0.0)
    finite = np.isfinite(d_exact)
    n_exact = int(finite.sum())
    print(f"\n{map_path}: exact d* finite={n_exact}  edges={n_edges}  "
          f"spawn_d*={spawn_exact:.1f}f")
    print(f"  {'grain':>11} {'edges':>9} {'reach%':>7} {'lost':>6} "
          f"{'mean|Δ|':>8} {'p90|Δ|':>8} {'max|Δ|':>8} {'spawn_d*':>10}")
    for (qp, qv) in grains:
        d_g, n_e, spawn_g = compute_dstar(audit, qp, qv)
        fin_g = np.isfinite(d_g)
        both = finite & fin_g
        lost = int((finite & ~fin_g).sum())
        reach = 100.0 * int(fin_g.sum()) / max(1, n_exact)
        diff = np.abs(d_g[both] - d_exact[both])
        mean_d = float(diff.mean()) if diff.size else 0.0
        p90 = float(np.percentile(diff, 90)) if diff.size else 0.0
        mx = float(diff.max()) if diff.size else 0.0
        sd = (f"{spawn_g:.1f}f" if np.isfinite(spawn_g)
              else "INF(lost)")
        print(f"  {qp:>4}px/{qv:<4} {n_e:>9} {reach:6.1f}% {lost:>6} "
              f"{mean_d:8.2f} {p90:8.2f} {mx:8.2f} {sd:>10}")


if __name__ == "__main__":
    grains = [(0.5, 0.05), (1.0, 0.1), (2.0, 0.2), (3.0, 0.25), (6.0, 0.5)]
    for mp in sys.argv[1:]:
        run(mp, grains)
