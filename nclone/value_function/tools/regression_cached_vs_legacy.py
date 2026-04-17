"""Regression harness: cached tile-neighborhood pipeline vs legacy enumerator.

For each test level we run edge enumeration twice - once with the legacy
native-sim per-node path and once with the three-phase cached pipeline -
and compare:

* number of edges and nodes
* per-node d_star to switch and exit (Dijkstra cost-to-go)
* fraction of cached launch / landing entries that had to be populated
  on the fly (cache-miss fallback rate)

This is the level-of-detail required by the plan's "regression harness"
milestone: keep ``d_star_switch`` and ``d_star_exit`` within 1e-3 frames
of the legacy baseline on simple maps.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from nclone.value_function.edge_enumeration import (
    enumerate_edges,
    populate_graph_edges,
)
from nclone.value_function.physics_step_wrapper import PhysicsStepWrapper
from nclone.value_function.surface_extraction import extract_surfaces
from nclone.value_function.surface_graph import (
    _build_reverse_adjacency,
    build_graph,
    compute_cost_to_go_with_tree,
)


@dataclass
class RunReport:
    level: str
    n_nodes: int
    n_edges_legacy: int
    n_edges_cached: int
    t_legacy_s: float
    t_cached_s: float
    max_dsw_abs_err: float  # max |d_star_switch_cached - d_star_switch_legacy|
    max_dex_abs_err: float
    med_dsw_abs_err: float
    med_dex_abs_err: float
    reachable_mismatch_sw: int
    reachable_mismatch_ex: int


def _load_map(map_path: str) -> Tuple[List[int], PhysicsStepWrapper]:
    with open(map_path, "rb") as f:
        raw = f.read()
    map_data = [int(b) for b in raw]
    physics = PhysicsStepWrapper(map_path=map_path)
    return map_data, physics


def _run_pipeline(
    map_data: list,
    physics: PhysicsStepWrapper,
    *,
    tile_cache_dir: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, int, int, float]:
    """Run extract -> build_graph -> enumerate -> Dijkstra.

    Returns ``(d_star_switch, d_star_exit, n_nodes, n_edges, elapsed_s)``.
    """
    tile_dict = physics.tile_dic
    mine_entries = []
    for e in physics.entity_dic.get(21, []):
        mine_entries.append((float(e.xpos), float(e.ypos)))
    for e in physics.entity_dic.get(1, []):
        mine_entries.append((float(e.xpos), float(e.ypos)))

    mine_list = mine_entries
    if mine_entries:
        mine_arr = np.asarray(mine_entries, dtype=np.float32)
    else:
        mine_arr = np.empty((0, 2), dtype=np.float32)

    t0 = time.time()
    surface_points = extract_surfaces(tile_dict, mine_positions=mine_list)
    graph = build_graph(surface_points)
    stats: dict = {}
    edges, _ = enumerate_edges(
        graph,
        map_data=map_data,
        num_workers=1,                   # single-process for fair comparison
        collect_waypoints=False,
        verbose=False,
        mine_positions=mine_arr,
        tile_cache_dir=tile_cache_dir,
        stats_out=stats,
    )
    if tile_cache_dir is not None and stats:
        print(
            f"    cache: launch_hit={stats.get('launch_hit_rate', 0.0)*100:.1f}% "
            f"({stats.get('launch_misses', 0)}/{stats.get('launch_lookups', 0)} miss), "
            f"landing_hit={stats.get('landing_hit_rate', 0.0)*100:.1f}% "
            f"({stats.get('landing_misses', 0)}/{stats.get('landing_lookups', 0)} miss)",
            flush=True,
        )
    populate_graph_edges(graph, edges)
    sw = physics.get_goal_position("switch")
    ex = physics.get_goal_position("exit")
    rev_adj = _build_reverse_adjacency(graph)
    d_star_switch, _, _ = compute_cost_to_go_with_tree(
        graph, sw[0], sw[1], "switch", rev_adj=rev_adj,
    )
    d_star_exit, _, _ = compute_cost_to_go_with_tree(
        graph, ex[0], ex[1], "exit", rev_adj=rev_adj,
    )
    elapsed = time.time() - t0
    return d_star_switch, d_star_exit, graph.n_nodes, len(edges), elapsed


def _finite_abs_err(a: np.ndarray, b: np.ndarray) -> Tuple[float, float, int]:
    """Return (max_abs_err, median_abs_err, reachability_mismatch)."""
    ok = np.isfinite(a) & np.isfinite(b)
    if not np.any(ok):
        return 0.0, 0.0, int(np.sum(np.isfinite(a) != np.isfinite(b)))
    diff = np.abs(a[ok] - b[ok])
    return (
        float(np.max(diff)),
        float(np.median(diff)),
        int(np.sum(np.isfinite(a) != np.isfinite(b))),
    )


def run_regression(
    map_path: str,
    *,
    keep_cache_dir: bool = False,
) -> RunReport:
    level = os.path.splitext(os.path.basename(map_path))[0]
    map_data, physics = _load_map(map_path)
    (
        dsw_leg, dex_leg, n_nodes, n_edges_leg, t_leg,
    ) = _run_pipeline(map_data, physics)

    with tempfile.TemporaryDirectory() as td:
        cache_dir = td if not keep_cache_dir else os.path.join(
            os.path.dirname(map_path), f"_cache_{level}",
        )
        os.makedirs(cache_dir, exist_ok=True)

        (
            dsw_cached, dex_cached, n_nodes_c, n_edges_c, t_cached,
        ) = _run_pipeline(map_data, physics, tile_cache_dir=cache_dir)

    max_sw, med_sw, mis_sw = _finite_abs_err(dsw_cached, dsw_leg)
    max_ex, med_ex, mis_ex = _finite_abs_err(dex_cached, dex_leg)

    assert n_nodes == n_nodes_c, (
        f"node count mismatch: {n_nodes} vs {n_nodes_c}"
    )
    return RunReport(
        level=level,
        n_nodes=n_nodes,
        n_edges_legacy=n_edges_leg,
        n_edges_cached=n_edges_c,
        t_legacy_s=t_leg,
        t_cached_s=t_cached,
        max_dsw_abs_err=max_sw,
        max_dex_abs_err=max_ex,
        med_dsw_abs_err=med_sw,
        med_dex_abs_err=med_ex,
        reachable_mismatch_sw=mis_sw,
        reachable_mismatch_ex=mis_ex,
    )


_DEFAULT_LEVELS = [
    "simple-walk",
    "only-jump",
    "rampjump",
    "fall-required",
    "minefield",
]


def _format_report(report: RunReport) -> str:
    speedup = report.t_legacy_s / max(report.t_cached_s, 1e-9)
    return (
        f"{report.level:20s} "
        f"nodes={report.n_nodes:>6d} "
        f"edges(leg/cache)={report.n_edges_legacy:>6d}/{report.n_edges_cached:>6d} "
        f"t={report.t_legacy_s:>6.1f}s/{report.t_cached_s:>6.1f}s ({speedup:.1f}x) "
        f"d*_switch max_err={report.max_dsw_abs_err:.3e} "
        f"med={report.med_dsw_abs_err:.3e} "
        f"d*_exit max_err={report.max_dex_abs_err:.3e} "
        f"med={report.med_dex_abs_err:.3e} "
        f"mismatch(sw/ex)={report.reachable_mismatch_sw}/{report.reachable_mismatch_ex}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--maps-dir",
        default=os.path.join(
            os.path.dirname(__file__), "..", "..", "test_maps",
        ),
        help="Directory containing test maps.",
    )
    parser.add_argument(
        "--levels", nargs="*", default=_DEFAULT_LEVELS,
        help="List of test-level basenames.",
    )
    parser.add_argument(
        "--tol", type=float, default=1e-3,
        help="Max allowed d_star absolute error (frames).",
    )
    args = parser.parse_args()

    reports = []
    failures = 0
    for level in args.levels:
        path = os.path.join(args.maps_dir, level)
        if not os.path.isfile(path):
            print(f"skip {level}: no such map")
            continue
        try:
            r = run_regression(path)
        except Exception as e:
            print(f"{level}: exception {e}")
            failures += 1
            continue
        reports.append(r)
        print(_format_report(r))
        if r.max_dsw_abs_err > args.tol or r.max_dex_abs_err > args.tol:
            failures += 1

    if failures:
        print(f"\nFAILED: {failures} regressions above tol={args.tol}")
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
