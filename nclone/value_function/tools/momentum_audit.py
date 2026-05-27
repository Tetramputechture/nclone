"""Momentum-loss audit for the lattice planner (read-only diagnostic).

Quantifies WHERE the single-axis velocity binning loses momentum, to gate the
momentum-preservation work (plan: parsed-purring-wave).  The physics
(``ninja.py``) zeroes the surface-NORMAL velocity at every node, so a 2D
``(vx,vy)`` grid is the wrong fix; the real losses are (a) bin granularity /
range of the surviving TANGENTIAL speed and (b) the earliest-contact truncation.
This tool measures both, plus how many curved tiles the extractor drops today.

What it reports per map:
  * per-bucket CLIP-RATE  — landings whose tangential speed saturates the bin
    grid (first/last bin).  High => the bin RANGE is too narrow.
  * per-bucket QUANT-DELTA — ``|raw_vel - bin_center|`` vs the local bin spacing.
    High fraction => the bin GRANULARITY is too coarse.
  * spawn-d* GRAIN sensitivity — how much the optimal route length (spawn d*)
    moves when velocity is coarsened (reuses ``grain_dstar_study``); flat =>
    the field is not velocity-sensitive and finer bins won't help.
  * curved-tile COUNT + baseline-route proximity — how many GridSegmentCircular
    tiles are DROPPED today and whether the baseline route passes near them.

Buckets: floor-high-speed (|vx| >= 0.75*MAX_HOR_SPEED), wall-upward (vy < 0:
the coarse -2/-1/-0.4 bins), wall-terminal (vy >= 5.0).

Gate decisions (printed): 1b fires per-bucket if clip>8% OR quant_p90 > local bin
width; 1c fires if wall-upward landings pile onto WALL_VEL_MIN with p50 vy<-2;
2d (circular extraction) fires if curved tiles exist on/near the route.

Usage:
    .venv/bin/python -m nclone.value_function.tools.momentum_audit \
        "nclone/maps/official/SL/250 pipeworks" [more maps...] \
        [--sample-nodes 250] [--no-grain] [--no-route]
"""
from __future__ import annotations

import argparse
import math
import random
import sys
from typing import List, Tuple

import numpy as np

from ...constants.physics_constants import (
    MAX_HOR_SPEED, NINJA_RADIUS, EXIT_SWITCH_RADIUS, EXIT_DOOR_RADIUS,
)
from ...entities import GridSegmentCircular
from ...utils.tile_segment_factory import TileSegmentFactory
from .. import edge_enumeration as EE
from ..physics_step_wrapper import PhysicsStepWrapper
from ..route_extraction import extract_and_simplify_route
from ..surface_extraction import (
    SURFACE_FLOOR, SURFACE_WALL, SURFACE_SLIDE, extract_surfaces, is_wall_like,
)
from ..surface_graph import (
    build_graph, node_id_to_parts, num_velocity_bins, velocity_bin_centers,
    WALL_VEL_MIN,
)
from .lattice_audit import LatticeAudit
from . import grain_dstar_study

_CLIP_GATE = 8.0          # % clipped above which the bin range is too narrow
_CURVE_NEAR_PX = 18.0     # route within this of a curved tile => 2d fires
_FLOOR_FAST = 0.75 * MAX_HOR_SPEED


def _bucket(stype: int, raw: float) -> str:
    if is_wall_like(stype):
        if raw < 0.0:
            return "wall-upward"
        if raw >= 5.0:
            return "wall-terminal"
        return "wall-mid"
    if abs(raw) >= _FLOOR_FAST:
        return "floor-high-speed"
    return "floor-low-speed"


def _spacing(stype: int, raw: float) -> float:
    """Local adjacent-bin spacing (px/f) near ``raw`` for this surface type."""
    centers = np.asarray(velocity_bin_centers(stype), dtype=np.float64)
    if centers.size < 2:
        return float("inf")
    i = int(np.argmin(np.abs(centers - raw)))
    lo = centers[i] - centers[i - 1] if i > 0 else centers[i + 1] - centers[i]
    hi = centers[i + 1] - centers[i] if i < centers.size - 1 else lo
    return float(min(lo, hi))


def _collect_landings(audit: LatticeAudit, sample_nodes: int):
    """Sample source nodes and record (bucket, |quant_delta|, clipped, raw) for
    every landing, by wrapping the real ``_landing_to_node`` during a pure-Python
    single-process enumeration (so we see the exact raw vs binned velocity)."""
    md = audit.map_data
    phys = PhysicsStepWrapper(map_data=md)
    graph = build_graph(extract_surfaces(phys.tile_dic, mine_positions=audit.all_mines))

    records: List[Tuple[str, float, bool, float]] = []
    orig = EE._landing_to_node

    def wrapped(g, landing, return_clip: bool = False):
        node, clipped = orig(g, landing, return_clip=True)
        if node is not None:
            pt, vb = node_id_to_parts(node, g.max_vel_bins)
            stype = g.surface_points[pt].surface_type
            _x, _y, vx, vy, _st = landing
            raw = vy if is_wall_like(stype) else vx
            centers = velocity_bin_centers(stype)
            center = float(centers[vb]) if vb < len(centers) else raw
            records.append((_bucket(stype, raw), abs(raw - center), bool(clipped), raw))
        return (node, clipped) if return_clip else node

    node_indices = [
        (pt, vb)
        for pt, s in enumerate(graph.surface_points)
        for vb in range(num_velocity_bins(s.surface_type))
    ]
    rng = random.Random(0)
    sample = rng.sample(node_indices, min(sample_nodes, len(node_indices)))
    mine_pos = (
        np.asarray(audit.all_mines, dtype=np.float32)
        if (audit.all_mines and audit.mine_inflation) else None
    )
    from nclone_native import Simulator as NativeSimulator
    sim = NativeSimulator.load(md)

    EE._landing_to_node = wrapped
    try:
        for (pt, vb) in sample:
            EE._enumerate_node_edges(
                sim, graph, pt, vb, wp_accum=None, mine_positions=mine_pos,
                max_arc_frames=EE.MAX_ARC_FRAMES_DEFAULT, goals=None,
            )
    finally:
        EE._landing_to_node = orig
    return records, graph


def _report_buckets(records) -> None:
    order = ["floor-high-speed", "floor-low-speed", "wall-upward",
             "wall-mid", "wall-terminal"]
    print(f"  {'bucket':>16} {'n':>7} {'clip%':>7} {'meanΔ':>7} "
          f"{'p90Δ':>7} {'spacing':>8} {'1b?':>4}")
    for b in order:
        rows = [r for r in records if r[0] == b]
        n = len(rows)
        if n == 0:
            continue
        deltas = np.array([r[1] for r in rows])
        clip = 100.0 * sum(1 for r in rows if r[2]) / n
        raws = np.array([r[3] for r in rows])
        # representative spacing from the median-velocity landing in the bucket
        stype = SURFACE_WALL if b.startswith("wall") else SURFACE_FLOOR
        spc = _spacing(stype, float(np.median(raws)))
        p90 = float(np.percentile(deltas, 90))
        fires = (clip > _CLIP_GATE) or (p90 > spc)
        print(f"  {b:>16} {n:>7} {clip:6.1f}% {deltas.mean():7.3f} "
              f"{p90:7.3f} {spc:8.3f} {'YES' if fires else '-':>4}")

    # 1c gate: wall-upward landings piling onto WALL_VEL_MIN.
    up = [r for r in records if r[0] == "wall-upward"]
    if up:
        raws = np.array([r[3] for r in up])
        at_min = 100.0 * sum(1 for r in up if r[2]) / len(up)  # clipped == at extreme bin
        p50 = float(np.median(raws))
        fires_1c = at_min > _CLIP_GATE and p50 < WALL_VEL_MIN
        print(f"  wall-upward: p50 vy={p50:.2f} (WALL_VEL_MIN={WALL_VEL_MIN}), "
              f"clip-to-extreme={at_min:.1f}%  => 1c {'FIRES' if fires_1c else 'defer'}")


def _curved_tiles(map_data) -> List[Tuple[float, float]]:
    """Representative contact points (arc midpoint + endpoints) of every curved
    tile, via the REAL segment geometry the collision uses."""
    phys = PhysicsStepWrapper(map_data=map_data)
    seg_dic = TileSegmentFactory.create_segment_dictionary(phys.tile_dic)
    pts: List[Tuple[float, float]] = []
    for _cell, segs in seg_dic.items():
        for s in segs:
            if isinstance(s, GridSegmentCircular):
                inv = 1.0 / math.sqrt(2.0)
                mid = (s.xpos + s.radius * s.hor * inv, s.ypos + s.radius * s.ver * inv)
                pts.extend([mid, s.p_hor, s.p_ver])
    return pts


def _route_proximity(audit: LatticeAudit, curved: List[Tuple[float, float]]) -> None:
    n_curved_tiles = len(curved) // 3
    if not curved:
        print("  curved tiles: 0  => 2d N/A on this map")
        return
    try:
        pc = audit.precompute(use_cache=True)
        g, d = pc["graph"], pc["d_comb"]
        spawn = audit._spawn_node(g, d)
        route = (extract_and_simplify_route(
            g, d, pc["pred_comb"], pc["best_comb"], spawn, dedup_loops=True)
            if spawn is not None else [])
    except Exception as e:  # noqa: BLE001 - diagnostic must not crash
        print(f"  curved tiles: {n_curved_tiles}  (route unavailable: {e})")
        return
    cur = np.asarray(curved)
    best = float("inf")
    for a in route:
        for (px, py) in ((a.launch_x, a.launch_y), (a.landing_x, a.landing_y)):
            d2 = ((cur[:, 0] - px) ** 2 + (cur[:, 1] - py) ** 2).min()
            best = min(best, math.sqrt(float(d2)))
    fires = (not route) or (best <= _CURVE_NEAR_PX)
    near = "no route" if not route else f"min route->curve dist={best:.1f}px"
    print(f"  curved tiles: {n_curved_tiles} (DROPPED today); {near}  "
          f"=> 2d {'FIRES' if fires else 'defer'}")


def run(map_path: str, sample_nodes: int, do_grain: bool, do_route: bool) -> None:
    print(f"\n=== {map_path} ===")
    audit = LatticeAudit(map_path=map_path, use_native=True)
    records, _g = _collect_landings(audit, sample_nodes)
    print(f"sampled {sample_nodes} source nodes -> {len(records)} landings")
    _report_buckets(records)
    if do_route:
        _route_proximity(audit, _curved_tiles(audit.map_data))
    if do_grain:
        print("  spawn-d* vs velocity grain (qpos=0):")
        for qv in (0.0, 0.1, 0.25):
            _d, _n, spawn = grain_dstar_study.compute_dstar(audit, 0.0, qv)
            sd = f"{spawn:.1f}f" if np.isfinite(spawn) else "INF"
            print(f"    qvel={qv:<5} spawn_d*={sd}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("maps", nargs="+")
    p.add_argument("--sample-nodes", type=int, default=250)
    p.add_argument("--no-grain", action="store_true")
    p.add_argument("--no-route", action="store_true")
    args = p.parse_args()
    for mp in args.maps:
        run(mp, args.sample_nodes, not args.no_grain, not args.no_route)
    return 0


if __name__ == "__main__":
    sys.exit(main())
