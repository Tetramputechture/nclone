"""Three-phase cached edge enumeration for the surface graph.

This is the fast path used when a tile-neighborhood cache is available.
For every ``(pt_idx, vel_bin)`` source node, we enumerate input classes
and, for each class, run:

1. **Launch phase** - 3x3 tile-neighborhood lookup around the source
   surface's cell.  Returns either a full in-window landing (weight +
   destination node) or a handoff state at the exact frame the ninja
   exited the 3x3 window.
2. **Ballistic phase** - analytical free-flight propagation of the
   handoff state against the level's ``tile_dict``.  On first solid
   tile contact it hands off to the landing phase.
3. **Landing phase** - 3x3 tile-neighborhood lookup around the impact
   cell.  Resolves the final surface/velocity/subcell plus a small
   frame offset for collision resolution.

On cache miss, the relevant entry is populated on the fly via the
harness simulator (so that subsequent lookups - including later edges
in this very run - hit the cache).  Fallback to the legacy per-node
enumeration is controlled by the caller.

This module intentionally reuses as much as possible from
:mod:`nclone.value_function.edge_enumeration` (EdgeResult,
_inflate_arc_cost, etc.) to keep the two paths' output schema aligned.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..constants import TILE_PIXEL_SIZE, NINJA_RADIUS
from .edge_enumeration import (
    ARC_WAYPOINT_INTERVAL_DEFAULT,
    EdgeResult,
    MAX_ARC_FRAMES_DEFAULT,
    _MINE_INFLATION_BETA_DEFAULT,
    _MINE_KILL_RADIUS,
    _inflate_arc_cost,
)
from .surface_extraction import SURFACE_FLOOR, SURFACE_WALL
from .surface_graph import (
    find_nearest_point,
    node_id,
    num_velocity_bins,
    velocity_bin_centers,
    velocity_to_bin,
)
from .tile_cache import (
    Landed,
    LaunchEntry,
    LandingEntry,
    TileCacheStore,
    classes_for_surface,
    decode_input_class,
    extract_3x3_signature,
    hash_signature,
    landing_lookup,
    launch_lookup,
    populate_landing_entry,
    populate_launch_entry,
    propagate_and_sweep,
    quantise_air_velocity,
    quantise_subcell,
    tile_dict_from_map_data,
)

logger = logging.getLogger(__name__)

_POS_SCALE_INV = 1.0            # positions quantised at 1 px in LaunchEntry
_VEL_SCALE_INV = 0.01           # velocities quantised at 0.01 px/frame
_INV_TILE = 1.0 / TILE_PIXEL_SIZE


def _source_cell_and_subcell(
    sp_x: float,
    sp_y: float,
    surface_type: int,
) -> Tuple[int, int, Tuple[int, int]]:
    """Return ``(cell_col, cell_row, (sx, sy))`` for a surface point.

    Positions are in the simulator's padded pixel frame, so the cell
    indices map directly into the 44x25 tile grid returned by
    :func:`tile_grid_from_map_data`.
    """
    col = int(math.floor(sp_x * _INV_TILE))
    row = int(math.floor(sp_y * _INV_TILE))
    in_x = sp_x - col * TILE_PIXEL_SIZE
    in_y = sp_y - row * TILE_PIXEL_SIZE
    return col, row, quantise_subcell(in_x, in_y)


def _launch_entry_world_pos(
    entry: LaunchEntry,
    sp_x: float,
    sp_y: float,
) -> Tuple[float, float]:
    """Recover the in-window landing position in world pixels."""
    # Note: LaunchEntry stores cell offsets relative to the harness centre,
    # which is (sp_col, sp_row) when this entry is applied to the level.
    # sub_* are absolute pixel offsets inside the landing cell (0..23).
    src_col, src_row, _ = _source_cell_and_subcell(sp_x, sp_y, SURFACE_FLOOR)
    land_col = src_col + entry.land_cell_dx
    land_row = src_row + entry.land_cell_dy
    return (
        land_col * TILE_PIXEL_SIZE + entry.land_subcell_x,
        land_row * TILE_PIXEL_SIZE + entry.land_subcell_y,
    )


def _landing_entry_world_pos(
    entry: LandingEntry,
    impact_col: int,
    impact_row: int,
) -> Tuple[float, float]:
    land_col = impact_col + entry.landing_cell_dx
    land_row = impact_row + entry.landing_cell_dy
    return (
        land_col * TILE_PIXEL_SIZE + entry.landing_subcell_x,
        land_row * TILE_PIXEL_SIZE + entry.landing_subcell_y,
    )


def _resolve_landing_to_node(
    graph,
    x: float,
    y: float,
    vel_bin: int,
    surface_type: int,
) -> Optional[int]:
    """Map a cached landing state to the nearest graph node."""
    pt_idx = find_nearest_point(graph, x, y, surface_type)
    if pt_idx is None:
        pt_idx = find_nearest_point(graph, x, y, surface_type=None)
        if pt_idx is None:
            return None
    sp = graph.surface_points[pt_idx]
    n_bins = num_velocity_bins(sp.surface_type)
    vb = max(0, min(n_bins - 1, int(vel_bin)))
    return node_id(pt_idx, vb, graph.max_vel_bins)


class _HarnessSimPool:
    """Keyed cache of ``nclone_native.Simulator`` instances by signature hash.

    A fresh simulator is ~1-10 ms to construct, so repeatedly calling
    ``populate_launch_entry(sim=None)`` inside an inner loop would dominate
    runtime.  This pool keeps one launch harness and one landing harness
    per distinct signature for the duration of a single enumeration.
    """

    def __init__(self) -> None:
        self._launch: Dict[int, object] = {}
        self._landing: Dict[int, object] = {}

    def launch(self, sig: np.ndarray) -> object:
        from nclone_native import Simulator as NativeSimulator
        from .tile_cache.harness import build_launch_harness_map

        key = int(hash_signature(sig))
        sim = self._launch.get(key)
        if sim is None:
            sim = NativeSimulator.load(build_launch_harness_map(list(sig)))
            self._launch[key] = sim
        return sim

    def landing(self, sig: np.ndarray) -> object:
        from nclone_native import Simulator as NativeSimulator
        from .tile_cache.harness import build_landing_harness_map

        key = int(hash_signature(sig))
        sim = self._landing.get(key)
        if sim is None:
            sim = NativeSimulator.load(build_landing_harness_map(list(sig)))
            self._landing[key] = sim
        return sim


def _lookup_or_populate_launch(
    store: TileCacheStore,
    sig: np.ndarray,
    surface_type: int,
    subcell: Tuple[int, int],
    vel_bin_in: int,
    input_class_id: int,
    vel_bin_centre: float,
    *,
    sim_pool: Optional[_HarnessSimPool] = None,
    miss_counter: Optional[List[int]] = None,
) -> LaunchEntry:
    """Return cached launch entry; populate on miss."""
    entry = launch_lookup(
        store, sig, surface_type, subcell, vel_bin_in, input_class_id,
    )
    if entry is not None:
        return entry
    if miss_counter is not None:
        miss_counter[0] += 1
    sim = sim_pool.launch(sig) if sim_pool is not None else None
    return populate_launch_entry(
        store, sig,
        surface_type=surface_type,
        subcell=subcell,
        vel_bin_in=vel_bin_in,
        input_class_id=input_class_id,
        vel_bin_centre=vel_bin_centre,
        velocity_to_bin=velocity_to_bin,
        sim=sim,
    )


def _lookup_or_populate_landing(
    store: TileCacheStore,
    sig: np.ndarray,
    entry_subcell: Tuple[int, int],
    entry_vx_bin_air: int,
    entry_vy_bin_air: int,
    *,
    sim_pool: Optional[_HarnessSimPool] = None,
    miss_counter: Optional[List[int]] = None,
) -> LandingEntry:
    entry = landing_lookup(
        store, sig, entry_subcell, entry_vx_bin_air, entry_vy_bin_air,
    )
    if entry is not None:
        return entry
    if miss_counter is not None:
        miss_counter[0] += 1
    sim = sim_pool.landing(sig) if sim_pool is not None else None
    return populate_landing_entry(
        store, sig,
        entry_subcell=entry_subcell,
        entry_vx_bin_air=entry_vx_bin_air,
        entry_vy_bin_air=entry_vy_bin_air,
        velocity_to_bin=velocity_to_bin,
        sim=sim,
    )


def _mine_penalty_for_points(
    w_base: float,
    points: List[Tuple[float, float]],
    mine_positions: Optional[np.ndarray],
    beta: float,
    kill_radius: float,
) -> float:
    """Apply the same mine inflation policy as the legacy path.

    ``points`` is a list of ``(x, y)`` samples along the arc.  A synthetic
    waypoint list with frame index 0 is passed into ``_inflate_arc_cost``
    (the inflation depends only on the geometry, not the frame index).
    """
    if mine_positions is None or len(mine_positions) == 0 or not points:
        return w_base
    synth = [(0, x, y, 0.0, 0.0) for (x, y) in points]
    return _inflate_arc_cost(w_base, synth, mine_positions, beta, kill_radius)


def enumerate_node_edges_cached(
    graph,
    pt_idx: int,
    vel_bin: int,
    *,
    tile_grid: np.ndarray,
    tile_dict: Dict[Tuple[int, int], int],
    store: TileCacheStore,
    n_cols: int,
    n_rows: int,
    mine_positions: Optional[np.ndarray] = None,
    mine_kill_radius: float = _MINE_KILL_RADIUS,
    mine_inflation_beta: float = _MINE_INFLATION_BETA_DEFAULT,
    max_arc_frames: int = MAX_ARC_FRAMES_DEFAULT,
    sim_pool: Optional["_HarnessSimPool"] = None,
    launch_miss_counter: Optional[List[int]] = None,
    landing_miss_counter: Optional[List[int]] = None,
    launch_hit_counter: Optional[List[int]] = None,
    landing_hit_counter: Optional[List[int]] = None,
) -> Tuple[List[EdgeResult], int, int]:
    """Three-phase enumeration of all edges from a single graph node.

    Mirrors the signature/return of
    :func:`edge_enumeration._enumerate_node_edges` so the two code
    paths can be swapped at the enumerator layer.
    """
    sp = graph.surface_points[pt_idx]
    src = node_id(pt_idx, vel_bin, graph.max_vel_bins)
    centers = velocity_bin_centers(sp.surface_type)
    if vel_bin >= len(centers):
        return [], 0, 0
    vel_centre = float(centers[vel_bin])

    src_col, src_row, subcell = _source_cell_and_subcell(
        sp.x, sp.y, sp.surface_type,
    )
    sig = extract_3x3_signature(tile_grid, src_col, src_row)

    edges: List[EdgeResult] = []
    n_clipped = 0
    n_landings = 0

    for input_class_id in classes_for_surface(sp.surface_type):
        ic = decode_input_class(input_class_id)
        if launch_hit_counter is not None:
            # Track lookup attempt; we'll decrement if it was a miss.
            launch_hit_counter[0] += 1
        entry = _lookup_or_populate_launch(
            store, sig,
            surface_type=sp.surface_type,
            subcell=subcell,
            vel_bin_in=vel_bin,
            input_class_id=input_class_id,
            vel_bin_centre=vel_centre,
            sim_pool=sim_pool,
            miss_counter=launch_miss_counter,
        )
        if entry.died == 1:
            continue

        if entry.escaped == 0:
            # Phase 1 only: full in-window landing.
            lx, ly = _launch_entry_world_pos(entry, sp.x, sp.y)
            dst = _resolve_landing_to_node(
                graph, lx, ly,
                vel_bin=entry.land_vel_bin,
                surface_type=entry.land_surface_type,
            )
            if dst is None or dst == src:
                continue
            w = float(entry.frames)
            w = _mine_penalty_for_points(
                w, [(sp.x, sp.y), (lx, ly)],
                mine_positions, mine_inflation_beta, mine_kill_radius,
            )
            if not np.isfinite(w):
                continue
            edges.append(EdgeResult(
                src, dst, w,
                jump_hold_frames=getattr(ic, "hold", 0),
                h_input_during=ic.h_during,
            ))
            n_landings += 1
            continue

        # Phase 2 + 3: ballistic free flight + landing resolution.
        exit_x = sp.x + entry.exit_pos_dx * _POS_SCALE_INV
        exit_y = sp.y + entry.exit_pos_dy * _POS_SCALE_INV
        exit_vx = entry.exit_vx_q * _VEL_SCALE_INV
        exit_vy = entry.exit_vy_q * _VEL_SCALE_INV

        # Analytical propagator currently uses the post-launch h-input
        # during the remaining hold frames and the h_after after release.
        h_during = ic.h_during
        h_after = getattr(ic, "h_after", 0)
        result = propagate_and_sweep(
            launch_pos=(exit_x, exit_y),
            launch_vel=(exit_vx, exit_vy),
            hold_remaining=int(entry.hold_remaining),
            h_during=int(h_during),
            h_after=int(h_after),
            tile_dict=tile_dict,
            n_cols=n_cols, n_rows=n_rows,
            max_frames=max(1, max_arc_frames - entry.frames),
        )
        if not isinstance(result, Landed):
            # Died or timed out - no usable edge from this primitive.
            continue

        impact_col, impact_row = result.cell
        land_sig = extract_3x3_signature(tile_grid, impact_col, impact_row)
        impact_abs_x, impact_abs_y = result.pos
        # Quantise ballistic entry state into the landing cache key schema.
        in_x = impact_abs_x - impact_col * TILE_PIXEL_SIZE
        in_y = impact_abs_y - impact_row * TILE_PIXEL_SIZE
        entry_sub = quantise_subcell(in_x, in_y)
        vx_bin, vy_bin = quantise_air_velocity(*result.vel)

        if landing_hit_counter is not None:
            landing_hit_counter[0] += 1
        land_entry = _lookup_or_populate_landing(
            store, land_sig,
            entry_subcell=entry_sub,
            entry_vx_bin_air=vx_bin,
            entry_vy_bin_air=vy_bin,
            sim_pool=sim_pool,
            miss_counter=landing_miss_counter,
        )
        if land_entry.landing_surface_type == 0 and (
            land_entry.landing_subcell_x == 0
            and land_entry.landing_subcell_y == 0
            and land_entry.landing_vel_bin == 0
        ):
            # Landing harness could not resolve inside the 3x3 - skip.
            continue

        lx, ly = _landing_entry_world_pos(land_entry, impact_col, impact_row)
        dst = _resolve_landing_to_node(
            graph, lx, ly,
            vel_bin=land_entry.landing_vel_bin,
            surface_type=land_entry.landing_surface_type,
        )
        if dst is None or dst == src:
            continue

        total_frames = (
            float(entry.frames)
            + float(result.frame)
            + float(land_entry.contact_frame_offset)
        )
        # Sample arc for mine penalty: start, mid, exit, landing.
        mid_x = 0.5 * (sp.x + impact_abs_x)
        mid_y = 0.5 * (sp.y + impact_abs_y)
        w = _mine_penalty_for_points(
            total_frames,
            [(sp.x, sp.y), (exit_x, exit_y), (mid_x, mid_y),
             (impact_abs_x, impact_abs_y), (lx, ly)],
            mine_positions, mine_inflation_beta, mine_kill_radius,
        )
        if not np.isfinite(w):
            continue

        # Clip telemetry — analogous to the legacy code.
        bins = velocity_bin_centers(land_entry.landing_surface_type)
        if land_entry.landing_vel_bin in (0, len(bins) - 1):
            n_clipped += 1
        n_landings += 1

        edges.append(EdgeResult(
            src, dst, w,
            jump_hold_frames=getattr(ic, "hold", 0),
            h_input_during=ic.h_during,
        ))

    return edges, n_clipped, n_landings


def enumerate_edges_cached(
    graph,
    map_data: Sequence[int],
    *,
    tile_cache_dir: str,
    num_shards: int = 64,
    mine_positions: Optional[np.ndarray] = None,
    mine_kill_radius: float = _MINE_KILL_RADIUS,
    mine_inflation_beta: float = _MINE_INFLATION_BETA_DEFAULT,
    max_arc_frames: int = MAX_ARC_FRAMES_DEFAULT,
    verbose: bool = False,
    stats_out: Optional[dict] = None,
) -> List[EdgeResult]:
    """Run the three-phase pipeline over every node in ``graph``.

    Unlike :func:`edge_enumeration.enumerate_edges`, this function is
    single-process: multiprocessing support is deferred until the
    cache's file-locking/IO pattern has been re-audited for concurrent
    writers.  For most real levels the cache's 100% hit rate after a
    single prewarm makes a single process fast enough.

    Args:
        graph: Surface graph.
        map_data: Raw level bytes (full list of ints).
        tile_cache_dir: Directory containing the sharded ``.npz`` cache.
        num_shards: Must match the on-disk cache.
        mine_positions, mine_kill_radius, mine_inflation_beta: Same
            semantics as the legacy enumerator.
        max_arc_frames: Per-arc frame budget (split across phases).
        verbose: Print progress.
        stats_out: Optional dict populated with clip / landing / hit-rate
            telemetry for the validation harness.
    """
    from .tile_cache.signature import tile_grid_from_map_data
    from .tile_cache.ballistic import tile_dict_from_map_data as _tdict

    tile_grid = tile_grid_from_map_data(list(map_data))
    tile_dict = _tdict(list(map_data))
    n_cols, n_rows = tile_grid.shape
    store = TileCacheStore(tile_cache_dir, num_shards=num_shards)

    total = 0
    for pt_idx, sp in enumerate(graph.surface_points):
        total += num_velocity_bins(sp.surface_type)

    all_edges: List[EdgeResult] = []
    n_clip = 0
    n_land = 0
    done = 0
    pool = _HarnessSimPool()
    launch_miss = [0]
    landing_miss = [0]
    launch_hit = [0]
    landing_hit = [0]
    for pt_idx, sp in enumerate(graph.surface_points):
        n_bins = num_velocity_bins(sp.surface_type)
        for vb in range(n_bins):
            e, c, l = enumerate_node_edges_cached(
                graph, pt_idx, vb,
                tile_grid=tile_grid, tile_dict=tile_dict,
                store=store, n_cols=n_cols, n_rows=n_rows,
                mine_positions=mine_positions,
                mine_kill_radius=mine_kill_radius,
                mine_inflation_beta=mine_inflation_beta,
                max_arc_frames=max_arc_frames,
                sim_pool=pool,
                launch_miss_counter=launch_miss,
                landing_miss_counter=landing_miss,
                launch_hit_counter=launch_hit,
                landing_hit_counter=landing_hit,
            )
            all_edges.extend(e)
            n_clip += c
            n_land += l
            done += 1
            if verbose and done % 500 == 0:
                total_launch = max(launch_hit[0], 1)
                total_landing = max(landing_hit[0], 1)
                logger.info(
                    "cached enum: %d/%d nodes, %d edges, "
                    "launch hit %.1f%%, landing hit %.1f%%",
                    done, total, len(all_edges),
                    100.0 * (1 - launch_miss[0] / total_launch),
                    100.0 * (1 - landing_miss[0] / total_landing),
                )

    # Dominated-pair prune (identical to the legacy path).
    best: Dict[Tuple[int, int], EdgeResult] = {}
    for e in all_edges:
        key = (e.src_node, e.dst_node)
        existing = best.get(key)
        if existing is None or e.weight < existing.weight:
            best[key] = e
    pruned = list(best.values())

    if stats_out is not None:
        stats_out["n_clipped_landings"] = int(n_clip)
        stats_out["n_total_landings"] = int(n_land)
        stats_out["pct_clipped"] = (
            100.0 * n_clip / max(1, n_land) if n_land > 0 else 0.0
        )
        stats_out["launch_lookups"] = int(launch_hit[0])
        stats_out["launch_misses"] = int(launch_miss[0])
        stats_out["landing_lookups"] = int(landing_hit[0])
        stats_out["landing_misses"] = int(landing_miss[0])
        stats_out["launch_hit_rate"] = (
            1.0 - launch_miss[0] / max(1, launch_hit[0])
        )
        stats_out["landing_hit_rate"] = (
            1.0 - landing_miss[0] / max(1, landing_hit[0])
        )
    if verbose:
        logger.info(
            "cached enum done: %d edges (%d pruned), clip=%.2f%%",
            len(pruned), len(all_edges) - len(pruned),
            100.0 * n_clip / max(1, n_land),
        )
    return pruned


__all__ = [
    "enumerate_edges_cached",
    "enumerate_node_edges_cached",
]
