"""Orchestrator for building the analytical reachability PBRS system.

Called once per level load. Constructs the full pipeline:
    1. Build hull collections (floor + wall jump hulls, singleton)
    2. Build velocity-augmented reachability graph (walk + hull-masked jump edges)
    3. Run backward Dijkstra for switch and exit goals
    4. Return a ReachabilityPBRS instance ready for O(1) potential lookup

Both floor and wall jump edges use precomputed hull masking with flood-fill
obstruction filtering (exact frame-by-frame physics).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from nclone.constants.physics_constants import TILE_PIXEL_SIZE

from .jump_hull import (
    get_or_build_hulls,
    get_or_build_wall_hulls,
)
from .reachability_graph import (
    build_reachability_graph,
    compute_cost_to_go,
)
from .reachability_pbrs import ReachabilityPBRS

logger = logging.getLogger(__name__)


def build_reachability_pbrs(
    graph_data: dict,
    level_data: Any,
    switch_pos: tuple[float, float],
    exit_pos: tuple[float, float],
) -> ReachabilityPBRS:
    """Build the full reachability PBRS system for a level.

    Args:
        graph_data: Output of GraphBuilder.build_graph(). Must contain
            'adjacency', 'node_physics', 'spatial_hash'.
        level_data: LevelData instance (for mine state invalidation).
        switch_pos: (x, y) of exit switch entity.
        exit_pos: (x, y) of exit door entity.

    Returns:
        ReachabilityPBRS ready for get_potential() calls.
    """
    t_start = time.perf_counter()

    # 1. Build hull collections (singleton, cached after first build)
    hull_collection = get_or_build_hulls()
    wall_hull_collection = get_or_build_wall_hulls()

    # 2. Build reachability graph
    t_graph = time.perf_counter()
    graph = build_reachability_graph(
        graph_data=graph_data,
        hull_collection=hull_collection,
        wall_hull_collection=wall_hull_collection,
    )
    graph_ms = (time.perf_counter() - t_graph) * 1000

    # 3. Backward Dijkstra for switch and exit
    # Goal positions from entities are in full-map coords; graph uses inner-grid.
    switch_gx = switch_pos[0] - TILE_PIXEL_SIZE
    switch_gy = switch_pos[1] - TILE_PIXEL_SIZE
    exit_gx = exit_pos[0] - TILE_PIXEL_SIZE
    exit_gy = exit_pos[1] - TILE_PIXEL_SIZE

    t_dijk = time.perf_counter()
    d_star_switch, d_max_switch = compute_cost_to_go(
        graph, switch_gx, switch_gy, "switch",
    )
    d_star_exit, d_max_exit = compute_cost_to_go(
        graph, exit_gx, exit_gy, "exit",
    )
    dijk_ms = (time.perf_counter() - t_dijk) * 1000

    # Store results on graph object for inspection
    graph.d_star_switch = d_star_switch
    graph.d_star_exit = d_star_exit
    graph.d_max_switch = d_max_switch
    graph.d_max_exit = d_max_exit

    # 4. Build PBRS
    pbrs = ReachabilityPBRS(
        graph=graph,
        d_star_switch=d_star_switch,
        d_star_exit=d_star_exit,
        d_max_switch=d_max_switch,
        d_max_exit=d_max_exit,
    )

    # Store mine state signature for invalidation
    pbrs.mine_state_signature = _mine_state_hash(level_data)

    total_ms = (time.perf_counter() - t_start) * 1000
    logger.info(
        f"Built reachability PBRS: "
        f"{graph.n_points} surface pts, {graph.n_nodes} nodes, "
        f"{len(graph.edges_src)} edges | "
        f"graph {graph_ms:.0f}ms, dijkstra {dijk_ms:.0f}ms, "
        f"total {total_ms:.0f}ms | "
        f"d_max switch={d_max_switch:.0f} exit={d_max_exit:.0f}"
    )

    return pbrs


def should_rebuild(
    current_pbrs: Optional[ReachabilityPBRS],
    level_data: Any,
) -> bool:
    """Check if the PBRS needs rebuilding due to mine state change.

    Args:
        current_pbrs: Current PBRS instance (None = definitely rebuild).
        level_data: LevelData for current mine states.

    Returns:
        True if rebuild is needed.
    """
    if current_pbrs is None:
        return True
    new_sig = _mine_state_hash(level_data)
    return new_sig != current_pbrs.mine_state_signature


def _mine_state_hash(level_data: Any) -> int:
    """Compute a hash of mine states for invalidation checking."""
    if not hasattr(level_data, "get_all_toggle_mines"):
        return 0
    mines = level_data.get_all_toggle_mines()
    if not mines:
        return 0
    # Hash tuple of (x, y, state) for each mine
    state_tuples = tuple(
        (m.get("x", 0), m.get("y", 0), m.get("state", 0))
        for m in mines
    )
    return hash(state_tuples)
