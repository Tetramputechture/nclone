"""Reachability-based solvability gate for procedurally generated maps.

Generated maps (especially curved / composite ones) can be geometrically
unsolvable -- the exit switch or door may sit in a surface component the ninja
can never reach. This module rejects such maps using the *same* grid-based
traversability graph the training reachability/PBRS stack uses
(``graph.reachability.GraphBuilder``), NOT the heavy lattice d*/trick-edge
precompute (~80s/level). A map is "solvable" iff both the exit switch and the
exit door land in the ninja-spawn's flood-fill reachable set.

Cost note: this loads the map into a headless simulator and builds the
reachability graph (order ~0.1-1s/level on small maps, more on large ones).
That is acceptable for the OFFLINE dataset-generation pipeline
(``scripts/generate_benchmark_dataset.py``), where it is amortized once into the
pickled dataset. For the RUNTIME size-curriculum path it is far too slow to run
every reset; the new generators are solvable-by-construction, so callers there
should leave it off (or sample it) -- see ``is_map_solvable``'s docstring.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

from ..constants import MAP_TILE_WIDTH, MAP_TILE_HEIGHT
from ..constants.entity_types import EntityType
from ..constants.physics_constants import TOGGLE_MINE_RADII
from ..ninja import NINJA_RADIUS

logger = logging.getLogger(__name__)

# Minimum spawn-to-mine CENTER clearance (px). A toggled mine kills the ninja
# within NINJA_RADIUS + TOGGLE_MINE_RADII[0] (= 14px) -> a mine that close makes
# the level impossible (frame-1 death); an UNtoggled mine the ninja merely
# overlaps at spawn (NINJA_RADIUS + TOGGLE_MINE_RADII[2] = 14.5px) toggles into a
# deadly mine the instant the ninja steps off it -> a spawn trap. We require a
# full extra ninja-radius of clearance beyond the overlap distance so the spawn
# is never on or immediately adjacent to a mine.
MINE_SPAWN_CLEARANCE_PX = NINJA_RADIUS + max(TOGGLE_MINE_RADII.values()) + NINJA_RADIUS  # 24.5


def spawn_mine_clearance_ok(
    level_data, clearance_px: float = MINE_SPAWN_CLEARANCE_PX
) -> bool:
    """True iff no toggle mine sits within ``clearance_px`` of the ninja spawn.

    A mine on/adjacent to the spawn makes a level impossible (the ninja dies on
    frame 1 to a toggled mine) or a spawn trap (an untoggled mine the ninja
    overlaps toggles deadly the moment it moves). Both classes are silent — the
    goals can still be "reachable" — so this is checked separately in the gate.
    Reusable so runtime generators can self-verify by construction too.
    """
    sx, sy = level_data.start_position
    c2 = float(clearance_px) * float(clearance_px)
    for mine in level_data.get_all_toggle_mines():
        mx, my = mine.get("x"), mine.get("y")
        if mx is None or my is None:
            continue
        if (float(mx) - sx) ** 2 + (float(my) - sy) ** 2 < c2:
            return False
    return True

# A single reusable headless sim -- instantiating one per call is wasteful.
_SHARED_NPLAY = None


def _get_nplay():
    global _SHARED_NPLAY
    if _SHARED_NPLAY is None:
        from ..nplay_headless import NPlayHeadless

        _SHARED_NPLAY = NPlayHeadless(
            render_mode=None,
            enable_animation=False,
            enable_logging=False,
            enable_debug_overlay=False,
            # Solvability only extracts tiles/entities; never render. Without this
            # the headless sim initialises a pygame display (render_mode=None still
            # left rendering enabled), which is wasted work and, when called from an
            # interactive window, resizes the shared display surface.
            enable_rendering=False,
        )
    return _SHARED_NPLAY


def _build_level_data(map_data: List[int]):
    """Load ``map_data`` into the shared sim and build a reachability LevelData.

    Mirrors the proven construction in
    ``graph.reachability.shared_cache_builder.build_shared_level_cache``.
    """
    from ..graph.level_data import LevelData, extract_start_position_from_map_data
    from ..gym_environment.entity_extractor import EntityExtractor

    nplay = _get_nplay()
    nplay.load_map_from_map_data(list(map_data))
    nplay.reset()

    tile_dic = nplay.get_tile_data()
    tiles = np.zeros((MAP_TILE_HEIGHT, MAP_TILE_WIDTH), dtype=np.int32)
    for (x, y), tile_id in tile_dic.items():
        inner_x, inner_y = x - 1, y - 1
        if 0 <= inner_x < MAP_TILE_WIDTH and 0 <= inner_y < MAP_TILE_HEIGHT:
            tiles[inner_y, inner_x] = int(tile_id)

    entities = EntityExtractor(nplay).extract_graph_entities()
    start_position = extract_start_position_from_map_data(list(map_data))
    return LevelData(
        start_position=start_position,
        tiles=tiles,
        entities=entities,
        switch_states={},
    )


def _reachable_set(level_data):
    """Return (reachable_node_set, adjacency) flood-filled from the ninja spawn."""
    from ..graph.reachability.graph_builder import GraphBuilder
    from ..graph.reachability.surface_area import flood_fill_reachable_nodes

    graph_data = GraphBuilder().build_graph(level_data)
    adjacency = graph_data.get("adjacency", {})
    if not adjacency:
        return set(), {}
    spawn_world = level_data.start_position
    reachable = flood_fill_reachable_nodes(spawn_world, adjacency)
    return reachable, adjacency


def is_map_solvable(map_data: List[int]) -> bool:
    """Return ``True`` iff the exit switch AND exit door are reachable from spawn.

    Uses the grid-based traversability graph (jump-aware), so this reflects
    physics feasibility far better than a plain tile flood-fill, while staying
    much cheaper than the lattice d* precompute.

    Intended for the OFFLINE dataset pipeline. Do not call this on every runtime
    reset -- the new generators are solvable-by-construction.

    Returns ``False`` (rather than raising) on any structural problem -- missing
    switch/door, empty graph, or an unreachable goal -- so generators can simply
    reseed and retry.
    """
    from ..graph.reachability.level_data_helpers import extract_goal_positions
    from ..graph.reachability.node_lookup import find_closest_node_to_position

    try:
        level_data = _build_level_data(map_data)
    except Exception as exc:  # malformed map / missing entities
        logger.debug("solvability: level build failed: %s", exc)
        return False

    # Must have exactly the goal pair.
    if not level_data.get_entities_by_type(EntityType.EXIT_SWITCH):
        return False
    if not level_data.get_entities_by_type(EntityType.EXIT_DOOR):
        return False

    # A mine on/adjacent to the spawn makes the level impossible (frame-1 mine
    # death) or a spawn trap, even when the goals are perfectly reachable. Reject
    # so the generator reseeds to a spawn-safe layout.
    if not spawn_mine_clearance_ok(level_data):
        logger.debug("solvability: a mine is within the spawn clearance radius")
        return False

    try:
        goals = extract_goal_positions(level_data)
        reachable, adjacency = _reachable_set(level_data)
    except Exception as exc:
        logger.debug("solvability: reachability failed: %s", exc)
        return False

    if not reachable or not adjacency:
        return False

    for goal_world, _goal_id in goals:
        node = find_closest_node_to_position(goal_world, adjacency, threshold=24.0)
        if node is None or node not in reachable:
            return False
    return True


def find_solvable_seed(
    factory_fn,
    base_seed: int,
    max_retry: int = 16,
) -> Optional[Tuple[int, List[int]]]:
    """Reseed-and-retry helper: return the first ``(seed, map_data)`` that passes.

    ``factory_fn(seed) -> map_data`` builds and serializes a level for a seed.
    Returns ``None`` if no seed in ``[base_seed, base_seed + max_retry)`` passes.
    """
    for attempt in range(max_retry):
        seed = base_seed + attempt * 1_000_000
        try:
            map_data = factory_fn(seed)
        except Exception as exc:
            logger.debug("solvability: generation failed at seed %s: %s", seed, exc)
            continue
        if is_map_solvable(map_data):
            return seed, map_data
    return None
