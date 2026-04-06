"""
Efficient reachability feature computation using adjacency graph and lookup tables.

This module provides graph-based feature computation that uses:
- Subcell lookup (O(1)) for node finding
- Spatial hash (O(1)) as fallback
- Precomputed level cache for distances (no pathfinding needed)
"""

import logging
import time
import numpy as np
from typing import Dict, Tuple, Any, Optional
from collections import OrderedDict

from .pathfinding_utils import (
    find_closest_node_to_position,
    extract_spatial_lookups_from_graph_data,
    get_cached_surface_area,
    NODE_WORLD_COORD_OFFSET,
)
from .directional_connectivity import compute_directional_platform_distances
from ...constants.entity_types import EntityType
from ...constants.physics_constants import (
    MAP_TILE_WIDTH,
    MAP_TILE_HEIGHT,
    EXIT_SWITCH_RADIUS,
    EXIT_DOOR_RADIUS,
    LOCKED_DOOR_SWITCH_RADIUS,
    NINJA_RADIUS,
)
from ...gym_environment.constants import LEVEL_DIAGONAL
from .subcell_node_lookup import SUB_NODE_SIZE
from .path_distance_calculator import CachedPathDistanceCalculator

logger = logging.getLogger(__name__)

# Global profiling data for reachability features (optional, disabled by default)
ENABLE_FEATURE_PROFILING = False
_feature_timings = {}

# ---------------------------------------------------------------------------
# Module-level reachability diagnostics
# Updated on every call to compute_reachability_features_from_graph.
# Read by the SAC trainer and written to TensorBoard for easy inspection.
# ---------------------------------------------------------------------------
_reach_diagnostics: Dict[str, Any] = {
    "area_scale": 0.0,        # sqrt(surface_area) * SUB_NODE_SIZE (normalization denominator)
    "surface_area": 0.0,      # number of reachable graph nodes
    "switch_distance_raw": float("inf"),  # last A* path distance to switch (px)
    "exit_distance_raw": float("inf"),    # last A* path distance to exit (px)
    "switch_pos": (0, 0),     # last switch world position (0,0 = entity not loaded)
    "exit_pos": (0, 0),       # last exit world position
    "inf_count": 0,           # cumulative calls where current-goal dist was inf
    "finite_count": 0,        # cumulative calls where current-goal dist was finite
}


def get_reachability_diagnostics() -> Dict[str, Any]:
    """Return a snapshot of the module-level reachability diagnostics dict.

    Values are updated on every call to compute_reachability_features_from_graph.
    Safe to call from any thread; returns a shallow copy.
    """
    return dict(_reach_diagnostics)


def reset_reachability_diagnostics() -> None:
    """Reset the cumulative inf/finite counters (call between training runs)."""
    _reach_diagnostics["inf_count"] = 0
    _reach_diagnostics["finite_count"] = 0


def _profile_section(name: str):
    """Context manager for profiling feature computation sections."""
    if not ENABLE_FEATURE_PROFILING:
        return _NullContext()
    return _ProfileContext(name)


class _NullContext:
    """No-op context manager when profiling disabled."""

    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False


class _ProfileContext:
    """Records elapsed time for a named section."""

    def __init__(self, name: str):
        self.name = name
        self.start = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        elapsed_ms = (time.perf_counter() - self.start) * 1000.0
        if self.name not in _feature_timings:
            _feature_timings[self.name] = []
        _feature_timings[self.name].append(elapsed_ms)
        return False

# Cache for exit path features (features 25-28) - static per level
# Stores numpy array [next_hop_x, next_hop_y, multi_hop_x, multi_hop_y]
_exit_path_features_cache: OrderedDict[str, np.ndarray] = OrderedDict()
MAX_EXIT_PATH_CACHE_SIZE = 100

# Cache for directional connectivity (features 30-37) - static per level/cell
# Stores numpy array [E, NE, N, NW, W, SW, S, SE] platform distances
# Key: (ninja_cell_x, ninja_cell_y, level_id)
_directional_connectivity_cache: OrderedDict[Tuple[int, int, str], np.ndarray] = (
    OrderedDict()
)
MAX_DIRECTIONAL_CACHE_SIZE = 500  # High cache size for cell-based caching
_directional_connectivity_hits = 0
_directional_connectivity_misses = 0


def get_cached_directional_connectivity(
    ninja_cell_x: int,
    ninja_cell_y: int,
    level_id: str,
    ninja_tile_pos: Tuple[int, int],
    adjacency: Dict[Tuple[int, int], list],
    physics_cache: Dict[Tuple[int, int], Any],
    spatial_hash: Optional[Dict[Tuple[int, int], list]],
) -> np.ndarray:
    """
    Get or compute directional connectivity features (30-37) with caching.

    These features show platform distances in 8 directions and are static per
    level/cell combination (ninja moves ~6px per second, so high cache hit rate).

    Args:
        ninja_cell_x: Ninja cell X coordinate (ninja_x // 24)
        ninja_cell_y: Ninja cell Y coordinate (ninja_y // 24)
        level_id: Unique identifier for level
        ninja_tile_pos: Ninja position in tile data space
        adjacency: Graph adjacency structure
        physics_cache: Physics cache for grounded node identification
        spatial_hash: Spatial hash for node lookups

    Returns:
        numpy array [E, NE, N, NW, W, SW, S, SE] platform distances (8 dims)
    """
    global _directional_connectivity_hits, _directional_connectivity_misses

    # Check cache first
    cache_key = (ninja_cell_x, ninja_cell_y, level_id)
    if cache_key in _directional_connectivity_cache:
        _directional_connectivity_hits += 1
        cached_features = _directional_connectivity_cache[cache_key]
        # Move to end to maintain LRU ordering
        _directional_connectivity_cache.move_to_end(cache_key)
        return cached_features.copy()  # Return copy to prevent external modification

    _directional_connectivity_misses += 1
    # Compute features
    directional_distances = compute_directional_platform_distances(
        ninja_tile_pos,
        adjacency,
        physics_cache,
        spatial_hash=spatial_hash,
        max_distance=500.0,
    )

    # Cache the result with LRU eviction
    _directional_connectivity_cache[cache_key] = directional_distances
    _directional_connectivity_cache.move_to_end(cache_key)

    # Enforce cache size limit
    while len(_directional_connectivity_cache) > MAX_DIRECTIONAL_CACHE_SIZE:
        _directional_connectivity_cache.popitem(
            last=False
        )  # Remove oldest (first) item

    return directional_distances.copy()


def get_cached_exit_path_features(
    level_id: str,
    switch_pos: Tuple[int, int],
    exit_pos: Tuple[int, int],
    adjacency: Dict[Tuple[int, int], list],
    spatial_hash: Dict[Tuple[int, int], list],
    subcell_lookup: Optional[Any],
    path_calculator: CachedPathDistanceCalculator,
) -> np.ndarray:
    """
    Get or compute exit path features (25-28) with caching.

    These features show the path from exit switch to exit door and are static
    per level (independent of ninja position or switch state).

    Args:
        level_id: Unique identifier for level (without switch states)
        switch_pos: Exit switch position in world coordinates
        exit_pos: Exit door position in world coordinates
        adjacency: Graph adjacency structure
        spatial_hash: Spatial hash for node lookups
        subcell_lookup: Subcell lookup structure
        path_calculator: Path calculator with level cache

    Returns:
        numpy array [next_hop_x, next_hop_y, multi_hop_x, multi_hop_y]
    """
    # Check cache first
    if level_id in _exit_path_features_cache:
        cached_features = _exit_path_features_cache[level_id]
        # Move to end to maintain LRU ordering
        _exit_path_features_cache.move_to_end(level_id)
        return cached_features.copy()  # Return copy to prevent external modification

    # Compute features
    exit_features = np.zeros(4, dtype=np.float32)

    # Find the exit switch node
    exit_switch_node = find_closest_node_to_position(
        switch_pos,
        adjacency,
        threshold=50.0,
        spatial_hash=spatial_hash,
        subcell_lookup=subcell_lookup,
    )

    if exit_switch_node is not None:
        # Features 0-1: Next hop from exit switch toward exit door
        exit_next_hop = path_calculator.level_cache.get_next_hop(
            exit_switch_node, "exit"
        )

        if exit_next_hop is not None:
            # Convert to world coordinates
            exit_next_hop_world_x = exit_next_hop[0] + NODE_WORLD_COORD_OFFSET
            exit_next_hop_world_y = exit_next_hop[1] + NODE_WORLD_COORD_OFFSET

            # Compute direction from exit switch to exit next_hop
            switch_x, switch_y = switch_pos
            dx = exit_next_hop_world_x - switch_x
            dy = exit_next_hop_world_y - switch_y
            dist = np.sqrt(dx * dx + dy * dy)

            if dist > 0.001:
                exit_features[0] = dx / dist  # X component
                exit_features[1] = dy / dist  # Y component

        # Features 2-3: Multi-hop direction from exit switch toward exit door
        exit_multi_hop_direction = path_calculator.level_cache.get_multi_hop_direction(
            exit_switch_node, "exit"
        )

        if exit_multi_hop_direction is not None:
            # Multi-hop direction is already normalized (unit vector)
            exit_features[2] = exit_multi_hop_direction[0]  # X component
            exit_features[3] = exit_multi_hop_direction[1]  # Y component

    # Cache the result with LRU eviction
    _exit_path_features_cache[level_id] = exit_features
    _exit_path_features_cache.move_to_end(level_id)

    # Enforce cache size limit
    while len(_exit_path_features_cache) > MAX_EXIT_PATH_CACHE_SIZE:
        _exit_path_features_cache.popitem(last=False)  # Remove oldest (first) item

    return exit_features.copy()


def compute_reachability_features_from_graph(
    adjacency: Dict[Tuple[int, int], list],
    graph_data: Dict[str, Any],
    level_data: Any,
    ninja_pos: Tuple[int, int],
    path_calculator: CachedPathDistanceCalculator,
    goal_positions: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, float, float]:
    """
    Compute 8-dimensional streamlined reachability features.

    Uses efficient O(1) lookups for node finding and precomputed level cache for distances.
    Removes redundant and misleading features from v1 (22 dims).

    Features (8 dims):
        0. Path distance to switch (normalized 0-1) - A* path length
        1. Path distance to exit (normalized 0-1) - A* path length
        2. Switch activated flag (0-1) - Markov property: which objective to pursue
        3. Next hop direction X (-1 to 1) - A* optimal path direction
        4. Next hop direction Y (-1 to 1)
        5. Platform ahead (0-1) - platform distance in next_hop direction
        6. Platform below (0-1) - platform distance directly below (for drops)
        7. Path distance to current goal (0-1) - phase-aware: switch if pre-activation, exit if post

    REMOVED: mine_clearance_dir - duplicate of escape_gradient in mine_sdf_features
    REMOVED: exit_reachable - near-constant, derivable from path distance
    REMOVED (Feature Experiment A): path_requires_detour - redundant with next_hop_dir

    REMOVED from v1 (22 dims → 7 dims):
        - reachable_area_ratio: global statistic, low information density
        - dist_to_next_objective_inv, dist_to_exit_inv: redundant inverses
        - direction_to_switch (x,y), direction_to_exit (x,y): MISLEADING Euclidean directions
          that can point into walls and contradict optimal policy
        - total_mines_norm, deadly_mine_ratio: global statistics, local info (8-nearest + SDF) sufficient
        - 6 of 8 directional connectivity features: keep only most relevant 2

    Args:
        adjacency: Graph adjacency structure from GraphBuilder
        graph_data: Full graph data dict with spatial_hash
        level_data: LevelData object with entities
        ninja_pos: Current ninja position (x, y) in world coordinates
        path_calculator: CachedPathDistanceCalculator instance
        goal_positions: Optional pre-computed goal positions from nplay_headless.get_goal_positions_for_features()
                       If provided, skips O(n) entity iteration for massive speedup

    Returns:
        Tuple of:
        - 8-dimensional numpy array with normalized features
        - switch_distance_raw (for PBRS caching)
        - exit_distance_raw (for PBRS caching)

    Raises:
        RuntimeError: If required data is missing (adjacency, graph_data, etc.)
    """
    if not adjacency:
        raise RuntimeError("Adjacency graph is empty")
    if graph_data is None:
        raise RuntimeError("graph_data is None")

    # Guard against severely out-of-bounds ninja positions (e.g. ninja fell below the
    # map after an OOB spawn). Such positions produce meaningless distances and corrupt
    # PBRS gradients.  Return zero features and inf distances so the caller can handle
    # or discard the step gracefully without touching the pathfinding hot path.
    _nx, _ny = ninja_pos
    if _nx < 0 or _nx > 1056 or _ny < 0 or _ny > 600:
        logger.warning(
            "compute_reachability_features_from_graph: ninja_pos (%.1f, %.1f) is "
            "outside valid level bounds [0,1056]x[0,600]; returning zero features.",
            _nx, _ny,
        )
        return np.zeros(8, dtype=np.float32), float("inf"), float("inf")

    with _profile_section("extract_spatial_lookups"):
        spatial_hash, subcell_lookup = extract_spatial_lookups_from_graph_data(
            graph_data
        )

    # Initialize 8-dimensional feature array
    features = np.zeros(8, dtype=np.float32)

    # Track raw distances for PBRS caching and goal positions
    switch_distance_raw = float("inf")
    exit_distance_raw = float("inf")
    switch_pos_for_computation = None
    exit_pos_for_computation = None

    # Compute reachable area scale for distance normalization.
    # NOTE: area_scale is tracked in diagnostics but is NO LONGER used to normalize
    # path distance features 0, 1, 7 (they use LEVEL_DIAGONAL instead).  Retaining
    # the computation keeps the diagnostic value accurate and avoids breaking any
    # future callers that might rely on it.
    with _profile_section("surface_area"):
        area_scale = LEVEL_DIAGONAL  # Fallback
        _diag_surface_area = 0.0
        try:
            start_position = getattr(level_data, "start_position", None)
            if start_position is not None:
                cache_key = level_data.get_cache_key_for_reachability(
                    include_switch_states=True
                )
                start_pos = (
                    int(start_position[0]) + NODE_WORLD_COORD_OFFSET,
                    int(start_position[1]) + NODE_WORLD_COORD_OFFSET,
                )
                surface_area = get_cached_surface_area(
                    cache_key, start_pos, adjacency, graph_data
                )
                _diag_surface_area = float(surface_area)
                area_scale = np.sqrt(surface_area) * SUB_NODE_SIZE
        except Exception:
            pass
    # Update normalization diagnostics (area_scale is now only used for diagnostics).
    _reach_diagnostics["area_scale"] = float(area_scale)
    _reach_diagnostics["surface_area"] = _diag_surface_area

    # OPTIMIZATION: Use pre-computed goal positions if provided (O(1) instead of O(n))
    # This eliminates entity iteration bottleneck by using nplay_headless helper methods
    if goal_positions is not None:
        # Extract from pre-computed dict (O(1) entity_dic lookups)
        switch_pos_precomputed = goal_positions.get("switch_pos", (0, 0))
        exit_pos_precomputed = goal_positions.get("exit_pos", (0, 0))
        switch_activated_precomputed = goal_positions.get("switch_activated", False)
        locked_door_switches_precomputed = goal_positions.get(
            "locked_door_switches", []
        )

        # Build entity-like structures for compatibility with existing code
        # (avoids rewriting all downstream logic)
        exit_switches = []
        if switch_pos_precomputed != (0, 0):
            # Create minimal entity representation with position and active state
            exit_switches = [
                {
                    "xpos": switch_pos_precomputed[0],
                    "ypos": switch_pos_precomputed[1],
                    "active": not switch_activated_precomputed,  # active=False means collected
                    "type": EntityType.EXIT_SWITCH,
                }
            ]

        exits = []
        if exit_pos_precomputed != (0, 0):
            exits = [
                {
                    "xpos": exit_pos_precomputed[0],
                    "ypos": exit_pos_precomputed[1],
                    "type": EntityType.EXIT_DOOR,
                }
            ]

        locked_door_switches = locked_door_switches_precomputed
    else:
        raise RuntimeError("goal_positions is None")

    # Helper function for pre-computed positions (dict format)
    def get_entity_position(entity):
        if isinstance(entity, dict):
            return (
                int(entity.get("xpos", entity.get("x", 0))),
                int(entity.get("ypos", entity.get("y", 0))),
            )
        else:
            return (
                int(getattr(entity, "xpos", 0)),
                int(getattr(entity, "ypos", 0)),
            )

    def get_entity_type(entity):
        if isinstance(entity, dict):
            return entity.get("type")
        else:
            return getattr(entity, "type", None)

    # Extract base_adjacency for physics checks
    base_adjacency = (
        graph_data.get("base_adjacency", adjacency) if graph_data else adjacency
    )

    # Compute distances to switch and exit for features 0-1
    with _profile_section("path_distances"):
        for switch in exit_switches:
            switch_active = switch.get("active", True) if isinstance(switch, dict) else getattr(switch, "active", True)
            if switch_active:  # Not collected yet
                switch_pos = get_entity_position(switch)
                distance = path_calculator.get_path_cost(
                    ninja_pos,
                    switch_pos,
                    adjacency,
                    base_adjacency,
                    level_data=level_data,
                    graph_data=graph_data,
                    entity_radius=EXIT_SWITCH_RADIUS,
                    ninja_radius=NINJA_RADIUS,
                )
                if distance != float("inf") and distance < switch_distance_raw:
                    switch_distance_raw = distance
                    switch_pos_for_computation = switch_pos

        for exit_entity in exits:
            exit_pos = get_entity_position(exit_entity)
            distance = path_calculator.get_path_cost(
                ninja_pos,
                exit_pos,
                adjacency,
                base_adjacency,
                level_data=level_data,
                graph_data=graph_data,
                entity_radius=EXIT_DOOR_RADIUS,
                ninja_radius=NINJA_RADIUS,
            )
            if distance != float("inf") and distance < exit_distance_raw:
                exit_distance_raw = distance
                exit_pos_for_computation = exit_pos

    # Update raw-distance and goal-position diagnostics.
    _reach_diagnostics["switch_distance_raw"] = float(switch_distance_raw)
    _reach_diagnostics["exit_distance_raw"] = float(exit_distance_raw)
    _reach_diagnostics["switch_pos"] = (
        switch_pos_for_computation if switch_pos_for_computation is not None else (0, 0)
    )
    _reach_diagnostics["exit_pos"] = (
        exit_pos_for_computation if exit_pos_for_computation is not None else (0, 0)
    )
    if switch_distance_raw == float("inf") or exit_distance_raw == float("inf"):
        _reach_diagnostics["inf_count"] += 1
    else:
        _reach_diagnostics["finite_count"] += 1

    # Features 0-1: Path distances normalized to [0, 1] using LEVEL_DIAGONAL.
    # LEVEL_DIAGONAL (~1215 px) is the maximum possible straight-line distance in the
    # level.  Using area_scale (sqrt(nodes)*6, typically ~200 px for a standard level)
    # caused almost all real distances to be clipped to 1.0 by np.clip, making the
    # feature useless as a learning signal.
    if switch_distance_raw != float("inf"):
        features[0] = np.clip(switch_distance_raw / LEVEL_DIAGONAL, 0.0, 1.0)
    else:
        features[0] = 1.0  # Max distance if unreachable

    if exit_distance_raw != float("inf"):
        features[1] = np.clip(exit_distance_raw / LEVEL_DIAGONAL, 0.0, 1.0)
    else:
        features[1] = 1.0  # Max distance if unreachable

    # Feature 2: Switch activated flag (EXPLICIT phase indicator for Markov property)
    # Critical: agent needs to know which objective to pursue (switch or exit)
    # For exit switches: active=True means not collected, active=False means activated
    switch_activated = False
    for switch in exit_switches:
        switch_active = switch.get("active", True) if isinstance(switch, dict) else getattr(switch, "active", True)
        if not switch_active:  # Switch has been activated
            switch_activated = True
            break
    features[2] = 1.0 if switch_activated else 0.0

    # ========================================================================
    # Features 3-7: Path Direction Features
    # Provides A* path guidance for navigation without misleading Euclidean directions
    # ========================================================================

    # Determine current goal for path direction features
    if not switch_activated:
        current_goal_id = "switch"
        current_goal_pos = switch_pos_for_computation
    else:
        current_goal_id = "exit"
        current_goal_pos = exit_pos_for_computation

    ninja_x, ninja_y = ninja_pos

    # Find ninja node for path computation (use path calculator cache when available)
    with _profile_section("ninja_node_and_path_direction"):
        ninja_node = None
        if (
            hasattr(path_calculator, "level_cache")
            and path_calculator.level_cache is not None
        ):
            if current_goal_pos is not None:
                goal_node = find_closest_node_to_position(
                    current_goal_pos,
                    adjacency,
                    threshold=50.0,
                    spatial_hash=spatial_hash,
                    subcell_lookup=subcell_lookup,
                )
                ninja_node = path_calculator._get_cached_or_find_ninja_node(
                    ninja_pos,
                    adjacency,
                    goal_id=current_goal_id,
                    goal_node=goal_node,
                    subcell_lookup=subcell_lookup,
                    ninja_radius=NINJA_RADIUS,
                )

    # Features 3-4: Next hop direction (A* optimal path direction)
    # Uses level_cache.get_next_hop() to get direction along optimal path
    # This solves "wrong direction" problem: tells agent to go LEFT even when goal is RIGHT
    next_hop_dir_x = 0.0
    next_hop_dir_y = 0.0
    if ninja_node is not None:
        next_hop = path_calculator.level_cache.get_next_hop(ninja_node, current_goal_id)

        if next_hop is not None:
            # Convert to world coordinates
            next_hop_world_x = next_hop[0] + NODE_WORLD_COORD_OFFSET
            next_hop_world_y = next_hop[1] + NODE_WORLD_COORD_OFFSET

            # Compute direction from ninja to next_hop
            dx = next_hop_world_x - ninja_x
            dy = next_hop_world_y - ninja_y
            dist = np.sqrt(dx * dx + dy * dy)

            if dist > 0.001:
                next_hop_dir_x = dx / dist
                next_hop_dir_y = dy / dist
                features[3] = next_hop_dir_x  # X component
                features[4] = next_hop_dir_y  # Y component

    # ========================================================================
    # Features 5-6: Critical Platform Distances
    with _profile_section("exit_reachable_and_platform"):
        # Features 5-6: Platform distances in critical directions
        # Use get_cached_directional_connectivity for cell-based LRU caching
        physics_cache = graph_data.get("node_physics") if graph_data else None
        if physics_cache is not None and (
            next_hop_dir_x != 0.0 or next_hop_dir_y != 0.0
        ):
            ninja_tile_x = ninja_x - NODE_WORLD_COORD_OFFSET
            ninja_tile_y = ninja_y - NODE_WORLD_COORD_OFFSET
            ninja_tile_pos = (int(ninja_tile_x), int(ninja_tile_y))
            level_id = getattr(level_data, "level_id", "unknown")
            ninja_cell_x = int(ninja_x // 24)
            ninja_cell_y = int(ninja_y // 24)

            directional_distances = get_cached_directional_connectivity(
                ninja_cell_x,
                ninja_cell_y,
                level_id,
                ninja_tile_pos,
                adjacency,
                physics_cache,
                spatial_hash,
            )

            directions = [
                (1.0, 0.0),
                (0.707, -0.707),
                (0.0, -1.0),
                (-0.707, -0.707),
                (-1.0, 0.0),
                (-0.707, 0.707),
                (0.0, 1.0),
                (0.707, 0.707),
            ]

            best_alignment = -2.0
            best_idx = 0
            for idx, (dir_x, dir_y) in enumerate(directions):
                alignment = next_hop_dir_x * dir_x + next_hop_dir_y * dir_y
                if alignment > best_alignment:
                    best_alignment = alignment
                    best_idx = idx

            features[5] = directional_distances[best_idx]
            features[6] = directional_distances[6]

    # Feature 7: Path distance to current goal (phase-aware, normalized 0-1).
    # Uses LEVEL_DIAGONAL (same as features 0-1) so the value meaningfully tracks
    # the agent's progress; area_scale was too small and clipped everything to 1.0.
    if switch_activated:
        current_goal_dist = exit_distance_raw
    else:
        current_goal_dist = switch_distance_raw
    if current_goal_dist != float("inf"):
        features[7] = np.clip(current_goal_dist / LEVEL_DIAGONAL, 0.0, 1.0)
    else:
        features[7] = 1.0

    # Return features and raw distances for PBRS caching optimization
    # This avoids duplicate distance computations in reward calculation
    return features, switch_distance_raw, exit_distance_raw


def get_feature_profiling_stats():
    """Get profiling statistics for reachability features (if enabled)."""
    if not ENABLE_FEATURE_PROFILING or not _feature_timings:
        return None

    stats = {}
    for name, timings in _feature_timings.items():
        if len(timings) > 0:
            stats[name] = {
                "avg_ms": np.mean(timings),
                "std_ms": np.std(timings),
                "min_ms": np.min(timings),
                "max_ms": np.max(timings),
                "count": len(timings),
            }
        else:
            stats[name] = {
                "avg_ms": 0.0,
                "std_ms": 0.0,
                "min_ms": 0.0,
                "max_ms": 0.0,
                "count": 0,
            }
    return stats


def reset_feature_profiling():
    """Reset profiling data."""
    global _feature_timings
    _feature_timings = {}


def get_directional_connectivity_cache_stats() -> dict:
    """Get hit/miss stats for directional connectivity cache (for auditing)."""
    total = _directional_connectivity_hits + _directional_connectivity_misses
    hit_rate = (
        _directional_connectivity_hits / total
        if total > 0
        else 0.0
    )
    return {
        "hits": _directional_connectivity_hits,
        "misses": _directional_connectivity_misses,
        "total": total,
        "hit_rate": hit_rate,
        "cache_size": len(_directional_connectivity_cache),
        "max_size": MAX_DIRECTIONAL_CACHE_SIZE,
    }
