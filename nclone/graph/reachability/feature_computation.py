"""
Efficient reachability feature computation using adjacency graph and lookup tables.

This module provides graph-based feature computation that uses:
- Subcell lookup (O(1)) for node finding
- Spatial hash (O(1)) as fallback
- Precomputed level cache for distances (no pathfinding needed)
"""

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

# Global profiling data for reachability features (optional, disabled by default)
ENABLE_FEATURE_PROFILING = False
_feature_timings = {}

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
    # Check cache first
    cache_key = (ninja_cell_x, ninja_cell_y, level_id)
    if cache_key in _directional_connectivity_cache:
        cached_features = _directional_connectivity_cache[cache_key]
        # Move to end to maintain LRU ordering
        _directional_connectivity_cache.move_to_end(cache_key)
        return cached_features.copy()  # Return copy to prevent external modification

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
    Compute 11-dimensional streamlined reachability features.

    Uses efficient O(1) lookups for node finding and precomputed level cache for distances.
    Removes redundant and misleading features from v1 (22 dims).

    Features (11 dims):
        0. Path distance to switch (normalized 0-1) - A* path length
        1. Path distance to exit (normalized 0-1) - A* path length
        2. Switch activated flag (0-1) - Markov property: which objective to pursue
        3. Next hop direction X (-1 to 1) - A* optimal path direction
        4. Next hop direction Y (-1 to 1)
        5. Path requires detour (0-1) - flag when A* opposes Euclidean direction
        6. Mine clearance direction X (-1 to 1) - SDF gradient (safe direction)
        7. Mine clearance direction Y (-1 to 1)
        8. Exit reachable flag (0-1) - can reach exit door
        9. Platform ahead (0-1) - platform distance in next_hop direction
        10. Platform below (0-1) - platform distance directly below (for drops)

    REMOVED from v1 (22 dims → 11 dims):
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
        - 11-dimensional numpy array with normalized features
        - switch_distance_raw (for PBRS caching)
        - exit_distance_raw (for PBRS caching)

    Raises:
        RuntimeError: If required data is missing (adjacency, graph_data, etc.)
    """
    if not adjacency:
        raise RuntimeError("Adjacency graph is empty")
    if graph_data is None:
        raise RuntimeError("graph_data is None")

    spatial_hash, subcell_lookup = extract_spatial_lookups_from_graph_data(graph_data)

    # Initialize 11-dimensional feature array
    features = np.zeros(11, dtype=np.float32)

    # Track raw distances for PBRS caching and goal positions
    switch_distance_raw = float("inf")
    exit_distance_raw = float("inf")
    switch_pos_for_computation = None
    exit_pos_for_computation = None

    # Compute reachable area scale for distance normalization
    # Use cached surface area calculation to avoid recomputation
    area_scale = LEVEL_DIAGONAL  # Fallback
    try:
        start_position = getattr(level_data, "start_position", None)
        if start_position is not None:
            # Generate cache key using LevelData utility method for consistency
            cache_key = level_data.get_cache_key_for_reachability(
                include_switch_states=True
            )

            # Convert start position to world space
            start_pos = (
                int(start_position[0]) + NODE_WORLD_COORD_OFFSET,
                int(start_position[1]) + NODE_WORLD_COORD_OFFSET,
            )

            # Use cached version to avoid recomputation for same level
            surface_area = get_cached_surface_area(
                cache_key, start_pos, adjacency, graph_data
            )
            area_scale = np.sqrt(surface_area) * SUB_NODE_SIZE
    except Exception:
        # Fallback to LEVEL_DIAGONAL if computation fails
        pass

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
    # First priority: Exit switch (if not collected)
    for switch in exit_switches:
        switch_active = getattr(switch, "active", True)
        if switch_active:  # Not collected yet
            switch_pos = get_entity_position(switch)
            distance = path_calculator.get_distance(
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

    # Compute distance to exit
    for exit_entity in exits:
        exit_pos = get_entity_position(exit_entity)
        distance = path_calculator.get_distance(
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

    # Features 0-1: Path distances (normalized 0-1, non-inverted)
    # Direct path lengths provide clear learning signal without redundant inverses
    if switch_distance_raw != float("inf"):
        features[0] = np.clip(switch_distance_raw / area_scale, 0.0, 1.0)
    else:
        features[0] = 1.0  # Max distance if unreachable

    if exit_distance_raw != float("inf"):
        features[1] = np.clip(exit_distance_raw / area_scale, 0.0, 1.0)
    else:
        features[1] = 1.0  # Max distance if unreachable

    # Feature 2: Switch activated flag (EXPLICIT phase indicator for Markov property)
    # Critical: agent needs to know which objective to pursue (switch or exit)
    # For exit switches: active=True means not collected, active=False means activated
    switch_activated = False
    for switch in exit_switches:
        switch_active = getattr(switch, "active", True)
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

    # Find ninja node for path computation
    ninja_node = None
    if (
        hasattr(path_calculator, "level_cache")
        and path_calculator.level_cache is not None
    ):
        from .pathfinding_utils import find_ninja_node

        if current_goal_pos is not None:
            ninja_node = find_ninja_node(
                ninja_pos,
                adjacency,
                spatial_hash=spatial_hash,
                subcell_lookup=subcell_lookup,
                ninja_radius=NINJA_RADIUS,
                level_cache=path_calculator.level_cache,
                goal_id=current_goal_id,
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

    # Feature 5: Path requires detour flag
    # Indicates when optimal path points away from goal (e.g., must go LEFT when goal is RIGHT)
    # Computed by comparing next_hop direction with Euclidean goal direction
    if (
        next_hop_dir_x != 0.0 or next_hop_dir_y != 0.0
    ) and current_goal_pos is not None:
        # Euclidean direction to goal
        goal_dx = current_goal_pos[0] - ninja_x
        goal_dy = current_goal_pos[1] - ninja_y
        goal_dist = np.sqrt(goal_dx * goal_dx + goal_dy * goal_dy)

        if goal_dist > 0.001:
            goal_dir_x = goal_dx / goal_dist
            goal_dir_y = goal_dy / goal_dist

            # Dot product: positive = aligned, negative = opposing
            alignment = next_hop_dir_x * goal_dir_x + next_hop_dir_y * goal_dir_y

            # Flag if next_hop points away from goal (< 0.5 = >60°)
            # Indicates detour required (e.g., go around wall)
            if alignment < 0.5:
                features[5] = 1.0  # Detour required

    # Features 6-7: Mine clearance direction (safe direction from SDF gradient)
    # Uses mine SDF to compute direction pointing away from nearest deadly mine
    # Provides continuous safety guidance complementing discrete mine avoidance
    if (
        hasattr(path_calculator, "level_cache")
        and path_calculator.level_cache is not None
    ):
        mine_sdf = getattr(path_calculator.level_cache, "mine_sdf", None)

        if mine_sdf is not None:
            # Get gradient at current position (points away from nearest mine)
            grad_x, grad_y = mine_sdf.get_gradient_at_position(ninja_x, ninja_y)

            # Check if near a mine (gradient magnitude > 0.01)
            grad_mag = np.sqrt(grad_x * grad_x + grad_y * grad_y)
            if grad_mag > 0.01:
                # Normalize gradient (direction away from mine)
                features[6] = grad_x / grad_mag
                features[7] = grad_y / grad_mag

    # ========================================================================
    # Features 8-10: Exit Reachability and Critical Platform Distances
    # Reduced from 8 directional connectivity features to 2 most relevant
    # ========================================================================

    # Feature 8: Exit reachable flag
    # Checks if the exit door is reachable (simplified from full completion path)
    exit_reachable = False
    for exit_entity in exits:
        exit_pos = get_entity_position(exit_entity)
        exit_node = find_closest_node_to_position(
            exit_pos,
            adjacency,
            threshold=50.0,
            spatial_hash=spatial_hash,
            subcell_lookup=subcell_lookup,
        )
        if exit_node is not None and exit_node in adjacency:
            exit_reachable = True
            break
    features[8] = 1.0 if exit_reachable else 0.0

    # Features 9-10: Platform distances in critical directions
    # Reduced from 8 directions to 2 most relevant: ahead (in next_hop direction) and below
    # These solve blind jump problem with minimal feature overhead
    physics_cache = graph_data.get("node_physics") if graph_data else None
    if physics_cache is not None and (next_hop_dir_x != 0.0 or next_hop_dir_y != 0.0):
        # Convert ninja_pos from world space to tile data space
        ninja_tile_x = ninja_x - NODE_WORLD_COORD_OFFSET
        ninja_tile_y = ninja_y - NODE_WORLD_COORD_OFFSET
        ninja_tile_pos = (int(ninja_tile_x), int(ninja_tile_y))

        # Import platform distance computation
        from .directional_connectivity import compute_directional_platform_distances

        # Compute platform distance in next_hop direction (feature 9)
        # This tells agent if there's a platform ahead in the direction they should move
        directional_distances = compute_directional_platform_distances(
            ninja_tile_pos,
            adjacency,
            physics_cache,
            spatial_hash=spatial_hash,
            max_distance=500.0,
        )

        # Find the direction index closest to next_hop direction
        # Directions: [E, NE, N, NW, W, SW, S, SE]
        directions = [
            (1.0, 0.0),  # E
            (0.707, -0.707),  # NE
            (0.0, -1.0),  # N
            (-0.707, -0.707),  # NW
            (-1.0, 0.0),  # W
            (-0.707, 0.707),  # SW
            (0.0, 1.0),  # S
            (0.707, 0.707),  # SE
        ]

        # Find closest direction to next_hop
        best_alignment = -2.0
        best_idx = 0
        for idx, (dir_x, dir_y) in enumerate(directions):
            alignment = next_hop_dir_x * dir_x + next_hop_dir_y * dir_y
            if alignment > best_alignment:
                best_alignment = alignment
                best_idx = idx

        features[9] = directional_distances[best_idx]  # Platform ahead
        features[10] = directional_distances[6]  # Platform below (S direction, index 6)

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
