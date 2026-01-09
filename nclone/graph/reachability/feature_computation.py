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
    Compute 38-dimensional reachability features using adjacency graph and lookup tables.

    Uses efficient O(1) lookups for node finding and precomputed level cache for distances.
    No linear searches or Euclidean distance calculations.

    Features (38 dims):
    Base features (4):
        0. Reachable area ratio (0-1)
        1. Distance to nearest switch (normalized, inverted)
        2. Distance to exit (normalized, inverted)
        3. Exit reachable flag (0-1)

    Path distances (2) - normalized raw distances for learning:
        4. Path distance to switch (normalized 0-1)
        5. Path distance to exit (normalized 0-1)

    Direction vectors (4) - unit vectors toward goals (Euclidean):
        6. Direction to switch X component (-1 to 1)
        7. Direction to switch Y component (-1 to 1)
        8. Direction to exit X component (-1 to 1)
        9. Direction to exit Y component (-1 to 1)

    Mine context (2):
        10. Total mines normalized (0-1)
        11. Deadly mine ratio (0-1)

    Phase indicator (1):
        12. Switch activated flag (0-1) - EXPLICIT phase indicator for Markov property

    Path direction features (8) - NEW for inflection point navigation:
        13. Next hop direction X (-1 to 1) - optimal path direction
        14. Next hop direction Y (-1 to 1)
        15. Waypoint direction X (-1 to 1) - toward active waypoint if present
        16. Waypoint direction Y (-1 to 1)
        17. Waypoint distance (0-1) - normalized distance to waypoint
        18. Path requires detour (0-1) - binary flag if next_hop points away from goal
        19. Mine clearance direction X (-1 to 1) - safe direction from SDF
        20. Mine clearance direction Y (-1 to 1)

    Path difficulty (1) - NEW for physics awareness:
        21. Path difficulty ratio (0-1) - physics_cost / geometric_distance
                                          Tells LSTM if path is hard (high ratio) or easy (low ratio)

    Path curvature features (3) - NEW for anticipatory turning:
        22. Multi-hop direction X (-1 to 1) - 8-hop weighted lookahead direction
        23. Multi-hop direction Y (-1 to 1) - anticipates upcoming path curvature
        24. Path curvature (0-1) - dot product of next_hop and multi_hop directions
                                    (1.0=straight path, 0.0=90° turn, -1.0=180° turn)

    Exit lookahead features (5) - Phase 4 for switch transition continuity:
        25. Exit switch to door next hop X (-1 to 1) - ONLY before switch activation
        26. Exit switch to door next hop Y (-1 to 1) - shows immediate exit path from switch
        27. Exit switch to door multi-hop X (-1 to 1) - 8-hop lookahead from switch to exit
        28. Exit switch to door multi-hop Y (-1 to 1) - anticipates exit path curvature
        29. Near-switch transition indicator (0-1) - ramps to 1.0 as agent approaches switch

    Directional connectivity (8) - Phase 5 for blind jump verification:
        30. Platform distance East [0, 1]
        31. Platform distance Northeast [0, 1]
        32. Platform distance North [0, 1]
        33. Platform distance Northwest [0, 1]
        34. Platform distance West [0, 1]
        35. Platform distance Southwest [0, 1]
        36. Platform distance South [0, 1]
        37. Platform distance Southeast [0, 1]

    Args:
        adjacency: Graph adjacency structure from GraphBuilder
        graph_data: Full graph data dict with spatial_hash
        level_data: LevelData object with entities
        ninja_pos: Current ninja position (x, y) in world coordinates
        path_calculator: CachedPathDistanceCalculator instance
        goal_positions: Optional pre-computed goal positions from nplay_headless.get_goal_positions_for_features()
                       If provided, skips O(n) entity iteration for massive speedup

    Returns:
        25-dimensional numpy array with normalized features

    Raises:
        RuntimeError: If required data is missing (adjacency, graph_data, etc.)
    """
    if not adjacency:
        raise RuntimeError("Adjacency graph is empty")
    if graph_data is None:
        raise RuntimeError("graph_data is None")

    spatial_hash, subcell_lookup = extract_spatial_lookups_from_graph_data(graph_data)

    # Total possible nodes (approximate: all traversable tiles)
    total_possible_nodes = MAP_TILE_WIDTH * MAP_TILE_HEIGHT

    features = np.zeros(38, dtype=np.float32)

    # Track raw distances and positions for direction vectors
    switch_distance_raw = float("inf")
    exit_distance_raw = float("inf")
    switch_pos_for_direction = None
    exit_pos_for_direction = None

    # Feature 1: Reachable area ratio
    features[0] = np.clip(len(adjacency) / max(total_possible_nodes, 1), 0.0, 1.0)

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

    # Feature 2: Distance to NEXT objective (normalized, inverted)
    # Uses objective hierarchy: exit switch -> locked door switches -> exit door
    next_obj_distances = []

    # Extract base_adjacency for physics checks
    base_adjacency = (
        graph_data.get("base_adjacency", adjacency) if graph_data else adjacency
    )

    # First priority: Exit switch (if not collected)
    for switch in exit_switches:
        # Check if switch is collected (for exit switch, check if active=False)
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
            if distance != float("inf"):
                next_obj_distances.append(distance)
                # Track for direction vector computation
                if distance < switch_distance_raw:
                    switch_distance_raw = distance
                    switch_pos_for_direction = switch_pos

    # Second priority: Locked door switches (if exit switch collected but doors still closed)
    if not next_obj_distances:
        for switch in locked_door_switches:
            switch_active = getattr(switch, "active", True)
            if switch_active:  # Door not opened yet
                switch_pos = get_entity_position(switch)
                distance = path_calculator.get_distance(
                    ninja_pos,
                    switch_pos,
                    adjacency,
                    base_adjacency,
                    level_data=level_data,
                    graph_data=graph_data,
                    entity_radius=LOCKED_DOOR_SWITCH_RADIUS,
                    ninja_radius=NINJA_RADIUS,
                )
                if distance != float("inf"):
                    next_obj_distances.append(distance)

    # Third priority: Exit door (if all switches collected)
    if not next_obj_distances:
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
            if distance != float("inf"):
                next_obj_distances.append(distance)

    if next_obj_distances:
        min_obj_dist = min(next_obj_distances)
        features[1] = np.clip(1.0 - (min_obj_dist / area_scale), 0.0, 1.0)

    # Feature 3: Distance to exit (normalized, inverted)
    exit_distances = []
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
        if distance != float("inf"):
            exit_distances.append(distance)
            # Track for direction vector computation
            if distance < exit_distance_raw:
                exit_distance_raw = distance
                exit_pos_for_direction = exit_pos
    if exit_distances:
        min_exit_dist = min(exit_distances)
        features[2] = np.clip(1.0 - (min_exit_dist / area_scale), 0.0, 1.0)

    # Feature 4: Exit reachable flag
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
    features[3] = 1.0 if exit_reachable else 0.0

    # Feature 5-6: Raw path distances (normalized 0-1)
    # These are non-inverted distances for learning signal
    if switch_distance_raw != float("inf"):
        features[4] = np.clip(switch_distance_raw / area_scale, 0.0, 1.0)
    else:
        features[4] = 1.0  # Max distance if unreachable

    if exit_distance_raw != float("inf"):
        features[5] = np.clip(exit_distance_raw / area_scale, 0.0, 1.0)
    else:
        features[5] = 1.0  # Max distance if unreachable

    # Features 7-10: Direction vectors toward goals (unit vectors)
    # Use Euclidean direction as approximation (path direction would require next-hop lookup)
    ninja_x, ninja_y = ninja_pos

    # Direction to switch (features 6-7)
    if switch_pos_for_direction is not None:
        dx = switch_pos_for_direction[0] - ninja_x
        dy = switch_pos_for_direction[1] - ninja_y
        dist = np.sqrt(dx * dx + dy * dy)
        if dist > 0.001:
            features[6] = dx / dist  # X component (-1 to 1)
            features[7] = dy / dist  # Y component (-1 to 1)
        # else: 0.0 (already at goal)

    # Direction to exit (features 8-9)
    if exit_pos_for_direction is not None:
        dx = exit_pos_for_direction[0] - ninja_x
        dy = exit_pos_for_direction[1] - ninja_y
        dist = np.sqrt(dx * dx + dy * dy)
        if dist > 0.001:
            features[8] = dx / dist  # X component (-1 to 1)
            features[9] = dy / dist  # Y component (-1 to 1)
        # else: 0.0 (already at goal)

    # Features 10-11: Mine context (simplified from detailed mine features)
    # Mine information is primarily handled by graph nodes, these provide high-level context
    # OPTIMIZATION: Use pre-computed mine entities if available (O(1) instead of O(n))
    if goal_positions is not None and "mine_entities" in goal_positions:
        # Direct entity_dic access - no iteration needed
        toggle_mines, toggled_mines = goal_positions["mine_entities"]

        # Feature 10: Total mines normalized (0-1)
        total_mines = len(toggle_mines) + len(toggled_mines)
        features[10] = np.clip(total_mines / 10.0, 0.0, 1.0)  # Max 10 mines

        # Feature 11: Deadly mine ratio (0-1)
        if total_mines > 0:
            deadly_mines = 0
            # Count deadly mines in toggle_mines (type 1)
            for mine in toggle_mines:
                if getattr(mine, "state", 1) == 0:  # state 0 = deadly
                    deadly_mines += 1
            # Count deadly mines in toggled_mines (type 21 - start deadly)
            for mine in toggled_mines:
                if getattr(mine, "state", 0) == 0:  # state 0 = deadly
                    deadly_mines += 1
            features[11] = deadly_mines / total_mines
        else:
            features[11] = 0.0
    else:
        raise RuntimeError("goal_positions is None")

    # Feature 12: Switch activated flag (EXPLICIT phase indicator)
    # Critical for Markov property: agent needs to know which objective to pursue
    # For exit switches: active=True means not collected, active=False means activated
    switch_activated = False
    for switch in exit_switches:
        switch_active = getattr(switch, "active", True)
        if not switch_active:  # Switch has been activated
            switch_activated = True
            break
    features[12] = 1.0 if switch_activated else 0.0

    # ========================================================================
    # Features 13-20: Path Direction Features (NEW - Phase 1.1)
    # Provides explicit path guidance for graph-free LSTM to solve inflection point navigation
    # ========================================================================

    # Determine current goal for path direction features
    if not switch_activated:
        current_goal_id = "switch"
        current_goal_pos = switch_pos_for_direction
    else:
        current_goal_id = "exit"
        current_goal_pos = exit_pos_for_direction

    # OPTIMIZATION: Compute ninja_node ONCE and reuse for multiple features (13-14, 22-24)
    # This avoids redundant find_ninja_node() calls with the same goal_id
    ninja_node = None
    if (
        hasattr(path_calculator, "level_cache")
        and path_calculator.level_cache is not None
    ):
        from .pathfinding_utils import find_ninja_node

        # Find current node (for current goal: switch or exit)
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

    # Features 13-14: Next hop direction (optimal path direction)
    # Uses level_cache.get_next_hop() to get direction along optimal path
    # This solves "wrong direction" problem: tells agent to go LEFT even when goal is RIGHT
    if ninja_node is not None:
        # Get next hop toward goal
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
                features[13] = dx / dist  # X component
                features[14] = dy / dist  # Y component

    # Features 15-17: Waypoint direction and distance (Phase 1.3)
    # Uses adaptive waypoint system to guide through inflection points
    # Waypoints are discovered from successful trajectories
    # This provides dense guidance toward intermediate goals on complex paths
    #
    # OPTIMIZATION: Waypoints are stored in level_cache._waypoints but typically
    # there are only 0-10 waypoints per level (max_waypoints_per_level=10), so
    # linear search is acceptable. For now, leave as placeholder - waypoints will
    # be populated when reward_calculator wires them to PBRS (Phase 1.3).
    #
    # Note: This feature will be zero until waypoints are discovered from successful
    # trajectories. Early in training (0% success), these features provide no signal.
    # Once agent starts succeeding, waypoints guide through discovered inflection points.
    # Get waypoints from path_calculator's associated reward_calculator if available
    # Note: This requires the path_calculator to have access to the reward system
    # For now, initialize as placeholder - will be populated by reward_calculator
    # once waypoint system is fully wired (waypoints stored in level_cache)
    if (
        hasattr(path_calculator, "level_cache")
        and path_calculator.level_cache is not None
    ):
        # Check if level_cache has waypoints stored
        waypoints = getattr(path_calculator.level_cache, "_waypoints", None)
        if waypoints and current_goal_id and len(waypoints) > 0:
            # OPTIMIZATION: Early exit if no waypoints (common case early in training)
            # Find nearest active waypoint (O(N) where N typically ≤ 10)
            nearest_waypoint = None
            nearest_dist_sq = float("inf")  # Use squared distance to avoid sqrt

            # Precompute goal distance squared for comparison
            goal_dist_sq = float("inf")
            if current_goal_pos is not None:
                goal_dx = current_goal_pos[0] - ninja_x
                goal_dy = current_goal_pos[1] - ninja_y
                goal_dist_sq = goal_dx * goal_dx + goal_dy * goal_dy

            for wp_pos, wp_value, wp_type, wp_count in waypoints:
                dx = wp_pos[0] - ninja_x
                dy = wp_pos[1] - ninja_y
                dist_sq = dx * dx + dy * dy  # Squared distance (avoid sqrt)

                # Check if waypoint is between ninja and goal (simple heuristic)
                if current_goal_pos is not None:
                    wp_to_goal_dx = current_goal_pos[0] - wp_pos[0]
                    wp_to_goal_dy = current_goal_pos[1] - wp_pos[1]
                    wp_to_goal_dist_sq = (
                        wp_to_goal_dx * wp_to_goal_dx + wp_to_goal_dy * wp_to_goal_dy
                    )

                    # Waypoint should be closer to goal than ninja is
                    if wp_to_goal_dist_sq < goal_dist_sq and dist_sq < nearest_dist_sq:
                        nearest_waypoint = (wp_pos, dist_sq)
                        nearest_dist_sq = dist_sq

            if nearest_waypoint is not None:
                wp_pos, wp_dist_sq = nearest_waypoint
                wp_dist = np.sqrt(wp_dist_sq)  # Only sqrt once for nearest
                dx = wp_pos[0] - ninja_x
                dy = wp_pos[1] - ninja_y
                if wp_dist > 0.001:
                    features[15] = dx / wp_dist  # X direction
                    features[16] = dy / wp_dist  # Y direction
                    features[17] = np.clip(wp_dist / area_scale, 0.0, 1.0)  # Distance

    # Features 18-19: Mine clearance direction (safe direction from SDF gradient)
    # Uses mine SDF to compute direction pointing away from nearest deadly mine
    # This provides continuous safety guidance complementing discrete mine avoidance
    if (
        hasattr(path_calculator, "level_cache")
        and path_calculator.level_cache is not None
    ):
        # Get mine SDF from level cache if available
        mine_sdf = getattr(path_calculator.level_cache, "mine_sdf", None)

        if mine_sdf is not None:
            # Get gradient at current position (points away from nearest mine)
            grad_x, grad_y = mine_sdf.get_gradient_at_position(ninja_x, ninja_y)

            # Check if near a mine (gradient magnitude > 0.01)
            grad_mag = np.sqrt(grad_x * grad_x + grad_y * grad_y)
            if grad_mag > 0.01:
                # Normalize gradient (direction away from mine)
                features[18] = grad_x / grad_mag
                features[19] = grad_y / grad_mag
                # Note: Direction AWAY from mine, so moving in this direction is safer

    # Feature 20: Path requires detour flag (NEW - critical for inflection points)
    # Indicates when optimal path points away from goal (e.g., must go LEFT when goal is RIGHT)
    # This is computed by comparing next_hop direction with Euclidean goal direction
    if features[13] != 0.0 or features[14] != 0.0:  # Next hop direction computed
        if current_goal_pos is not None:
            # Euclidean direction to goal
            goal_dx = current_goal_pos[0] - ninja_x
            goal_dy = current_goal_pos[1] - ninja_y
            goal_dist = np.sqrt(goal_dx * goal_dx + goal_dy * goal_dy)

            if goal_dist > 0.001:
                goal_dir_x = goal_dx / goal_dist
                goal_dir_y = goal_dy / goal_dist

                # Dot product: positive = aligned, negative = opposing
                next_hop_dir_x = features[13]
                next_hop_dir_y = features[14]
                alignment = next_hop_dir_x * goal_dir_x + next_hop_dir_y * goal_dir_y

                # Flag if next_hop points significantly away from goal (< -0.3 = >107°)
                # This tells LSTM "you need to go the opposite direction first"
                if alignment < -0.3:
                    features[20] = 1.0  # Detour required

    # Feature 21: Path difficulty ratio (Phase 3.3)
    # Ratio of physics cost to geometric distance - tells LSTM if path direction is hard or easy
    # High ratio (e.g., 5.0+) = difficult path requiring jumps, aerial movement, mine detours
    # Low ratio (e.g., 0.2) = easy path with mostly grounded horizontal movement
    #
    # This complements the physics-weighted PBRS by making difficulty explicit in observations
    # LSTM can learn: "high difficulty ahead → build momentum" or "low difficulty → direct route"
    #
    # OPTIMIZATION: Reuse physics distances already computed earlier (switch_distance_raw / exit_distance_raw)
    # This avoids redundant get_distance() call. Only need to compute geometric distance.
    if (
        hasattr(path_calculator, "level_cache")
        and path_calculator.level_cache is not None
        and current_goal_pos is not None
    ):
        # OPTIMIZATION: Reuse physics cost computed earlier based on current goal
        if current_goal_id == "switch":
            physics_cost = switch_distance_raw
        else:  # current_goal_id == "exit"
            physics_cost = exit_distance_raw

        # Geometric distance (pixels) - still need to compute this as it's different from physics
        geometric_dist = float("inf")
        if physics_cost != float("inf"):
            # Only compute geometric distance if physics path exists
            base_adjacency_for_difficulty = (
                graph_data.get("base_adjacency", adjacency) if graph_data else adjacency
            )
            geometric_dist = path_calculator.get_geometric_distance(
                ninja_pos,
                current_goal_pos,
                adjacency,
                base_adjacency_for_difficulty,
                level_data=level_data,
                graph_data=graph_data,
                entity_radius=EXIT_SWITCH_RADIUS
                if not switch_activated
                else EXIT_DOOR_RADIUS,
                ninja_radius=NINJA_RADIUS,
                goal_id=current_goal_id,
            )

        if (
            geometric_dist != float("inf")
            and physics_cost != float("inf")
            and geometric_dist > 0.001
        ):
            # Compute ratio: physics_cost / geometric_distance
            # Typical values:
            #   Pure grounded horizontal: 0.15
            #   Mix of grounded + jumps: 0.5-2.0
            #   Complex aerial navigation: 3.0-10.0
            #   Mine detours with jumps: 5.0-20.0
            difficulty_ratio = physics_cost / geometric_dist

            # Normalize to [0, 1] range
            # Use log scale since ratios can vary widely (0.15 to 40.0)
            # log(0.15) ≈ -1.9, log(40.0) ≈ 3.7, range ≈ 5.6
            # Map to [0, 1]: (log(ratio) + 2.0) / 6.0
            import math

            log_difficulty = math.log(max(0.1, difficulty_ratio))
            normalized_difficulty = np.clip((log_difficulty + 2.0) / 6.0, 0.0, 1.0)
            features[21] = normalized_difficulty

    # ========================================================================
    # Features 22-24: Path Curvature Features (NEW)
    # Provides 8-hop lookahead for anticipatory turning and curvature awareness
    # ========================================================================

    # Features 22-23: Multi-hop direction (8-hop weighted lookahead)
    # This anticipates upcoming path curvature by looking ahead 8 hops with exponential decay
    # Unlike next_hop (features 13-14) which shows immediate direction, this shows where
    # the path is heading in the mid-term, enabling agents to anticipate sharp turns
    # OPTIMIZATION: Reuse ninja_node computed earlier instead of calling find_ninja_node again
    if ninja_node is not None:
        # Get multi-hop lookahead direction from level cache
        multi_hop_direction = path_calculator.level_cache.get_multi_hop_direction(
            ninja_node, current_goal_id
        )

        if multi_hop_direction is not None:
            # Multi-hop direction is already normalized (unit vector)
            features[22] = multi_hop_direction[0]  # X component
            features[23] = multi_hop_direction[1]  # Y component

            # Feature 24: Path curvature (dot product of next_hop and multi_hop)
            # This explicitly tells the agent if the path curves ahead:
            #   1.0 = straight path (next_hop aligned with multi_hop)
            #   0.0 = 90° turn coming up
            #  -1.0 = 180° turn (must go opposite direction first)
            # This is critical for episodes with sharp turns near hazards
            if features[13] != 0.0 or features[14] != 0.0:  # Next hop computed
                next_hop_dir_x = features[13]
                next_hop_dir_y = features[14]
                multi_hop_dir_x = features[22]
                multi_hop_dir_y = features[23]

                # Dot product: measures alignment (-1 to 1)
                curvature = (
                    next_hop_dir_x * multi_hop_dir_x + next_hop_dir_y * multi_hop_dir_y
                )

                # Normalize to [0, 1] for easier learning: (curvature + 1) / 2
                # 1.0 = straight, 0.5 = 90° turn, 0.0 = 180° turn
                features[24] = (curvature + 1.0) / 2.0

    # ========================================================================
    # Features 25-29: Exit Lookahead Features (Phase 4 - Switch Transition Continuity)
    # Shows path from exit switch to exit door BEFORE switch activation
    # ========================================================================

    # Features 25-28: Exit switch to exit door path (ONLY before switch activation)
    # This gives agent context about the path from switch to exit before they activate it
    # After activation, features 22-23 (multi-hop) provide guidance toward exit
    # ONLY computed when switch not yet activated
    # Uses caching since these features are static per level
    if (
        not switch_activated
        and switch_pos_for_direction is not None
        and exit_pos_for_direction is not None
    ):
        try:
            # Generate cache key WITHOUT switch states (path geometry is static)
            exit_path_cache_key = level_data.get_cache_key_for_reachability(
                include_switch_states=False
            )

            # Get cached or compute exit path features
            exit_path_features = get_cached_exit_path_features(
                exit_path_cache_key,
                switch_pos_for_direction,
                exit_pos_for_direction,
                adjacency,
                spatial_hash,
                subcell_lookup,
                path_calculator,
            )

            # Copy cached features into output array
            features[25:29] = exit_path_features
        except Exception:
            # If caching fails, features remain at 0.0 (safe fallback)
            pass

    # Feature 29: Near-switch transition indicator
    # Signals when agent is approaching switch activation (provides temporal context)
    # Ramps from 0.0 (far from switch) to 1.0 (at switch)
    # This helps LSTM prepare for goal transition
    if switch_pos_for_direction is not None:
        dx = switch_pos_for_direction[0] - ninja_x
        dy = switch_pos_for_direction[1] - ninja_y
        dist_to_switch = np.sqrt(dx * dx + dy * dy)
        # Ramp over 50px radius: 1.0 at switch, 0.0 at 50px+ away
        features[29] = 1.0 - np.clip(dist_to_switch / 50.0, 0.0, 1.0)

    # ========================================================================
    # Features 30-37: Directional Connectivity (Phase 5)
    # Solves blind jump problem via platform distance verification in 8 directions
    # ========================================================================

    # Requires physics_cache for grounded node identification
    physics_cache = graph_data.get("node_physics") if graph_data else None
    if physics_cache is not None:
        # Convert ninja_pos from world space to tile data space
        ninja_tile_x = ninja_x - NODE_WORLD_COORD_OFFSET
        ninja_tile_y = ninja_y - NODE_WORLD_COORD_OFFSET
        ninja_tile_pos = (int(ninja_tile_x), int(ninja_tile_y))

        directional_distances = compute_directional_platform_distances(
            ninja_tile_pos,
            adjacency,
            physics_cache,
            spatial_hash=spatial_hash,
            max_distance=500.0,
        )
        features[30:38] = directional_distances
    else:
        # Fallback: physics cache not available, fill with zeros
        # This shouldn't happen if graph building is correct
        import logging

        logger = logging.getLogger(__name__)
        logger.warning("Physics cache not available for directional connectivity")
        features[30:38] = 0.0

    # Return features and raw distances for PBRS caching optimization
    # This avoids duplicate distance computations in reward calculation
    return features, switch_distance_raw, exit_distance_raw


def get_feature_profiling_stats():
    """Get profiling statistics for reachability features (if enabled)."""
    if not ENABLE_FEATURE_PROFILING or not _feature_timings:
        return None

    stats = {}
    for name, timings in _feature_timings.items():
        stats[name] = {
            "avg_ms": np.mean(timings),
            "std_ms": np.std(timings),
            "min_ms": np.min(timings),
            "max_ms": np.max(timings),
            "count": len(timings),
        }
    return stats


def reset_feature_profiling():
    """Reset profiling data."""
    global _feature_timings
    _feature_timings = {}
