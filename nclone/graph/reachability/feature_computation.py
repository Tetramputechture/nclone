"""
Efficient reachability feature computation using adjacency graph and lookup tables.

This module provides graph-based feature computation that uses:
- Subcell lookup (O(1)) for node finding
- Spatial hash (O(1)) as fallback
- Precomputed level cache for distances (no pathfinding needed)
"""

import numpy as np
from typing import Dict, Tuple, Any

from .pathfinding_utils import (
    find_closest_node_to_position,
    extract_spatial_lookups_from_graph_data,
    get_cached_surface_area,
    NODE_WORLD_COORD_OFFSET,
)
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


def compute_reachability_features_from_graph(
    adjacency: Dict[Tuple[int, int], list],
    graph_data: Dict[str, Any],
    level_data: Any,
    ninja_pos: Tuple[int, int],
    path_calculator: CachedPathDistanceCalculator,
) -> np.ndarray:
    """
    Compute 22-dimensional reachability features using adjacency graph and lookup tables.

    Uses efficient O(1) lookups for node finding and precomputed level cache for distances.
    No linear searches or Euclidean distance calculations.

    Features (22 dims):
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

    Args:
        adjacency: Graph adjacency structure from GraphBuilder
        graph_data: Full graph data dict with spatial_hash
        level_data: LevelData object with entities
        ninja_pos: Current ninja position (x, y) in world coordinates
        path_calculator: CachedPathDistanceCalculator instance

    Returns:
        22-dimensional numpy array with normalized features

    Raises:
        RuntimeError: If required data is missing (adjacency, graph_data, etc.)
    """
    if not adjacency:
        raise RuntimeError("Adjacency graph is empty")
    if graph_data is None:
        raise RuntimeError("graph_data is None")

    # Extract spatial lookups for O(1) node finding
    spatial_hash, subcell_lookup = extract_spatial_lookups_from_graph_data(graph_data)

    # Total possible nodes (approximate: all traversable tiles)
    total_possible_nodes = MAP_TILE_WIDTH * MAP_TILE_HEIGHT

    features = np.zeros(22, dtype=np.float32)

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

    # Get entities from level_data
    entities = level_data.entities if hasattr(level_data, "entities") else []

    # Helper function to extract position from entity (handles both dict and Entity object)
    def get_entity_position(entity):
        """Extract position from entity, handling both dict and Entity object formats."""
        if isinstance(entity, dict):
            return (int(entity.get("x", 0)), int(entity.get("y", 0)))
        else:
            # Entity object with xpos/ypos attributes
            return (int(getattr(entity, "xpos", 0)), int(getattr(entity, "ypos", 0)))

    # Helper function to get entity type
    def get_entity_type(entity):
        """Extract type from entity, handling both dict and Entity object formats."""
        if isinstance(entity, dict):
            return entity.get("type")
        else:
            return getattr(entity, "type", None)

    # Find switches and exit using efficient lookups
    switches = [
        e
        for e in entities
        if get_entity_type(e) == EntityType.EXIT_SWITCH
        or get_entity_type(e) == EntityType.LOCKED_DOOR_SWITCH
    ]
    exits = [e for e in entities if get_entity_type(e) == EntityType.EXIT_DOOR]

    # Separate switch types for objective hierarchy
    exit_switches = [
        e for e in switches if get_entity_type(e) == EntityType.EXIT_SWITCH
    ]
    locked_door_switches = [
        e for e in switches if get_entity_type(e) == EntityType.LOCKED_DOOR_SWITCH
    ]

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
        locked_door_switches = [
            e for e in switches if get_entity_type(e) == EntityType.LOCKED_DOOR_SWITCH
        ]
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
    hazard_types = [
        EntityType.TOGGLE_MINE,
        EntityType.TOGGLE_MINE_TOGGLED,
    ]
    hazards = [e for e in entities if get_entity_type(e) in hazard_types]

    # Feature 10: Total mines normalized (0-1)
    total_mines = len(hazards)
    features[10] = np.clip(total_mines / 10.0, 0.0, 1.0)  # Max 10 mines

    # Feature 11: Deadly mine ratio (0-1)
    if total_mines > 0:
        deadly_mines = 0
        for hazard in hazards:
            hazard_state = getattr(hazard, "state", 1)
            # For TOGGLE_MINE_TOGGLED, state is always 0 (deadly)
            if getattr(hazard, "entity_type", None) == EntityType.TOGGLE_MINE_TOGGLED:
                hazard_state = 0
            if hazard_state == 0:  # Deadly mine
                deadly_mines += 1
        features[11] = deadly_mines / total_mines
    else:
        features[11] = 0.0

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

    # Features 13-14: Next hop direction (optimal path direction)
    # Uses level_cache.get_next_hop() to get direction along optimal path
    # This solves "wrong direction" problem: tells agent to go LEFT even when goal is RIGHT
    if (
        hasattr(path_calculator, "level_cache")
        and path_calculator.level_cache is not None
        and current_goal_pos is not None
    ):
        from .pathfinding_utils import find_ninja_node

        # Find current node
        ninja_node = find_ninja_node(
            ninja_pos,
            adjacency,
            spatial_hash=spatial_hash,
            subcell_lookup=subcell_lookup,
            ninja_radius=NINJA_RADIUS,
            level_cache=path_calculator.level_cache,
            goal_id=current_goal_id,
        )

        if ninja_node is not None:
            # Get next hop toward goal
            next_hop = path_calculator.level_cache.get_next_hop(
                ninja_node, current_goal_id
            )

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
    if (
        hasattr(path_calculator, "level_cache")
        and path_calculator.level_cache is not None
        and current_goal_pos is not None
    ):
        # Extract base_adjacency for physics checks (defensive - already extracted above at line 186)
        # Re-extract here for self-contained Feature 21 calculation
        base_adjacency_for_difficulty = (
            graph_data.get("base_adjacency", adjacency) if graph_data else adjacency
        )

        # Geometric distance (pixels)
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

        # Physics cost (difficulty-weighted)
        physics_cost = path_calculator.get_distance(
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

    return features
