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


def compute_reachability_features_from_graph(
    adjacency: Dict[Tuple[int, int], list],
    graph_data: Dict[str, Any],
    level_data: Any,
    ninja_pos: Tuple[int, int],
    path_calculator: Any,
) -> np.ndarray:
    """
    Compute 7-dimensional reachability features using adjacency graph and lookup tables.

    Uses efficient O(1) lookups for node finding and precomputed level cache for distances.
    No linear searches or Euclidean distance calculations.

    Features (7 dims):
    1. Reachable area ratio (0-1)
    2. Distance to nearest switch (normalized, inverted)
    3. Distance to exit (normalized, inverted)
    4. Exit reachable flag (0-1)
    5. Total mines normalized (0-1)
    6. Deadly mine ratio (0-1)
    7. Switch activated flag (0-1) - EXPLICIT phase indicator for Markov property

    Args:
        adjacency: Graph adjacency structure from GraphBuilder
        graph_data: Full graph data dict with spatial_hash
        level_data: LevelData object with entities
        ninja_pos: Current ninja position (x, y) in world coordinates
        path_calculator: CachedPathDistanceCalculator instance

    Returns:
        7-dimensional numpy array with normalized features [0-1]

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

    features = np.zeros(7, dtype=np.float32)

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
                level_data=level_data,
                graph_data=graph_data,
                entity_radius=EXIT_SWITCH_RADIUS,
                ninja_radius=NINJA_RADIUS,
            )
            if distance != float("inf"):
                next_obj_distances.append(distance)

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
            level_data=level_data,
            graph_data=graph_data,
            entity_radius=EXIT_DOOR_RADIUS,
            ninja_radius=NINJA_RADIUS,
        )
        if distance != float("inf"):
            exit_distances.append(distance)
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

    # Features 5-6: Mine context (simplified from detailed mine features)
    # Mine information is primarily handled by graph nodes, these provide high-level context
    hazard_types = [
        EntityType.TOGGLE_MINE,
        EntityType.TOGGLE_MINE_TOGGLED,
    ]
    hazards = [e for e in entities if get_entity_type(e) in hazard_types]

    # Feature 5: Total mines normalized (0-1)
    total_mines = len(hazards)
    features[4] = np.clip(total_mines / 10.0, 0.0, 1.0)  # Max 10 mines

    # Feature 6: Deadly mine ratio (0-1)
    if total_mines > 0:
        deadly_mines = 0
        for hazard in hazards:
            hazard_state = getattr(hazard, "state", 1)
            # For TOGGLE_MINE_TOGGLED, state is always 0 (deadly)
            if getattr(hazard, "entity_type", None) == EntityType.TOGGLE_MINE_TOGGLED:
                hazard_state = 0
            if hazard_state == 0:  # Deadly mine
                deadly_mines += 1
        features[5] = deadly_mines / total_mines
    else:
        features[5] = 0.0

    # Feature 7: Switch activated flag (EXPLICIT phase indicator)
    # Critical for Markov property: agent needs to know which objective to pursue
    # For exit switches: active=True means not collected, active=False means activated
    switch_activated = False
    for switch in exit_switches:
        switch_active = getattr(switch, "active", True)
        if not switch_active:  # Switch has been activated
            switch_activated = True
            break
    features[6] = 1.0 if switch_activated else 0.0

    return features
