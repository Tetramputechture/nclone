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
    NODE_WORLD_COORD_OFFSET,
)
from ...constants.entity_types import EntityType
from ...constants.physics_constants import (
    MAP_TILE_WIDTH,
    MAP_TILE_HEIGHT,
    EXIT_SWITCH_RADIUS,
    EXIT_DOOR_RADIUS,
    LOCKED_DOOR_SWITCH_RADIUS,
    TOGGLE_MINE_RADII,
    NINJA_RADIUS,
)
from ...gym_environment.constants import LEVEL_DIAGONAL
from .subcell_node_lookup import SUB_NODE_SIZE
from collections import deque


def compute_reachability_features_from_graph(
    adjacency: Dict[Tuple[int, int], list],
    graph_data: Dict[str, Any],
    level_data: Any,
    ninja_pos: Tuple[int, int],
    path_calculator: Any,
) -> np.ndarray:
    """
    Compute 8-dimensional reachability features using adjacency graph and lookup tables.

    Uses efficient O(1) lookups for node finding and precomputed level cache for distances.
    No linear searches or Euclidean distance calculations.

    Args:
        adjacency: Graph adjacency structure from GraphBuilder
        graph_data: Full graph data dict with spatial_hash
        level_data: LevelData object with entities
        ninja_pos: Current ninja position (x, y) in world coordinates
        path_calculator: CachedPathDistanceCalculator instance

    Returns:
        8-dimensional numpy array with normalized features [0-1]

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

    features = np.zeros(8, dtype=np.float32)

    # Feature 1: Reachable area ratio
    features[0] = np.clip(len(adjacency) / max(total_possible_nodes, 1), 0.0, 1.0)

    # Compute reachable area scale for distance normalization
    # Use flood-fill from start position to count reachable nodes
    area_scale = LEVEL_DIAGONAL  # Fallback
    try:
        start_position = getattr(level_data, "start_position", None)
        if start_position is not None:
            # Convert start position to world space
            start_pos = (
                int(start_position[0]) + NODE_WORLD_COORD_OFFSET,
                int(start_position[1]) + NODE_WORLD_COORD_OFFSET,
            )

            # Find closest node to start position
            closest_node = find_closest_node_to_position(
                start_pos,
                adjacency,
                threshold=NINJA_RADIUS,
                spatial_hash=spatial_hash,
                subcell_lookup=subcell_lookup,
            )

            if closest_node is None:
                closest_node = find_closest_node_to_position(
                    start_pos,
                    adjacency,
                    threshold=50.0,
                    spatial_hash=spatial_hash,
                    subcell_lookup=subcell_lookup,
                )

            if closest_node is not None and closest_node in adjacency:
                # Flood fill from start node
                reachable_nodes = set()
                queue = deque([closest_node])
                visited = set([closest_node])

                while queue:
                    current = queue.popleft()
                    reachable_nodes.add(current)

                    neighbors = adjacency.get(current, [])
                    for neighbor_info in neighbors:
                        if isinstance(neighbor_info, tuple) and len(neighbor_info) >= 2:
                            neighbor_pos = neighbor_info[0]
                            if neighbor_pos not in visited:
                                visited.add(neighbor_pos)
                                queue.append(neighbor_pos)

                if len(reachable_nodes) > 0:
                    surface_area = float(len(reachable_nodes))
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

    # Feature 4: Objective path quality (normalized)
    # Measures path smoothness/quality to next objective (replaces switch count)
    # Uses path distance vs Euclidean distance ratio as quality metric
    path_quality = 1.0  # Default: perfect path

    if next_obj_distances:
        # Find next objective position
        next_obj_pos = None
        for switch in exit_switches:
            if getattr(switch, "active", True):
                next_obj_pos = get_entity_position(switch)
                break
        if next_obj_pos is None:
            for switch in locked_door_switches:
                if getattr(switch, "active", True):
                    next_obj_pos = get_entity_position(switch)
                    break
        if next_obj_pos is None and exits:
            next_obj_pos = get_entity_position(exits[0])

        if next_obj_pos:
            path_dist = min(next_obj_distances)
            euclidean_dist = np.sqrt(
                (next_obj_pos[0] - ninja_pos[0]) ** 2
                + (next_obj_pos[1] - ninja_pos[1]) ** 2
            )
            if euclidean_dist > 0:
                # Quality = 1.0 if path is direct (ratio = 1.0), decreases as path becomes longer
                path_ratio = path_dist / euclidean_dist
                path_quality = np.clip(
                    2.0 - path_ratio, 0.0, 1.0
                )  # 1.0 = direct, 0.0 = very indirect

    features[3] = path_quality

    # Feature 5: Deadly mines on optimal path (normalized)
    # Counts deadly (toggled) mines near the optimal path to next objective
    hazard_types = [
        EntityType.TOGGLE_MINE,
        EntityType.TOGGLE_MINE_TOGGLED,
    ]
    hazards = [e for e in entities if get_entity_type(e) in hazard_types]

    deadly_mines_on_path = 0
    if next_obj_distances:
        # Find next objective position for path analysis
        next_obj_pos = None
        for switch in exit_switches:
            if getattr(switch, "active", True):
                next_obj_pos = get_entity_position(switch)
                break
        if next_obj_pos is None:
            for switch in locked_door_switches:
                if getattr(switch, "active", True):
                    next_obj_pos = get_entity_position(switch)
                    break
        if next_obj_pos is None and exits:
            next_obj_pos = get_entity_position(exits[0])

        if next_obj_pos:
            # Count deadly mines near the path
            # A mine is "on path" if it's within 50 pixels of the optimal path
            path_threshold = 50.0
        for hazard in hazards:
            hazard_state = getattr(hazard, "state", 1)
            # For TOGGLE_MINE_TOGGLED, state is always 0 (deadly)
            if getattr(hazard, "entity_type", None) == EntityType.TOGGLE_MINE_TOGGLED:
                hazard_state = 0

            if hazard_state == 0:  # Deadly mine
                hazard_pos = get_entity_position(hazard)
                # Check if mine is near the path (simplified: check if closer to path than threshold)
                # Use path distance to mine as proxy for "on path"
                # Get mine radius based on state (0=toggled/deadly)
                mine_radius = TOGGLE_MINE_RADII.get(hazard_state, 4.0)
                mine_path_dist = path_calculator.get_distance(
                    ninja_pos,
                    hazard_pos,
                    adjacency,
                    level_data=level_data,
                    graph_data=graph_data,
                    entity_radius=mine_radius,
                    ninja_radius=NINJA_RADIUS,
                )
                obj_path_dist = min(next_obj_distances)
                # If mine is between ninja and objective (path distance similar), count it
                if mine_path_dist < obj_path_dist + path_threshold:
                    deadly_mines_on_path += 1

    features[4] = np.clip(
        deadly_mines_on_path / 5.0, 0.0, 1.0
    )  # Max 5 deadly mines on path

    # Feature 6: Connectivity score (edge density)
    total_edges = sum(len(neighbors) for neighbors in adjacency.values())
    if len(adjacency) > 0:
        features[5] = np.clip(total_edges / (len(adjacency) * 4.0), 0.0, 1.0)
    else:
        features[5] = 0.0

    # Feature 7: Next objective reachable (replaces "exit reachable")
    # Checks if the next objective in hierarchy is reachable
    next_obj_reachable = False
    if next_obj_distances:
        # If we found distances, the objective is reachable
        next_obj_reachable = True
    features[6] = 1.0 if next_obj_reachable else 0.0

    # Feature 8: Full completion path exists (switch→door→exit)
    # Verifies that a complete path exists through all objectives
    full_path_exists = False
    if exits and switches:
        # Check if exit is reachable
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

        # Check if at least one switch is reachable
        switch_reachable = False
        for switch in switches:
            switch_pos = get_entity_position(switch)
            switch_node = find_closest_node_to_position(
                switch_pos,
                adjacency,
                threshold=50.0,
                spatial_hash=spatial_hash,
                subcell_lookup=subcell_lookup,
            )
            if switch_node is not None and switch_node in adjacency:
                switch_reachable = True
                break

        # Full path exists if both switch and exit are reachable
        full_path_exists = exit_reachable and switch_reachable

    features[7] = 1.0 if full_path_exists else 0.0

    return features
