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
)
from ...constants.entity_types import EntityType
from ...constants.physics_constants import (
    MAP_TILE_WIDTH,
    MAP_TILE_HEIGHT,
    FULL_MAP_WIDTH_PX,
    FULL_MAP_HEIGHT_PX,
)


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
    max_distance = np.sqrt(FULL_MAP_WIDTH_PX**2 + FULL_MAP_HEIGHT_PX**2)

    features = np.zeros(8, dtype=np.float32)

    # Feature 1: Reachable area ratio
    features[0] = np.clip(len(adjacency) / max(total_possible_nodes, 1), 0.0, 1.0)

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

    # Feature 2: Distance to nearest switch (normalized, inverted)
    switch_distances = []
    for switch in switches:
        switch_pos = get_entity_position(switch)
        distance = path_calculator.get_distance(
            ninja_pos,
            switch_pos,
            adjacency,
            level_data=level_data,
            graph_data=graph_data,
        )
        if distance != float("inf"):
            switch_distances.append(distance)
    if switch_distances:
        min_switch_dist = min(switch_distances)
        features[1] = np.clip(1.0 - (min_switch_dist / max_distance), 0.0, 1.0)

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
        )
        if distance != float("inf"):
            exit_distances.append(distance)
    if exit_distances:
        min_exit_dist = min(exit_distances)
        features[2] = np.clip(1.0 - (min_exit_dist / max_distance), 0.0, 1.0)

    # Feature 4: Reachable switches count (normalized)
    reachable_switches = 0
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
            reachable_switches += 1
    features[3] = np.clip(reachable_switches / 5.0, 0.0, 1.0)  # Max 5 switches

    # Feature 5: Reachable hazards count (normalized)
    # Only consider toggle mines (type 1: TOGGLE_MINE, type 21: TOGGLE_MINE_TOGGLED)
    hazard_types = [
        EntityType.TOGGLE_MINE,
        EntityType.TOGGLE_MINE_TOGGLED,
    ]
    hazards = [e for e in entities if get_entity_type(e) in hazard_types]

    # Cache hazard nodes since hazard positions never change during a level
    hazard_node_cache = graph_data.get("_hazard_node_cache")
    if hazard_node_cache is None:
        # Build cache: map hazard position -> closest node (or None)
        hazard_node_cache = {}
        for hazard in hazards:
            hazard_pos = get_entity_position(hazard)
            hazard_node = find_closest_node_to_position(
                hazard_pos,
                adjacency,
                threshold=200.0,  # Larger threshold for hazards
                spatial_hash=spatial_hash,
                subcell_lookup=subcell_lookup,
            )
            hazard_node_cache[hazard_pos] = hazard_node
        # Store cache in graph_data for future use
        graph_data["_hazard_node_cache"] = hazard_node_cache

    # Count reachable hazards using cached nodes
    reachable_hazards = 0
    for hazard in hazards:
        hazard_pos = get_entity_position(hazard)
        hazard_node = hazard_node_cache.get(hazard_pos)
        if hazard_node is not None and hazard_node in adjacency:
            reachable_hazards += 1
    features[4] = np.clip(reachable_hazards / 10.0, 0.0, 1.0)  # Max 10 hazards

    # Feature 6: Connectivity score (edge density)
    total_edges = sum(len(neighbors) for neighbors in adjacency.values())
    if len(adjacency) > 0:
        features[5] = np.clip(total_edges / (len(adjacency) * 4.0), 0.0, 1.0)
    else:
        features[5] = 0.0

    # Feature 7: Exit reachable flag
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
    features[6] = 1.0 if exit_reachable else 0.0

    # Feature 8: Switch-to-exit path exists
    switch_to_exit_path = False
    if exit_reachable and reachable_switches > 0:
        # Check if any switch and exit are both reachable
        switch_to_exit_path = True
    features[7] = 1.0 if switch_to_exit_path else 0.0

    return features
