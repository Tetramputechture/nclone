"""
Helper functions for building graph data from adjacency information.

This module provides utilities for converting GraphBuilder adjacency graphs
into GraphData format with comprehensive node and edge features for ML models.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from .common import NodeType, GraphData, Edge
from .level_data import LevelData
from ..constants.physics_constants import TILE_PIXEL_SIZE
from ..constants.entity_types import EntityType
from .common import N_MAX_NODES, E_MAX_EDGES, NODE_FEATURE_DIM


def _extract_entity_info(
    pos: Tuple[int, int], level_data: LevelData
) -> Optional[Dict[str, Any]]:
    """
    Extract entity information for a position if an entity is present.

    Args:
        pos: Position in pixel coordinates (x, y)
        level_data: Complete level data including entities

    Returns:
        Entity info dict with type, active, state, radius, closed (for doors)
        or None if no entity at this position
    """
    x, y = pos

    # Check for entities at this position
    entities_here = level_data.get_entities_in_region(
        x - TILE_PIXEL_SIZE // 2,
        y - TILE_PIXEL_SIZE // 2,
        x + TILE_PIXEL_SIZE // 2,
        y + TILE_PIXEL_SIZE // 2,
    )

    if not entities_here:
        return None

    # Return first relevant entity (same priority as node type determination)
    for entity in entities_here:
        entity_type = entity.get("type", 0)

        # Only extract info for entities we care about
        if entity_type in [
            EntityType.EXIT_DOOR,
            EntityType.EXIT_SWITCH,
            EntityType.TOGGLE_MINE,
            EntityType.TOGGLE_MINE_TOGGLED,
            EntityType.LOCKED_DOOR,
        ]:
            entity_info = {
                "type": entity_type,
                "active": entity.get("active", True),
                "state": entity.get("state", 0.0),
                "radius": entity.get("radius", 0.0),
            }

            # Add closed status for doors
            if entity_type == EntityType.LOCKED_DOOR:
                entity_info["closed"] = entity.get("closed", True)

            return entity_info

    return None


def _determine_node_type(pos: Tuple[int, int], level_data: LevelData) -> NodeType:
    """
    Determine node type for a REACHABLE position.

    Note: This function assumes pos is already validated as reachable by
    GraphBuilder's flood fill. No tile-type traversability checking needed.
    All positions passed to this function are guaranteed to be reachable from
    player spawn, so we only need to classify by entity type.

    Args:
        pos: Position in pixel coordinates (x, y) - guaranteed reachable
        level_data: Complete level data including entities

    Returns:
        NodeType enum value representing the type of node at this position
    """
    x, y = pos

    # 1. Check spawn point (fast - single position check)
    if level_data.player and level_data.player.position:
        player_x, player_y = level_data.player.position
        # Check if this position is close to player spawn (within one tile)
        if abs(x - player_x) < TILE_PIXEL_SIZE and abs(y - player_y) < TILE_PIXEL_SIZE:
            return NodeType.SPAWN

    # 2. Check for relevant entities only (4 types: exit switch, exit door, toggle mines, locked doors)
    entities_here = level_data.get_entities_in_region(
        x - TILE_PIXEL_SIZE // 2,
        y - TILE_PIXEL_SIZE // 2,
        x + TILE_PIXEL_SIZE // 2,
        y + TILE_PIXEL_SIZE // 2,
    )

    if entities_here:
        for entity in entities_here:
            entity_type = entity.get("type", 0)

            # Priority order: exits first, then interactive entities
            if entity_type == EntityType.EXIT_DOOR:
                return NodeType.EXIT_DOOR
            if entity_type == EntityType.EXIT_SWITCH:
                return NodeType.EXIT_SWITCH
            if entity_type in [EntityType.TOGGLE_MINE, EntityType.TOGGLE_MINE_TOGGLED]:
                return NodeType.TOGGLE_MINE
            if entity_type == EntityType.LOCKED_DOOR:
                return NodeType.LOCKED_DOOR

    # 3. Default: empty traversable space
    # (We know it's reachable because GraphBuilder included it)
    return NodeType.EMPTY


def create_graph_data(edges: List[Edge], level_data: LevelData) -> GraphData:
    """
    Create GraphData from edges with comprehensive 50-dim node and 6-dim edge features.

    This function converts the edge representation into the standard GraphData format
    expected by the rest of the system, using NodeFeatureBuilder and EdgeFeatureBuilder
    to create full feature representations with proper ReachabilitySystem integration.

    Args:
        edges: List of Edge objects
        level_data: LevelData object
    """
    from .feature_builder import NodeFeatureBuilder
    from .reachability.reachability_system import ReachabilitySystem

    # Extract unique positions from edges
    positions = set()
    for edge in edges:
        positions.add(edge.source)
        positions.add(edge.target)

    # If no positions, create at least one node
    if not positions:
        positions.add((0, 0))

    # DEFENSIVE: Check for mixed types in positions before sorting
    import logging

    logger = logging.getLogger(__name__)
    position_types = {type(pos) for pos in positions}
    if len(position_types) > 1:
        logger.error(f"Mixed types in positions set: {position_types}")
        logger.error(f"  Sample positions: {list(positions)[:10]}")
        # Filter to only keep tuples
        positions = {pos for pos in positions if isinstance(pos, tuple)}
        logger.error(f"  After filtering, kept {len(positions)} tuple positions")

    # Create position to index mapping
    try:
        pos_to_idx = {pos: idx for idx, pos in enumerate(sorted(positions))}
    except TypeError as e:
        logger.error(f"Failed to sort positions: {e}")
        logger.error(f"  Position types: {position_types}")
        logger.error(f"  Sample positions: {list(positions)[:10]}")
        raise

    num_nodes = len(positions)
    num_edges = len(edges)

    # Initialize fixed-size arrays with full feature dimensions
    # MEMORY OPTIMIZATION: Use efficient data types
    node_features = np.zeros((N_MAX_NODES, NODE_FEATURE_DIM), dtype=np.float32)
    node_types = np.zeros(N_MAX_NODES, dtype=np.uint8)  # 7 node types (0-6)
    node_mask = np.zeros(N_MAX_NODES, dtype=np.uint8)  # Binary 0/1

    edge_index = np.zeros(
        (2, E_MAX_EDGES), dtype=np.uint16
    )  # Max node index 4500 < 65535
    edge_mask = np.zeros(E_MAX_EDGES, dtype=np.uint8)  # Binary 0/1

    # Initialize feature builders
    node_builder = NodeFeatureBuilder()
    # No edge feature builder needed - all edges are simple adjacency

    # Fill node data with comprehensive features
    for i, pos in enumerate(sorted(positions)):
        if i >= N_MAX_NODES:
            break

        # Determine node type
        node_type = _determine_node_type(pos, level_data)

        # Extract entity info if this is an entity node
        entity_info = _extract_entity_info(pos, level_data)

        built_features = node_builder.build_node_features(
            node_pos=pos,
            node_type=node_type,
            entity_info=entity_info,
        )

        node_features[i] = built_features
        node_types[i] = node_type
        node_mask[i] = 1

    # Extract mine nodes for mine danger computation
    mine_nodes = []
    entities = level_data.entities if hasattr(level_data, "entities") else []
    for entity in entities:
        entity_type = entity.get("type", 0)
        from ..constants.entity_types import EntityType

        if entity_type in [EntityType.TOGGLE_MINE, EntityType.TOGGLE_MINE_TOGGLED]:
            # Extract mine data
            mine_state_raw = entity.get("state", 0.0)
            # Convert to -1/0/+1 encoding
            if mine_state_raw == 0.0:
                mine_state = -1.0  # Deadly (toggled)
            elif mine_state_raw == 2.0:
                mine_state = 0.0  # Transitioning
            else:
                mine_state = 1.0  # Safe (untoggled)

            mine_nodes.append(
                {
                    "position": (entity.get("x", 0), entity.get("y", 0)),
                    "radius": entity.get("radius", 4.0),
                    "state": mine_state,
                }
            )

    # Fill edge connectivity (no features or types - all edges are simple adjacency)
    for i, edge in enumerate(edges):
        if i >= E_MAX_EDGES:
            break

        source_idx = pos_to_idx[edge.source]
        target_idx = pos_to_idx[edge.target]

        edge_index[0, i] = source_idx
        edge_index[1, i] = target_idx
        edge_mask[i] = 1

    return GraphData(
        node_features=node_features,
        edge_index=edge_index,
        node_mask=node_mask,
        edge_mask=edge_mask,
        node_types=node_types,
        num_nodes=num_nodes,
        num_edges=num_edges,
    )
