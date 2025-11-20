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
from .common import N_MAX_NODES, E_MAX_EDGES, NODE_FEATURE_DIM, EDGE_FEATURE_DIM


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
    from .feature_builder import NodeFeatureBuilder, EdgeFeatureBuilder
    from .reachability.reachability_system import ReachabilitySystem

    # Extract unique positions from edges
    positions = set()
    for edge in edges:
        positions.add(edge.source)
        positions.add(edge.target)

    # If no positions, create at least one node
    if not positions:
        positions.add((0, 0))

    # Create position to index mapping
    pos_to_idx = {pos: idx for idx, pos in enumerate(sorted(positions))}

    num_nodes = len(positions)
    num_edges = len(edges)

    # Initialize fixed-size arrays with full feature dimensions
    # MEMORY OPTIMIZATION: Use efficient data types
    node_features = np.zeros((N_MAX_NODES, NODE_FEATURE_DIM), dtype=np.float32)
    node_types = np.zeros(N_MAX_NODES, dtype=np.uint8)  # 7 node types (0-6)
    node_mask = np.zeros(N_MAX_NODES, dtype=np.uint8)  # Binary 0/1

    edge_index = np.zeros((2, E_MAX_EDGES), dtype=np.uint16)  # Max node index 4500 < 65535
    edge_features = np.zeros((E_MAX_EDGES, EDGE_FEATURE_DIM), dtype=np.float32)
    edge_types = np.zeros(E_MAX_EDGES, dtype=np.uint8)  # 4 edge types (0-3)
    edge_mask = np.zeros(E_MAX_EDGES, dtype=np.uint8)  # Binary 0/1

    # Initialize feature builders
    node_builder = NodeFeatureBuilder()
    edge_builder = EdgeFeatureBuilder()

    # Extract ninja position from level_data for reachability analysis only
    ninja_pos = None
    if hasattr(level_data, "ninja") and level_data.ninja is not None:
        ninja_pos = (level_data.ninja.position.x, level_data.ninja.position.y)

    # Initialize reachability system for connectivity analysis
    reachability_sys = ReachabilitySystem(debug=False)
    reachability_result = None

    # Compute reachability if ninja position is available
    if ninja_pos is not None:
        try:
            # Get current switch states (default to empty if not available)
            switch_states = {}
            if hasattr(level_data, "switch_states"):
                switch_states = level_data.switch_states

            reachability_result = reachability_sys.analyze_reachability(
                level_data=level_data,
                ninja_position=ninja_pos,
                switch_states=switch_states,
            )
        except Exception:
            # If reachability analysis fails, continue without it
            reachability_result = None

    # Compute topological features for all nodes (Phase 5)
    from .topological_features import compute_batch_topological_features

    # Get current objective position (exit switch if not collected, else exit door)
    objective_pos = (0.0, 0.0)  # Default
    if hasattr(level_data, "entities") and level_data.entities:
        from ..constants.entity_types import EntityType

        # Find exit switch and door
        exit_switch = None
        exit_door = None
        for entity in level_data.entities:
            if entity.get("type") == EntityType.EXIT_SWITCH:
                exit_switch = (entity.get("x", 0), entity.get("y", 0))
            elif entity.get("type") == EntityType.EXIT_DOOR:
                exit_door = (entity.get("x", 0), entity.get("y", 0))

        # Use exit switch as objective (agent must collect it first)
        if exit_switch:
            objective_pos = exit_switch
        elif exit_door:
            objective_pos = exit_door

    # Build adjacency dict from edges for topological computation
    adjacency_dict: Dict[Tuple[float, float], List[Tuple[float, float]]] = {}
    for edge in edges:
        if edge.source not in adjacency_dict:
            adjacency_dict[edge.source] = []
        adjacency_dict[edge.source].append(edge.target)

    # Compute topological features for all nodes
    node_positions_array = np.array(sorted(positions), dtype=np.float32)
    batch_topological = compute_batch_topological_features(
        node_positions_array, adjacency_dict, objective_pos, sample_size=100
    )

    # Fill node data with comprehensive features
    for i, pos in enumerate(sorted(positions)):
        if i >= N_MAX_NODES:
            break

        # Determine node type
        node_type = _determine_node_type(pos, level_data)

        # Extract entity info if this is an entity node
        entity_info = _extract_entity_info(pos, level_data)

        # Build reachability info for this node
        reachability_info = None
        if reachability_result is not None:
            reachability_info = {
                "reachable_from_ninja": reachability_result.is_position_reachable(pos),
            }

        # Extract topological info for this node (only objective-relative features)
        topological_info = None
        if batch_topological and len(batch_topological["objective_dx"]) > i:
            topological_info = {
                "objective_dx": float(batch_topological["objective_dx"][i]),
                "objective_dy": float(batch_topological["objective_dy"][i]),
                "objective_hops": float(batch_topological["objective_hops"][i]),
            }

        # Build optimized node features (17 dimensions in Phase 6 - Memory Optimized)
        node_features[i] = node_builder.build_node_features(
            node_pos=pos,
            node_type=node_type,
            entity_info=entity_info,
            reachability_info=reachability_info,
            topological_info=topological_info,
        )
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

    # Compute geometric features for all edges in batch (vectorized for performance)
    from .geometric_features import compute_batch_geometric_features

    if len(edges) > 0:
        # Extract source and target positions
        edge_sources = np.array([edge.source for edge in edges], dtype=np.float32)
        edge_targets = np.array([edge.target for edge in edges], dtype=np.float32)

        # Compute geometric features in batch
        batch_geometric = compute_batch_geometric_features(edge_sources, edge_targets)
    else:
        batch_geometric = None

    # Compute mine danger features for all edges in batch
    from .mine_proximity import batch_compute_mine_features

    if len(edges) > 0 and len(mine_nodes) > 0:
        batch_mine = batch_compute_mine_features(edges, mine_nodes)
    else:
        batch_mine = None

    # Fill edge data with enhanced features
    for i, edge in enumerate(edges):
        if i >= E_MAX_EDGES:
            break

        source_idx = pos_to_idx[edge.source]
        target_idx = pos_to_idx[edge.target]

        edge_index[0, i] = source_idx
        edge_index[1, i] = target_idx

        # Build reachability confidence for edge
        edge_reachability_confidence = 1.0
        if reachability_result is not None:
            # Edge is confident if both endpoints are reachable
            source_reachable = reachability_result.is_position_reachable(edge.source)
            target_reachable = reachability_result.is_position_reachable(edge.target)
            if source_reachable and target_reachable:
                edge_reachability_confidence = reachability_result.confidence
            else:
                edge_reachability_confidence = 0.0

        # Extract geometric features for this edge
        geometric_dict = None
        if batch_geometric is not None:
            geometric_dict = {
                "dx_norm": float(batch_geometric["dx_norm"][i]),
                "dy_norm": float(batch_geometric["dy_norm"][i]),
                "distance": float(batch_geometric["distance"][i]),
                "movement_category": float(batch_geometric["movement_category"][i]),
            }

        # Extract mine danger features for this edge
        mine_dict = None
        if batch_mine is not None:
            mine_dict = {
                "nearest_mine_distance": float(batch_mine["nearest_mine_distance"][i]),
                "passes_deadly_mine": float(batch_mine["passes_deadly_mine"][i]),
                "mine_threat_level": float(batch_mine["mine_threat_level"][i]),
                "num_mines_nearby": float(batch_mine["num_mines_nearby"][i]),
            }

        # Build enhanced edge features (14 dimensions)
        edge_features[i] = edge_builder.build_edge_features(
            edge_type=edge.edge_type,
            weight=edge.weight,
            reachability_confidence=edge_reachability_confidence,
            geometric_features=geometric_dict,
            mine_features=mine_dict,
        )
        edge_types[i] = edge.edge_type
        edge_mask[i] = 1

    return GraphData(
        node_features=node_features,
        edge_index=edge_index,
        edge_features=edge_features,
        node_mask=node_mask,
        edge_mask=edge_mask,
        node_types=node_types,
        edge_types=edge_types,
        num_nodes=num_nodes,
        num_edges=num_edges,
    )
