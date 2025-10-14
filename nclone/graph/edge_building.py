"""
Simplified edge building for strategic RL representation.

This module creates basic graph edges using only connectivity information from
the fast flood fill reachability system. It focuses on simple adjacency and reachability relationships.
"""

import numpy as np
from typing import List, Tuple, Set
from .reachability.reachability_system import ReachabilitySystem
from .common import EdgeType, NodeType, GraphData, Edge
from .level_data import LevelData
from ..constants.physics_constants import TILE_PIXEL_SIZE
from ..constants.entity_types import EntityType


class EdgeBuilder:
    """
    Edge builder using basic connectivity.

    This approach focuses on simple connectivity information:
    - Adjacent edges for direct movement possibilities
    - Basic reachability edges from flood fill analysis
    """

    def __init__(
        self, debug: bool = False, reachability_system: ReachabilitySystem = None
    ):
        """Initialize simplified edge builder."""
        self.debug = debug
        self.reachability_system = reachability_system or ReachabilitySystem()

    def build_edges(self, level_data: LevelData) -> List[Edge]:
        """
        Build simplified edges using basic connectivity.

        Args:
            level_data: Complete level data including tiles, entities, and player state

        Returns:
            List of edges
        """
        edges = []

        # Get traversable positions from level data
        traversable_positions = self._get_traversable_positions(level_data)

        # 1. Create adjacent edges (basic connectivity)
        adjacent_edges = self._create_adjacent_edges(traversable_positions)
        edges.extend(adjacent_edges)

        # 2. Create reachable edges (using flood fill results)
        reachable_edges = []
        if level_data.player and level_data.player.position:
            reachable_edges = self._create_reachable_edges(
                level_data.player.position, level_data, traversable_positions
            )
            edges.extend(reachable_edges)

        if self.debug:
            print(f"Created {len(edges)} edges:")
            print(f"  Adjacent: {len(adjacent_edges)}")
            print(f"  Reachable: {len(reachable_edges)}")

        return edges

    def _get_traversable_positions(self, level_data: LevelData) -> Set[Tuple[int, int]]:
        """Get all positions that are not solid walls."""
        traversable = set()

        for row in range(level_data.height):
            for col in range(level_data.width):
                tile_id = level_data.tiles[row, col]
                # Check if tile is traversable based on approximate tile analysis
                # Tile 0: Empty space (fully traversable)
                # Tiles 2-33: Complex geometry but generally traversable
                # Tile 1 and 34+: Solid walls or glitched (not traversable)
                if tile_id == 0 or (2 <= tile_id <= 33):
                    # Convert to pixel coordinates (center of tile)
                    pixel_x = col * TILE_PIXEL_SIZE + TILE_PIXEL_SIZE // 2
                    pixel_y = row * TILE_PIXEL_SIZE + TILE_PIXEL_SIZE // 2
                    traversable.add((pixel_x, pixel_y))

        return traversable

    def _create_adjacent_edges(
        self, traversable_positions: Set[Tuple[int, int]]
    ) -> List[Edge]:
        """Create edges between adjacent traversable positions."""
        edges = []

        # Use 4-connectivity (up, down, left, right)
        directions = [
            (0, TILE_PIXEL_SIZE),
            (0, -TILE_PIXEL_SIZE),
            (TILE_PIXEL_SIZE, 0),
            (-TILE_PIXEL_SIZE, 0),
        ]

        for pos in traversable_positions:
            x, y = pos
            for dx, dy in directions:
                neighbor = (x + dx, y + dy)
                if neighbor in traversable_positions:
                    edges.append(
                        Edge(
                            source=pos,
                            target=neighbor,
                            edge_type=EdgeType.ADJACENT,  # Direct adjacency between nodes
                            weight=1.0,
                        )
                    )

        return edges

    def _create_reachable_edges(
        self,
        ninja_pos: Tuple[int, int],
        level_data: LevelData,
        traversable_positions: Set[Tuple[int, int]],
    ) -> List[Edge]:
        """Create simplified edges based on reachability analysis."""
        edges = []

        # Get reachable positions from flood fill
        switch_states = getattr(level_data, "switch_states", {})
        result = self.reachability_system.analyze_reachability(
            level_data=level_data,
            ninja_position=ninja_pos,
            switch_states=switch_states,
        )

        if hasattr(result, "reachable_positions"):
            reachable_positions = result.reachable_positions

            # Create simple reachable edges from ninja position to all reachable positions
            # This provides strategic information about what's accessible
            for pos in reachable_positions:
                if pos != ninja_pos and pos in traversable_positions:
                    distance = np.sqrt(
                        (ninja_pos[0] - pos[0]) ** 2 + (ninja_pos[1] - pos[1]) ** 2
                    )
                    # Only create edges for positions that are reasonably close
                    if distance < TILE_PIXEL_SIZE * 10:  # Within 10 tiles
                        edges.append(
                            Edge(
                                source=ninja_pos,
                                target=pos,
                                edge_type=EdgeType.REACHABLE,  # Can reach via movement
                                weight=distance / TILE_PIXEL_SIZE,
                            )
                        )

        return edges


def _determine_node_type(pos: Tuple[int, int], level_data: LevelData) -> NodeType:
    """
    Determine the node type for a position based on tiles and entities.

    Args:
        pos: Position in pixel coordinates (x, y)
        level_data: Complete level data including tiles and entities

    Returns:
        NodeType enum value representing the type of node at this position
    """
    x, y = pos

    # Convert pixel coordinates to tile coordinates
    tile_col = int(x // TILE_PIXEL_SIZE)
    tile_row = int(y // TILE_PIXEL_SIZE)

    # Check if position is within level bounds
    if not (0 <= tile_row < level_data.height and 0 <= tile_col < level_data.width):
        return NodeType.WALL  # Out of bounds treated as wall

    # Get tile type at this position
    tile_id = level_data.tiles[tile_row, tile_col]

    # Check for player spawn point
    if level_data.player and level_data.player.position:
        player_x, player_y = level_data.player.position
        # Check if this position is close to player spawn (within one tile)
        if abs(x - player_x) < TILE_PIXEL_SIZE and abs(y - player_y) < TILE_PIXEL_SIZE:
            return NodeType.SPAWN

    # Check for entities at this position (entities take priority over tiles)
    entities_here = level_data.get_entities_in_region(
        x - TILE_PIXEL_SIZE // 2,
        y - TILE_PIXEL_SIZE // 2,
        x + TILE_PIXEL_SIZE // 2,
        y + TILE_PIXEL_SIZE // 2,
    )

    if entities_here:
        # Determine node type based on entity types present
        for entity in entities_here:
            entity_type = entity.get("type", 0)

            # Exit entities (doors and switches)
            if entity_type in [EntityType.EXIT_DOOR, EntityType.EXIT_SWITCH]:
                return NodeType.EXIT

            # Hazardous entities
            if entity_type in [
                EntityType.TOGGLE_MINE,
                EntityType.TOGGLE_MINE_TOGGLED,
                EntityType.DRONE_ZAP,
                EntityType.DRONE_CHASER,
                EntityType.MINI_DRONE,
                EntityType.THWUMP,
                EntityType.SHWUMP,
                EntityType.DEATH_BALL,
            ]:
                return NodeType.HAZARD

            # Interactive entities (doors, switches, etc.)
            if entity_type in [
                EntityType.REGULAR_DOOR,
                EntityType.LOCKED_DOOR,
                EntityType.TRAP_DOOR,
                EntityType.GOLD,
                EntityType.LAUNCH_PAD,
                EntityType.ONE_WAY,
                EntityType.BOUNCE_BLOCK,
                EntityType.BOOST_PAD,
            ]:
                return NodeType.ENTITY

    # Determine node type based on tile
    if tile_id == 0:
        return NodeType.EMPTY  # Empty/traversable space
    elif tile_id == 1:
        return NodeType.WALL  # Solid wall
    else:
        # For complex tile types, determine if they're generally traversable
        # Based on tile definitions, most tiles are some form of geometry that can be navigated
        # Only fully solid tiles (type 1) are completely impassable
        # Half-tiles (2-5), slopes (6-9), quarter circles/pipes (10-17), and mild/steep slopes (18-33)
        # are all partially or fully traversable depending on ninja position and movement
        # For strategic RL purposes, we'll treat them as EMPTY since they allow some form of traversal
        if 2 <= tile_id <= 33:
            return NodeType.EMPTY  # Complex geometry but traversable
        else:
            # Glitched tiles (34-37) or unknown tiles - treat as walls for safety
            return NodeType.WALL


def create_graph_data(edges: List[Edge], level_data: LevelData) -> GraphData:
    """
    Create GraphData from edges with comprehensive 56-dim node and 6-dim edge features.

    This function converts the edge representation into the standard GraphData format
    expected by the rest of the system, using NodeFeatureBuilder and EdgeFeatureBuilder
    to create full feature representations.
    """
    from .common import N_MAX_NODES, E_MAX_EDGES, NODE_FEATURE_DIM, EDGE_FEATURE_DIM
    from .feature_builder import NodeFeatureBuilder, EdgeFeatureBuilder

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
    node_features = np.zeros((N_MAX_NODES, NODE_FEATURE_DIM), dtype=np.float32)
    node_types = np.zeros(N_MAX_NODES, dtype=np.int32)
    node_mask = np.zeros(N_MAX_NODES, dtype=np.int32)

    edge_index = np.zeros((2, E_MAX_EDGES), dtype=np.int32)
    edge_features = np.zeros((E_MAX_EDGES, EDGE_FEATURE_DIM), dtype=np.float32)
    edge_types = np.zeros(E_MAX_EDGES, dtype=np.int32)
    edge_mask = np.zeros(E_MAX_EDGES, dtype=np.int32)

    # Initialize feature builders
    node_builder = NodeFeatureBuilder()
    edge_builder = EdgeFeatureBuilder()

    # Fill node data with comprehensive features
    for i, pos in enumerate(sorted(positions)):
        if i >= N_MAX_NODES:
            break

        # Determine node type
        node_type = _determine_node_type(pos, level_data)

        # Build comprehensive node features (56 dimensions)
        node_features[i] = node_builder.build_node_features(
            position=pos,
            node_type=node_type,
            level_data=level_data,
            resolution_level=1,  # Default resolution
        )
        node_types[i] = node_type
        node_mask[i] = 1

    # Fill edge data with comprehensive features
    for i, edge in enumerate(edges):
        if i >= E_MAX_EDGES:
            break

        source_idx = pos_to_idx[edge.source]
        target_idx = pos_to_idx[edge.target]

        edge_index[0, i] = source_idx
        edge_index[1, i] = target_idx

        # Build comprehensive edge features (6 dimensions)
        edge_features[i] = edge_builder.build_edge_features(
            edge=edge,
            source_pos=edge.source,
            target_pos=edge.target,
            level_data=level_data,
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
