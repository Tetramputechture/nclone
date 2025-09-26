"""
Simplified edge building for strategic RL representation.

This module creates basic graph edges using only connectivity information from
the fast flood fill reachability system. It removes complex physics calculations
and focuses on simple adjacency and reachability relationships.
"""

import numpy as np
from typing import List, Tuple, Set, Dict, Optional
from dataclasses import dataclass
from .reachability.reachability_system import ReachabilitySystem
from .common import EdgeType, NodeType, GraphData
from .level_data import LevelData
from ..constants.physics_constants import TILE_PIXEL_SIZE


@dataclass
class Edge:
    """edge representation for strategic RL."""

    source: Tuple[int, int]
    target: Tuple[int, int]
    edge_type: EdgeType
    weight: float = 1.0
    metadata: Optional[Dict] = None


class EdgeBuilder:
    """
    Simplified edge builder using basic connectivity.

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
        if level_data.player and level_data.player.position:
            reachable_edges = self._create_reachable_edges(
                level_data.player.position, level_data, traversable_positions
            )
            edges.extend(reachable_edges)

        if self.debug:
            print(f"Created {len(edges)} simplified edges:")
            print(f"  Adjacent: {len(adjacent_edges)}")
            print(f"  Reachable: {len(reachable_edges)}")

        return edges

    def _get_traversable_positions(self, level_data: LevelData) -> Set[Tuple[int, int]]:
        """Get all positions that are not solid walls."""
        traversable = set()

        for row in range(level_data.height):
            for col in range(level_data.width):
                tile_id = level_data.tiles[row, col]
                # Assume tile_id 0 is empty space (traversable)
                # This is a simplification - in reality we'd check tile properties
                if tile_id == 0:
                    # Convert to pixel coordinates
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
                            edge_type=EdgeType.WALK,  # Use WALK instead of ADJACENT
                            weight=1.0,
                        )
                    )

        return edges

    def _create_reachable_edges(
        self, ninja_pos: Tuple[int, int], level_data: LevelData, traversable_positions: Set[Tuple[int, int]]
    ) -> List[Edge]:
        """Create simplified edges based on reachability analysis."""
        edges = []

        # Get reachable positions from flood fill
        switch_states = getattr(level_data, 'switch_states', {})
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
                                edge_type=EdgeType.JUMP,  # Use JUMP to indicate reachability
                                weight=distance / TILE_PIXEL_SIZE,
                            )
                        )

        return edges


def create_simplified_graph_data(edges: List[Edge], level_data: LevelData) -> GraphData:
    """
    Create simplified GraphData from edges.

    This function converts the edge representation into the
    standard GraphData format expected by the rest of the system.
    """
    from .common import N_MAX_NODES, E_MAX_EDGES

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

    # Initialize fixed-size arrays
    node_features = np.zeros((N_MAX_NODES, 3), dtype=np.float32)  # [x, y, node_type]
    node_types = np.zeros(N_MAX_NODES, dtype=np.int32)
    node_mask = np.zeros(N_MAX_NODES, dtype=np.int32)

    edge_index = np.zeros((2, E_MAX_EDGES), dtype=np.int32)
    edge_features = np.zeros((E_MAX_EDGES, 1), dtype=np.float32)  # Just weight
    edge_types = np.zeros(E_MAX_EDGES, dtype=np.int32)
    edge_mask = np.zeros(E_MAX_EDGES, dtype=np.int32)

    # Fill node data
    for i, pos in enumerate(sorted(positions)):
        if i >= N_MAX_NODES:
            break

        # Simple node type assignment
        node_type = NodeType.EMPTY  # Default to empty space

        node_features[i] = [pos[0], pos[1], float(node_type)]
        node_types[i] = node_type
        node_mask[i] = 1

    # Fill edge data
    for i, edge in enumerate(edges):
        if i >= E_MAX_EDGES:
            break

        source_idx = pos_to_idx[edge.source]
        target_idx = pos_to_idx[edge.target]

        edge_index[0, i] = source_idx
        edge_index[1, i] = target_idx
        edge_features[i, 0] = edge.weight
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

