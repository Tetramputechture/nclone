"""
Common graph components and constants for N++ level representations.

This module contains shared classes, enums, and constants used by
both the standard and hierarchical graph builders.
"""

import numpy as np
from dataclasses import dataclass
from enum import IntEnum
from typing import Tuple, Optional, Dict

from ..constants import TILE_PIXEL_SIZE
from ..constants.physics_constants import (
    MAP_TILE_WIDTH,
    MAP_TILE_HEIGHT,
    FULL_MAP_WIDTH,
    FULL_MAP_HEIGHT,
)

# Graph configuration constants
# GraphBuilder uses 2x2 sub-nodes per tile at 12px resolution
# See graph_builder.py: SUB_NODE_OFFSETS = [(6,6), (18,6), (6,18), (18,18)]
SUB_NODES_PER_TILE = 4  # 2×2 grid per 24px tile

# Map dimensions (from physics_constants.py)
# MAP_TILE_WIDTH = 42, MAP_TILE_HEIGHT = 23 (playable area)
# FULL_MAP_WIDTH = 44, FULL_MAP_HEIGHT = 25 (with 1-tile padding)
FULL_TILES = FULL_MAP_WIDTH * FULL_MAP_HEIGHT  # 1,100 tiles

# Theoretical maximum (if all tiles fully traversable):
THEORETICAL_MAX_NODES = FULL_TILES * SUB_NODES_PER_TILE  # 4,400
# Add buffer for edge cases and entity nodes
N_MAX_NODES = THEORETICAL_MAX_NODES + 100  # 4,500

# GraphBuilder uses 4-connectivity (N, E, S, W cardinal only, no diagonals)
# See graph_builder.py: directions = {"N": (0, -12), "E": (12, 0), "S": (0, 12), "W": (-12, 0)}
MAX_NEIGHBORS_PER_NODE = 4
THEORETICAL_MAX_EDGES = N_MAX_NODES * MAX_NEIGHBORS_PER_NODE  # 18,000
# Add buffer for functional edges (entity interactions)
E_MAX_EDGES = THEORETICAL_MAX_EDGES + 500  # 18,500

# Legacy constants for backward compatibility (deprecated)
SUB_GRID_RESOLUTION = 2  # Matches actual 2x2 sub-nodes
SUB_CELL_SIZE = TILE_PIXEL_SIZE // SUB_GRID_RESOLUTION  # 12 pixels per sub-cell
SUB_GRID_WIDTH = MAP_TILE_WIDTH * SUB_GRID_RESOLUTION  # 84 sub-cells wide (42*2)
SUB_GRID_HEIGHT = MAP_TILE_HEIGHT * SUB_GRID_RESOLUTION  # 46 sub-cells tall (23*2)

# Feature dimensions for  observation space
NODE_FEATURE_DIM = 17
EDGE_FEATURE_DIM = 12

# Type counts (derived from IntEnum classes below)
N_NODE_TYPES = (
    7  # NodeType: EMPTY, WALL, TOGGLE_MINE, LOCKED_DOOR, SPAWN, EXIT_SWITCH, EXIT_DOOR
)
N_EDGE_TYPES = 4  # EdgeType: ADJACENT, REACHABLE, FUNCTIONAL, BLOCKED

# Entity tracking limits
MAX_TOGGLE_MINES = 128  # Maximum toggled mines per level
MAX_UNTOGGLE_MINES = 128  # Maximum untoggled mines per level
MAX_MINES_TOTAL = MAX_TOGGLE_MINES + MAX_UNTOGGLE_MINES  # 256 total
MAX_LOCKED_DOORS = 16  # Maximum locked doors per level

# Map constants - playable area is 42x23 tiles (1008x552 pixels)
# with a 1-tile border making full map 44x25 tiles (1056x600 pixels)
GRID_WIDTH = MAP_TILE_WIDTH  # 42 tiles (playable area)
GRID_HEIGHT = MAP_TILE_HEIGHT  # 23 tiles (playable area)
CELL_SIZE = TILE_PIXEL_SIZE  # 24 pixels


class NodeType(IntEnum):
    """Simplified node types for strategic RL representation.

    Note: All node positions are guaranteed to be reachable from player spawn
    via GraphBuilder's flood fill. No tile-type traversability checking needed.
    """

    EMPTY = 0  # Traversable space (reachable via flood fill)
    WALL = 1  # Solid obstacle (unused if we only create nodes for reachable positions)
    TOGGLE_MINE = 2  # Toggle mine (EntityType.TOGGLE_MINE or TOGGLE_MINE_TOGGLED)
    LOCKED_DOOR = 3  # Locked door (EntityType.LOCKED_DOOR)
    SPAWN = 4  # Player spawn point
    EXIT_SWITCH = 5  # Exit switch (EntityType.EXIT_SWITCH)
    EXIT_DOOR = 6  # Exit door (EntityType.EXIT_DOOR)


class EdgeType(IntEnum):
    """Simplified edge types for connectivity-based representation."""

    ADJACENT = 0  # Simple adjacency (can move between nodes)
    REACHABLE = 1  # Can reach via movement (jump/fall possible)
    FUNCTIONAL = 2  # Entity interaction edge
    BLOCKED = 3  # Currently blocked (door without key)


@dataclass
class Edge:
    """edge representation for strategic RL."""

    source: Tuple[int, int]
    target: Tuple[int, int]
    edge_type: EdgeType
    weight: float = 1.0
    metadata: Optional[Dict] = None


@dataclass
class GraphData:
    """Container for graph data with fixed-size arrays for Gym compatibility."""

    node_features: np.ndarray  # [N_MAX_NODES, F_node]
    edge_index: np.ndarray  # [2, E_MAX_EDGES]
    edge_features: np.ndarray  # [E_MAX_EDGES, F_edge]
    node_mask: np.ndarray  # [N_MAX_NODES] - 1 for valid nodes, 0 for padding
    edge_mask: np.ndarray  # [E_MAX_EDGES] - 1 for valid edges, 0 for padding
    node_types: np.ndarray  # [N_MAX_NODES] - NodeType enum values
    edge_types: np.ndarray  # [E_MAX_EDGES] - EdgeType enum values
    num_nodes: int
    num_edges: int


@dataclass
class SparseGraphData:
    """Container for sparse graph data in COO (Coordinate) format.

    Memory-optimized format that stores only valid nodes/edges without padding.
    Achieves ~95% memory reduction compared to dense GraphData format.

    This format is mathematically lossless - exact same values as dense format,
    just stored more efficiently. Conversion to/from dense is deterministic.

    Compatible with PyTorch Geometric and can be efficiently converted to dense
    on GPU during training forward pass.
    """

    node_features: np.ndarray  # [num_nodes, NODE_FEATURE_DIM] - only valid nodes
    edge_index: np.ndarray  # [2, num_edges] - COO format edge indices
    edge_features: np.ndarray  # [num_edges, EDGE_FEATURE_DIM] - only valid edges
    node_types: np.ndarray  # [num_nodes] - NodeType enum values
    edge_types: np.ndarray  # [num_edges] - EdgeType enum values
    num_nodes: int  # Actual number of valid nodes
    num_edges: int  # Actual number of valid edges

    def to_dense(self) -> GraphData:
        """Convert sparse format to dense format with padding.

        Returns:
            GraphData with padded arrays matching N_MAX_NODES and E_MAX_EDGES
        """
        # Initialize dense arrays with zeros (padding)
        dense_node_features = np.zeros(
            (N_MAX_NODES, NODE_FEATURE_DIM), dtype=np.float32
        )
        dense_edge_index = np.zeros((2, E_MAX_EDGES), dtype=np.int32)
        dense_edge_features = np.zeros(
            (E_MAX_EDGES, EDGE_FEATURE_DIM), dtype=np.float32
        )
        dense_node_mask = np.zeros(N_MAX_NODES, dtype=np.int32)
        dense_edge_mask = np.zeros(E_MAX_EDGES, dtype=np.int32)
        dense_node_types = np.zeros(N_MAX_NODES, dtype=np.int32)
        dense_edge_types = np.zeros(E_MAX_EDGES, dtype=np.int32)

        # Copy valid data
        dense_node_features[: self.num_nodes] = self.node_features
        dense_edge_index[:, : self.num_edges] = self.edge_index
        dense_edge_features[: self.num_edges] = self.edge_features
        dense_node_types[: self.num_nodes] = self.node_types
        dense_edge_types[: self.num_edges] = self.edge_types

        # Set masks
        dense_node_mask[: self.num_nodes] = 1
        dense_edge_mask[: self.num_edges] = 1

        return GraphData(
            node_features=dense_node_features,
            edge_index=dense_edge_index,
            edge_features=dense_edge_features,
            node_mask=dense_node_mask,
            edge_mask=dense_edge_mask,
            node_types=dense_node_types,
            edge_types=dense_edge_types,
            num_nodes=self.num_nodes,
            num_edges=self.num_edges,
        )

    @staticmethod
    def from_dense(graph_data: GraphData) -> "SparseGraphData":
        """Convert dense format to sparse format by removing padding.

        Args:
            graph_data: Dense GraphData with padded arrays

        Returns:
            SparseGraphData with only valid nodes/edges
        """
        # Extract only valid nodes/edges using masks or num_nodes/num_edges
        num_nodes = graph_data.num_nodes
        num_edges = graph_data.num_edges

        return SparseGraphData(
            node_features=graph_data.node_features[:num_nodes].copy(),
            edge_index=graph_data.edge_index[:, :num_edges].copy(),
            edge_features=graph_data.edge_features[:num_edges].copy(),
            node_types=graph_data.node_types[:num_nodes].copy(),
            edge_types=graph_data.edge_types[:num_edges].copy(),
            num_nodes=num_nodes,
            num_edges=num_edges,
        )

    def to_torch_geometric(self):
        """Convert to PyTorch Geometric Data format (optional utility).

        This is useful for direct PyG integration if needed.
        Requires torch to be installed.

        Returns:
            torch_geometric.data.Data object
        """
        try:
            import torch
            from torch_geometric.data import Data

            # Convert numpy arrays to torch tensors
            x = torch.from_numpy(self.node_features).float()
            edge_index = torch.from_numpy(self.edge_index).long()
            edge_attr = torch.from_numpy(self.edge_features).float()

            return Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                num_nodes=self.num_nodes,
            )
        except ImportError:
            raise ImportError(
                "PyTorch Geometric not installed. "
                "Install with: pip install torch-geometric"
            )

    def memory_bytes(self) -> int:
        """Calculate actual memory usage in bytes.

        Returns:
            Total bytes used by all arrays
        """
        return (
            self.node_features.nbytes
            + self.edge_index.nbytes
            + self.edge_features.nbytes
            + self.node_types.nbytes
            + self.edge_types.nbytes
        )
