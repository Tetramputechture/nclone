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
from ..constants.physics_constants import MAP_TILE_WIDTH, MAP_TILE_HEIGHT

# Graph configuration constants
# Sub-grid resolution for better spatial accuracy
SUB_GRID_RESOLUTION = 4  # Divide each tile into 4x4 sub-cells (6x6 pixels each)
SUB_CELL_SIZE = TILE_PIXEL_SIZE // SUB_GRID_RESOLUTION  # 6 pixels per sub-cell

# Calculate sub-grid dimensions using inner playable area (42x23 tiles)
SUB_GRID_WIDTH = MAP_TILE_WIDTH * SUB_GRID_RESOLUTION  # 168 sub-cells wide (42*4)
SUB_GRID_HEIGHT = MAP_TILE_HEIGHT * SUB_GRID_RESOLUTION  # 92 sub-cells tall (23*4)

# Keep generous upper bounds to allow entity padding in observations
N_MAX_NODES = (
    SUB_GRID_WIDTH * SUB_GRID_HEIGHT + 400
)  # Sub-grid + entities buffer (~18000)
E_MAX_EDGES = N_MAX_NODES * 8  # Up to 8 directions per node (4 cardinal + 4 diagonal)

# Feature dimensions for enhanced observation space
NODE_FEATURE_DIM = 56  # Comprehensive node features (reduced from 61, removed 5 unused entity indices)
EDGE_FEATURE_DIM = 6  # Edge features (type one-hot, weight, reachability confidence)

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
    """Simplified node types for strategic RL representation."""

    EMPTY = 0  # Traversable space
    WALL = 1  # Solid obstacle
    ENTITY = 2  # Interactive entity (switch, door, locked door, etc.)
    HAZARD = 3  # Dangerous area (mines, drones, etc.)
    SPAWN = 4  # Player spawn point
    EXIT = 5  # Level exit


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
