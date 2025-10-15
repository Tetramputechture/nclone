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

# Graph configuration constants - Optimized for single tile-level graph
# Tile-level resolution (24px) - no sub-grid needed for efficiency
TILE_LEVEL_RESOLUTION = TILE_PIXEL_SIZE  # 24 pixels per node

# Optimized graph configuration (down from hierarchical multi-resolution)
N_MAX_NODES = 1000  # Conservative upper bound (down from ~18,000)
E_MAX_EDGES = 4000  # 4 adjacency edges per node average (down from ~144,000)

# Feature dimensions for optimized observation space
NODE_FEATURE_DIM = 19  # Compact features (down from 55)
EDGE_FEATURE_DIM = 6  # Comprehensive edge features (unchanged)

# Type counts (derived from IntEnum classes below)
N_NODE_TYPES = 6  # NodeType: EMPTY, WALL, ENTITY, HAZARD, SPAWN, EXIT
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
class StaticGraphStructure:
    """Static graph topology that doesn't change during episode."""

    node_positions: np.ndarray  # [num_nodes, 2] - (x, y) positions
    node_types: np.ndarray  # [num_nodes] - NodeType enum values
    tile_categories: np.ndarray  # [num_nodes] - 0=empty, 1=solid, 2=navigable
    edge_index: np.ndarray  # [2, num_edges] - adjacency structure
    edge_types: np.ndarray  # [num_edges] - EdgeType enum values
    entity_node_indices: Dict[str, int]  # entity_id -> node_index mapping
    tile_to_node_index: Dict[Tuple[int, int], int]  # (x, y) -> node_index
    num_nodes: int
    num_edges: int


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
