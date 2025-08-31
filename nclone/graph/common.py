"""
Common graph components and constants for N++ level representations.

This module contains shared classes, enums, and constants used by
both the standard and hierarchical graph builders.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import IntEnum

# Use shared constants from the simulator
from ..constants import MAP_TILE_WIDTH, MAP_TILE_HEIGHT, TILE_PIXEL_SIZE

# Graph configuration constants
# Sub-grid resolution for better spatial accuracy
SUB_GRID_RESOLUTION = 4  # Divide each tile into 4x4 sub-cells (6x6 pixels each)
SUB_CELL_SIZE = TILE_PIXEL_SIZE // SUB_GRID_RESOLUTION  # 6 pixels per sub-cell

# Calculate sub-grid dimensions
SUB_GRID_WIDTH = MAP_TILE_WIDTH * SUB_GRID_RESOLUTION  # 168 sub-cells wide
SUB_GRID_HEIGHT = MAP_TILE_HEIGHT * SUB_GRID_RESOLUTION  # 92 sub-cells tall

# Keep generous upper bounds to allow entity padding in observations
N_MAX_NODES = SUB_GRID_WIDTH * SUB_GRID_HEIGHT + 400  # Sub-grid + entities buffer (~15856)
E_MAX_EDGES = N_MAX_NODES * 8  # Up to 8 directions per node (4 cardinal + 4 diagonal)

# Mirror simulator map constants to avoid divergence
GRID_WIDTH = MAP_TILE_WIDTH
GRID_HEIGHT = MAP_TILE_HEIGHT
CELL_SIZE = TILE_PIXEL_SIZE


class NodeType(IntEnum):
    """Types of nodes in the graph."""
    GRID_CELL = 0
    ENTITY = 1
    NINJA = 2


class EdgeType(IntEnum):
    """Types of edges in the graph."""
    WALK = 0
    JUMP = 1
    WALL_SLIDE = 2
    FALL = 3
    ONE_WAY = 4
    FUNCTIONAL = 5  # switch->door, launchpad->target, etc.


@dataclass
class GraphData:
    """Container for graph data with fixed-size arrays for Gym compatibility."""
    node_features: np.ndarray  # [N_MAX_NODES, F_node]
    edge_index: np.ndarray     # [2, E_MAX_EDGES]
    edge_features: np.ndarray  # [E_MAX_EDGES, F_edge]
    node_mask: np.ndarray      # [N_MAX_NODES] - 1 for valid nodes, 0 for padding
    edge_mask: np.ndarray      # [E_MAX_EDGES] - 1 for valid edges, 0 for padding
    node_types: np.ndarray     # [N_MAX_NODES] - NodeType enum values
    edge_types: np.ndarray     # [E_MAX_EDGES] - EdgeType enum values
    num_nodes: int
    num_edges: int