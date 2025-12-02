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

# Realistic maximum nodes based on actual level analysis:
# - Most N++ levels have 40-70% solid tiles (platforms, walls, etc.)
# - Flood-fill filtering removes isolated/unreachable areas
# - Typical levels have 300-800 reachable nodes
# - Very open levels (80-100% type 0 tiles) can have 3000-3500 nodes
# - Setting 2500 accommodates very open levels while reducing memory ~44%
#
# Memory impact (per observation):
# - Old (4500 nodes): ~295 KB graph data
# - New (2500 nodes): ~166 KB graph data
# - Rollout buffer: ~22 GB with 64 envs, 1024 steps (vs ~38 GB before)
N_MAX_NODES = 2500  # Accommodates very open levels (80-100% traversable)

# GraphBuilder uses 8-connectivity (N, E, S, W, NE, SE, SW, NW - cardinal + diagonal)
# See graph_builder.py line 936-945 for direction definitions
MAX_NEIGHBORS_PER_NODE = 8
# Edge count scales with node count (8 neighbors per node max)
E_MAX_EDGES = N_MAX_NODES * MAX_NEIGHBORS_PER_NODE  # 20,000

# Legacy constants for backward compatibility (deprecated)
SUB_GRID_RESOLUTION = 2  # Matches actual 2x2 sub-nodes
SUB_CELL_SIZE = TILE_PIXEL_SIZE // SUB_GRID_RESOLUTION  # 12 pixels per sub-cell
SUB_GRID_WIDTH = MAP_TILE_WIDTH * SUB_GRID_RESOLUTION  # 84 sub-cells wide (42*2)
SUB_GRID_HEIGHT = MAP_TILE_HEIGHT * SUB_GRID_RESOLUTION  # 46 sub-cells tall (23*2)

# Feature dimensions for observation space
# GCN-optimized: Reduced from 17 to 6 to 4, then restored spatial for goal-directed navigation
NODE_FEATURE_DIM = 6  # Spatial(2) + Mine(2) + Entity(2)
# Note: Reachability removed - all nodes in graph are reachable (flood fill filtered)
# Note: Spatial features restored - explicit coordinates help with precise distance estimation
#       and goal-directed navigation, enabling faster convergence in early training
EDGE_FEATURE_DIM = (
    0  # No edge features - all edges are simple adjacency between reachable nodes
)

# Type counts (derived from IntEnum classes below)
N_NODE_TYPES = (
    6  # NodeType: EMPTY, TOGGLE_MINE, LOCKED_DOOR, SPAWN, EXIT_SWITCH, EXIT_DOOR
)
N_EDGE_TYPES = 1  # EdgeType: ADJACENT (all edges are adjacency between reachable nodes)

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
    TOGGLE_MINE = 1  # Toggle mine (EntityType.TOGGLE_MINE or TOGGLE_MINE_TOGGLED)
    LOCKED_DOOR = 2  # Locked door (EntityType.LOCKED_DOOR)
    SPAWN = 3  # Player spawn point
    EXIT_SWITCH = 4  # Exit switch (EntityType.EXIT_SWITCH)
    EXIT_DOOR = 5  # Exit door (EntityType.EXIT_DOOR)


class EdgeType(IntEnum):
    """Edge type enum (kept for backward compatibility but not used).

    All edges represent simple adjacency between reachable nodes.
    Edge types are no longer distinguished - graph structure is sufficient.
    """

    ADJACENT = 0  # All edges are this type (not actively used)


@dataclass
class Edge:
    """Edge representation for strategic RL.

    Simplified: All edges represent adjacency between reachable nodes.
    No edge features or types needed - graph structure is sufficient.
    """

    source: Tuple[int, int]
    target: Tuple[int, int]
    metadata: Optional[Dict] = None


@dataclass
class GraphData:
    """Container for graph data with fixed-size arrays for Gym compatibility.

    Simplified: No edge features or edge types - all edges are simple adjacency.
    """

    node_features: np.ndarray  # [N_MAX_NODES, F_node]
    edge_index: np.ndarray  # [2, E_MAX_EDGES]
    node_mask: np.ndarray  # [N_MAX_NODES] - 1 for valid nodes, 0 for padding
    edge_mask: np.ndarray  # [E_MAX_EDGES] - 1 for valid edges, 0 for padding
    node_types: np.ndarray  # [N_MAX_NODES] - NodeType enum values
    num_nodes: int
    num_edges: int
