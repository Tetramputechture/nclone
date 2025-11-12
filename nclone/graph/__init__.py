"""
Graph-based structural representations for N++ environments.
"""

# Note: FeatureExtractor moved to npp-rl project
# from .feature_extraction import FeatureExtractor
from .level_data import LevelData, ensure_level_data, create_level_data_dict
from .common import (
    GraphData,
    NodeType,
    EdgeType,
    SUB_CELL_SIZE,
    SUB_GRID_WIDTH,
    SUB_GRID_HEIGHT,
    N_MAX_NODES,
    E_MAX_EDGES,
)

__all__ = [
    "LevelData",
    "ensure_level_data",
    "create_level_data_dict",
    "GraphData",
    "NodeType",
    "EdgeType",
    "SUB_CELL_SIZE",
    "SUB_GRID_WIDTH",
    "SUB_GRID_HEIGHT",
    "N_MAX_NODES",
    "E_MAX_EDGES",
]
