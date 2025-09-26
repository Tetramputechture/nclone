"""
Graph-based structural representations for N++ environments.
"""

from .hierarchical_builder import (
    HierarchicalGraphBuilder,
    HierarchicalGraphData,
    ResolutionLevel,
)
from .feature_extraction import FeatureExtractor
from .edge_building import EdgeBuilder
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
    "HierarchicalGraphBuilder",
    "HierarchicalGraphData",
    "ResolutionLevel",
    "FeatureExtractor",
    "EdgeBuilder",
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
