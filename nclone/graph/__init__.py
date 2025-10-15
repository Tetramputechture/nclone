"""
Graph-based structural representations for N++ environments.
"""

from .graph_builder import (
    GraphBuilder,
    HierarchicalGraphData,
)

# Backward compatibility alias
HierarchicalGraphBuilder = GraphBuilder
# ResolutionLevel no longer used (single resolution)
ResolutionLevel = None
# Note: FeatureExtractor moved to npp-rl project
# from .feature_extraction import FeatureExtractor
from .edge_building import EdgeBuilder
from .level_data import LevelData, ensure_level_data, create_level_data_dict
from .common import (
    GraphData,
    StaticGraphStructure,
    NodeType,
    EdgeType,
    N_MAX_NODES,
    E_MAX_EDGES,
    NODE_FEATURE_DIM,
    EDGE_FEATURE_DIM,
)

__all__ = [
    "GraphBuilder",
    "HierarchicalGraphBuilder",  # Backward compatibility
    "HierarchicalGraphData",
    "ResolutionLevel",  # Backward compatibility (None)
    "EdgeBuilder",
    "LevelData",
    "ensure_level_data",
    "create_level_data_dict",
    "GraphData",
    "StaticGraphStructure",
    "NodeType",
    "EdgeType",
    "N_MAX_NODES",
    "E_MAX_EDGES",
    "NODE_FEATURE_DIM",
    "EDGE_FEATURE_DIM",
]
