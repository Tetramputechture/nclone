"""
Graph-based structural representations for N++ environments.

This package provides hierarchical multi-resolution graph processing
for N++ levels, enabling both precise local movement decisions and
strategic global pathfinding.

Key Components:
    - HierarchicalGraphBuilder: Multi-resolution graph builder (6px, 24px, 96px)
    - HierarchicalGraphData: Container for multi-resolution graph data
    - ResolutionLevel: Enumeration of resolution levels

Note: The standard GraphBuilder has been moved to archive/ in favor
of the more advanced hierarchical approach.
"""

from .hierarchical_builder import (
    HierarchicalGraphBuilder,
    HierarchicalGraphData,
    ResolutionLevel
)
from .base_builder import GraphBuilder
from .common import (
    GraphData,
    NodeType,
    EdgeType,
    SUB_CELL_SIZE,
    SUB_GRID_WIDTH,
    SUB_GRID_HEIGHT,
    N_MAX_NODES,
    E_MAX_EDGES
)

__all__ = [
    'HierarchicalGraphBuilder',
    'HierarchicalGraphData', 
    'ResolutionLevel',
    'GraphBuilder',
    'GraphData',
    'NodeType',
    'EdgeType',
    'SUB_CELL_SIZE',
    'SUB_GRID_WIDTH',
    'SUB_GRID_HEIGHT',
    'N_MAX_NODES',
    'E_MAX_EDGES'
]