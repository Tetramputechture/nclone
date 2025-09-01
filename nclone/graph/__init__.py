"""
Graph-based structural representations for N++ environments.

This package provides hierarchical multi-resolution graph processing
for N++ levels, enabling both precise local movement decisions and
strategic global pathfinding with comprehensive bounce block mechanics.

Key Components:
    - HierarchicalGraphBuilder: Consolidated multi-resolution graph builder with bounce block support
    - HierarchicalGraphData: Container for multi-resolution graph data
    - ResolutionLevel: Enumeration of resolution levels

The HierarchicalGraphBuilder now includes all bounce block-aware graph building
functionality in a single, consolidated class.
"""

from .hierarchical_builder import (
    HierarchicalGraphBuilder,
    HierarchicalGraphData,
    ResolutionLevel
)
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
    'GraphData',
    'NodeType',
    'EdgeType',
    'SUB_CELL_SIZE',
    'SUB_GRID_WIDTH',
    'SUB_GRID_HEIGHT',
    'N_MAX_NODES',
    'E_MAX_EDGES'
]