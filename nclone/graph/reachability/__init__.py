"""
Reachability analysis package for hierarchical graph optimization.

The components work together to determine which areas of a level are
accessible to the player from their starting position.

New path-aware reward shaping components:
- tile_connectivity_precomputer: Offline tile traversability precomputation
- tile_connectivity_loader: Runtime O(1) lookups
- entity_mask: Dynamic entity state masking (doors, mines)
- graph_builder: Combined tile + entity graph builder
- path_distance_calculator: BFS/A* pathfinding for accurate distances
"""

from .reachability_system import ReachabilitySystem
from .reachability_types import (
    ReachabilityApproximation,
)

# Path-aware components (optional imports - may not have precomputed data yet)
try:
    from .tile_connectivity_loader import TileConnectivityLoader
    from .tile_connectivity_precomputer import TileConnectivityPrecomputer
    from .entity_mask import EntityMask
    from .graph_builder import GraphBuilder
    from .path_distance_calculator import (
        PathDistanceCalculator,
        CachedPathDistanceCalculator,
    )
    from .pathfinding_utils import (
        find_closest_node_to_position,
        bfs_distance_from_start,
        find_shortest_path_with_parents,
    )
    from .path_visualization_cache import PathVisualizationCache
    from .subcell_node_lookup import (
        SubcellNodeLookupPrecomputer,
        SubcellNodeLookupLoader,
    )

    __all__ = [
        "ReachabilitySystem",
        "ReachabilityApproximation",
        "TileConnectivityLoader",
        "TileConnectivityPrecomputer",
        "EntityMask",
        "GraphBuilder",
        "PathDistanceCalculator",
        "CachedPathDistanceCalculator",
        "find_closest_node_to_position",
        "bfs_distance_from_start",
        "find_shortest_path_with_parents",
        "PathVisualizationCache",
        "SubcellNodeLookupPrecomputer",
        "SubcellNodeLookupLoader",
    ]
except ImportError:
    # Precomputed data not available yet
    __all__ = [
        "ReachabilitySystem",
        "ReachabilityApproximation",
    ]
