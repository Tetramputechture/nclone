"""
Pathfinding utilities - backward-compatible re-export shim.

Re-exports from cost_functions, node_lookup, pathfinding_algorithms, surface_area.
"""

from .cost_functions import (
    NODE_WORLD_COORD_OFFSET,
    SUB_NODE_SIZE,
    _is_node_grounded_util,
)
from .node_lookup import (
    _get_subcell_lookup_loader,
    extract_spatial_lookups_from_graph_data,
    find_closest_node_to_position,
    _is_node_grounded,
    find_goal_node_closest_to_start,
    find_ninja_node,
)
from .pathfinding_algorithms import (
    bfs_distance_from_start,
    calculate_geometric_path_distance,
    find_shortest_path,
    find_nearest_stable_position,
    find_shortest_path_with_fallback,
)
from .surface_area import (
    flood_fill_reachable_nodes,
    compute_reachable_surface_area,
    get_cached_surface_area,
    clear_surface_area_cache,
)

__all__ = [
    "NODE_WORLD_COORD_OFFSET",
    "SUB_NODE_SIZE",
    "_is_node_grounded_util",
    "_get_subcell_lookup_loader",
    "extract_spatial_lookups_from_graph_data",
    "find_closest_node_to_position",
    "_is_node_grounded",
    "find_goal_node_closest_to_start",
    "find_ninja_node",
    "bfs_distance_from_start",
    "calculate_geometric_path_distance",
    "find_shortest_path",
    "find_nearest_stable_position",
    "find_shortest_path_with_fallback",
    "flood_fill_reachable_nodes",
    "compute_reachable_surface_area",
    "get_cached_surface_area",
    "clear_surface_area_cache",
]
