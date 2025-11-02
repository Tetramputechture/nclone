"""
Utility functions for pathfinding operations.

Shared utilities for node finding, BFS operations, and path reconstruction
used by both performance-critical pathfinding and visualization systems.
"""

import logging
from typing import Dict, Tuple, List, Optional, Any
from collections import deque

# Node coordinate offset for world coordinate conversion
# Nodes are in tile data space, entity positions in world space differ by 24px
NODE_WORLD_COORD_OFFSET = 24

# Logger for pathfinding utilities
_logger = logging.getLogger(__name__)

# Cache for subcell lookup loader singleton (lazy-loaded)
_subcell_lookup_loader_cache = None


def _get_subcell_lookup_loader():
    """
    Get or create SubcellNodeLookupLoader singleton instance.

    Returns:
        SubcellNodeLookupLoader instance if available, None otherwise
    """
    global _subcell_lookup_loader_cache

    # Return cached instance if already loaded
    if _subcell_lookup_loader_cache is not None:
        return _subcell_lookup_loader_cache

    # Try to load the singleton
    try:
        from .subcell_node_lookup import SubcellNodeLookupLoader

        loader = SubcellNodeLookupLoader()
        # Verify it's actually loaded
        if loader._lookup_table is not None:
            _subcell_lookup_loader_cache = loader
            _logger.info(
                f"Subcell lookup loader initialized successfully: "
                f"shape={loader._lookup_table.shape}, "
                f"size={loader._lookup_table.nbytes / 1024:.2f} KB"
            )
            return loader
        else:
            _logger.warning("Subcell lookup loader created but table is None")
            return None
    except FileNotFoundError as e:
        _logger.debug(f"Subcell lookup file not found: {e}")
        return None
    except RuntimeError as e:
        _logger.debug(f"Subcell lookup runtime error: {e}")
        return None
    except Exception as e:
        _logger.warning(f"Subcell lookup unexpected error: {type(e).__name__}: {e}")
        return None


def extract_spatial_lookups_from_graph_data(
    graph_data: Optional[Dict[str, Any]],
) -> Tuple[Optional[any], Optional[any]]:
    """
    Extract spatial_hash and subcell_lookup from graph_data.

    Helper function to get both spatial lookup mechanisms from graph_data.
    This ensures consistent access across all call sites.

    Args:
        graph_data: Graph data dict (may contain spatial_hash)

    Returns:
        Tuple of (spatial_hash, subcell_lookup):
        - spatial_hash: SpatialHash instance from graph_data, or None
        - subcell_lookup: SubcellNodeLookupLoader singleton, or None if unavailable
    """
    spatial_hash = None
    if graph_data is not None:
        spatial_hash = graph_data.get("spatial_hash")
        _logger.debug(
            f"extract_spatial_lookups: graph_data provided, spatial_hash={spatial_hash is not None}"
        )
    else:
        _logger.debug("extract_spatial_lookups: graph_data is None")

    # Always try to load subcell lookup (singleton, loads once)
    subcell_lookup = _get_subcell_lookup_loader()

    if subcell_lookup is not None:
        _logger.debug("extract_spatial_lookups: subcell_lookup available")
    else:
        _logger.debug("extract_spatial_lookups: subcell_lookup not available")

    return spatial_hash, subcell_lookup


def find_closest_node_to_position(
    world_pos: Tuple[int, int],
    adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
    threshold: float = 50.0,
    spatial_hash: Optional[any] = None,
    subcell_lookup: Optional[any] = None,
) -> Optional[Tuple[int, int]]:
    """
    Find the closest node to a world position with optional spatial indexing.

    Priority order:
    1. Precomputed subcell lookup (fastest, O(1) direct array access)
    2. Spatial hash (O(1) grid-based lookup)
    3. Linear search (fallback, O(N))

    Coordinate Systems:
    - Node positions (adjacency keys): Tile data space (excludes 1-tile padding)
    - World positions (entities, ninja): Full map space (includes 1-tile padding)
    - Offset: Add +24 to node coords to convert to world coords for comparison

    Args:
        world_pos: World position (x, y) in pixels (full map space)
        adjacency: Graph adjacency structure (keys in tile data space)
        threshold: Maximum distance threshold (default 50 pixels)
        spatial_hash: Optional SpatialHash instance for O(1) lookup
        subcell_lookup: Optional SubcellNodeLookupLoader instance for fastest lookup

    Returns:
        Closest node position (in tile data space), or None if no node within threshold
    """
    if not adjacency:
        return None

    world_x, world_y = world_pos

    # Convert query position from world space to tile data space
    query_x = world_x - NODE_WORLD_COORD_OFFSET
    query_y = world_y - NODE_WORLD_COORD_OFFSET

    # Fastest path: Use precomputed subcell lookup if available (O(1) direct access)
    if subcell_lookup is not None:
        try:
            # Use threshold as max_radius for entity radius handling (4-12px entities)
            closest_node = subcell_lookup.find_closest_node_position(
                query_x, query_y, adjacency, max_radius=threshold
            )
            return closest_node
        except IndexError:
            # Out of bounds - fall through to spatial_hash or linear search
            _logger.warning(
                f"Subcell lookup out of bounds for query ({query_x}, {query_y}), "
                f"falling back to spatial_hash or linear search"
            )
        except Exception as e:
            # Other errors (e.g., lookup table not loaded)
            _logger.warning(
                f"Subcell lookup failed: {e}, falling back to spatial_hash or linear search"
            )

    # Fast path: Use spatial hash if available (O(1))
    if spatial_hash is not None:
        # Spatial hash operates in tile data space
        closest_node = spatial_hash.find_closest(query_x, query_y, threshold)
        return closest_node

    # Fallback: Linear search for backward compatibility (O(N))
    _logger.warning(
        f"No spatial lookup available for query ({query_x}, {query_y}). "
        f"subcell_lookup={subcell_lookup is not None} (type={type(subcell_lookup).__name__ if subcell_lookup is not None else 'None'}), "
        f"spatial_hash={spatial_hash is not None} (type={type(spatial_hash).__name__ if spatial_hash is not None else 'None'}), "
        f"adjacency size={len(adjacency)}. "
        f"Using O(N) linear search - this is slow for large graphs."
    )
    closest_node = None
    min_dist = float("inf")

    for pos in adjacency.keys():
        x, y = pos
        # Convert node coords to world coords (+24) for comparison with entity position
        dist = (
            (x + NODE_WORLD_COORD_OFFSET - world_x) ** 2
            + (y + NODE_WORLD_COORD_OFFSET - world_y) ** 2
        ) ** 0.5
        if dist < min_dist:
            min_dist = dist
            closest_node = pos

    # Only return if within threshold
    if closest_node is not None and min_dist < threshold:
        return closest_node

    return None


def bfs_distance_from_start(
    start_node: Tuple[int, int],
    target_node: Optional[Tuple[int, int]],
    adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
    max_distance: Optional[float] = None,
) -> Tuple[Dict[Tuple[int, int], float], Optional[float]]:
    """
    Calculate distances from start node using BFS.

    Matches the exact logic used in debug_overlay_renderer.py visualization.
    Returns distances dict and optionally the distance to target_node if specified.

    Args:
        start_node: Starting node position
        target_node: Optional target node to find distance to (early termination)
        adjacency: Graph adjacency structure
        max_distance: Optional maximum distance to compute (for early termination)

    Returns:
        Tuple of (distances_dict, target_distance):
        - distances_dict: Map of node -> distance from start
        - target_distance: Distance to target_node if found, None otherwise
    """
    distances = {start_node: 0.0}
    queue = deque([start_node])

    while queue:
        current = queue.popleft()
        current_dist = distances[current]

        # Early termination if we found target and it's requested
        if target_node is not None and current == target_node:
            return distances, current_dist

        # Early termination if we've exceeded max distance
        if max_distance is not None and current_dist > max_distance:
            continue

        neighbors = adjacency.get(current, [])
        for neighbor_info in neighbors:
            neighbor_pos, cost = neighbor_info
            if neighbor_pos not in distances:
                distances[neighbor_pos] = current_dist + cost
                queue.append(neighbor_pos)

    # Return distances dict and target distance (None if target not found)
    target_distance = distances.get(target_node) if target_node else None
    return distances, target_distance


def find_shortest_path_with_parents(
    start_node: Tuple[int, int],
    end_node: Tuple[int, int],
    adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
) -> Tuple[Optional[List[Tuple[int, int]]], float]:
    """
    Find shortest path from start to end node using BFS.

    Returns both the path (list of nodes) and the distance.
    Matches the logic used in debug_overlay_renderer.py visualization.

    Args:
        start_node: Starting node position
        end_node: Target node position
        adjacency: Graph adjacency structure

    Returns:
        Tuple of (path, distance):
        - path: List of node positions from start to end, or None if unreachable
        - distance: Total path distance, or float('inf') if unreachable
    """
    if start_node == end_node:
        return [start_node], 0.0

    distances = {start_node: 0.0}
    parents = {start_node: None}
    queue = deque([start_node])

    while queue:
        current = queue.popleft()

        if current == end_node:
            # Reconstruct path
            path = []
            node = end_node
            while node is not None:
                path.append(node)
                node = parents.get(node)
            path.reverse()
            return path, distances[end_node]

        current_dist = distances[current]
        neighbors = adjacency.get(current, [])
        for neighbor_info in neighbors:
            neighbor_pos, cost = neighbor_info
            if neighbor_pos not in distances:
                distances[neighbor_pos] = current_dist + cost
                parents[neighbor_pos] = current
                queue.append(neighbor_pos)

    return None, float("inf")
