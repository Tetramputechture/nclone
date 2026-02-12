"""
Surface area computation and flood fill reachability analysis.

Provides flood fill from start positions to find reachable nodes,
and cached surface area computation for level reachability metrics.
"""

import logging
from typing import Any, Dict, Tuple, List, Optional
from collections import deque, OrderedDict

from .cost_functions import NODE_WORLD_COORD_OFFSET
from .node_lookup import (
    find_closest_node_to_position,
    _get_subcell_lookup_loader,
    extract_spatial_lookups_from_graph_data,
)

_logger = logging.getLogger(__name__)

# Module-level cache for surface area by level ID
# Uses OrderedDict for LRU eviction to prevent unbounded growth
_surface_area_cache: OrderedDict[str, float] = OrderedDict()
_SURFACE_AREA_CACHE_MAX_SIZE = 1000


def flood_fill_reachable_nodes(
    start_pos: Tuple[int, int],
    adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
    spatial_hash: Optional[any] = None,
    subcell_lookup: Optional[any] = None,
    player_radius: float = 10.0,
) -> set:
    """
    Perform flood fill from start position to find all reachable nodes.

    Handles coordinate space conversion and uses optimal spatial lookups.
    This consolidates the flood fill logic used across the codebase.

    Coordinate Systems:
    - start_pos is in world space (includes 1-tile padding)
    - adjacency keys are in tile data space (excludes 1-tile padding)
    - Conversion: tile_data = world - NODE_WORLD_COORD_OFFSET (24px)

    Args:
        start_pos: Starting position in world space (x, y) pixels
        adjacency: Graph adjacency structure (keys in tile data space, for pathfinding)
        spatial_hash: Optional SpatialHash instance for O(1) lookup
        subcell_lookup: Optional SubcellNodeLookupLoader instance
        player_radius: Player collision radius in pixels (default 10.0)

    Returns:
        Set of reachable node positions (in tile data space)

    Raises:
        RuntimeError: If adjacency is empty or None
    """
    if not adjacency:
        raise RuntimeError(
            "flood_fill_reachable_nodes: adjacency graph is empty or None"
        )

    # Get subcell_lookup if not provided
    if subcell_lookup is None:
        subcell_lookup = _get_subcell_lookup_loader()

    # Find closest node(s) within player radius using optimal lookups
    closest_node = find_closest_node_to_position(
        start_pos,
        adjacency,
        threshold=player_radius,
        spatial_hash=spatial_hash,
        subcell_lookup=subcell_lookup,
    )

    # Fallback: try with larger threshold if nothing found
    if closest_node is None:
        closest_node = find_closest_node_to_position(
            start_pos,
            adjacency,
            threshold=50.0,
            spatial_hash=spatial_hash,
            subcell_lookup=subcell_lookup,
        )

    # Check if we found a valid starting node
    if closest_node is None or closest_node not in adjacency:
        # Return empty set to allow caller to handle gracefully
        return set()

    # Perform BFS flood fill from starting node(s)
    reachable = set()
    queue = deque([closest_node])
    visited = set([closest_node])

    while queue:
        current = queue.popleft()
        reachable.add(current)

        # Get neighbors from adjacency
        neighbors = adjacency.get(current, [])
        for neighbor_info in neighbors:
            # Handle both tuple format (neighbor_pos, cost) and just neighbor_pos
            if isinstance(neighbor_info, tuple) and len(neighbor_info) >= 2:
                neighbor_pos = neighbor_info[0]
            else:
                neighbor_pos = neighbor_info

            if neighbor_pos not in visited:
                visited.add(neighbor_pos)
                queue.append(neighbor_pos)

    return reachable


def compute_reachable_surface_area(
    start_pos: Tuple[int, int],
    adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
    graph_data: Optional[Dict[str, Any]] = None,
) -> float:
    """
    Compute reachable surface area as number of nodes from start position.

    This provides a normalized scale for distance calculations based on the
    actual reachable area of the level. Uses flood fill to count all nodes
    reachable from the starting position.

    Args:
        start_pos: Start position in world space (x, y) pixels
        adjacency: Graph adjacency structure (keys in tile data space)
        graph_data: Optional graph data dict with spatial_hash for optimization

    Returns:
        Total number of reachable nodes (float)

    Raises:
        RuntimeError: If adjacency is empty or no nodes reachable from start
    """
    if not adjacency:
        raise RuntimeError(
            "compute_reachable_surface_area: adjacency graph is empty or None.\n"
            "Surface area calculation requires valid graph data."
        )

    # Extract spatial lookups from graph_data for optimization
    spatial_hash, subcell_lookup = extract_spatial_lookups_from_graph_data(graph_data)

    # Perform flood fill to find all reachable nodes
    reachable_nodes = flood_fill_reachable_nodes(
        start_pos,
        adjacency,
        spatial_hash,
        subcell_lookup,
    )

    if not reachable_nodes:
        sample_keys = list(adjacency.keys())[:10] if adjacency else []
        converted_start = (
            start_pos[0] - NODE_WORLD_COORD_OFFSET,
            start_pos[1] - NODE_WORLD_COORD_OFFSET,
        )  # Convert to tile data space

        raise RuntimeError(
            f"compute_reachable_surface_area: no nodes reachable from start position.\n"
            f"start_pos={start_pos} (world space) = {converted_start} (tile data space)\n"
            f"adjacency_size={len(adjacency)}, sample_nodes={sample_keys}\n"
            f"This indicates either:\n"
            f"  1. Degenerate map (too few traversable tiles)\n"
            f"  2. Start position is isolated from graph\n"
            f"  3. Coordinate space mismatch\n"
            f"CRITICAL: This is likely a map generation bug and should not occur in training."
        )

    return float(len(reachable_nodes))


def get_cached_surface_area(
    level_id: str,
    start_pos: Tuple[int, int],
    adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
    graph_data: Optional[Dict[str, Any]] = None,
    force_recompute: bool = False,
) -> float:
    """
    Get or compute reachable surface area with caching by level ID.

    Caches surface area per unique level configuration. Cache key is level_id,
    which should uniquely identify the level tiles, entities, and start position.
    This avoids expensive flood fill recomputation for the same level.

    Args:
        level_id: Unique identifier for level configuration (should include
                 switch states if they affect reachability)
        start_pos: Start position in world space (x, y) pixels
        adjacency: Graph adjacency structure (keys in tile data space)
        graph_data: Optional graph data dict with spatial_hash for optimization
        force_recompute: If True, bypass cache and recompute (default False)

    Returns:
        Reachable surface area (float) - number of nodes reachable from start

    Raises:
        RuntimeError: If surface area computation fails
    """
    # Check cache unless force recompute requested
    if not force_recompute and level_id in _surface_area_cache:
        cached_value = _surface_area_cache[level_id]
        # Move to end to maintain LRU ordering
        _surface_area_cache.move_to_end(level_id)
        _logger.debug(
            f"[SURFACE_AREA_CACHE] HIT - cache_key={level_id[:50]}..., "
            f"surface_area={cached_value:.1f}"
        )
        return cached_value

    # Compute surface area
    surface_area = compute_reachable_surface_area(start_pos, adjacency, graph_data)

    # Cache the result with LRU eviction
    _surface_area_cache[level_id] = surface_area
    _surface_area_cache.move_to_end(level_id)

    # Evict oldest entry if cache exceeds max size
    if len(_surface_area_cache) > _SURFACE_AREA_CACHE_MAX_SIZE:
        _surface_area_cache.popitem(last=False)

    return surface_area


def clear_surface_area_cache(level_id: Optional[str] = None) -> None:
    """
    Clear surface area cache.

    This should be called when level data changes or when memory cleanup is needed.

    Args:
        level_id: If provided, clear only this specific level from cache.
                 If None, clear the entire cache.
    """
    if level_id is None:
        _surface_area_cache.clear()
    else:
        _surface_area_cache.pop(level_id, None)
