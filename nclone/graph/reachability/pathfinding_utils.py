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

    Auto-generates the lookup file if it doesn't exist.

    Returns:
        SubcellNodeLookupLoader instance (always returns a valid instance)
    """
    global _subcell_lookup_loader_cache

    # Return cached instance if already loaded
    if _subcell_lookup_loader_cache is not None:
        return _subcell_lookup_loader_cache

    # Try to load the singleton
    try:
        from .subcell_node_lookup import (
            SubcellNodeLookupLoader,
            SubcellNodeLookupPrecomputer,
        )

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
            _logger.error("Subcell lookup loader created but table is None")
            raise RuntimeError("Subcell lookup table is None")
    except FileNotFoundError as e:
        # Auto-generate the lookup file if it doesn't exist
        _logger.info(f"Subcell lookup file not found, auto-generating: {e}")
        try:
            import os
            from .subcell_node_lookup import SubcellNodeLookupPrecomputer

            # Get the data directory path (match path from subcell_node_lookup.py)
            data_path = os.path.join(
                os.path.dirname(__file__), "../../data/subcell_node_lookup.pkl.gz"
            )
            data_dir = os.path.dirname(data_path)

            # Ensure data directory exists
            os.makedirs(data_dir, exist_ok=True)

            # Generate the lookup table
            _logger.info("Precomputing subcell node lookup table...")
            precomputer = SubcellNodeLookupPrecomputer()
            lookup = precomputer.precompute_all(verbose=False)
            precomputer.save_to_file(lookup, data_path, verbose=False)
            _logger.info(f"Subcell lookup table generated successfully at {data_path}")

            # Now try loading again
            loader = SubcellNodeLookupLoader()
            if loader._lookup_table is not None:
                _subcell_lookup_loader_cache = loader
                _logger.info(
                    f"Subcell lookup loader initialized successfully: "
                    f"shape={loader._lookup_table.shape}, "
                    f"size={loader._lookup_table.nbytes / 1024:.2f} KB"
                )
                return loader
            else:
                raise RuntimeError("Failed to load generated lookup table")
        except Exception as gen_error:
            _logger(
                f"Failed to auto-generate subcell lookup file: {gen_error}. "
                f"Falling back to loader error."
            )
            raise
    except RuntimeError as e:
        _logger(f"Subcell lookup runtime error: {e}")
        raise
    except Exception as e:
        _logger(f"Subcell lookup unexpected error: {type(e).__name__}: {e}")
        raise


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

    # Always try to load subcell lookup (singleton, loads once, auto-generates if needed)
    try:
        subcell_lookup = _get_subcell_lookup_loader()
        _logger.debug("extract_spatial_lookups: subcell_lookup available")
    except Exception as e:
        _logger(f"Failed to load subcell lookup: {e}")
        subcell_lookup = None

    return spatial_hash, subcell_lookup


def find_closest_node_to_position(
    world_pos: Tuple[int, int],
    adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
    threshold: Optional[float] = None,
    entity_radius: float = 0.0,
    ninja_radius: float = 10.0,
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
        threshold: Maximum distance threshold (if None, calculated as ninja_radius + entity_radius)
        entity_radius: Collision radius of the entity at world_pos (default 0.0)
        ninja_radius: Collision radius of the ninja (default 10.0)
        spatial_hash: Optional SpatialHash instance for O(1) lookup
        subcell_lookup: Optional SubcellNodeLookupLoader instance for fastest lookup

    Returns:
        Closest node position (in tile data space), or None if no node within threshold
    """
    if not adjacency:
        return None

    # Calculate threshold from radii if not provided
    if threshold is None:
        threshold = ninja_radius + entity_radius

    world_x, world_y = world_pos

    # Convert query position from world space to tile data space
    query_x = world_x - NODE_WORLD_COORD_OFFSET
    query_y = world_y - NODE_WORLD_COORD_OFFSET

    # Get subcell_lookup if not provided (auto-loads and auto-generates if needed)
    if subcell_lookup is None:
        try:
            subcell_lookup = _get_subcell_lookup_loader()
        except Exception as e:
            _logger(f"Failed to load subcell lookup: {e}")
            subcell_lookup = None

    # Fastest path: Use precomputed subcell lookup if available (O(1) direct access)
    if subcell_lookup is not None:
        try:
            # Use threshold as max_radius for entity radius handling (4-12px entities)
            closest_node = subcell_lookup.find_closest_node_position(
                query_x, query_y, adjacency, max_radius=threshold
            )
            if closest_node is not None:
                return closest_node
        except Exception as e:
            # Other errors (e.g., lookup table not loaded)
            _logger(
                f"Subcell lookup failed: {e}. Falling back to spatial hash or linear search."
            )

    # Fallback to spatial hash if available
    if spatial_hash is not None:
        try:
            # Spatial hash lookup
            candidates = spatial_hash.query(query_x, query_y, radius=threshold)
            if candidates:
                # Find closest candidate
                min_dist = float("inf")
                closest = None
                for candidate in candidates:
                    if candidate in adjacency:
                        dist_sq = (candidate[0] - query_x) ** 2 + (
                            candidate[1] - query_y
                        ) ** 2
                        if dist_sq < min_dist:
                            min_dist = dist_sq
                            closest = candidate
                if closest is not None and min_dist <= threshold * threshold:
                    return closest
        except Exception as e:
            _logger.debug(f"Spatial hash lookup failed: {e}")

    # Final fallback: Linear search (O(N))
    min_dist = float("inf")
    closest = None
    for node_pos in adjacency.keys():
        nx, ny = node_pos
        dist_sq = (nx - query_x) ** 2 + (ny - query_y) ** 2
        if dist_sq < min_dist:
            min_dist = dist_sq
            closest = node_pos

    if closest is not None and min_dist <= threshold * threshold:
        return closest

    return None


def find_start_node_for_player(
    player_pos: Tuple[int, int],
    adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
    player_radius: float = 10.0,
    spatial_hash: Optional[any] = None,
    subcell_lookup: Optional[any] = None,
    goal_pos: Optional[Tuple[int, int]] = None,
) -> Optional[Tuple[int, int]]:
    """
    Find best start node for player position, preferring overlapped nodes.

    Priority:
    1. Nodes whose centers are within player radius (overlapped)
       - If multiple overlapped and goal_pos provided: select closest to goal
       - Otherwise: select topmost (lowest y value)
    2. If no overlapped nodes, closest node within threshold

    Args:
        player_pos: Player world position (x, y) in pixels
        adjacency: Graph adjacency structure
        player_radius: Player collision radius (default 10.0)
        spatial_hash: Optional SpatialHash for fast lookup
        subcell_lookup: Optional SubcellNodeLookupLoader for fastest lookup
        goal_pos: Optional goal position to prefer nodes in goal direction

    Returns:
        Best start node, or None if none found
    """
    if not adjacency:
        return None

    world_x, world_y = player_pos
    query_x = world_x - NODE_WORLD_COORD_OFFSET
    query_y = world_y - NODE_WORLD_COORD_OFFSET

    # Get candidates within larger search radius
    search_radius = player_radius * 2.0  # Search wider initially

    overlapped_nodes = []
    nearby_nodes = []

    # Get subcell_lookup if not provided
    if subcell_lookup is None:
        try:
            subcell_lookup = _get_subcell_lookup_loader()
        except Exception:
            subcell_lookup = None

    # Use spatial indexing to get candidates
    candidates = []
    if subcell_lookup is not None:
        try:
            # Use subcell lookup to find nodes within search radius
            closest = subcell_lookup.find_closest_node_position(
                query_x, query_y, adjacency, max_radius=search_radius
            )
            if closest is not None:
                # Get all nearby candidates using spatial hash or linear search
                if spatial_hash is not None:
                    candidates = spatial_hash.query(
                        query_x, query_y, radius=search_radius
                    )
                    candidates = [c for c in candidates if c in adjacency]
                else:
                    # Fallback: check all nodes
                    for node in adjacency.keys():
                        nx, ny = node
                        dist_sq = (nx - query_x) ** 2 + (ny - query_y) ** 2
                        if dist_sq <= search_radius * search_radius:
                            candidates.append(node)
        except Exception:
            candidates = []

    if not candidates:
        if spatial_hash is not None:
            try:
                candidates = spatial_hash.query(query_x, query_y, radius=search_radius)
                candidates = [c for c in candidates if c in adjacency]
            except Exception:
                candidates = []

    # Final fallback: all nodes
    if not candidates:
        candidates = list(adjacency.keys())

    # Categorize candidates
    for node in candidates:
        nx, ny = node
        dist_sq = (nx - query_x) ** 2 + (ny - query_y) ** 2
        dist = dist_sq**0.5

        if dist <= player_radius:
            # Node center is within player radius (overlapped)
            overlapped_nodes.append((node, dist))
        elif dist <= search_radius:
            nearby_nodes.append((node, dist))

    # Prefer overlapped nodes
    if overlapped_nodes:
        if len(overlapped_nodes) > 1:
            if goal_pos is not None:
                # Prefer node closest to goal (most relevant direction)
                goal_x = goal_pos[0] - NODE_WORLD_COORD_OFFSET
                goal_y = goal_pos[1] - NODE_WORLD_COORD_OFFSET
                overlapped_nodes.sort(
                    key=lambda x: (x[0][0] - goal_x) ** 2 + (x[0][1] - goal_y) ** 2
                )
                return overlapped_nodes[0][0]
            else:
                # Prefer topmost node (lowest y value) for consistency
                overlapped_nodes.sort(key=lambda x: x[0][1])
                return overlapped_nodes[0][0]
        else:
            return overlapped_nodes[0][0]

    # Fallback to closest nearby node
    if nearby_nodes:
        nearby_nodes.sort(key=lambda x: x[1])
        return nearby_nodes[0][0]

    return None


def find_goal_node_closest_to_start(
    goal_pos: Tuple[int, int],
    start_node: Tuple[int, int],
    adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
    entity_radius: float = 0.0,
    ninja_radius: float = 10.0,
    spatial_hash: Optional[any] = None,
    subcell_lookup: Optional[any] = None,
) -> Optional[Tuple[int, int]]:
    """
    Find goal node closest to start among nodes overlapped by entity radius.

    Strategy:
    1. Find all nodes within entity_radius of goal position (overlapped nodes)
    2. Among those, select the one closest to start_node
    3. If no overlapped nodes, expand search to ninja_radius + entity_radius

    Args:
        goal_pos: Goal world position (x, y) in pixels
        start_node: Start node position (for proximity check)
        adjacency: Graph adjacency structure
        entity_radius: Entity collision radius
        ninja_radius: Ninja collision radius
        spatial_hash: Optional SpatialHash
        subcell_lookup: Optional SubcellNodeLookupLoader

    Returns:
        Best goal node, or None if none found
    """
    if not adjacency or start_node is None:
        return None

    world_x, world_y = goal_pos
    query_x = world_x - NODE_WORLD_COORD_OFFSET
    query_y = world_y - NODE_WORLD_COORD_OFFSET

    # Get subcell_lookup if not provided
    if subcell_lookup is None:
        try:
            subcell_lookup = _get_subcell_lookup_loader()
        except Exception:
            subcell_lookup = None

    # First try to find nodes overlapped by entity radius
    overlapped_nodes = []
    nearby_nodes = []

    # Search radius for candidates
    search_radius = ninja_radius + entity_radius

    # Get candidates within search radius
    candidates = []
    if spatial_hash is not None:
        try:
            candidates = spatial_hash.query(query_x, query_y, radius=search_radius)
            candidates = [c for c in candidates if c in adjacency]
        except Exception:
            pass

    # Fallback: linear search if no spatial hash
    if not candidates:
        for node in adjacency.keys():
            nx, ny = node
            dist_sq = (nx - query_x) ** 2 + (ny - query_y) ** 2
            if dist_sq <= search_radius * search_radius:
                candidates.append(node)

    if not candidates:
        return None

    # Categorize candidates by distance to goal
    for candidate in candidates:
        cx, cy = candidate
        dist_to_goal = ((cx - query_x) ** 2 + (cy - query_y) ** 2) ** 0.5

        if dist_to_goal <= entity_radius:
            # Node overlapped by entity radius
            overlapped_nodes.append(candidate)
        else:
            nearby_nodes.append(candidate)

    # Prefer nodes overlapped by entity radius
    search_set = overlapped_nodes if overlapped_nodes else nearby_nodes

    if not search_set:
        return None

    # Among candidates, find closest to start_node
    sx, sy = start_node
    min_dist_to_start = float("inf")
    best_node = None

    for candidate in search_set:
        cx, cy = candidate
        dist_sq = (cx - sx) ** 2 + (cy - sy) ** 2
        if dist_sq < min_dist_to_start:
            min_dist_to_start = dist_sq
            best_node = candidate

    return best_node


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


def find_shortest_path(
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
