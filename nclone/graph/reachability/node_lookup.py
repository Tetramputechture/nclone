"""
Node lookup and spatial indexing utilities for pathfinding.

Provides functions for finding graph nodes from world positions,
spatial hash indexing, and subcell lookup optimization.
"""

import logging
from typing import Dict, Tuple, List, Optional, Any

from .cost_functions import NODE_WORLD_COORD_OFFSET, _is_node_grounded_util

_logger = logging.getLogger(__name__)
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
            _logger.error(
                f"Failed to auto-generate subcell lookup file: {gen_error}. "
                f"Falling back to loader error."
            )
            raise
    except RuntimeError as e:
        _logger.error(f"Subcell lookup runtime error: {e}")
        raise
    except Exception as e:
        _logger.error(f"Subcell lookup unexpected error: {type(e).__name__}: {e}")
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
        _logger.warning(f"Failed to load subcell lookup: {e}")
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
        adjacency: Graph adjacency structure (keys in tile data space, for candidate filtering)
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
            _logger.warning(f"Failed to load subcell lookup: {e}")
            subcell_lookup = None

    # Fastest path: Use precomputed subcell lookup if available (O(1) direct access)
    if subcell_lookup is not None:
        try:
            # Use threshold as max_radius for entity radius handling (4-12px entities)
            closest_node = subcell_lookup.find_closest_node_position(
                query_x,
                query_y,
                adjacency,
                max_radius=threshold,
            )
            if closest_node is not None:
                return closest_node
        except Exception as e:
            # Other errors (e.g., lookup table not loaded)
            _logger.debug(
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


def _is_node_grounded(
    node_pos: Tuple[int, int],
    physics_cache: Optional[Dict[Tuple[int, int], Dict[str, bool]]] = None,
    base_adjacency: Optional[
        Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]]
    ] = None,
) -> bool:
    """
    Check if a node is grounded (has solid/blocked surface directly below).

    Uses physics_cache for O(1) lookup if available, otherwise falls back to
    base_adjacency traversal. IMPORTANT: Uses base_adjacency (tile geometry only),
    NOT entity-masked adjacency, to ensure mines/doors don't affect grounding status.

    A node is grounded if there's a solid surface preventing downward movement.
    In screen coordinates: y=0 is top, y increases going DOWN.

    Args:
        node_pos: Node position (x, y) in pixels
        physics_cache: Optional pre-computed physics properties for O(1) lookup
        base_adjacency: Optional base adjacency (pre-entity-mask) for grounding checks

    Returns:
        True if node is grounded (on a surface), False otherwise (mid-air)

    Raises:
        ValueError: If neither physics_cache nor base_adjacency is provided
    """
    # Prefer physics_cache for O(1) lookup (consistent with pathfinding)
    if physics_cache is not None and node_pos in physics_cache:
        return physics_cache[node_pos]["grounded"]

    # Fallback to base_adjacency traversal
    if base_adjacency is not None:
        return _is_node_grounded_util(node_pos, base_adjacency)

    # Error: neither source provided
    raise ValueError(
        "_is_node_grounded requires either physics_cache or base_adjacency. "
        "This ensures grounding is determined by tile geometry only, not mines/doors."
    )


def find_goal_node_closest_to_start(
    goal_pos: Tuple[int, int],
    start_node: Tuple[int, int],
    adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
    entity_radius: float = 0.0,
    ninja_radius: float = 10.0,
    spatial_hash: Optional[any] = None,
    subcell_lookup: Optional[any] = None,
    search_radius_override: Optional[float] = None,
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
        search_radius_override: Optional override for search radius (default: ninja_radius + entity_radius)

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

    # Search radius for candidates (use override if provided)
    search_radius = (
        search_radius_override
        if search_radius_override is not None
        else (ninja_radius + entity_radius)
    )

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


def find_ninja_node(
    ninja_pos: Tuple[int, int],
    adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
    spatial_hash: Optional[any] = None,
    subcell_lookup: Optional[any] = None,
    ninja_radius: float = 10.0,
    goal_node: Optional[Tuple[int, int]] = None,
    level_cache: Optional[any] = None,
    goal_id: Optional[str] = None,
    search_radius_override: Optional[float] = None,
) -> Optional[Tuple[int, int]]:
    """
    Find the node representing the ninja's current position.

    This is the canonical function for finding the start node for the ninja,
    used by both pathfinding and debug visualization to ensure consistency.

    Strategy:
    1. Find all nodes within ninja_radius (10px by default - player collision radius)
    2. Among overlapping nodes:
       - If level_cache provided: use cached PATH distance to goal (correct behavior)
       - Else if goal_node provided: use Euclidean distance (legacy, may cause issues)
       - Otherwise: return the closest one to ninja center
    3. If no nodes overlap, fall back to visual matching (within 5px for blue highlight)

    Args:
        ninja_pos: Ninja world position (x, y) in pixels
        adjacency: Graph adjacency structure
        spatial_hash: Optional SpatialHash for fast lookup
        subcell_lookup: Optional SubcellNodeLookupLoader for fastest lookup
        ninja_radius: Ninja collision radius in pixels (default 10.0)
        goal_node: Optional goal node for path-aware start node selection
        level_cache: Optional LevelBasedPathDistanceCache for correct path distance selection
        goal_id: Optional goal identifier for level_cache lookup (e.g., "switch" or "exit")
        search_radius_override: Optional override for search radius (default: ninja_radius)

    Returns:
        Ninja node position, or None if no suitable node found
    """
    if not ninja_pos or not adjacency:
        return None

    # Convert ninja position to tile data space for comparison with nodes
    ninja_x = ninja_pos[0] - NODE_WORLD_COORD_OFFSET
    ninja_y = ninja_pos[1] - NODE_WORLD_COORD_OFFSET

    # Use search_radius_override if provided, otherwise use ninja_radius
    search_radius = (
        search_radius_override if search_radius_override is not None else ninja_radius
    )

    # OPTIMIZATION: Use spatial indexing if available (avoid O(N) linear search)
    overlapping_nodes = []

    if spatial_hash is not None:
        # Fast O(1) spatial hash lookup
        try:
            candidates = spatial_hash.query(ninja_x, ninja_y, radius=search_radius)
            for pos in candidates:
                if pos in adjacency:
                    x, y = pos
                    dist_sq = (x - ninja_x) ** 2 + (y - ninja_y) ** 2
                    if dist_sq <= search_radius * search_radius:
                        overlapping_nodes.append((pos, dist_sq))
        except Exception:
            # Fallback to linear search if spatial hash fails
            pass

    # Fallback: Linear search if no spatial hash or it failed
    if not overlapping_nodes:
        for pos in adjacency.keys():
            x, y = pos
            dist_sq = (x - ninja_x) ** 2 + (y - ninja_y) ** 2
            if dist_sq <= search_radius * search_radius:
                overlapping_nodes.append((pos, dist_sq))

    # If we found overlapping nodes, select the best one
    if overlapping_nodes:
        # If goal provided and multiple candidates, pick node closest to goal
        if goal_node is not None and len(overlapping_nodes) > 1:
            best_node = None
            best_distance = float("inf")
            gx, gy = goal_node
            candidate_info = []  # For diagnostic logging
            use_path_distance = level_cache is not None and goal_id is not None

            # Track if any fallbacks occurred for diagnostic logging
            euclidean_fallback_count = 0

            for node_pos, dist_sq_to_ninja in overlapping_nodes:
                nx, ny = node_pos

                if use_path_distance:
                    # FIX: Use cached PATH distance instead of Euclidean distance
                    # This correctly handles levels where the path goes away from the goal first
                    dist_to_goal = level_cache.get_geometric_distance(
                        node_pos, goal_node, goal_id
                    )
                    # Fallback to Euclidean if cache miss (can happen for nodes outside flood-fill)
                    if dist_to_goal == float("inf"):
                        dist_to_goal = ((gx - nx) ** 2 + (gy - ny) ** 2) ** 0.5
                        euclidean_fallback_count += 1
                        _logger.debug(
                            f"[PBRS] Cache miss for node {node_pos}, using Euclidean fallback."
                        )
                else:
                    # Legacy: Euclidean distance to goal
                    # WARNING: This can cause incorrect node selection when path requires
                    # going away from the goal first (e.g., going left when goal is right)
                    dist_to_goal = ((gx - nx) ** 2 + (gy - ny) ** 2) ** 0.5
                    euclidean_fallback_count += 1

                candidate_info.append((node_pos, dist_to_goal, dist_sq_to_ninja**0.5))
                if dist_to_goal < best_distance:
                    best_distance = dist_to_goal
                    best_node = node_pos

            # DIAGNOSTIC: Log when multiple nodes compete or Euclidean fallback was used
            if best_node is not None:
                # Check if nodes have significantly different distances
                distances = [info[1] for info in candidate_info]
                dist_spread = (
                    max(distances) - min(distances) if len(distances) > 1 else 0
                )

                # Log at debug level if Euclidean fallback was used (no level_cache)
                # This is expected in some code paths (e.g., visualization, cache miss)
                if not use_path_distance and len(overlapping_nodes) > 1:
                    _logger.debug(
                        f"[PBRS] Node selection without path cache. "
                        f"Using Euclidean distance. "
                        f"ninja_pos={ninja_pos}, candidates={len(overlapping_nodes)}"
                    )
                elif dist_spread > 10.0:  # Nodes differ by >10px
                    distance_type = "PATH" if use_path_distance else "EUCLIDEAN"
                    log_level = _logger.debug if use_path_distance else _logger.warning
                    log_level(
                        f"[PBRS_DIAG] Node selection using {distance_type} distance to goal. "
                        f"ninja_pos={ninja_pos}, selected={best_node}, "
                        f"dist_to_goal={best_distance:.1f}px, "
                        f"candidates (node, dist_to_goal, dist_to_ninja): {candidate_info}, "
                        f"spread={dist_spread:.1f}px."
                        + (
                            ""
                            if use_path_distance
                            else " WARNING: May select wrong node if path goes away from goal first!"
                        )
                    )
                return best_node

        # Otherwise, return closest to ninja center
        overlapping_nodes.sort(key=lambda node: node[1])  # Sort by distance
        return overlapping_nodes[0][0]

    # Fallback: visual matching within 24 pixels (1 tile, for airborne ninja tolerance)
    # This handles edge cases where ninja is airborne between nodes
    # Increased from 5px to 24px to be more tolerant when ninja is in midair
    for pos in adjacency.keys():
        x, y = pos
        if (
            abs(x + NODE_WORLD_COORD_OFFSET - ninja_pos[0]) < 24
            and abs(y + NODE_WORLD_COORD_OFFSET - ninja_pos[1]) < 24
        ):
            return pos

    return None
