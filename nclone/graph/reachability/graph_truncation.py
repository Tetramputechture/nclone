"""
Smart graph truncation for memory optimization.

When the reachable node count exceeds N_MAX_NODES, this module provides
intelligent truncation that preserves path-relevant nodes while reducing
graph size.

Key principle: Full flood-fill playspace is preserved when possible.
Only apply truncation when absolutely necessary (exceeding the limit),
and even then prioritize keeping path-relevant nodes.
"""

import logging
from collections import deque
from typing import Dict, List, Set, Tuple, Optional

from ..common import N_MAX_NODES

logger = logging.getLogger(__name__)


def _bfs_reachable_from(
    start_node: Tuple[int, int],
    adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
) -> Set[Tuple[int, int]]:
    """
    Simple BFS to find all nodes reachable from start (no physics costs).

    Args:
        start_node: Starting node position
        adjacency: Graph adjacency structure

    Returns:
        Set of reachable node positions
    """
    if start_node not in adjacency:
        return set()

    reachable = set()
    queue = deque([start_node])
    visited = {start_node}

    while queue:
        current = queue.popleft()
        reachable.add(current)

        for neighbor_pos, _ in adjacency.get(current, []):
            if neighbor_pos not in visited:
                visited.add(neighbor_pos)
                queue.append(neighbor_pos)

    return reachable


def _bfs_distance_from(
    start_node: Tuple[int, int],
    adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
    max_nodes: Optional[int] = None,
) -> Dict[Tuple[int, int], int]:
    """
    BFS to compute hop distances from start node (no physics costs).

    Args:
        start_node: Starting node position
        adjacency: Graph adjacency structure
        max_nodes: Optional limit on nodes to explore

    Returns:
        Dict mapping node positions to hop distance from start
    """
    if start_node not in adjacency:
        return {}

    distances = {start_node: 0}
    queue = deque([start_node])

    while queue:
        if max_nodes and len(distances) >= max_nodes:
            break

        current = queue.popleft()
        current_dist = distances[current]

        for neighbor_pos, _ in adjacency.get(current, []):
            if neighbor_pos not in distances:
                distances[neighbor_pos] = current_dist + 1
                queue.append(neighbor_pos)

    return distances


def _find_nodes_on_paths(
    start_node: Tuple[int, int],
    goal_nodes: List[Tuple[int, int]],
    adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
) -> Set[Tuple[int, int]]:
    """
    Find nodes that lie on shortest paths between start and any goal.

    Uses bidirectional BFS approach:
    1. BFS from start -> get distances to all nodes
    2. BFS from each goal -> get distances from all nodes
    3. Node is on path if: dist_from_start[n] + dist_to_goal[n] == dist_start_to_goal

    Args:
        start_node: Starting node position
        goal_nodes: List of goal node positions
        adjacency: Graph adjacency structure

    Returns:
        Set of nodes that lie on optimal paths to goals
    """
    if start_node not in adjacency:
        return set()

    # BFS from start
    dist_from_start = _bfs_distance_from(start_node, adjacency)

    path_nodes = {start_node}

    for goal_node in goal_nodes:
        if goal_node not in adjacency or goal_node not in dist_from_start:
            continue

        # BFS from goal (reverse direction conceptually, but same graph)
        dist_from_goal = _bfs_distance_from(goal_node, adjacency)

        # Optimal distance from start to goal
        optimal_dist = dist_from_start.get(goal_node)
        if optimal_dist is None:
            continue

        # Find all nodes on optimal paths
        for node, d_start in dist_from_start.items():
            d_goal = dist_from_goal.get(node)
            if d_goal is not None and d_start + d_goal == optimal_dist:
                path_nodes.add(node)

    return path_nodes


def truncate_graph_if_needed(
    adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
    reachable_nodes: Set[Tuple[int, int]],
    start_pos: Tuple[int, int],
    goal_positions: List[Tuple[int, int]],
    max_nodes: int = N_MAX_NODES,
    path_buffer_distance: int = 5,
) -> Tuple[
    Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
    Set[Tuple[int, int]],
    bool,
]:
    """
    Truncate graph only if reachable nodes exceed max_nodes.

    Preserves full exploration space when possible. Only truncates when
    the level is exceptionally open, and then intelligently prunes the
    least useful exploration areas.

    Priority for keeping nodes:
    1. Nodes on shortest paths to objectives (highest priority)
    2. Nodes within buffer distance of path nodes (medium priority)
    3. Remaining nodes by distance to nearest objective (lowest priority)

    Args:
        adjacency: Full adjacency graph
        reachable_nodes: Set of all reachable node positions
        start_pos: Start position (spawn) in pixel coordinates
        goal_positions: List of goal positions (switches, exits) in pixel coordinates
        max_nodes: Maximum allowed nodes (default: N_MAX_NODES)
        path_buffer_distance: Hop distance buffer around path nodes to preserve

    Returns:
        Tuple of:
        - Filtered adjacency graph
        - Filtered reachable nodes set
        - Boolean indicating if truncation was applied
    """
    # Check if truncation is needed
    if len(reachable_nodes) <= max_nodes:
        logger.debug(
            f"Graph within limits: {len(reachable_nodes)} nodes <= {max_nodes} max"
        )
        return adjacency, reachable_nodes, False

    logger.warning(
        f"Graph truncation needed: {len(reachable_nodes)} nodes > {max_nodes} max. "
        f"Applying smart truncation..."
    )

    # Find start node in graph (closest to start_pos)
    start_node = _find_closest_node(start_pos, reachable_nodes)
    if start_node is None:
        logger.error("Could not find start node in reachable set")
        return adjacency, reachable_nodes, False

    # Find goal nodes in graph (closest to each goal_position)
    goal_nodes = []
    for goal_pos in goal_positions:
        goal_node = _find_closest_node(goal_pos, reachable_nodes)
        if goal_node is not None:
            goal_nodes.append(goal_node)

    if not goal_nodes:
        logger.warning(
            "No goal nodes found in reachable set, using distance-based truncation only"
        )

    # Step 1: Find nodes on paths to objectives (highest priority)
    path_nodes = _find_nodes_on_paths(start_node, goal_nodes, adjacency)
    logger.debug(f"Found {len(path_nodes)} nodes on paths to objectives")

    # Step 2: Find nodes within buffer distance of path nodes
    buffer_nodes = set()
    for path_node in path_nodes:
        nearby = _bfs_distance_from(
            path_node, adjacency, max_nodes=path_buffer_distance * 10
        )
        for node, dist in nearby.items():
            if dist <= path_buffer_distance:
                buffer_nodes.add(node)

    priority_nodes = path_nodes | buffer_nodes
    logger.debug(
        f"Priority nodes (path + buffer): {len(priority_nodes)} "
        f"({len(path_nodes)} path + {len(buffer_nodes - path_nodes)} buffer)"
    )

    # Step 3: If priority nodes alone exceed limit, keep only path nodes + closest buffer
    if len(priority_nodes) > max_nodes:
        logger.warning(
            f"Even priority nodes ({len(priority_nodes)}) exceed limit ({max_nodes}). "
            f"Keeping only path nodes and closest buffer."
        )
        # Start with path nodes
        keep_nodes = set(path_nodes)

        # Add buffer nodes by distance until we hit limit
        remaining_slots = max_nodes - len(keep_nodes)
        if remaining_slots > 0:
            # Sort buffer nodes by minimum distance to any path node
            non_path_buffer = buffer_nodes - path_nodes
            buffer_with_dist = []
            for node in non_path_buffer:
                min_dist = min(
                    abs(node[0] - pn[0]) + abs(node[1] - pn[1]) for pn in path_nodes
                )
                buffer_with_dist.append((node, min_dist))
            buffer_with_dist.sort(key=lambda x: x[1])

            for node, _ in buffer_with_dist[:remaining_slots]:
                keep_nodes.add(node)
    else:
        # Step 4: Fill remaining slots with other reachable nodes by distance to goals
        keep_nodes = set(priority_nodes)
        remaining_slots = max_nodes - len(keep_nodes)

        if remaining_slots > 0 and goal_nodes:
            # Get remaining nodes not in priority set
            other_nodes = reachable_nodes - priority_nodes

            # Sort by minimum distance to any goal
            nodes_with_dist = []
            for node in other_nodes:
                min_dist = min(
                    abs(node[0] - gn[0]) + abs(node[1] - gn[1]) for gn in goal_nodes
                )
                nodes_with_dist.append((node, min_dist))
            nodes_with_dist.sort(key=lambda x: x[1])

            for node, _ in nodes_with_dist[:remaining_slots]:
                keep_nodes.add(node)

    # Build filtered adjacency
    filtered_adjacency = {}
    for node in keep_nodes:
        if node in adjacency:
            # Filter neighbors to only include kept nodes
            valid_neighbors = [
                (neighbor, cost)
                for neighbor, cost in adjacency[node]
                if neighbor in keep_nodes
            ]
            filtered_adjacency[node] = valid_neighbors

    logger.info(
        f"Graph truncated: {len(reachable_nodes)} -> {len(keep_nodes)} nodes "
        f"({100 * len(keep_nodes) / len(reachable_nodes):.1f}% retained)"
    )

    return filtered_adjacency, keep_nodes, True


def _find_closest_node(
    pos: Tuple[int, int],
    nodes: Set[Tuple[int, int]],
    max_distance: float = 50.0,
) -> Optional[Tuple[int, int]]:
    """
    Find the closest node to a position.

    Args:
        pos: Target position (x, y) in pixels
        nodes: Set of node positions to search
        max_distance: Maximum distance to consider (pixels)

    Returns:
        Closest node position, or None if none within max_distance
    """
    if not nodes:
        return None

    closest = None
    min_dist_sq = max_distance * max_distance

    for node in nodes:
        dist_sq = (node[0] - pos[0]) ** 2 + (node[1] - pos[1]) ** 2
        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
            closest = node

    return closest
