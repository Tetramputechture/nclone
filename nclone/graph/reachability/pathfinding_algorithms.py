"""
Pathfinding algorithms for graph traversal.

Provides Dijkstra's algorithm (find_shortest_path), BFS distance calculation,
and fallback pathfinding for unstable positions. Supports both geometric
(euclidean pixel) and physics-aware edge costs.
"""

import logging
import heapq
from typing import Dict, Tuple, List, Optional, Any
from collections import deque

from .cost_functions import (
    NODE_WORLD_COORD_OFFSET,
    SUB_NODE_SIZE,
    _is_node_grounded_util,
    _calculate_physics_aware_cost,
)
from .node_lookup import (
    find_closest_node_to_position,
    find_ninja_node,
)

_logger = logging.getLogger(__name__)

# Sub-node spacing for geometric distance calculation
_SUB_NODE_SIZE = 12


def _violates_horizontal_rule(
    current: Tuple[int, int],
    neighbor: Tuple[int, int],
    parents: Dict[Tuple[int, int], Optional[Tuple[int, int]]],
    base_adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
    physics_cache: Optional[Dict[Tuple[int, int], Dict[str, bool]]] = None,
) -> bool:
    """
    Check if edge violates consecutive horizontal non-grounded rule.

    Rule: Allow one horizontal non-grounded edge for corrections, but if the
    previous edge was also horizontal, the current horizontal edge must be grounded.
    """
    curr_x, curr_y = current
    neigh_x, neigh_y = neighbor

    if curr_y != neigh_y or curr_x == neigh_x:
        return False

    if physics_cache is None:
        return False

    current_physics = physics_cache.get(current, {"grounded": True, "walled": False})
    neighbor_physics = physics_cache.get(neighbor, {"grounded": True, "walled": False})
    current_grounded = current_physics["grounded"]
    neighbor_grounded = neighbor_physics["grounded"]

    if current_grounded and neighbor_grounded:
        return False

    parent = parents.get(current)
    if parent is None:
        return False

    par_x, par_y = parent
    return par_y == curr_y and par_x != curr_x


def _is_node_walled(
    node_pos: Tuple[int, int],
    base_adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
) -> bool:
    """Check if node has wall (left or right neighbor unreachable)."""
    x, y = node_pos
    left_pos = (x - SUB_NODE_SIZE, y)
    right_pos = (x + SUB_NODE_SIZE, y)
    return left_pos not in base_adjacency or right_pos not in base_adjacency


def bfs_distance_from_start(
    start_node: Tuple[int, int],
    target_node: Optional[Tuple[int, int]],
    adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
    base_adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
    max_distance: Optional[float] = None,
    return_parents: bool = False,
    physics_cache: Optional[Dict[Tuple[int, int], Dict[str, bool]]] = None,
    level_data: Optional[Any] = None,
    mine_proximity_cache: Optional[Any] = None,
    use_geometric_costs: bool = True,
    track_geometric_distances: bool = False,
) -> Tuple[
    Dict[Tuple[int, int], float],
    Optional[float],
    Optional[Dict],
    Optional[Dict[Tuple[int, int], float]],
]:
    """
    Calculate distances from start node using Dijkstra's algorithm.

    Supports two cost modes:
    - use_geometric_costs=True (default): Euclidean pixel costs (faster, no cache needed)
    - use_geometric_costs=False: Physics-aware costs (requires physics_cache, level_data,
      mine_proximity_cache). Uses N++ movement physics for path preference.

    When using physics costs, track_geometric_distances=True returns geometric distances
    along the physics-optimal path (for PBRS normalization).

    Args:
        start_node: Starting node position
        target_node: Optional target node to find distance to (early termination)
        adjacency: Masked graph adjacency structure (for pathfinding)
        base_adjacency: Base graph adjacency structure (pre-entity-mask)
        max_distance: Optional maximum distance to compute (for early termination)
        return_parents: If True, return the parents dict for path reconstruction
        physics_cache: Pre-computed physics properties (grounded, walled) for physics costs
        level_data: LevelData for mine proximity (required when use_geometric_costs=False)
        mine_proximity_cache: MineProximityCostCache for O(1) mine cost (required when use_geometric_costs=False)
        use_geometric_costs: If True, use Euclidean costs; if False, use physics-aware costs
        track_geometric_distances: If True and use_geometric_costs=False, also track pixel distances

    Returns:
        Tuple of (distances_dict, target_distance, parents_dict, geometric_distances_dict)
    """
    use_physics = not use_geometric_costs and physics_cache is not None
    if use_physics and (level_data is None or mine_proximity_cache is None):
        use_physics = False
        use_geometric_costs = True

    pq = [(0.0, start_node)]
    distances = {start_node: 0.0}
    parents = {start_node: None}
    visited = set()
    grandparents = {start_node: None} if use_physics else {}
    aerial_chains = {start_node: 0} if use_physics else {}
    geometric_distances: Optional[Dict[Tuple[int, int], float]] = (
        {start_node: 0.0} if (track_geometric_distances and use_physics) else None
    )

    euclidean_to_target = None
    if target_node is not None:
        dx_target = target_node[0] - start_node[0]
        dy_target = target_node[1] - start_node[1]
        euclidean_to_target = (dx_target * dx_target + dy_target * dy_target) ** 0.5
        early_termination_distance = euclidean_to_target * 10.0

    while pq:
        current_dist, current = heapq.heappop(pq)

        if current in visited:
            continue
        visited.add(current)

        if target_node is not None and current == target_node:
            parents_result = parents if return_parents else None
            return distances, current_dist, parents_result, geometric_distances

        if max_distance is not None and current_dist > max_distance:
            continue

        if (
            euclidean_to_target is not None
            and current_dist > early_termination_distance
            and target_node in distances
        ):
            break

        current_physics = physics_cache.get(current, {}) if use_physics else {}
        current_grounded = current_physics.get("grounded", True)
        current_chain = aerial_chains.get(current, 0) if use_physics else 0
        current_parent = parents.get(current) if use_physics else None
        current_grandparent = grandparents.get(current) if use_physics else None

        neighbors = adjacency.get(current, [])

        for neighbor_info in neighbors:
            neighbor_pos, _ = neighbor_info

            if neighbor_pos in visited:
                continue

            if use_physics:
                if _violates_horizontal_rule(
                    current, neighbor_pos, parents, base_adjacency, physics_cache
                ):
                    continue

                dy = neighbor_pos[1] - current[1]
                # Track ALL airborne moves (not just upward) so chain is correct for
                # upward/diagonal after horizontal air, and for horizontal-after-horizontal
                is_aerial = not current_grounded
                new_chain = current_chain + 1 if is_aerial else 0

                cost = _calculate_physics_aware_cost(
                    current,
                    neighbor_pos,
                    base_adjacency,
                    current_parent,
                    physics_cache,
                    level_data,
                    mine_proximity_cache,
                    new_chain,
                    current_grandparent,
                )
            else:
                dx = neighbor_pos[0] - current[0]
                dy = neighbor_pos[1] - current[1]
                cost = (dx * dx + dy * dy) ** 0.5

            new_dist = current_dist + cost

            if neighbor_pos not in distances or new_dist < distances[neighbor_pos]:
                distances[neighbor_pos] = new_dist
                parents[neighbor_pos] = current
                if use_physics:
                    grandparents[neighbor_pos] = current_parent
                    aerial_chains[neighbor_pos] = new_chain
                heapq.heappush(pq, (new_dist, neighbor_pos))

                if geometric_distances is not None:
                    gdx = neighbor_pos[0] - current[0]
                    gdy = neighbor_pos[1] - current[1]
                    geo_edge = (gdx * gdx + gdy * gdy) ** 0.5
                    geometric_distances[neighbor_pos] = (
                        geometric_distances.get(current, 0.0) + geo_edge
                    )

    target_distance = distances.get(target_node) if target_node else None
    parents_result = parents if return_parents else None
    return distances, target_distance, parents_result, geometric_distances


def calculate_geometric_path_distance(
    start_pos: Tuple[int, int],
    goal_pos: Tuple[int, int],
    adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
    base_adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
    physics_cache: Optional[Dict[Tuple[int, int], Dict[str, bool]]] = None,
    spatial_hash: Optional[Any] = None,
    subcell_lookup: Optional[Any] = None,
    entity_radius: float = 0.0,
    ninja_radius: float = 10.0,
    level_data: Optional[Any] = None,
    mine_proximity_cache: Optional[Any] = None,
    mine_sdf: Optional[Any] = None,
    ninja_velocity: Optional[Tuple[float, float]] = None,
    allow_high_cost_edges: bool = False,
) -> float:
    """
    Calculate geometric (pixel) path distance from start to goal.

    Uses pure geometric BFS. physics_cache, mine_proximity_cache, etc. are deprecated
    and ignored. Kept for backward compatibility during migration.
    """
    start_node = find_ninja_node(
        start_pos,
        adjacency,
        spatial_hash=spatial_hash,
        subcell_lookup=subcell_lookup,
        ninja_radius=ninja_radius,
    )
    goal_node = find_closest_node_to_position(
        goal_pos,
        adjacency,
        threshold=None,
        entity_radius=entity_radius,
        ninja_radius=ninja_radius,
        spatial_hash=spatial_hash,
        subcell_lookup=subcell_lookup,
    )

    if start_node is None or goal_node is None:
        return float("inf")
    if start_node == goal_node:
        return 0.0

    distances, _, parents, _ = bfs_distance_from_start(
        start_node,
        goal_node,
        adjacency,
        base_adjacency,
        return_parents=True,
    )

    target_geometric_distance = distances.get(goal_node)
    if target_geometric_distance is None or target_geometric_distance == float("inf"):
        return float("inf")

    # === SUB-NODE RESOLUTION FOR DENSE PBRS (0.05-3.33px movements) ===
    # Agent can move 0.05-3.33px per step, but our graph is 12px spaced.
    # To provide dense rewards for small movements, we project the agent's
    # actual position onto the optimal path direction (start_node → next_hop).
    #
    # This ensures:
    # - Moving 0.1px toward next_hop: distance decreases by ~0.1px (dense feedback)
    # - Moving 0.1px away from next_hop: distance increases by ~0.1px (immediate penalty)
    # - Path-aware: Uses next_hop direction, not Euclidean (handles winding paths)
    #
    # Example: Agent at (312.5, 444.2), start_node at (300, 444), next_hop at (312, 444)
    # - Path direction: (312-300, 444-444) = (12, 0) normalized = (1, 0)
    # - Agent offset from node: (312.5-300, 444.2-444) = (12.5, 0.2)
    # - Projection: 12.5*1 + 0.2*0 = 12.5px (agent is 12.5px ahead of node)
    # - Adjusted distance: node_distance - 12.5px (agent is closer by 12.5px)

    # Extract next hop from path (second node after start_node)
    next_hop = None
    if parents is not None:
        # Reconstruct path from start to goal by following parents backward
        path = []
        current = goal_node
        while current is not None and len(path) < 1000:  # Safety limit
            path.append(current)
            current = parents.get(current)
        path.reverse()

        # Next hop is the second node in the path (first step from start_node)
        if len(path) >= 2 and path[0] == start_node:
            next_hop = path[1]

    if next_hop is not None:
        # Convert node positions to world coordinates for projection calculation
        start_node_world_x = start_node[0] + NODE_WORLD_COORD_OFFSET
        start_node_world_y = start_node[1] + NODE_WORLD_COORD_OFFSET
        next_hop_world_x = next_hop[0] + NODE_WORLD_COORD_OFFSET
        next_hop_world_y = next_hop[1] + NODE_WORLD_COORD_OFFSET

        # Vector from start_node to next_hop (optimal path direction)
        path_dx = next_hop_world_x - start_node_world_x
        path_dy = next_hop_world_y - start_node_world_y
        path_len = (path_dx * path_dx + path_dy * path_dy) ** 0.5

        if path_len > 0.001:
            # Normalize path direction
            path_dir_x = path_dx / path_len
            path_dir_y = path_dy / path_len

            # Vector from start_node to player's actual position
            player_dx = start_pos[0] - start_node_world_x
            player_dy = start_pos[1] - start_node_world_y

            # Project player offset onto path direction
            # Positive projection = player ahead (toward next_hop) = closer to goal
            # Negative projection = player behind (away from next_hop) = further from goal
            projection = player_dx * path_dir_x + player_dy * path_dir_y

            # Adjust distance by projection and collision radii
            # Positive projection reduces distance (agent ahead of node toward goal)
            adjusted_distance = max(
                0.0,
                target_geometric_distance - projection - (ninja_radius + entity_radius),
            )

            # DIAGNOSTIC: Log sub-node projection in BFS fallback path
            # This helps verify slow path produces same results as fast path
            _logger.debug(
                f"[SUB_NODE_SLOW] projection={projection:.3f}px, "
                f"node_dist={target_geometric_distance:.1f}px, "
                f"adjusted_dist={adjusted_distance:.1f}px, "
                f"player_offset=({player_dx:.2f}, {player_dy:.2f}), "
                f"path_dir=({path_dir_x:.3f}, {path_dir_y:.3f}), "
                f"path_length={len(path)} nodes"
            )

            return adjusted_distance

    # Fallback: No next_hop available (single-node path or path reconstruction failed)
    # Just use node-to-node distance with radius adjustment
    return max(0.0, target_geometric_distance - (ninja_radius + entity_radius))


def find_shortest_path(
    start_node: Tuple[int, int],
    end_node: Tuple[int, int],
    adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
    base_adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
    physics_cache: Optional[Dict[Tuple[int, int], Dict[str, bool]]] = None,
    level_data: Optional[Any] = None,
    mine_proximity_cache: Optional[Any] = None,
    ninja_velocity: Optional[Tuple[float, float]] = None,
    allow_high_cost_edges: bool = False,
) -> Tuple[Optional[List[Tuple[int, int]]], float]:
    """
    Find shortest path from start to end node using Dijkstra.

    Uses physics-aware costs when physics_cache, level_data, and mine_proximity_cache
    are all provided; otherwise uses geometric (Euclidean) costs.
    ninja_velocity and allow_high_cost_edges are reserved for future use.
    """
    if start_node == end_node:
        return [start_node], 0.0

    use_physics = (
        physics_cache is not None
        and level_data is not None
        and mine_proximity_cache is not None
    )

    pq = [(0.0, start_node)]
    distances = {start_node: 0.0}
    parents = {start_node: None}
    visited = set()
    grandparents = {start_node: None} if use_physics else {}
    aerial_chains = {start_node: 0} if use_physics else {}

    while pq:
        current_dist, current = heapq.heappop(pq)

        if current in visited:
            continue
        visited.add(current)

        if current == end_node:
            path = []
            node = end_node
            while node is not None:
                path.append(node)
                node = parents.get(node)
            path.reverse()
            return path, distances[end_node]

        current_physics = physics_cache.get(current, {}) if use_physics else {}
        current_grounded = current_physics.get("grounded", True)
        current_chain = aerial_chains.get(current, 0) if use_physics else 0
        current_parent = parents.get(current) if use_physics else None
        current_grandparent = grandparents.get(current) if use_physics else None

        neighbors = adjacency.get(current, [])
        for neighbor_info in neighbors:
            neighbor_pos, _ = neighbor_info

            if neighbor_pos in visited:
                continue

            if use_physics:
                if _violates_horizontal_rule(
                    current, neighbor_pos, parents, base_adjacency, physics_cache
                ):
                    continue

                dy = neighbor_pos[1] - current[1]
                # Track ALL airborne moves (not just upward) so chain is correct for
                # upward/diagonal after horizontal air, and for horizontal-after-horizontal
                is_aerial = not current_grounded
                new_chain = current_chain + 1 if is_aerial else 0

                cost = _calculate_physics_aware_cost(
                    current,
                    neighbor_pos,
                    base_adjacency,
                    current_parent,
                    physics_cache,
                    level_data,
                    mine_proximity_cache,
                    new_chain,
                    current_grandparent,
                )
            else:
                dx = neighbor_pos[0] - current[0]
                dy = neighbor_pos[1] - current[1]
                cost = (dx * dx + dy * dy) ** 0.5

            new_dist = current_dist + cost

            if neighbor_pos not in distances or new_dist < distances[neighbor_pos]:
                distances[neighbor_pos] = new_dist
                parents[neighbor_pos] = current
                if use_physics:
                    grandparents[neighbor_pos] = current_parent
                    aerial_chains[neighbor_pos] = new_chain
                heapq.heappush(pq, (new_dist, neighbor_pos))

    return None, float("inf")


def find_nearest_stable_position(
    start_node: Tuple[int, int],
    adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
    base_adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
    physics_cache: Optional[Dict[Tuple[int, int], Dict[str, bool]]] = None,
    level_data: Optional[Any] = None,
    mine_proximity_cache: Optional[Any] = None,
) -> Optional[Tuple[int, int]]:
    """
    Find the nearest SAFE grounded or walled position reachable from start_node.

    Uses base_adjacency for grounding/walled checks. physics_cache is deprecated.
    """
    if base_adjacency is None:
        return None

    # Check if start is already stable
    if _is_node_grounded_util(start_node, base_adjacency) or _is_node_walled(
        start_node, base_adjacency
    ):
        return start_node

    queue = deque([(start_node, 0.0)])
    visited = {start_node}
    best_safe_stable = None
    best_safe_distance = float("inf")
    best_any_stable = None
    best_any_distance = float("inf")

    while queue:
        current, dist = queue.popleft()

        is_stable = _is_node_grounded_util(current, base_adjacency) or _is_node_walled(
            current, base_adjacency
        )
        if is_stable:
            # Check if it's also SAFE (has masked edges available)
            current_masked_edges = adjacency.get(current, [])
            is_safe = len(current_masked_edges) > 0

            if is_safe:
                # Found a SAFE stable position - this is preferred!
                if dist < best_safe_distance:
                    best_safe_stable = current
                    best_safe_distance = dist
                    # Early exit if we found a reasonably close SAFE stable position
                    if dist < 2400.0:  # ~16 nodes away - worth traveling for safety
                        return best_safe_stable
            else:
                # Found a stable position but it's in a dangerous area
                if dist < best_any_distance:
                    best_any_stable = current
                    best_any_distance = dist

        # Explore neighbors with much larger search radius to find safe areas
        if dist < 2400.0:  # Max search distance (~100 tiles, can span entire map)
            # Use base_adjacency to allow escape from dangerous positions (e.g., near mines)
            # The agent exists at start_node, so we need to allow movement away from it
            neighbors = base_adjacency.get(current, [])
            for neighbor_pos, _ in neighbors:
                if neighbor_pos not in visited:
                    visited.add(neighbor_pos)
                    # Calculate actual edge cost
                    edge_dist = (
                        (neighbor_pos[0] - current[0]) ** 2
                        + (neighbor_pos[1] - current[1]) ** 2
                    ) ** 0.5
                    queue.append((neighbor_pos, dist + edge_dist))

    # Prefer safe stable positions over dangerous ones, even if farther away
    # Only use dangerous stable positions if no safe ones were found
    if best_safe_stable is not None:
        return best_safe_stable
    elif best_any_stable is not None:
        return best_any_stable
    else:
        # Always return at least the start_node as fallback
        # The agent exists there currently and should be treated as a valid starting point
        return start_node


def find_shortest_path_with_fallback(
    start_node: Tuple[int, int],
    end_node: Tuple[int, int],
    adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
    base_adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
    physics_cache: Optional[Dict[Tuple[int, int], Dict[str, bool]]] = None,
    level_data: Optional[Any] = None,
    mine_proximity_cache: Optional[Any] = None,
    ninja_velocity: Optional[Tuple[float, float]] = None,
    debug: bool = False,
    allow_high_cost_edges: bool = False,
) -> Tuple[Optional[List[Tuple[int, int]]], float]:
    """
    Find shortest path with two-phase fallback for unstable or dangerous starting positions.

    Strategy:
    1. Try direct pathfinding from start to end using masked adjacency (safe routing)
    2. If that fails (including when masked adjacency blocks all edges from start):
       a. Find nearest SAFE stable position (grounded + has masked edges) using BFS
          - Searches up to 2400px (~100 tiles, can span entire map)
          - Prioritizes safe positions (with masked edges) over dangerous ones
          - Only uses dangerous positions as last resort
       b. Pathfind from start to stable using base_adjacency (escape dangerous area)
       c. Pathfind from stable to end using:
          - masked adjacency IF stable is in safe area (has masked edges) - NORMAL CASE
          - base_adjacency IF stable is still in danger zone (0 masked edges) - RARE FALLBACK
       d. Concatenate paths: start→stable→end

    This allows agents to:
    - Fall long distances to reach SAFE platforms
    - Jump towards distant SAFE platforms and continue from there
    - ESCAPE from dangerous positions (e.g., airborne near mines) to safe ground
    - AVOID pathing through deadly hazards by finding safe routes around them
    - Handle situations where direct path is impossible due to physics or hazards

    Args:
        start_node: Starting node position
        end_node: Target node position
        adjacency: Masked graph adjacency structure (hazards blocked)
        base_adjacency: Base graph adjacency structure (all physically possible edges)
        physics_cache: Optional pre-computed physics properties
        level_data: Optional LevelData for mine proximity checks
        mine_proximity_cache: Optional MineProximityCostCache
        ninja_velocity: Optional ninja velocity (xspeed, yspeed) for first edge cost adjustment
        debug: Enable diagnostic logging
        allow_high_cost_edges: Allow edges with infinite cost (for visualization/debugging)

    Returns:
        Tuple of (path, distance):
        - path: List of node positions from start to end, or None if unreachable
        - distance: Total path distance, or float('inf') if unreachable
    """
    # if debug:
    #     print("\n" + "=" * 80)
    #     print("[PATHFIND_DIAG] find_shortest_path_with_fallback called")
    #     print(f"  start_node: {start_node}")
    #     print(f"  end_node: {end_node}")
    #     print(f"  ninja_velocity: {ninja_velocity}")

    #     # Check adjacency availability
    #     masked_neighbors = adjacency.get(start_node, [])
    #     base_neighbors = base_adjacency.get(start_node, [])
    #     print(f"  start_node masked edges: {len(masked_neighbors)}")
    #     print(f"  start_node base edges: {len(base_neighbors)}")

    #     # Check physics
    #     if physics_cache:
    #         start_physics = physics_cache.get(start_node, {})
    #         print(
    #             f"  start_node physics: grounded={start_physics.get('grounded')}, walled={start_physics.get('walled')}"
    #         )

    #     # Check mine proximity
    #     if mine_proximity_cache and hasattr(mine_proximity_cache, "cache"):
    #         mine_cache_size = len(mine_proximity_cache.cache)
    #         is_near_mine = start_node in mine_proximity_cache.cache
    #         print(
    #             f"  mine cache size: {mine_cache_size}, start near mine: {is_near_mine}"
    #         )

    # Phase 1: Try direct pathfinding
    # if debug:
    #     print("\n[PATHFIND_DIAG] Phase 1: Direct pathfinding (masked adjacency)")

    path, cost = find_shortest_path(
        start_node,
        end_node,
        adjacency,
        base_adjacency,
        physics_cache=physics_cache,
        level_data=level_data,
        mine_proximity_cache=mine_proximity_cache,
    )

    if path is not None:
        # if debug:
        #     print(
        #         f"[PATHFIND_DIAG] Phase 1 SUCCESS: path length={len(path)}, cost={cost:.2f}"
        #     )
        return path, cost

    if debug:
        print("[PATHFIND_DIAG] Phase 1 FAILED: No direct path found")

    # Phase 2: Fallback - find nearest stable position and path from there
    # This fallback is critical for escaping dangerous positions (e.g., airborne near mines)
    # where the masked adjacency has blocked all edges from start_node
    if debug:
        print(
            "\n[PATHFIND_DIAG] Phase 2: Fallback pathfinding (base adjacency for escape)"
        )

    # Note: We always attempt fallback even if start is stable, because Phase 1 failed
    # This handles cases where masked adjacency blocks all edges from start (e.g., mines)
    # but base_adjacency allows escape

    # Find nearest stable position using unmasked graph to allow escape
    # This will always return at least start_node as fallback, ensuring we always
    # have a valid starting point for pathfinding (the agent exists there currently)
    if debug:
        print(f"[PATHFIND_DIAG] Finding nearest stable position from {start_node}...")

    stable_node = find_nearest_stable_position(
        start_node,
        adjacency,
        base_adjacency,
    )

    if debug:
        is_same = stable_node == start_node
        print(
            f"[PATHFIND_DIAG] Found stable_node: {stable_node} (same as start: {is_same})"
        )

    # Path from start to stable position (simple geometric path - falling/jumping)
    # If stable_node == start_node, this will return [start_node] with cost 0.0,
    # and we'll proceed to path from current position to goal
    # Use base_adjacency (unmasked) to allow escape from dangerous positions (e.g., near mines)
    # The agent exists at start_node, so we must allow movement away from it
    if debug:
        print(
            "[PATHFIND_DIAG] Phase 2a: Path from start to stable (using base_adjacency)"
        )

    path_to_stable, cost_to_stable = find_shortest_path(
        start_node,
        stable_node,
        base_adjacency,  # Use unmasked graph to allow escape from dangerous positions
        base_adjacency,
        physics_cache=physics_cache,
        level_data=level_data,
        mine_proximity_cache=mine_proximity_cache,
    )

    if path_to_stable is None:
        if debug:
            print("[PATHFIND_DIAG] Phase 2a FAILED: No path from start to stable")
            print("  This should not happen - base_adjacency should allow movement")
            # Show what edges are available
            base_neighbors = base_adjacency.get(start_node, [])
            print(f"  Available base edges from start: {len(base_neighbors)}")
            if base_neighbors and len(base_neighbors) <= 10:
                print(f"  Edges: {[n[0] for n in base_neighbors[:10]]}")
        return None, float("inf")

    if debug:
        print(
            f"[PATHFIND_DIAG] Phase 2a SUCCESS: path length={len(path_to_stable)}, cost={cost_to_stable:.2f}"
        )

    # Path from stable position to goal
    # Stable position is now prioritized to be in a SAFE area (has masked edges)
    # But as a fallback, if stable_node == start_node OR has no masked edges (still in danger zone),
    # use base_adjacency to allow continued escape (though this should be rare now)
    # Otherwise use masked adjacency for safe routing from stable position
    stable_masked_edges = adjacency.get(stable_node, [])
    use_unmasked_for_escape = stable_node == start_node or len(stable_masked_edges) == 0

    if debug:
        adjacency_type = "base (escape)" if use_unmasked_for_escape else "masked (safe)"
        print(
            f"[PATHFIND_DIAG] Phase 2b: Path from stable to goal (using {adjacency_type} adjacency)"
        )
        if stable_node == start_node:
            print(
                "  Using base_adjacency because stable_node == start_node (still at dangerous position)"
            )
        elif len(stable_masked_edges) == 0:
            print(
                "  Using base_adjacency because stable_node has 0 masked edges (in danger zone)"
            )

        # Check edge availability from stable_node
        if use_unmasked_for_escape:
            stable_edges = base_adjacency.get(stable_node, [])
        else:
            stable_edges = adjacency.get(stable_node, [])
        print(f"  Edges available from stable_node: {len(stable_edges)}")

    path_from_stable, cost_from_stable = find_shortest_path(
        stable_node,
        end_node,
        base_adjacency if use_unmasked_for_escape else adjacency,
        base_adjacency,
        physics_cache=physics_cache,
        level_data=level_data,
        mine_proximity_cache=mine_proximity_cache,
    )

    if path_from_stable is None:
        if debug:
            print("[PATHFIND_DIAG] Phase 2b FAILED: No path from stable to goal")
            print(f"  stable_node: {stable_node}")
            print(f"  end_node: {end_node}")
            print(f"  adjacency type used: {adjacency_type}")

            # Check if nodes exist in graph
            stable_in_graph = stable_node in base_adjacency
            end_in_graph = end_node in base_adjacency
            print(f"  stable_node in base_adjacency: {stable_in_graph}")
            print(f"  end_node in base_adjacency: {end_in_graph}")

            if not end_in_graph:
                print("  ISSUE: end_node not in graph! Cannot path to it.")

            print("=" * 80 + "\n")
        return None, float("inf")

    if debug:
        print(
            f"[PATHFIND_DIAG] Phase 2b SUCCESS: path length={len(path_from_stable)}, cost={cost_from_stable:.2f}"
        )

    # Concatenate paths (remove duplicate stable_node)
    combined_path = path_to_stable[:-1] + path_from_stable
    combined_cost = cost_to_stable + cost_from_stable

    if debug:
        print("[PATHFIND_DIAG] Phase 2 SUCCESS: Combined path")
        print(f"  Total path length: {len(combined_path)}")
        print(f"  Total cost: {combined_cost:.2f}")
        print("=" * 80 + "\n")

    return combined_path, combined_cost
