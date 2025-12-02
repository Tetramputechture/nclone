"""
Core pathfinding algorithms for shortest path calculation.

Provides BFS and A* implementations for finding shortest paths on traversability graphs.
"""

from typing import Dict, Tuple, List, Optional, Any
from collections import deque
import heapq


# Constants
SUB_NODE_SIZE = 12  # Sub-node spacing in pixels


def _is_node_grounded(
    node_pos: Tuple[int, int],
    base_adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
) -> bool:
    """
    Check if a node is grounded (has solid surface below).

    Uses base_adjacency (pre-entity-mask) to determine actual level geometry,
    not affected by mines or other dynamic entities.
    """
    x, y = node_pos
    below_pos = (x, y + SUB_NODE_SIZE)

    if below_pos not in base_adjacency:
        return True

    if node_pos in base_adjacency:
        neighbors = base_adjacency[node_pos]
        for neighbor_pos, _ in neighbors:
            if neighbor_pos == below_pos:
                return False

    return True


def _is_horizontal_edge(from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> bool:
    """Check if edge is horizontal (dy == 0, dx != 0)."""
    return to_pos[1] == from_pos[1] and to_pos[0] != from_pos[0]


def _violates_horizontal_rule(
    current: Tuple[int, int],
    neighbor: Tuple[int, int],
    parents: Dict[Tuple[int, int], Optional[Tuple[int, int]]],
    base_adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
) -> bool:
    """
    Check if edge violates consecutive horizontal non-grounded rule.

    Rule: Allow one horizontal non-grounded edge for corrections, but if the
    previous edge was also horizontal, the current horizontal edge must be grounded.

    Uses base_adjacency (pre-entity-mask) for grounding checks.
    """
    if not _is_horizontal_edge(current, neighbor):
        return False

    current_grounded = _is_node_grounded(current, base_adjacency)
    neighbor_grounded = _is_node_grounded(neighbor, base_adjacency)

    if current_grounded and neighbor_grounded:
        return False

    parent = parents.get(current)
    if parent is None:
        return False

    if _is_horizontal_edge(parent, current):
        return True

    return False


def _calculate_mine_proximity_cost(
    pos: Tuple[int, int],
    level_data: Optional[Any],
    mine_proximity_cache: Optional[Any] = None,
) -> float:
    """Calculate cost multiplier based on proximity to deadly mines.

    Uses cached values when available for O(1) lookup. Falls back to
    direct calculation only if cache is not provided (backward compatibility).

    Args:
        pos: Node position (x, y) in pixels
        level_data: LevelData instance containing mine entities (optional)
        mine_proximity_cache: MineProximityCostCache instance (optional)

    Returns:
        float: Cost multiplier in range [1.0, MINE_HAZARD_COST_MULTIPLIER]
               1.0 if far from mines or no penalty applies
               Higher values when close to deadly mines
    """
    from ...gym_environment.reward_calculation.reward_constants import (
        MINE_HAZARD_COST_MULTIPLIER,
        MINE_HAZARD_RADIUS,
        MINE_PENALIZE_DEADLY_ONLY,
    )
    from ...constants.entity_types import EntityType

    # Early exit if hazard avoidance is disabled
    if MINE_HAZARD_COST_MULTIPLIER <= 1.0:
        return 1.0

    # Use cache if available (O(1) lookup)
    if mine_proximity_cache is not None:
        return mine_proximity_cache.get_cost_multiplier(pos)

    # Fallback: Direct calculation if no cache provided (backward compatibility)
    if level_data is None:
        return 1.0

    # Get all toggle mines from level_data
    mines = level_data.get_entities_by_type(
        EntityType.TOGGLE_MINE
    ) + level_data.get_entities_by_type(EntityType.TOGGLE_MINE_TOGGLED)

    if not mines:
        return 1.0

    # Find closest deadly mine
    min_distance = float("inf")
    for mine in mines:
        # Check if mine is deadly (state 0 = toggled/deadly)
        mine_state = mine.get("state", 0)
        if MINE_PENALIZE_DEADLY_ONLY and mine_state != 0:
            continue  # Skip safe mines (state 1) and toggling mines (state 2)

        mine_x = mine.get("x", 0)
        mine_y = mine.get("y", 0)

        dx = pos[0] - mine_x
        dy = pos[1] - mine_y
        distance = (dx * dx + dy * dy) ** 0.5

        min_distance = min(min_distance, distance)

    # Apply cost based on proximity
    if min_distance < MINE_HAZARD_RADIUS:
        # Linear interpolation:
        # - At mine center: full multiplier
        # - At radius edge: 1.0 (no penalty)
        proximity_factor = 1.0 - (min_distance / MINE_HAZARD_RADIUS)
        multiplier = 1.0 + proximity_factor * (MINE_HAZARD_COST_MULTIPLIER - 1.0)
        return multiplier

    return 1.0


def _calculate_physics_aware_cost(
    src_pos: Tuple[int, int],
    dst_pos: Tuple[int, int],
    base_adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
    parent_pos: Optional[Tuple[int, int]] = None,
    physics_cache: Optional[Dict[Tuple[int, int], Dict[str, bool]]] = None,
    level_data: Optional[Any] = None,
    mine_proximity_cache: Optional[Any] = None,
) -> float:
    """
    Calculate edge cost based on N++ movement physics and mine hazard proximity.

    Movement costs reflect actual N++ physics (from sim_mechanics_doc.md):
    - Grounded horizontal: FAST (ground accel 0.0667, max speed 3.333)
    - Air horizontal: SLOW (air accel 0.0444, ~33% slower)
    - Vertical up: EXPENSIVE (requires jump, fights gravity 0.0667)
    - Vertical down: CHEAP (gravity assists 0.0667)
    - Diagonal movement: Combines horizontal and vertical physics
    - Momentum preservation: Continuing in same X direction is cheaper than changing

    Hazard avoidance (NEW):
    - Paths near deadly mines incur additional cost multiplier
    - Makes PBRS naturally guide agent along safer paths
    - Preserves policy invariance (optimal policy still reaches goal)

    Cost multipliers are tuned so A* naturally prefers physically efficient paths
    (e.g., running on ground > air movement, falling > jumping).

    Args:
        src_pos: Source node position (x, y) in pixels
        dst_pos: Destination node position (x, y) in pixels
        base_adjacency: Base graph adjacency for grounding/wall checks (pre-entity-mask)
        parent_pos: Optional parent position for momentum/direction checking
        physics_cache: Optional pre-computed physics properties for O(1) lookups
        level_data: Optional LevelData for mine proximity checks (fallback)
        mine_proximity_cache: Optional MineProximityCostCache for O(1) mine cost lookup

    Returns:
        Edge cost (float) where 1.0 = baseline movement cost
    """
    dx = dst_pos[0] - src_pos[0]
    dy = dst_pos[1] - src_pos[1]

    # Check if X direction is changing (for momentum considerations)
    x_direction_change = False
    if parent_pos is not None:
        parent_dx = src_pos[0] - parent_pos[0]
        # X direction changed if signs differ and both are non-zero
        if parent_dx != 0 and dx != 0:
            x_direction_change = (parent_dx > 0) != (dx > 0)

    # PERFORMANCE OPTIMIZATION: Use pre-computed physics cache if available
    # This eliminates expensive per-edge checks during A* (O(1) vs O(degree))
    if physics_cache is not None:
        src_physics = physics_cache.get(src_pos, {"grounded": True, "walled": False})
        dst_physics = physics_cache.get(dst_pos, {"grounded": True, "walled": False})
        src_grounded = src_physics["grounded"]
        dst_grounded = dst_physics["grounded"]
        src_walled = src_physics["walled"]
    else:
        raise ValueError("Physics cache is required for physics-aware cost calculation")

    # Base geometric cost (Euclidean distance)
    if dx != 0 and dy != 0:
        base_cost = 1.414  # sqrt(2) for diagonal
    else:
        base_cost = 1.0  # Unit cost for cardinal directions

    # Physics multipliers based on movement type
    # Screen coordinates: y increases downward

    # Special case: Diagonal falling along wall (sliding down wall)
    if dx != 0 and dy > 0 and not src_grounded and src_walled:
        # Wall-sliding downward - very efficient (gravity + wall friction control)
        # This is a common N++ movement pattern
        multiplier = 1.5
    # Special case: Diagonal falling from air (prefer over horizontal air movement)
    elif dx != 0 and dy > 0 and not src_grounded and not src_walled:
        # Diagonal downward from air - very efficient (gravity + momentum)
        # Cheaper than horizontal air movement to encourage diagonal falling
        multiplier = 0.6
    # Special case: Diagonal upward wall-assisted move (valid N++ mechanic)
    elif dx != 0 and dy < 0 and not src_grounded and src_walled:
        # Wall-assisted diagonal upward (wall-jump or wall-climb) - efficient N++ mechanic
        # More expensive than grounded diagonal jump (0.3) but still reasonable
        if x_direction_change:
            # Changing X direction in air is expensive (losing momentum, changing direction)
            # Cost: 1.414 × 1.5 = 2.121 (expensive but not impossible)
            multiplier = 0.25
        else:
            # Continuing same X direction preserves momentum - efficient!
            # Must be cheap enough that chaining diagonals beats staircase grounded path
            # Staircase 2 steps: 1.0 + 1.0 = 2.0, Diagonal chain: 0.707 + 0.707 = 1.414
            # Cost: 1.414 × 0.5 = 0.707 (same as grounded diagonal for momentum!)
            multiplier = 0.5
    # Special case: Diagonal upward from ground (efficient N++ movement)
    elif dx != 0 and dy < 0 and src_grounded:
        # Grounded diagonal jump - most efficient upward movement in N++
        # Combines horizontal momentum with jump, more efficient than pure vertical
        # This initiates the diagonal upward trajectory
        # Must be cheaper than zig-zag step: 1.414 × 0.5 = 0.707 vs 0.5 + 0.5 = 1.0
        # Cost: 1.414 × 0.5 = 0.707
        multiplier = 0.707
    # Special case: Diagonal upward from air without wall (chaining diagonal jumps)
    elif dx != 0 and dy < 0 and not src_grounded and not src_walled:
        # Diagonal upward from air - allows chaining diagonal jumps
        # In N++ you can continue diagonal upward movement while airborne with momentum
        # jump diagonal is only valid if the direction of the wall is opposite of the diagonal of the jump
        if x_direction_change:
            # Changing X direction in air is expensive (losing momentum, changing direction)
            # Cost: 1.414 × 1.5 = 2.121 (expensive but not impossible)
            multiplier = 20
        else:
            # Continuing same X direction preserves momentum - efficient!
            # Must be cheap enough that chaining diagonals beats staircase grounded path
            # Staircase 2 steps: 1.0 + 1.0 = 2.0, Diagonal chain: 0.707 + 0.707 = 1.414
            # Cost: 1.414 × 0.5 = 0.707 (same as grounded diagonal for momentum!)
            multiplier = 0.5
    elif dy < 0:  # Moving up (against gravity) - vertical only
        if src_grounded:
            # Vertical jump from ground - reasonable but not free
            # Can jump ~2.0 pixels/frame vertically (JUMP_FLAT_GROUND_Y)
            multiplier = 0.5
        else:
            # Vertical air movement upward - very expensive
            # Limited air accel (0.0444 vs 0.0667 ground), fighting gravity
            multiplier = 10.0
    elif dy > 0:  # Moving down (with gravity) - vertical or grounded diagonal
        # Gravity assists falling (0.0667 pixels/frame²)
        # Cheap regardless of grounding state
        multiplier = 0.8
    else:  # Horizontal (dy == 0)
        if src_grounded and dst_grounded:
            # Grounded horizontal - FASTEST movement
            # Ground accel 0.0667, max speed 3.333 pixels/frame
            # This is the most efficient movement in N++
            multiplier = 0.5
        else:
            # Air horizontal - more expensive than diagonal falling
            # Air accel 0.0444 (~33% slower than ground)
            # Higher cost to prefer diagonal falling over horizontal air movement
            multiplier = 5.0

    # Apply mine hazard proximity cost multiplier
    # This makes paths near deadly mines more expensive, guiding PBRS toward safer routes
    # Uses cache for O(1) lookup when available
    mine_multiplier = _calculate_mine_proximity_cost(
        dst_pos, level_data, mine_proximity_cache
    )

    return base_cost * multiplier * mine_multiplier


class PathDistanceCalculator:
    """
    Calculate shortest navigable path distances using BFS or A*.

    Operates on precomputed traversability graph for maximum performance.

    BFS: Guaranteed shortest path, explores uniformly
    A*: Faster with Manhattan heuristic, still optimal
    """

    def __init__(self, use_astar: bool = True):
        """
        Initialize path distance calculator.

        Args:
            use_astar: Use A* (True) or BFS (False) for pathfinding
        """
        self.use_astar = use_astar

    def calculate_distance(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
        base_adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
        physics_cache: Optional[Dict[Tuple[int, int], Dict[str, bool]]] = None,
        level_data: Optional[Any] = None,
        mine_proximity_cache: Optional[Any] = None,
    ) -> float:
        """
        Calculate shortest navigable path distance.

        Args:
            start: Starting position (x, y) in pixels
            goal: Goal position (x, y) in pixels
            adjacency: Masked graph adjacency structure (for pathfinding)
            base_adjacency: Base graph adjacency structure (pre-entity-mask, for physics checks)
            physics_cache: Optional pre-computed physics properties for O(1) lookups
            level_data: Optional LevelData for mine proximity checks (fallback)
            mine_proximity_cache: Optional MineProximityCostCache for O(1) mine cost lookup

        Returns:
            Shortest path distance in pixels, or float('inf') if unreachable
        """
        # Quick checks
        if start not in adjacency or goal not in adjacency:
            return float("inf")
        if start == goal:
            return 0.0

        if physics_cache is None:
            raise ValueError(
                "Physics cache is required for physics-aware cost calculation"
            )
        if level_data is None:
            raise ValueError(
                "Level data is required for mine proximity cost calculation"
            )
        if mine_proximity_cache is None:
            raise ValueError(
                "Mine proximity cache is required for mine proximity cost calculation"
            )

        # Choose algorithm
        if self.use_astar:
            return self._astar_distance(
                start,
                goal,
                adjacency,
                base_adjacency,
                physics_cache,
                level_data,
                mine_proximity_cache,
            )
        else:
            return self._bfs_distance(
                start,
                goal,
                adjacency,
                base_adjacency,
                physics_cache,
                level_data,
                mine_proximity_cache,
            )

    def _bfs_distance(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        adjacency: Dict,
        base_adjacency: Dict,
        physics_cache: Optional[Dict[Tuple[int, int], Dict[str, bool]]] = None,
        level_data: Optional[Any] = None,
        mine_proximity_cache: Optional[Any] = None,
    ) -> float:
        """BFS pathfinding - guaranteed shortest path with physics validation."""
        queue = deque([(start, 0.0)])
        visited = {start}
        parents = {start: None}  # Add parent tracking

        while queue:
            current, dist = queue.popleft()

            if current == goal:
                return dist

            # Explore neighbors from adjacency graph (masked for pathfinding)
            for neighbor, _ in adjacency.get(current, []):
                if neighbor not in visited:
                    # Check horizontal rule before accepting edge (uses base_adjacency)
                    if _violates_horizontal_rule(
                        current, neighbor, parents, base_adjacency
                    ):
                        continue

                    # Calculate physics-aware edge cost with parent for momentum checking (uses base_adjacency)
                    edge_cost = _calculate_physics_aware_cost(
                        current,
                        neighbor,
                        base_adjacency,
                        parents.get(current),
                        physics_cache,
                        level_data,
                        mine_proximity_cache,
                    )

                    visited.add(neighbor)
                    parents[neighbor] = current  # Track parent
                    queue.append((neighbor, dist + edge_cost))

        return float("inf")

    def _astar_distance(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        adjacency: Dict,
        base_adjacency: Dict,
        physics_cache: Optional[Dict[Tuple[int, int], Dict[str, bool]]] = None,
        level_data: Optional[Any] = None,
        mine_proximity_cache: Optional[Any] = None,
    ) -> float:
        """A* pathfinding - faster than BFS with heuristic and physics validation."""

        def manhattan_heuristic(pos: Tuple[int, int]) -> float:
            """Manhattan distance heuristic (admissible)."""
            return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

        # Priority queue: (f_score, g_score, position)
        open_set = [(manhattan_heuristic(start), 0.0, start)]
        g_score = {start: 0.0}
        visited = set()
        parents = {start: None}  # Add parent tracking

        while open_set:
            _, current_g, current = heapq.heappop(open_set)

            if current in visited:
                continue
            visited.add(current)

            if current == goal:
                return current_g

            # Explore neighbors from adjacency graph (masked for pathfinding)
            for neighbor, _ in adjacency.get(current, []):
                if neighbor in visited:
                    continue

                # Check horizontal rule before accepting edge (uses base_adjacency)
                if _violates_horizontal_rule(
                    current, neighbor, parents, base_adjacency
                ):
                    continue

                # Calculate physics-aware edge cost with parent for momentum checking (uses base_adjacency)
                edge_cost = _calculate_physics_aware_cost(
                    current,
                    neighbor,
                    base_adjacency,
                    parents.get(current),
                    physics_cache,
                    level_data,
                    mine_proximity_cache,
                )

                tentative_g = current_g + edge_cost

                if tentative_g < g_score.get(neighbor, float("inf")):
                    g_score[neighbor] = tentative_g
                    parents[neighbor] = current  # Track parent
                    f_score = tentative_g + manhattan_heuristic(neighbor)
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor))

        return float("inf")
