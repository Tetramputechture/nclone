"""
Core pathfinding algorithms for shortest path calculation.

Provides BFS and A* implementations for finding shortest paths on traversability graphs.
"""

from typing import Dict, Tuple, List, Optional, Any
from collections import deque
import heapq


# Constants
SUB_NODE_SIZE = 12  # Sub-node spacing in pixels

# Aerial upward movement cost constants
# N++ physics: A jump from ground can realistically cover ~2-3 sub-nodes (24-36px)
# of upward travel before gravity and momentum make further upward movement
# increasingly difficult. We use a tight threshold to prevent impossible paths.
AERIAL_UPWARD_CHAIN_THRESHOLD = 2  # Max chain before applying blocking cost
AERIAL_UPWARD_BLOCKING_COST = 100.0  # High cost for moves beyond threshold
AERIAL_UPWARD_BASE_MULTIPLIER = 3.0  # Base multiplier for aerial upward (not 1.0)

# Momentum-aware pathfinding cost constants
# These multipliers adjust edge costs based on whether movement continues or reverses momentum
MOMENTUM_CONTINUE_MULTIPLIER = 0.7  # Cheaper to maintain momentum (30% discount)
MOMENTUM_REVERSE_MULTIPLIER = 2.5  # Expensive to reverse direction (2.5x penalty)
MOMENTUM_BUILDING_THRESHOLD = (
    12  # Min horizontal displacement (1 sub-node) to have momentum
)


def _get_aerial_chain_multiplier(chain_count: int) -> float:
    """Multiplicative cost scaling for consecutive aerial upward moves.

    Models N++ jump physics where upward movement in air is valid while
    continuing a jump trajectory, but becomes quickly expensive to prevent
    impossible long-distance aerial paths.

    Chain 0: First aerial upward - 3x (immediate penalty for leaving ground/wall)
    Chain 1: Second aerial upward - 9x (3^2)
    Chain 2: Third aerial upward - 27x (3^3)
    Chain 3+: Beyond physics limits - apply blocking cost (100x base)

    Args:
        chain_count: Number of consecutive aerial upward moves

    Returns:
        Cost multiplier reflecting physics feasibility
    """
    if chain_count <= AERIAL_UPWARD_CHAIN_THRESHOLD:
        # Within jump physics limits - exponential increase with base 3
        # Chain 0: 3, Chain 1: 9, Chain 2: 27
        return AERIAL_UPWARD_BASE_MULTIPLIER ** (chain_count + 1)
    else:
        # Beyond physics limits - apply blocking cost with continued scaling
        # Chain 3: 100 * 3 = 300, Chain 4: 100 * 9 = 900, etc.
        excess_chain = chain_count - AERIAL_UPWARD_CHAIN_THRESHOLD
        return AERIAL_UPWARD_BLOCKING_COST * (
            AERIAL_UPWARD_BASE_MULTIPLIER**excess_chain
        )


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


def _infer_momentum_direction(
    parent_pos: Optional[Tuple[int, int]],
    current_pos: Tuple[int, int],
    grandparent_pos: Optional[Tuple[int, int]],
) -> int:
    """Infer horizontal momentum direction from recent trajectory.

    Analyzes the last two moves to determine if the agent has built momentum
    in a consistent horizontal direction. This enables momentum-aware cost
    adjustments that make momentum-preserving paths cheaper.

    Args:
        parent_pos: Position of parent node (one step back)
        current_pos: Position of current node
        grandparent_pos: Position of grandparent node (two steps back)

    Returns:
        int: -1 (leftward momentum), 0 (no/stationary), +1 (rightward momentum)
    """
    if parent_pos is None or grandparent_pos is None:
        return 0  # No momentum without trajectory history

    # Calculate horizontal displacement for last two moves
    recent_dx = current_pos[0] - parent_pos[0]
    prev_dx = parent_pos[0] - grandparent_pos[0]

    # Check if moving consistently in same horizontal direction
    # Both moves must be in same direction and exceed threshold
    if recent_dx * prev_dx > 0 and abs(recent_dx) >= MOMENTUM_BUILDING_THRESHOLD:
        # Consistent movement in same direction = momentum
        return -1 if recent_dx < 0 else 1

    return 0  # No consistent momentum


def _calculate_momentum_multiplier(
    momentum_direction: int,
    edge_dx: int,
) -> float:
    """Calculate cost multiplier based on momentum preservation or reversal.

    Momentum-aware pathfinding recognizes that in N++ physics:
    - Continuing movement in the same direction is efficient (momentum preserved)
    - Reversing direction wastes built-up momentum (expensive)
    - Building momentum for future jumps is a valid strategy

    This makes paths that temporarily move away from goal to build momentum
    cheaper than naive direct paths, solving the key issue with momentum-dependent
    navigation (e.g., running left to build speed before jumping right over mines).

    Args:
        momentum_direction: Current momentum (-1 = left, 0 = none, +1 = right)
        edge_dx: Horizontal displacement of the edge being evaluated

    Returns:
        float: Cost multiplier (0.7 = cheaper, 1.0 = neutral, 2.5 = expensive)
    """
    if momentum_direction == 0 or edge_dx == 0:
        return 1.0  # No momentum or vertical edge = neutral cost

    # Check if edge continues or reverses momentum
    continues_momentum = (momentum_direction * edge_dx) > 0

    if continues_momentum:
        # Preserving momentum: make this path cheaper (30% discount)
        # This rewards paths that build and maintain speed
        return MOMENTUM_CONTINUE_MULTIPLIER
    else:
        # Reversing momentum: make this path expensive (2.5x penalty)
        # This discourages wasteful zigzag patterns
        return MOMENTUM_REVERSE_MULTIPLIER


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

    # Apply cost based on proximity using QUADRATIC falloff
    # Quadratic is gentler at radius edge, steeper near mines
    # This provides better hazard avoidance without excessive path detours
    if min_distance < MINE_HAZARD_RADIUS:
        # Quadratic interpolation (changed from linear per plan Phase 6.1):
        # - At mine center (distance=0): full multiplier
        # - At radius edge (distance=RADIUS): 1.0 (no penalty)
        # - Quadratic curve: gentler near edge, steeper near center
        proximity_factor = 1.0 - (min_distance / MINE_HAZARD_RADIUS)
        # Apply quadratic scaling for more aggressive close-range avoidance
        multiplier = 1.0 + (proximity_factor**2) * (MINE_HAZARD_COST_MULTIPLIER - 1.0)
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
    aerial_upward_chain: int = 0,
    grandparent_pos: Optional[Tuple[int, int]] = None,
) -> float:
    """
    Calculate edge cost based on N++ movement physics, momentum, and mine hazard proximity.

    Movement costs reflect actual N++ physics (from sim_mechanics_doc.md):
    - Grounded horizontal: FAST (ground accel 0.0667, max speed 3.333)
    - Air horizontal: SLOW (air accel 0.0444, ~33% slower)
    - Vertical up: EXPENSIVE (requires jump, fights gravity 0.0667)
    - Vertical down: CHEAP (gravity assists 0.0667)
    - Diagonal movement: Combines horizontal and vertical physics
    - Momentum preservation: Continuing in same X direction is cheaper than changing

    Momentum-aware cost adjustments (NEW):
    - Tracks horizontal momentum from trajectory (requires grandparent_pos)
    - Paths that preserve momentum get 30% discount (0.7x multiplier)
    - Paths that reverse momentum get 2.5x penalty
    - Makes momentum-building "detours" cheaper than naive direct paths
    - Critical for levels requiring backtracking to build speed (e.g., long jumps)

    Mid-air jump prevention:
    - Upward movement while airborne (not grounded, not walled) is blocked with very high cost
    - Consecutive aerial upward moves accumulate multiplicative cost penalties
    - This reflects N++ physics where mid-air jumps are impossible

    Hazard avoidance:
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
        aerial_upward_chain: Number of consecutive aerial upward moves in current path
        grandparent_pos: Optional grandparent position for momentum inference

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

    # Check if Y direction is changing from falling to rising (physically impossible in air)
    # In N++, once you're falling (dy > 0), you can't start rising (dy < 0) without
    # touching ground or wall - gravity always pulls you down
    y_direction_change_to_rising = False
    if parent_pos is not None:
        parent_dy = src_pos[1] - parent_pos[1]
        # Was falling or stationary (parent_dy >= 0) and now moving up (dy < 0)
        if parent_dy >= 0 and dy < 0:
            y_direction_change_to_rising = True

    src_physics = physics_cache[src_pos]
    dst_physics = physics_cache[dst_pos]
    src_grounded = src_physics["grounded"]
    dst_grounded = dst_physics["grounded"]
    src_walled = src_physics["walled"]

    # Base geometric cost (Euclidean distance)
    if dx != 0 and dy != 0:
        base_cost = 1.414  # sqrt(2) for diagonal
    else:
        base_cost = 1.0  # Unit cost for cardinal directions

    # Physics multipliers based on movement type
    # Screen coordinates: y increases downward

    # Special case: Diagonal falling from air (prefer over horizontal air movement)
    if dx != 0 and dy > 0 and not src_grounded and not src_walled:
        if x_direction_change:
            # Changing X direction while falling in mid-air is nearly impossible
            # In N++, you have very limited air control and can't reverse direction
            # This prevents impossible zigzag paths through the air
            multiplier = 100.0
        else:
            # Diagonal downward continuing same direction - efficient (gravity + momentum)
            multiplier = 0.6
    # Special case: Diagonal upward wall-assisted move (valid N++ mechanic)
    elif dx != 0 and dy < 0 and not src_grounded and src_walled:
        # Wall-assisted diagonal upward (wall-jump) - valid but more expensive than ground
        if x_direction_change:
            # Changing X direction while wall jumping is expensive
            # Cost: 1.414 × 1.5 = 2.12
            multiplier = 1.5
        else:
            # Continuing same X direction - more efficient wall jump
            # Cost: 1.414 × 1.0 = 1.414
            multiplier = 1.0
    # Special case: Diagonal upward from ground (efficient N++ movement)
    elif dx != 0 and dy < 0 and src_grounded:
        # Grounded diagonal jump - efficient upward movement
        # Cost: 1.414 × 0.6 = 0.85
        multiplier = 0.6
    # Special case: Diagonal upward from air without wall (aerial jump continuation)
    elif dx != 0 and dy < 0 and not src_grounded and not src_walled:
        if y_direction_change_to_rising:
            # Transitioning from falling/stationary to rising in mid-air is IMPOSSIBLE
            # Can't reverse vertical momentum without ground or wall contact
            # This prevents zigzag paths that go down then up in the air
            multiplier = 200.0
        elif x_direction_change:
            # Changing X direction while moving upward in mid-air is nearly impossible
            # You can't reverse horizontal momentum without wall or ground contact
            multiplier = 100.0
        else:
            # Aerial upward movement - valid as jump continuation, cost scales with chain
            # Chain 0-2: Reasonable cost (continuing jump trajectory)
            # Chain 3+: Very expensive (beyond physics limits, effectively blocked)
            multiplier = _get_aerial_chain_multiplier(aerial_upward_chain)
    elif dy < 0:  # Moving up (against gravity) - vertical only
        if src_grounded:
            # Vertical jump from ground - reasonable but not free
            # Cost: 1.0 × 0.7 = 0.7
            multiplier = 0.7
        elif src_walled:
            # Wall jump (vertical) - more expensive than ground jump
            # Cost: 1.0 × 1.2 = 1.2
            multiplier = 1.2
        elif y_direction_change_to_rising:
            # Transitioning from falling/stationary to rising in mid-air is IMPOSSIBLE
            # Can't reverse vertical momentum without ground or wall contact
            multiplier = 200.0
        else:
            # Vertical upward from air without wall (aerial jump continuation)
            # Same chain-based cost as diagonal aerial upward
            multiplier = _get_aerial_chain_multiplier(aerial_upward_chain)
    elif dy > 0:  # Moving down (with gravity) - vertical or grounded diagonal
        # Gravity assists falling (0.0667 pixels/frame²)
        # Cheap regardless of grounding state
        multiplier = 0.5
    else:  # Horizontal (dy == 0)
        if src_grounded and dst_grounded:
            # Grounded horizontal - FASTEST and CHEAPEST movement
            # Ground accel 0.0667, max speed 3.333 pixels/frame
            # This is the most efficient movement in N++
            # Base cost: 1.0 × 0.15 = 0.15
            multiplier = 0.15
        elif x_direction_change and not src_grounded and not src_walled:
            # Changing horizontal direction in mid-air is nearly impossible
            # Very limited air control in N++
            multiplier = 100.0
        else:
            # Air horizontal - more expensive than diagonal falling
            # Air accel 0.0444 (~33% slower than ground)
            # Higher cost to prefer diagonal falling over horizontal air movement
            multiplier = 40.0

    # Apply momentum-aware cost adjustment for grounded horizontal movement
    # This makes paths that build and preserve momentum cheaper, enabling
    # momentum-dependent strategies (e.g., running left to build speed before jumping right)
    momentum_multiplier = 1.0
    if dy == 0 and src_grounded and dst_grounded and grandparent_pos is not None:
        # Infer momentum from recent trajectory
        momentum_direction = _infer_momentum_direction(
            parent_pos, src_pos, grandparent_pos
        )
        # Calculate multiplier based on whether edge continues or reverses momentum
        momentum_multiplier = _calculate_momentum_multiplier(momentum_direction, dx)

    # Apply mine hazard proximity cost multiplier
    # This makes paths near deadly mines more expensive, guiding PBRS toward safer routes
    # Uses cache for O(1) lookup when available
    mine_multiplier = _calculate_mine_proximity_cost(
        dst_pos, level_data, mine_proximity_cache
    )

    return base_cost * multiplier * momentum_multiplier * mine_multiplier


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
        """BFS pathfinding - guaranteed shortest path with physics and momentum validation."""
        queue = deque([(start, 0.0)])
        visited = {start}
        parents = {start: None}  # Track parents for momentum inference
        grandparents = {start: None}  # Track grandparents for momentum inference
        aerial_chains = {start: 0}  # Track consecutive aerial upward moves

        while queue:
            current, dist = queue.popleft()

            if current == goal:
                return dist

            # Get current node's physics properties for chain tracking
            current_physics = physics_cache[current]
            current_grounded = current_physics["grounded"]
            current_walled = current_physics["walled"]
            current_chain = aerial_chains.get(current, 0)

            # Explore neighbors from adjacency graph (masked for pathfinding)
            for neighbor, _ in adjacency.get(current, []):
                if neighbor not in visited:
                    # Check horizontal rule before accepting edge (uses base_adjacency)
                    if _violates_horizontal_rule(
                        current, neighbor, parents, base_adjacency
                    ):
                        continue

                    # Determine if this edge is aerial upward (for chain tracking)
                    dy = neighbor[1] - current[1]
                    is_aerial_upward = (
                        not current_grounded and not current_walled and dy < 0
                    )

                    # Calculate new chain count: increment for aerial upward, reset otherwise
                    new_chain = current_chain + 1 if is_aerial_upward else 0

                    # Calculate physics-aware edge cost with momentum tracking
                    edge_cost = _calculate_physics_aware_cost(
                        current,
                        neighbor,
                        base_adjacency,
                        parents.get(current),
                        physics_cache,
                        level_data,
                        mine_proximity_cache,
                        current_chain,  # Pass current chain count for cost calculation
                        grandparents.get(
                            current
                        ),  # Pass grandparent for momentum inference
                    )

                    visited.add(neighbor)
                    parents[neighbor] = current  # Track parent
                    grandparents[neighbor] = parents.get(current)  # Track grandparent
                    aerial_chains[neighbor] = new_chain  # Track chain count
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
        """A* pathfinding - faster than BFS with heuristic, physics, and momentum validation."""

        def manhattan_heuristic(pos: Tuple[int, int]) -> float:
            """Manhattan distance heuristic (admissible)."""
            return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

        # Priority queue: (f_score, g_score, position)
        open_set = [(manhattan_heuristic(start), 0.0, start)]
        g_score = {start: 0.0}
        visited = set()
        parents = {start: None}  # Track parents for momentum inference
        grandparents = {start: None}  # Track grandparents for momentum inference
        aerial_chains = {start: 0}  # Track consecutive aerial upward moves

        while open_set:
            _, current_g, current = heapq.heappop(open_set)

            if current in visited:
                continue
            visited.add(current)

            if current == goal:
                return current_g

            # Get current node's physics properties for chain tracking
            current_physics = physics_cache[current]
            current_grounded = current_physics["grounded"]
            current_walled = current_physics["walled"]
            current_chain = aerial_chains.get(current, 0)

            # Explore neighbors from adjacency graph (masked for pathfinding)
            for neighbor, _ in adjacency.get(current, []):
                if neighbor in visited:
                    continue

                # Check horizontal rule before accepting edge (uses base_adjacency)
                if _violates_horizontal_rule(
                    current, neighbor, parents, base_adjacency
                ):
                    continue

                # Determine if this edge is aerial upward (for chain tracking)
                dy = neighbor[1] - current[1]
                is_aerial_upward = (
                    not current_grounded and not current_walled and dy < 0
                )

                # Calculate new chain count: increment for aerial upward, reset otherwise
                new_chain = current_chain + 1 if is_aerial_upward else 0

                # Calculate physics-aware edge cost with momentum tracking
                edge_cost = _calculate_physics_aware_cost(
                    current,
                    neighbor,
                    base_adjacency,
                    parents.get(current),
                    physics_cache,
                    level_data,
                    mine_proximity_cache,
                    current_chain,  # Pass current chain count for cost calculation
                    grandparents.get(
                        current
                    ),  # Pass grandparent for momentum inference
                )

                tentative_g = current_g + edge_cost

                if tentative_g < g_score.get(neighbor, float("inf")):
                    g_score[neighbor] = tentative_g
                    parents[neighbor] = current  # Track parent
                    grandparents[neighbor] = parents.get(current)  # Track grandparent
                    aerial_chains[neighbor] = new_chain  # Track chain count
                    f_score = tentative_g + manhattan_heuristic(neighbor)
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor))

        return float("inf")
