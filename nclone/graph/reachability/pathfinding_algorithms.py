"""
Core pathfinding algorithms for shortest path calculation.

Provides BFS and A* implementations for finding shortest paths on traversability graphs.
"""

from typing import Dict, Tuple, List, Optional, Any
from collections import deque
import heapq

# Import shared utilities from pathfinding_utils to avoid duplication
# All physics cost calculation logic is now centralized in pathfinding_utils.py
from .pathfinding_utils import (
    _is_node_grounded_util as _is_node_grounded,
    _is_horizontal_edge_util as _is_horizontal_edge,
    _violates_horizontal_rule_util as _violates_horizontal_rule,
    _get_aerial_chain_multiplier,
    _calculate_mine_proximity_cost,
    _calculate_physics_aware_cost,
)


# Re-export for backward compatibility
__all__ = [
    "_is_node_grounded",
    "_is_horizontal_edge",
    "_violates_horizontal_rule",
    "_get_aerial_chain_multiplier",
    "_calculate_mine_proximity_cost",
    "_calculate_physics_aware_cost",
]


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
        mine_sdf: Optional[Any] = None,
        hazard_cost_multiplier: Optional[float] = None,
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
            mine_sdf: Optional MineSignedDistanceField for velocity-aware hazard costs
            hazard_cost_multiplier: Optional curriculum-adaptive mine hazard cost multiplier

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
                mine_sdf,
                hazard_cost_multiplier,
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
                mine_sdf,
                hazard_cost_multiplier,
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
        mine_sdf: Optional[Any] = None,
        hazard_cost_multiplier: Optional[float] = None,
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

            # OPTIMIZATION: Cache parent/grandparent lookups to avoid repeated dict.get()
            current_parent = parents.get(current)
            current_grandparent = grandparents.get(current)

            # Explore neighbors from adjacency graph (masked for pathfinding)
            for neighbor, _ in adjacency.get(current, []):
                if neighbor not in visited:
                    # Check horizontal rule before accepting edge (uses base_adjacency + physics_cache)
                    # OPTIMIZATION: Pass physics_cache for O(1) grounding checks
                    if _violates_horizontal_rule(
                        current, neighbor, parents, base_adjacency, physics_cache
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
                    # OPTIMIZATION: Pass cached parent/grandparent
                    edge_cost = _calculate_physics_aware_cost(
                        current,
                        neighbor,
                        base_adjacency,
                        current_parent,
                        physics_cache,
                        level_data,
                        mine_proximity_cache,
                        current_chain,  # Pass current chain count for cost calculation
                        current_grandparent,  # Pass cached grandparent
                        mine_sdf,  # Pass SDF for velocity-aware mine costs
                        hazard_cost_multiplier,  # Curriculum-adaptive mine costs
                    )

                    visited.add(neighbor)
                    parents[neighbor] = current  # Track parent
                    grandparents[neighbor] = current_parent  # Track cached grandparent
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
        mine_sdf: Optional[Any] = None,
        hazard_cost_multiplier: Optional[float] = None,
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

            # OPTIMIZATION: Cache parent/grandparent lookups to avoid repeated dict.get()
            current_parent = parents.get(current)
            current_grandparent = grandparents.get(current)

            # Explore neighbors from adjacency graph (masked for pathfinding)
            for neighbor, _ in adjacency.get(current, []):
                if neighbor in visited:
                    continue

                # Check horizontal rule before accepting edge (uses base_adjacency + physics_cache)
                # OPTIMIZATION: Pass physics_cache for O(1) grounding checks
                if _violates_horizontal_rule(
                    current, neighbor, parents, base_adjacency, physics_cache
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
                # OPTIMIZATION: Pass cached parent/grandparent
                edge_cost = _calculate_physics_aware_cost(
                    current,
                    neighbor,
                    base_adjacency,
                    current_parent,
                    physics_cache,
                    level_data,
                    mine_proximity_cache,
                    current_chain,  # Pass current chain count for cost calculation
                    current_grandparent,  # Pass cached grandparent
                    mine_sdf,  # Pass SDF for velocity-aware mine costs
                    hazard_cost_multiplier,  # Curriculum-adaptive mine costs
                )

                tentative_g = current_g + edge_cost

                # OPTIMIZATION: Direct comparison instead of .get() with default
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    parents[neighbor] = current  # Track parent
                    grandparents[neighbor] = current_parent  # Track cached grandparent
                    aerial_chains[neighbor] = new_chain  # Track chain count
                    f_score = tentative_g + manhattan_heuristic(neighbor)
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor))

        return float("inf")
