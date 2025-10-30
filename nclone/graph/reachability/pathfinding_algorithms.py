"""
Core pathfinding algorithms for shortest path calculation.

Provides BFS and A* implementations for finding shortest paths on traversability graphs.
"""

from typing import Dict, Tuple, List
from collections import deque
import heapq


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
    ) -> float:
        """
        Calculate shortest navigable path distance.

        Args:
            start: Starting position (x, y) in pixels
            goal: Goal position (x, y) in pixels
            adjacency: Graph adjacency structure

        Returns:
            Shortest path distance in pixels, or float('inf') if unreachable
        """
        # Quick checks
        if start not in adjacency or goal not in adjacency:
            return float("inf")
        if start == goal:
            return 0.0

        # Choose algorithm
        if self.use_astar:
            return self._astar_distance(start, goal, adjacency)
        else:
            return self._bfs_distance(start, goal, adjacency)

    def _bfs_distance(
        self, start: Tuple[int, int], goal: Tuple[int, int], adjacency: Dict
    ) -> float:
        """BFS pathfinding - guaranteed shortest path."""
        queue = deque([(start, 0.0)])
        visited = {start}

        while queue:
            current, dist = queue.popleft()

            if current == goal:
                return dist

            # Explore neighbors from adjacency graph
            for neighbor, edge_cost in adjacency.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + edge_cost))

        return float("inf")

    def _astar_distance(
        self, start: Tuple[int, int], goal: Tuple[int, int], adjacency: Dict
    ) -> float:
        """A* pathfinding - faster than BFS with heuristic."""

        def manhattan_heuristic(pos: Tuple[int, int]) -> float:
            """Manhattan distance heuristic (admissible)."""
            return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

        # Priority queue: (f_score, g_score, position)
        open_set = [(manhattan_heuristic(start), 0.0, start)]
        g_score = {start: 0.0}
        visited = set()

        while open_set:
            _, current_g, current = heapq.heappop(open_set)

            if current in visited:
                continue
            visited.add(current)

            if current == goal:
                return current_g

            # Explore neighbors
            for neighbor, edge_cost in adjacency.get(current, []):
                if neighbor in visited:
                    continue

                tentative_g = current_g + edge_cost

                if tentative_g < g_score.get(neighbor, float("inf")):
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + manhattan_heuristic(neighbor)
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor))

        return float("inf")

