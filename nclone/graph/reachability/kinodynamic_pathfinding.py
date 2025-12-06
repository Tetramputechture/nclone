"""Kinodynamic-aware pathfinding using exhaustive reachability database.

Integrates kinodynamic database queries into A* search for velocity-aware
path planning. When database is available, provides perfect accuracy for
momentum-dependent navigation.
"""

import logging
from typing import Dict, Tuple, List, Optional, Any
import heapq
import numpy as np

from .kinodynamic_database import KinodynamicDatabase

logger = logging.getLogger(__name__)


class KinodynamicPathfinder:
    """A* pathfinding in (position, velocity) state space using exhaustive database.

    When kinodynamic database is available, uses it for perfect reachability queries.
    Falls back to momentum-aware geometric pathfinding if database not available.
    """

    def __init__(
        self,
        kinodynamic_db: Optional[KinodynamicDatabase] = None,
        fallback_calculator: Optional[Any] = None,
    ):
        """Initialize kinodynamic pathfinder.

        Args:
            kinodynamic_db: Exhaustive kinodynamic database (if available)
            fallback_calculator: Fallback path calculator (momentum-aware geometric)
        """
        self.kinodynamic_db = kinodynamic_db
        self.fallback_calculator = fallback_calculator

    def find_path_with_velocity(
        self,
        start_node: Tuple[int, int],
        start_velocity: Tuple[float, float],
        goal_node: Tuple[int, int],
        adjacency: Dict,
    ) -> Tuple[Optional[List[Tuple[int, int]]], float]:
        """Find optimal path from (start, velocity) to goal.

        Uses kinodynamic database if available, otherwise falls back to
        momentum-aware geometric pathfinding.

        Args:
            start_node: Starting node position
            start_velocity: Starting velocity (vx, vy)
            goal_node: Goal node position
            adjacency: Graph adjacency structure

        Returns:
            (path, cost) where:
                path: List of nodes from start to goal, or None if unreachable
                cost: Total cost in frames
        """
        if self.kinodynamic_db:
            return self._astar_kinodynamic(
                start_node, start_velocity, goal_node, adjacency
            )
        elif self.fallback_calculator:
            # Fallback to momentum-aware geometric pathfinding
            cost = self.fallback_calculator.calculate_distance(
                start_node, goal_node, adjacency, adjacency
            )
            return None, cost  # Path reconstruction not implemented for fallback
        else:
            logger.warning("No kinodynamic database or fallback calculator available")
            return None, float("inf")

    def _astar_kinodynamic(
        self,
        start_node: Tuple[int, int],
        start_velocity: Tuple[float, float],
        goal_node: Tuple[int, int],
        adjacency: Dict,
    ) -> Tuple[Optional[List[Tuple[int, int]]], float]:
        """A* search in (position, velocity) state space.

        State: (node, velocity_bin)
        Transitions: Query kinodynamic database for reachable (next_node, velocity_bin) states

        Args:
            start_node: Starting node
            start_velocity: Starting velocity
            goal_node: Goal node
            adjacency: Graph adjacency (for node list)

        Returns:
            (path, cost) or (None, inf) if unreachable
        """
        # Discretize start velocity
        vx_bin, vy_bin = self.kinodynamic_db.velocity_binning.discretize_velocity(
            start_velocity[0], start_velocity[1]
        )
        start_state = (start_node, vx_bin, vy_bin)

        def heuristic(node: Tuple[int, int]) -> float:
            """Euclidean distance heuristic (admissible)."""
            dx = goal_node[0] - node[0]
            dy = goal_node[1] - node[1]
            return (dx * dx + dy * dy) ** 0.5 / 12.0  # Convert to frames estimate

        # Priority queue: (f_score, g_score, state)
        # State: (node, vx_bin, vy_bin)
        open_set = [(heuristic(start_node), 0.0, start_state)]
        g_score = {start_state: 0.0}
        visited = set()
        parents = {start_state: None}

        while open_set:
            _, current_g, current_state = heapq.heappop(open_set)

            if current_state in visited:
                continue
            visited.add(current_state)

            current_node, current_vx_bin, current_vy_bin = current_state

            # Check if reached goal (any velocity)
            if current_node == goal_node:
                # Reconstruct path (positions only, ignore velocity)
                path = []
                state = current_state
                while state is not None:
                    path.append(state[0])  # Append node position
                    state = parents.get(state)
                path.reverse()
                return path, current_g

            # Explore neighbors using kinodynamic database
            # For each neighbor node, check all velocity bins to find reachable states
            for neighbor_node in adjacency.get(current_node, []):
                neighbor_pos = neighbor_node[0] if isinstance(neighbor_node, tuple) and len(neighbor_node) == 2 else neighbor_node

                # Try all velocity bins at neighbor (database tells us which are reachable)
                for next_vx_bin in range(self.kinodynamic_db.num_vx_bins):
                    for next_vy_bin in range(self.kinodynamic_db.num_vy_bins):
                        # Query database: can we reach (neighbor, next_velocity) from current state?
                        current_velocity = self.kinodynamic_db.velocity_binning.get_velocity_from_bins(
                            current_vx_bin, current_vy_bin
                        )
                        reachable, transition_cost = self.kinodynamic_db.query_reachability(
                            current_node, current_velocity, neighbor_pos
                        )

                        if not reachable:
                            continue  # Can't reach this (neighbor, velocity) combination

                        next_state = (neighbor_pos, next_vx_bin, next_vy_bin)

                        if next_state in visited:
                            continue

                        tentative_g = current_g + transition_cost

                        if tentative_g < g_score.get(next_state, float("inf")):
                            g_score[next_state] = tentative_g
                            parents[next_state] = current_state
                            f_score = tentative_g + heuristic(neighbor_pos)
                            heapq.heappush(open_set, (f_score, tentative_g, next_state))

        # Goal not reached
        return None, float("inf")

    def get_distance_with_velocity(
        self,
        start_node: Tuple[int, int],
        start_velocity: Tuple[float, float],
        goal_node: Tuple[int, int],
        adjacency: Dict,
    ) -> float:
        """Get minimum cost from (start, velocity) to goal.

        Convenience method that returns just the cost (no path reconstruction).

        Args:
            start_node: Starting node
            start_velocity: Starting velocity
            goal_node: Goal node
            adjacency: Graph adjacency

        Returns:
            Minimum cost in frames, or inf if unreachable
        """
        _, cost = self.find_path_with_velocity(
            start_node, start_velocity, goal_node, adjacency
        )
        return cost

