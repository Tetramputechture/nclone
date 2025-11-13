"""
Graph-Based Reachability System for performance-optimized RL training.

This module implements reachability analysis using graph-based flood fill
from the adjacency graph, providing exact reachability without rendering overhead.

This approach is consistent with PBRS surface area calculation and eliminates
the deprecated OpenCV-based flood fill dependency.
"""

from typing import Tuple, List, Dict, Optional, Any

from .reachability_types import ReachabilityResult


class ReachabilitySystem:
    """
    Graph-based reachability analysis system.

    Provides exact reachability calculation using graph flood fill from adjacency.
    Uses the same approach as PBRS surface area calculation for consistency.
    """

    def __init__(self, debug: bool = False):
        """
        Initialize graph-based reachability system.

        Args:
            debug: Enable debug output and performance logging
        """
        self.debug = debug

    @property
    def subgoal_planner(self):
        """Lazy initialization of hierarchical subgoal planning system."""
        if not hasattr(self, "_subgoal_planner") or self._subgoal_planner is None:
            # Use LevelCompletionPlanner with hierarchical completion algorithm
            # Lazy import to avoid circular dependency
            from ...planning import LevelCompletionPlanner

            self._subgoal_planner = LevelCompletionPlanner()
        return self._subgoal_planner

    def analyze_reachability(
        self,
        adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
        graph_data: Dict[str, Any],
        ninja_position: Tuple[int, int],
    ) -> ReachabilityResult:
        """
        Analyze reachability using graph-based flood fill.

        Following the PBRS pattern: assumes graph is already built by GraphMixin,
        avoiding redundant graph construction. Uses the existing adjacency to
        perform flood fill from the ninja position.

        Args:
            adjacency: Pre-built graph adjacency structure from GraphMixin
            graph_data: Graph data dict with spatial_hash and other metadata
            ninja_position: Current ninja position (x, y)

        Returns:
            Exact graph-based reachability result
        """
        # Use graph data already built by GraphMixin - avoid redundant graph building
        # If 'reachable' is already in graph_data from GraphBuilder.build_graph(),
        # use it directly. Otherwise, perform flood fill.
        if "reachable" in graph_data:
            reachable_positions = graph_data["reachable"]
        else:
            # Perform flood fill using the existing adjacency graph
            from ...gym_environment.reward_calculation.pbrs_potentials import (
                _flood_fill_reachable_nodes,
            )

            reachable_positions = _flood_fill_reachable_nodes(
                ninja_position, adjacency, graph_data
            )

        return ReachabilityResult(
            reachable_positions=reachable_positions,
        )

    def create_hierarchical_completion_plan(
        self,
        ninja_position: Tuple[int, int],
        level_data,
        entities: List[Any],
        switch_states: Optional[Dict[str, bool]] = None,
    ):
        """
        Create hierarchical completion plan using enhanced SubgoalPlanner.

        This method provides the strategic completion analysis required for Deep RL
        by analyzing switch-door dependencies and creating optimal completion sequences.

        Args:
            ninja_position: Current ninja position (x, y)
            level_data: Level tile data
            entities: List of entities in the level
            switch_states: Current state of switches (activated/not activated)

        Returns:
            SubgoalPlan with hierarchical completion strategy, or None if impossible
        """
        # Use the enhanced SubgoalPlanner with hierarchical completion algorithm
        return self.subgoal_planner.create_hierarchical_completion_plan(
            ninja_position=ninja_position,
            level_data=level_data,
            entities=entities,
            switch_states=switch_states,
            reachability_analyzer=self,  # Pass self for reachability analysis
        )
