"""
Ultra-Fast Reachability System for performance-optimized RL training.

This module implements a simplified flood fill reachability analysis system
using ultra-fast OpenCV approximation (<1ms) suitable for real-time RL training.

This simplification aligns with the principle that deep RL agents should learn
complex movement patterns emergently rather than having them pre-computed.
"""

import time
from typing import Tuple, List, Dict, Optional, Any
import numpy as np

from .reachability_types import ReachabilityApproximation
from .opencv_flood_fill import OpenCVFloodFill


class ReachabilitySystem:
    """
    Ultra-fast reachability analysis system using OpenCV flood fill.

    Provides ultra-fast flood fill approximation (<1ms) for real-time RL training.
    Uses connectivity-based analysis rather than complex physics calculations.
    """

    def __init__(self, debug: bool = False):
        """
        Initialize ultra-fast reachability system.

        Args:
            debug: Enable debug output and performance logging
        """
        self.debug = debug

        # Initialize flood fill implementation (lazy loading for performance)
        self._flood_fill = None

        # Performance tracking
        self.performance_history = {"times": [], "accuracies": []}

    @property
    def flood_fill(self):
        """Lazy initialization of ultra-fast OpenCV flood fill."""
        if self._flood_fill is None:
            # Use OpenCV with 0.125x scale for ultra-fast performance (<1ms)
            self._flood_fill = OpenCVFloodFill(render_scale=0.125, debug=self.debug)
        return self._flood_fill

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
        level_data,
        ninja_position: Tuple[int, int],
        switch_states: Dict[str, bool],
    ) -> ReachabilityApproximation:
        """
        Analyze reachability using ultra-fast flood fill approximation.

        Args:
            level_data: Level data structure
            ninja_position: Current ninja position (x, y)
            switch_states: Current switch states

        Returns:
            Ultra-fast reachability approximation result
        """
        start_time = time.perf_counter()

        if self.debug:
            print("DEBUG: Using ultra-fast flood fill analysis")

        # Use ultra-fast flood fill analysis
        entities = getattr(level_data, "entities", [])
        result = self.flood_fill.quick_check(
            ninja_position, level_data, switch_states, entities=entities
        )

        # Record performance metrics
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self._record_performance(elapsed_ms, result.confidence)

        return result

    def _record_performance(self, time_ms: float, accuracy: float):
        """Record performance metrics."""
        self.performance_history["times"].append(time_ms)
        self.performance_history["accuracies"].append(accuracy)

        # Keep only recent history (last 100 measurements)
        if len(self.performance_history["times"]) > 100:
            self.performance_history["times"] = self.performance_history["times"][-100:]
            self.performance_history["accuracies"] = self.performance_history[
                "accuracies"
            ][-100:]

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for ultra-fast reachability analysis."""
        history = self.performance_history

        if history["times"]:
            return {
                "avg_time_ms": float(np.mean(history["times"])),
                "p95_time_ms": float(np.percentile(history["times"], 95)),
                "max_time_ms": float(np.max(history["times"])),
                "min_time_ms": float(np.min(history["times"])),
                "avg_accuracy": float(np.mean(history["accuracies"]))
                if history["accuracies"]
                else 0.0,
                "min_accuracy": float(np.min(history["accuracies"]))
                if history["accuracies"]
                else 0.0,
                "max_accuracy": float(np.max(history["accuracies"]))
                if history["accuracies"]
                else 0.0,
                "sample_count": len(history["times"]),
            }
        else:
            return {
                "avg_time_ms": 0.0,
                "p95_time_ms": 0.0,
                "max_time_ms": 0.0,
                "min_time_ms": 0.0,
                "avg_accuracy": 0.0,
                "min_accuracy": 0.0,
                "max_accuracy": 0.0,
                "sample_count": 0,
            }

    def reset_performance_history(self):
        """Reset performance tracking history."""
        self.performance_history = {"times": [], "accuracies": []}

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
            reachability_analyzer=self.flood_fill,  # Use flood fill for strategic analysis
        )
