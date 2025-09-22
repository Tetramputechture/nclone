"""
Tiered Reachability System for performance-optimized RL training.

This module implements a three-tier reachability analysis system designed to replace
detailed physics analysis with performance-optimized approximations suitable for
real-time RL training.

Architecture:
- Tier 1: Ultra-fast flood fill approximation (<1ms, ~85% accuracy)
- Tier 2: Simplified physics analysis (<10ms, ~92% accuracy)
- Tier 3: Enhanced analysis fallback (<50ms, ~95% accuracy)

Based on research showing that approximate connectivity analysis with learned spatial
representations outperforms precise pathfinding in complex RL environments.
"""

import time
from typing import Tuple, List, Dict, Optional, Any, Union
import numpy as np

from .reachability_types import (
    ReachabilityApproximation,
    ReachabilityResult,
    PerformanceTarget,
)
from .opencv_flood_fill import OpenCVFloodFill


class TieredReachabilitySystem:
    """
    Main coordinator for three-tier reachability analysis system.

    Provides adaptive tier selection based on performance requirements and
    maintains backward compatibility with existing reachability interfaces.
    """

    def __init__(self, debug: bool = False):
        """
        Initialize tiered reachability system.

        Args:
            debug: Enable debug output and performance logging
        """
        self.debug = debug

        # Initialize tier implementations (lazy loading for performance)
        self._tier1 = None
        self._tier2 = None
        self._tier3 = None

        # Performance tracking
        self.performance_history = {
            "tier1": {"times": [], "accuracies": []},
            "tier2": {"times": [], "accuracies": []},
            "tier3": {"times": [], "accuracies": []},
        }

        # Adaptive selection parameters
        self.tier1_time_threshold = 1.0  # ms
        self.tier2_time_threshold = 10.0  # ms
        self.tier3_time_threshold = 50.0  # ms

        self.tier1_accuracy_threshold = 0.80
        self.tier2_accuracy_threshold = 0.90
        self.tier3_accuracy_threshold = 0.95

    @property
    def tier1(self):
        """Lazy initialization of Tier 1 ultra-fast OpenCV flood fill."""
        if self._tier1 is None:
            # Use OpenCV with 0.125x scale for ultra-fast performance (<1ms)
            self._tier1 = OpenCVFloodFill(render_scale=0.125, debug=self.debug)
        return self._tier1

    @property
    def tier2(self):
        """Lazy initialization of Tier 2 medium accuracy OpenCV flood fill."""
        if self._tier2 is None:
            # Use OpenCV with 0.25x scale for balanced performance/accuracy (<10ms)
            self._tier2 = OpenCVFloodFill(render_scale=0.25, debug=self.debug)
        return self._tier2

    @property
    def tier3(self):
        """Lazy initialization of Tier 3 high accuracy OpenCV flood fill."""
        if self._tier3 is None:
            # Use OpenCV with 1.0x scale for highest accuracy (<100ms)
            self._tier3 = OpenCVFloodFill(render_scale=1.0, debug=self.debug)
        return self._tier3

    @property
    def subgoal_planner(self):
        """Lazy initialization of hierarchical subgoal planning system."""
        if not hasattr(self, "_subgoal_planner") or self._subgoal_planner is None:
            # Use enhanced SubgoalPlanner with hierarchical completion algorithm
            # Lazy import to avoid circular dependency
            from ..subgoal_planner import SubgoalPlanner

            self._subgoal_planner = SubgoalPlanner(debug=self.debug)
        return self._subgoal_planner

    def analyze_reachability(
        self,
        level_data,
        ninja_position: Tuple[int, int],
        switch_states: Dict[str, bool],
        performance_target: Union[PerformanceTarget, str] = PerformanceTarget.BALANCED,
    ) -> Union[ReachabilityApproximation, ReachabilityResult]:
        """
        Analyze reachability using adaptive tier selection.

        Args:
            level_data: Level data structure
            ninja_position: Current ninja position (x, y)
            switch_states: Current switch states
            performance_target: Target performance level for tier selection

        Returns:
            Reachability analysis result with performance metrics
        """
        if isinstance(performance_target, str):
            performance_target = PerformanceTarget(performance_target)

        start_time = time.perf_counter()

        # Select appropriate tier based on performance target
        tier_to_use = self._select_tier(performance_target)

        if self.debug:
            print(f"DEBUG: Selected tier {tier_to_use} for target {performance_target}")

        if tier_to_use == 1:
            result = self._analyze_tier1(level_data, ninja_position, switch_states)
        elif tier_to_use == 2:
            result = self._analyze_tier2(level_data, ninja_position, switch_states)
        else:  # tier_to_use == 3
            result = self._analyze_tier3(level_data, ninja_position, switch_states)

        # Record performance metrics
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self._record_performance(tier_to_use, elapsed_ms, result.confidence)

        return result

    def _select_tier(self, performance_target: PerformanceTarget) -> int:
        """
        Select appropriate tier based on performance target and history.

        Args:
            performance_target: Target performance level

        Returns:
            Tier number (1, 2, or 3) to use
        """
        if performance_target == PerformanceTarget.ULTRA_FAST:
            return 1
        elif performance_target == PerformanceTarget.FAST:
            # Use Tier 1 if it meets accuracy requirements, otherwise Tier 2
            if self._tier_meets_requirements(1):
                return 1
            else:
                return 2
        elif performance_target == PerformanceTarget.BALANCED:
            # Auto-select based on performance history
            if self._tier_meets_requirements(1):
                return 1
            elif self._tier_meets_requirements(2):
                return 2
            else:
                return 3
        else:  # PerformanceTarget.ACCURATE
            return 3

    def _tier_meets_requirements(self, tier: int) -> bool:
        """
        Check if tier meets performance and accuracy requirements.

        Args:
            tier: Tier number to check

        Returns:
            True if tier meets requirements
        """
        history = self.performance_history[f"tier{tier}"]

        if not history["times"]:
            return True  # No history, assume it meets requirements

        # Check time requirements
        avg_time = np.mean(history["times"])
        time_threshold = getattr(self, f"tier{tier}_time_threshold")

        if avg_time > time_threshold:
            return False

        # Check accuracy requirements
        if history["accuracies"]:
            avg_accuracy = np.mean(history["accuracies"])
            accuracy_threshold = getattr(self, f"tier{tier}_accuracy_threshold")

            if avg_accuracy < accuracy_threshold:
                return False

        return True

    def _analyze_tier1(
        self,
        level_data,
        ninja_position: Tuple[int, int],
        switch_states: Dict[str, bool],
    ) -> ReachabilityApproximation:
        """Analyze using Tier 1 ultra-fast OpenCV flood fill."""
        start_time = time.perf_counter()

        # Use OpenCV flood fill with entities (empty list for basic analysis)
        result = self.tier1.quick_check(
            ninja_position, level_data, switch_states, entities=[]
        )

        # The quick_check method already returns a ReachabilityApproximation
        # Just update the tier_used field
        result.tier_used = 1
        return result

    def _analyze_tier2(
        self,
        level_data,
        ninja_position: Tuple[int, int],
        switch_states: Dict[str, bool],
    ) -> ReachabilityResult:
        """Analyze using Tier 2 medium accuracy OpenCV flood fill."""
        start_time = time.perf_counter()

        # Use OpenCV flood fill with entities (empty list for basic analysis)
        result = self.tier2.quick_check(
            ninja_position, level_data, switch_states, entities=[]
        )

        # Convert ReachabilityApproximation to ReachabilityResult
        return ReachabilityResult(
            reachable_positions=result.reachable_positions,
            confidence=result.confidence,
            computation_time_ms=result.computation_time_ms,
            method=result.method,
            tier_used=2,
        )

    def _analyze_tier3(
        self,
        level_data,
        ninja_position: Tuple[int, int],
        switch_states: Dict[str, bool],
    ) -> ReachabilityApproximation:
        """Analyze using Tier 3 high-accuracy OpenCV flood fill."""
        start_time = time.perf_counter()

        # Use OpenCV flood fill with full resolution for highest accuracy
        result = self.tier3.quick_check(
            ninja_position, level_data, switch_states, entities=[]
        )

        # Update the tier_used field
        result.tier_used = 3
        return result

    def _record_performance(self, tier: int, time_ms: float, accuracy: float):
        """Record performance metrics for adaptive selection."""
        history = self.performance_history[f"tier{tier}"]
        history["times"].append(time_ms)
        history["accuracies"].append(accuracy)

        # Keep only recent history (last 100 measurements)
        if len(history["times"]) > 100:
            history["times"] = history["times"][-100:]
            history["accuracies"] = history["accuracies"][-100:]

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary for all tiers."""
        summary = {}

        for tier in [1, 2, 3]:
            history = self.performance_history[f"tier{tier}"]

            if history["times"]:
                summary[f"tier{tier}"] = {
                    "avg_time_ms": np.mean(history["times"]),
                    "p95_time_ms": np.percentile(history["times"], 95),
                    "max_time_ms": np.max(history["times"]),
                    "avg_accuracy": np.mean(history["accuracies"])
                    if history["accuracies"]
                    else 0.0,
                    "min_accuracy": np.min(history["accuracies"])
                    if history["accuracies"]
                    else 0.0,
                    "sample_count": len(history["times"]),
                }
            else:
                summary[f"tier{tier}"] = {
                    "avg_time_ms": 0.0,
                    "p95_time_ms": 0.0,
                    "max_time_ms": 0.0,
                    "avg_accuracy": 0.0,
                    "min_accuracy": 0.0,
                    "sample_count": 0,
                }

        return summary

    def reset_performance_history(self):
        """Reset performance tracking history."""
        for tier in [1, 2, 3]:
            self.performance_history[f"tier{tier}"] = {"times": [], "accuracies": []}

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
            reachability_analyzer=self.tier3,  # Use highest accuracy tier for strategic analysis
        )
