"""Potential-Based Reward Shaping (PBRS) potential functions.

This module implements reusable potential functions Φ(s) for reward shaping
following the theory of Ng, Harada, and Russell (1999): "Policy Invariance
Under Reward Transformations: Theory and Application to Reward Shaping".

PBRS provides dense reward signals without changing the optimal policy by using
the formula: F(s,s') = γ * Φ(s') - Φ(s)

Key properties:
- Policy invariance: Optimal policy unchanged under PBRS
- Dense rewards: Provides gradient at every step
- Normalization: All potentials normalized to [0, 1] range
- Composability: Multiple potentials can be combined with weights

All constants defined in reward_constants.py with full documentation.
"""

import logging
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from .reward_constants import (
    PBRS_SWITCH_DISTANCE_SCALE,
    PBRS_EXIT_DISTANCE_SCALE,
)

logger = logging.getLogger(__name__)

# Sub-node size for surface area calculations (from graph builder)
# The graph builder creates a 2x2 grid of sub-nodes per 24px tile
SUB_NODE_SIZE = 12  # pixels per sub-node
PLAYER_RADIUS = 10  # Player collision radius in pixels (from graph_builder)


class PBRSPotentials:
    """Collection of potential functions for reward shaping.

    Each potential function Φ(s) represents a heuristic estimate of state value.
    Potentials are normalized to [0, 1] range for consistent scaling.

    Available potentials:
    - objective_distance_potential: Distance to current objective (switch/exit)
    - hazard_proximity_potential: Proximity to dangerous hazards
    - impact_risk_potential: Risk of high-velocity collisions
    - exploration_potential: Novelty of current state
    """

    @staticmethod
    def objective_distance_potential(
        state: Dict[str, Any],
        adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
        level_data: Any,
        path_calculator: Any,
        graph_data: Optional[Dict[str, Any]] = None,
        scale_factor: float = 1.0,
    ) -> float:
        """Potential based on shortest path distance to objective (curriculum-aware).

        STRICT REQUIREMENTS:
        - adjacency, level_data, and path_calculator are REQUIRED
        - No fallback to Euclidean distance

        Returns higher potential when closer to the current objective:
        - Switch when inactive
        - Exit when switch is active

        Args:
            state: Game state dictionary (must contain _pbrs_surface_area)
            adjacency: Graph adjacency structure (REQUIRED)
            level_data: Level data object (REQUIRED)
            path_calculator: CachedPathDistanceCalculator instance (REQUIRED)
            graph_data: Graph data dict with spatial_hash for O(1) node lookup (optional)
            scale_factor: Curriculum normalization adjustment (default 1.0)
                         <1.0 = stronger gradients (early training)
                         1.0 = full normalization (late training)

        Returns:
            float: Potential in range [0.0, 1.0], higher when closer to objective

        Raises:
            RuntimeError: If required data is missing or invalid
        """
        # Validate required parameters
        if not adjacency:
            raise RuntimeError("PBRS requires adjacency graph")
        if not level_data:
            raise RuntimeError("PBRS requires level_data")
        if not path_calculator:
            raise RuntimeError("PBRS requires path_calculator")

        # Determine goal position
        from ...constants.physics_constants import (
            EXIT_SWITCH_RADIUS,
            EXIT_DOOR_RADIUS,
            NINJA_RADIUS,
        )
        from ..constants import LEVEL_DIAGONAL

        if not state["switch_activated"]:
            goal_pos = (int(state["switch_x"]), int(state["switch_y"]))
            cache_key = "switch"
            entity_radius = EXIT_SWITCH_RADIUS
        else:
            goal_pos = (int(state["exit_door_x"]), int(state["exit_door_y"]))
            cache_key = "exit"
            entity_radius = EXIT_DOOR_RADIUS

        player_pos = (int(state["player_x"]), int(state["player_y"]))

        # Calculate shortest path distance
        try:
            distance = path_calculator.get_distance(
                player_pos,
                goal_pos,
                adjacency,
                cache_key=cache_key,
                level_data=level_data,
                graph_data=graph_data,
                entity_radius=entity_radius,
                ninja_radius=NINJA_RADIUS,
            )
        except Exception as e:
            raise RuntimeError(
                f"PBRS path distance calculation failed: {e}\n"
                "Check player/goal positions are within level bounds."
            ) from e

        # Get surface area for normalization
        surface_area = state.get("_pbrs_surface_area")
        if not surface_area:
            raise RuntimeError(
                "Missing '_pbrs_surface_area' in state. "
                "Should be set by PBRSCalculator.calculate_combined_potential()."
            )

        # Handle unreachable goals
        if distance == float("inf"):
            normalized_distance = 1.0
        else:
            # Base area scale: sqrt converts 2D area to 1D distance
            area_scale = np.sqrt(surface_area) * SUB_NODE_SIZE
            
            # Apply curriculum scale factor (reduces normalization for stronger gradients)
            area_scale = area_scale * scale_factor
            
            # Clip to prevent over-normalization on large/complex levels
            max_scale = LEVEL_DIAGONAL * 0.5  # 50% of level diagonal
            area_scale = min(area_scale, max_scale)
            
            normalized_distance = min(1.0, distance / area_scale)

        potential = 1.0 - normalized_distance
        return max(0.0, min(1.0, potential))

    # REMOVED: hazard_proximity_potential() - Death penalty provides clearer signal
    # REMOVED: impact_risk_potential() - Death penalty provides clearer signal
    # REMOVED: exploration_potential() - PBRS provides via distance gradients


class PBRSCalculator:
    """Calculator for completion-focused potential functions.

    Combines multiple PBRS potentials with configurable weights to create
    a composite potential function. The calculator maintains state across
    steps and computes the shaping reward: F(s,s') = γ * Φ(s') - Φ(s)

    Uses shortest path distances for objective distance calculations, respecting
    level geometry and obstacles. REQUIRES adjacency graph and level_data when
    calculating potentials.

    All constants imported from reward_constants.py for consistency.
    """

    def __init__(
        self,
        path_calculator: Optional[Any] = None,
    ):
        """Initialize simplified PBRS calculator (objective potential only).

        Args:
            path_calculator: CachedPathDistanceCalculator instance for shortest path distances
        """
        # Initialize path distance calculator for path-aware reward shaping
        self.path_calculator = path_calculator

        # Cache for surface area per level (invalidated when level or switch states change)
        self._cached_surface_area: Optional[float] = None
        self._cached_level_id: Optional[str] = None
        self._cached_switch_states: Optional[Tuple[Tuple[str, bool], ...]] = None

    def _compute_reachable_surface_area(
        self,
        adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
        level_data: Any,
        graph_data: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Compute total reachable surface area as number of sub-nodes with caching.

        STRICT - NO FALLBACKS: Throws error if adjacency is empty or None.

        Uses flood-fill from player start position to count only nodes actually
        reachable from spawn, matching the behavior shown in debug overlay visualization.
        This ensures surface area calculation respects actual navigable space,
        not just all nodes in the adjacency graph.

        Surface area scaling rationale:
        - Small confined levels: fewer nodes → less normalization → stronger gradients
        - Large open levels: more nodes → more normalization → consistent gradients
        - Scale-invariant: gradient strength proportional to level complexity

        Args:
            adjacency: Graph adjacency structure (may include unreachable nodes)
            level_data: Level data object (must have start_position attribute)
            graph_data: Optional graph data dict with spatial_hash for O(1) node lookup

        Returns:
            Total number of reachable sub-nodes from start position (surface area metric)

        Raises:
            RuntimeError: If adjacency is None or empty, or start_position is missing
        """
        from ...graph.reachability.pathfinding_utils import get_cached_surface_area

        if not adjacency:
            raise RuntimeError(
                "PBRS surface area calculation failed: adjacency graph is empty or None.\n"
                "PBRS requires valid graph data to compute surface-area-based normalization.\n"
                "This typically means:\n"
                "  1. Graph building is not enabled in environment config\n"
                "  2. Graph builder failed to create reachable nodes\n"
                "  3. Level has no traversable space\n"
            )

        # Get player start position from level_data
        start_position = getattr(level_data, "start_position", None)
        if start_position is None:
            raise RuntimeError(
                "PBRS surface area calculation failed: level_data missing start_position.\n"
                "Surface area calculation requires player start position to compute reachability.\n"
                "Verify that level_data.start_position is set correctly."
            )

        # Generate cache key using LevelData utility method for consistency
        # This ensures all components use the same cache key format
        cache_key = level_data.get_cache_key_for_reachability(include_switch_states=True)

        # start_position is already in world space (no conversion needed)
        # This matches the coordinate space used by ninja_position() and flood_fill_reachable_nodes()
        start_pos = (
            int(start_position[0]),
            int(start_position[1]),
        )

        # Use cached surface area calculation to avoid recomputation
        # Wrap in try-except for graceful fallback on degenerate maps
        try:
            return get_cached_surface_area(cache_key, start_pos, adjacency, graph_data)
        except RuntimeError as e:
            # Critical: Surface area computation failed
            logger.error(
                f"CRITICAL: Failed to compute surface area: {e}\n"
                f"Using fallback value. This indicates a map generation bug!"
            )
            # Return a reasonable fallback (diagonal of typical level ~1400px)
            # This prevents crashes while still allowing detection of the bug
            return 1000.0  # Fallback to avoid crash

    # REMOVED: _precompute_reachable_mines() - No longer needed (hazard potential removed)

    def calculate_combined_potential(
        self,
        state: Dict[str, Any],
        adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
        level_data: Any,
        graph_data: Optional[Dict[str, Any]] = None,
        objective_weight: float = 1.0,
        scale_factor: float = 1.0,
    ) -> float:
        """Calculate ONLY objective distance potential (simplified, curriculum-aware).

        Removed components:
        - Hazard proximity potential (death penalty is clearer signal)
        - Impact risk potential (death penalty is clearer signal)
        - Exploration potential (PBRS provides via distance gradients)

        Args:
            state: Game state dictionary
            adjacency: Graph adjacency structure (REQUIRED)
            level_data: Level data object (REQUIRED)
            graph_data: Graph data dict with spatial_hash for O(1) node lookup (optional)
            objective_weight: Curriculum-scaled weight from RewardConfig (default 1.0)
            scale_factor: Normalization adjustment from RewardConfig (default 1.0)

        Returns:
            float: Objective distance potential (curriculum-scaled)

        Raises:
            RuntimeError: If required data is missing or invalid
        """
        # Strict validation
        if adjacency is None:
            raise RuntimeError(
                "PBRS requires adjacency graph. "
                "Ensure graph building is enabled in environment config."
            )

        if level_data is None:
            raise RuntimeError(
                "PBRS requires level_data. "
                "This indicates a configuration error in the environment."
            )

        if self.path_calculator is None:
            raise RuntimeError(
                "PBRS calculator missing path_calculator. "
                "Check PBRSCalculator initialization in RewardCalculator."
            )

        # Get level ID for caching
        level_id = getattr(level_data, "level_id", None)
        if level_id is None:
            level_id = str(getattr(level_data, "start_position", "unknown"))

        # Get switch states for cache invalidation
        switch_states = getattr(level_data, "switch_states", {})
        switch_states_signature = (
            tuple(sorted(switch_states.items())) if switch_states else ()
        )

        # Check if cache needs invalidation
        cache_invalid = (
            self._cached_level_id != level_id
            or self._cached_switch_states != switch_states_signature
            or self._cached_surface_area is None
        )

        # Compute and cache surface area per level
        if cache_invalid:
            self._cached_surface_area = self._compute_reachable_surface_area(
                adjacency, level_data, graph_data
            )
            self._cached_level_id = level_id
            self._cached_switch_states = switch_states_signature

        # Add surface area to state for potential calculation
        state_with_metrics = dict(state)
        state_with_metrics["_pbrs_surface_area"] = self._cached_surface_area

        # Calculate ONLY objective distance potential (no hazard/impact)
        objective_pot = PBRSPotentials.objective_distance_potential(
            state_with_metrics,
            adjacency=adjacency,
            level_data=level_data,
            path_calculator=self.path_calculator,
            graph_data=graph_data,
            scale_factor=scale_factor,  # Apply curriculum scaling
        )

        # Apply phase-specific scaling (switch vs exit) and curriculum weight
        if not state.get("switch_activated", False):
            potential = PBRS_SWITCH_DISTANCE_SCALE * objective_pot * objective_weight
        else:
            potential = PBRS_EXIT_DISTANCE_SCALE * objective_pot * objective_weight

        return max(0.0, potential)

    # REMOVED: _update_visited_positions() - No longer needed (exploration potential removed)

    # REMOVED: get_potential_components() - No longer needed with simplified PBRS
    # For debugging, use calculate_combined_potential() directly which returns the only component (objective)

    def reset(self) -> None:
        """Reset calculator state for new episode."""
        # Keep surface area cache - it's per level, not per episode
        # No other state to reset (visited_positions removed)
        # Will be invalidated when level_data or switch_states change
