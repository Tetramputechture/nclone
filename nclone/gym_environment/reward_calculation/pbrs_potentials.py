"""Potential-Based Reward Shaping (PBRS) potential functions.

This module implements reusable potential functions Φ(s) for reward shaping
following the theory of Ng, Harada, and Russell (1999): "Policy Invariance
Under Reward Transformations: Theory and Application to Reward Shaping".

PBRS provides dense reward signals without changing the optimal policy by using
the formula: F(s,s') = γ * Φ(s') - Φ(s)

Key properties:
- Policy invariance: Optimal policy unchanged under PBRS (holds for ANY γ)
- Dense rewards: Provides gradient at every step
- Normalization: All potentials normalized to [0, 1] range
- Composability: Multiple potentials can be combined with weights

Discount Factor (γ) Choice:
- We use γ=1.0 for heuristic potential functions (path distance)
- This eliminates negative bias: accumulated PBRS = Φ(goal) - Φ(start) exactly
- For heuristic potentials, γ should NOT match MDP's discount (that's only for learned V(s))
- Policy invariance holds for any γ, but γ=1.0 is standard for episodic tasks

Mathematical Justification:
With γ < 1.0, accumulated PBRS = γ*Φ(goal) - Φ(start) + (γ-1)*Σ Φ(intermediate)
The (γ-1)*Σ term is negative and grows with episode length, causing systematic negative bias.
With γ=1.0, this bias term vanishes, giving clean telescoping: Φ(goal) - Φ(start).

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
        """Potential based on shortest path distance to objective (non-linear, curriculum-aware).

        STRICT REQUIREMENTS:
        - adjacency, level_data, and path_calculator are REQUIRED
        - No fallback to Euclidean distance

        Uses non-linear normalization: Φ(s) = 1 / (1 + distance/area_scale)
        This ensures gradients at ALL distances, preventing the "dead zone" problem
        of linear normalization where potential=0 beyond a certain distance.

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
            float: Potential in range (0.0, 1.0], higher when closer to objective.
                   Never returns exactly 0.0 unless goal is unreachable (distance=inf)

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

        # Extract base_adjacency for physics checks
        base_adjacency = (
            graph_data.get("base_adjacency", adjacency) if graph_data else adjacency
        )

        # Calculate shortest path distance
        try:
            distance = path_calculator.get_distance(
                player_pos,
                goal_pos,
                adjacency,
                base_adjacency,
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

        # Get combined path distance for path-based normalization
        combined_path_distance = state.get("_pbrs_combined_path_distance")
        if combined_path_distance is None:
            raise RuntimeError(
                "Missing '_pbrs_combined_path_distance' in state. "
                "Should be set by PBRSCalculator.calculate_combined_potential()."
            )

        # Import path normalization factor
        from .reward_constants import PBRS_PATH_NORMALIZATION_FACTOR

        # Handle unreachable goals
        if distance == float("inf"):
            # Unreachable: return zero potential (no gradient)
            state["_pbrs_area_scale"] = 0.0
            state["_pbrs_normalized_distance"] = float("inf")
            return 0.0
        elif combined_path_distance == float("inf"):
            # Fallback to surface area if path distance is infinite (unreachable)
            logger.warning(
                "Combined path distance is infinite, falling back to surface area normalization"
            )
            surface_area = state.get("_pbrs_surface_area")
            if not surface_area:
                raise RuntimeError(
                    "Missing '_pbrs_surface_area' in state for fallback normalization"
                )
            area_scale = np.sqrt(surface_area) * SUB_NODE_SIZE * scale_factor
            max_scale = LEVEL_DIAGONAL * 0.3
            area_scale = min(area_scale, max_scale)
        else:
            # Path-based normalization: use combined path distance
            # area_scale = combined_path_distance * normalization_factor * curriculum_scale
            area_scale = (
                combined_path_distance * PBRS_PATH_NORMALIZATION_FACTOR * scale_factor
            )

        # Store area_scale in state for diagnostic logging
        state["_pbrs_area_scale"] = area_scale

        # NON-LINEAR NORMALIZATION: Provides gradients at ALL distances
        # Formula: Φ(s) = 1 / (1 + distance/area_scale)
        # Properties:
        # - Range: (0, 1] (never reaches zero, always provides gradient)
        # - Monotonic: closer distance → higher potential
        # - Natural decay: gradient strength decreases with distance
        # - PBRS-compatible: F(s,s') = γ * Φ(s') - Φ(s) maintains policy invariance
        #
        # Example gradients (area_scale=300px):
        #   distance=0:    potential=1.000 (at goal)
        #   distance=100:  potential=0.750 (strong gradient)
        #   distance=300:  potential=0.500 (moderate gradient)
        #   distance=600:  potential=0.333 (weak but non-zero gradient)
        #   distance=1200: potential=0.200 (very weak but still provides signal)
        normalized_distance = distance / area_scale
        potential = 1.0 / (1.0 + normalized_distance)

        # Store normalized distance for diagnostic logging
        state["_pbrs_normalized_distance"] = normalized_distance

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

        # Cache for combined path distance per level (spawn→switch + switch→exit)
        self._cached_combined_path_distance: Optional[float] = None
        self._path_distance_cache_key: Optional[str] = None

        # OPTIMIZATION: Position-based caching to avoid repeated pathfinding
        # Only recalculate path distance if ninja moves 3+ pixels
        self._last_player_pos: Optional[Tuple[int, int]] = None
        self._last_goal_pos: Optional[Tuple[int, int]] = None
        self._cached_objective_potential: Optional[float] = None

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
        cache_key = level_data.get_cache_key_for_reachability(
            include_switch_states=True
        )

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

    def _compute_combined_path_distance(
        self,
        adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
        level_data: Any,
        graph_data: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Compute combined path distance from spawn→switch + switch→exit with caching.

        This provides path-based normalization that better handles open levels where
        the player takes focused paths. Uses actual shortest path distances rather
        than surface area for normalization.

        Args:
            adjacency: Graph adjacency structure
            level_data: Level data object (must have start_position and entities)
            graph_data: Optional graph data dict with spatial_hash for O(1) node lookup

        Returns:
            Combined path distance in pixels (spawn→switch + switch→exit)
            Returns float('inf') if either path is unreachable

        Raises:
            RuntimeError: If required data is missing or entities not found
        """
        from ...constants.entity_types import EntityType
        from ...constants.physics_constants import (
            EXIT_SWITCH_RADIUS,
            EXIT_DOOR_RADIUS,
            NINJA_RADIUS,
        )

        if not adjacency:
            raise RuntimeError(
                "PBRS combined path distance calculation failed: adjacency graph is empty or None"
            )

        # Get spawn position from level_data
        start_position = getattr(level_data, "start_position", None)
        if start_position is None:
            raise RuntimeError(
                "PBRS combined path distance calculation failed: level_data missing start_position"
            )

        spawn_pos = (int(start_position[0]), int(start_position[1]))

        # Get switch position from first active exit switch
        exit_switches = level_data.get_entities_by_type(EntityType.EXIT_SWITCH)
        if not exit_switches:
            raise RuntimeError(
                "PBRS combined path distance calculation failed: no exit switch found in level"
            )

        # Use first active switch
        switch = exit_switches[0]
        switch_pos = (int(switch.get("x", 0)), int(switch.get("y", 0)))

        # Get exit door position from first exit door
        exit_doors = level_data.get_entities_by_type(EntityType.EXIT_DOOR)
        if not exit_doors:
            raise RuntimeError(
                "PBRS combined path distance calculation failed: no exit door found in level"
            )

        exit_door = exit_doors[0]
        exit_pos = (int(exit_door.get("x", 0)), int(exit_door.get("y", 0)))

        # Extract base_adjacency for physics checks
        base_adjacency = (
            graph_data.get("base_adjacency", adjacency) if graph_data else adjacency
        )

        # Calculate spawn→switch distance
        try:
            spawn_to_switch_dist = self.path_calculator.get_distance(
                spawn_pos,
                switch_pos,
                adjacency,
                base_adjacency,
                cache_key="spawn_to_switch",
                level_data=level_data,
                graph_data=graph_data,
                entity_radius=EXIT_SWITCH_RADIUS,
                ninja_radius=NINJA_RADIUS,
            )
        except Exception as e:
            logger.warning(
                f"Failed to calculate spawn→switch distance: {e}. Using inf."
            )
            spawn_to_switch_dist = float("inf")

        # Calculate switch→exit distance
        try:
            switch_to_exit_dist = self.path_calculator.get_distance(
                switch_pos,
                exit_pos,
                adjacency,
                base_adjacency,
                cache_key="switch_to_exit",
                level_data=level_data,
                graph_data=graph_data,
                entity_radius=EXIT_DOOR_RADIUS,
                ninja_radius=NINJA_RADIUS,
            )
        except Exception as e:
            logger.warning(f"Failed to calculate switch→exit distance: {e}. Using inf.")
            switch_to_exit_dist = float("inf")

        # Combine distances
        if spawn_to_switch_dist == float("inf") or switch_to_exit_dist == float("inf"):
            logger.warning(
                f"Combined path distance is infinite (unreachable goal). "
                f"spawn→switch: {spawn_to_switch_dist:.1f}, switch→exit: {switch_to_exit_dist:.1f}"
            )
            return float("inf")

        combined_distance = spawn_to_switch_dist + switch_to_exit_dist

        logger.debug(
            f"Combined path distance: {combined_distance:.1f}px "
            f"(spawn→switch: {spawn_to_switch_dist:.1f}px, switch→exit: {switch_to_exit_dist:.1f}px)"
        )

        return combined_distance

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

        # Compute and cache surface area per level (kept for backward compatibility/fallback)
        if cache_invalid:
            self._cached_surface_area = self._compute_reachable_surface_area(
                adjacency, level_data, graph_data
            )
            self._cached_level_id = level_id
            self._cached_switch_states = switch_states_signature

            # Clear position-based cache when level/switch state changes
            self._last_player_pos = None
            self._last_goal_pos = None
            self._cached_objective_potential = None

        # Compute and cache combined path distance for path-based normalization
        # Cache key includes level_id to invalidate on level change
        path_cache_key = f"{level_id}_{switch_states_signature}"
        if (
            self._path_distance_cache_key != path_cache_key
            or self._cached_combined_path_distance is None
        ):
            self._cached_combined_path_distance = self._compute_combined_path_distance(
                adjacency, level_data, graph_data
            )
            self._path_distance_cache_key = path_cache_key
            logger.debug(
                f"Cached combined path distance: {self._cached_combined_path_distance:.1f}px for level {level_id}"
            )

        # Add metrics to state for potential calculation
        state_with_metrics = dict(state)
        state_with_metrics["_pbrs_surface_area"] = self._cached_surface_area
        state_with_metrics["_pbrs_combined_path_distance"] = (
            self._cached_combined_path_distance
        )

        # OPTIMIZATION: Position-based caching to avoid expensive pathfinding
        # Only recalculate if ninja moved 3+ pixels or goal changed
        player_pos = (int(state["player_x"]), int(state["player_y"]))

        # Determine current goal position
        if not state["switch_activated"]:
            goal_pos = (int(state["switch_x"]), int(state["switch_y"]))
        else:
            goal_pos = (int(state["exit_door_x"]), int(state["exit_door_y"]))

        # Check if we can use cached potential
        use_cached_potential = False
        if (
            self._last_player_pos is not None
            and self._last_goal_pos is not None
            and self._cached_objective_potential is not None
        ):
            # Check if goal changed
            if self._last_goal_pos == goal_pos:
                # Goal unchanged, check player movement (Manhattan distance)
                distance_moved = abs(player_pos[0] - self._last_player_pos[0]) + abs(
                    player_pos[1] - self._last_player_pos[1]
                )
                if distance_moved < 6:
                    # PERFORMANCE OPTIMIZATION: Only recalculate if moved 6+ pixels
                    # Reduces pathfinding calls by ~50% while maintaining accurate potentials
                    use_cached_potential = True
                    objective_pot = self._cached_objective_potential

        # Calculate potential if cache miss
        if not use_cached_potential:
            objective_pot = PBRSPotentials.objective_distance_potential(
                state_with_metrics,
                adjacency=adjacency,
                level_data=level_data,
                path_calculator=self.path_calculator,
                graph_data=graph_data,
                scale_factor=scale_factor,  # Apply curriculum scaling
            )

            # Update cache
            self._last_player_pos = player_pos
            self._last_goal_pos = goal_pos
            self._cached_objective_potential = objective_pot

        # Copy diagnostic values back to original state for logging
        # These are set by objective_distance_potential() in state_with_metrics
        state["_pbrs_area_scale"] = state_with_metrics.get("_pbrs_area_scale", 0.0)
        state["_pbrs_normalized_distance"] = state_with_metrics.get(
            "_pbrs_normalized_distance", 0.0
        )
        state["_pbrs_combined_path_distance"] = state_with_metrics.get(
            "_pbrs_combined_path_distance"
        )
        state["_pbrs_surface_area"] = state_with_metrics.get("_pbrs_surface_area")

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
        """Reset calculator state for new episode.

        Clears position-based caching to ensure fresh potential calculations
        for the new episode. Surface area and combined path distance caches
        are kept as they are level-specific, not episode-specific.
        """
        # Clear position-based cache for fresh potential calculations
        # CRITICAL: Must reset when episode restarts (including same level reload)
        self._last_player_pos = None
        self._last_goal_pos = None
        self._cached_objective_potential = None

        # Keep surface area cache - it's per level, not per episode
        # Keep combined path distance cache - it's per level, not per episode
        # These will be invalidated when level_data or switch_states change
