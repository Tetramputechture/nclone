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

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from ..util.util import calculate_distance
from .reward_constants import (
    PBRS_MAX_VELOCITY,
    PBRS_EXPLORATION_VISIT_THRESHOLD,
    PBRS_EXPLORATION_RADIUS,
    PBRS_SWITCH_DISTANCE_SCALE,
    PBRS_EXIT_DISTANCE_SCALE,
    PBRS_OBJECTIVE_WEIGHT,
    PBRS_HAZARD_WEIGHT,
    PBRS_IMPACT_WEIGHT,
    PBRS_EXPLORATION_WEIGHT,
)

from ..observation_processor import compute_hazard_from_entity_states

# Sub-node size for surface area calculations (from graph builder)
# The graph builder creates a 2x2 grid of sub-nodes per 24px tile
SUB_NODE_SIZE = 12  # pixels per sub-node


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
    ) -> float:
        """Potential based on shortest path distance to nearest objective.

        STRICT REQUIREMENTS - NO FALLBACKS:
        - adjacency, level_data, and path_calculator are REQUIRED
        - No fallback to Euclidean distance
        - Throws descriptive RuntimeErrors if any required data is missing

        Returns higher potential when closer to the current objective:
        - Switch when inactive
        - Exit when switch is active

        Args:
            state: Game state dictionary (must contain _pbrs_surface_area)
            adjacency: Graph adjacency structure (REQUIRED)
            level_data: Level data object (REQUIRED)
            path_calculator: CachedPathDistanceCalculator instance (REQUIRED)
            graph_data: Graph data dict with spatial_hash for O(1) node lookup (optional)

        Returns:
            float: Potential in range [0.0, 1.0], higher when closer to objective

        Raises:
            RuntimeError: If required data is missing or invalid
        """
        # STRICT: Validate all required parameters
        if adjacency is None:
            raise RuntimeError(
                "PBRS objective_distance_potential requires adjacency graph.\n"
                "Adjacency is None, which means graph building failed or is disabled.\n"
                "Fix: Ensure 'enable_graph_for_pbrs: true' in environment config."
            )

        if level_data is None:
            raise RuntimeError(
                "PBRS objective_distance_potential requires level_data.\n"
                "Level data is None, which should never happen in normal operation.\n"
                "This indicates a serious configuration or initialization error."
            )

        if path_calculator is None:
            raise RuntimeError(
                "PBRS objective_distance_potential requires path_calculator.\n"
                "Path calculator is None, which means PBRS was not properly initialized.\n"
                "This should never happen - PBRS is always enabled and should auto-initialize."
            )

        # Determine goal position
        if not state["switch_activated"]:
            goal_pos = (int(state["switch_x"]), int(state["switch_y"]))
            cache_key = "switch"
        else:
            goal_pos = (int(state["exit_door_x"]), int(state["exit_door_y"]))
            cache_key = "exit"

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
            )
        except Exception as e:
            raise RuntimeError(
                f"PBRS path distance calculation failed: {e}\n"
                "This indicates a problem with:\n"
                "  1. Graph adjacency structure (corrupted or invalid)\n"
                "  2. Player or goal position (outside graph bounds)\n"
                "  3. Path calculator internal error\n"
                "Check that player and goal positions are within level bounds."
            ) from e

        # Get surface area for normalization (REQUIRED)
        surface_area = state.get("_pbrs_surface_area")
        if surface_area is None:
            raise RuntimeError(
                "PBRS potential calculation requires '_pbrs_surface_area' in state.\n"
                "This should be set by PBRSCalculator.calculate_combined_potential().\n"
                "If you see this error, there's a bug in the PBRS calculation flow."
            )

        # Handle unreachable goals
        if distance == float("inf"):
            # Goal is unreachable - return minimum potential
            # This is NOT an error - some goals may be legitimately unreachable
            normalized_distance = 1.0
        else:
            # Surface area scaling: sqrt converts 2D area to 1D distance scale
            # Multiply by SUB_NODE_SIZE (12px) to get pixel distance equivalent
            # Result: scale grows with sqrt(area), keeping gradients consistent
            area_scale = np.sqrt(surface_area) * SUB_NODE_SIZE
            normalized_distance = min(1.0, distance / area_scale)

        potential = 1.0 - normalized_distance
        return max(0.0, min(1.0, potential))

    @staticmethod
    def hazard_proximity_potential(state: Dict[str, Any]) -> float:
        """Potential penalty based on proximity to active hazards (mines).

        Returns lower potential when close to dangerous mines, encouraging
        the agent to maintain safe distance from hazards.

        Args:
            state: Game state dictionary containing entity_states and player position

        Returns:
            float: Potential in range [0.0, 1.0], lower when close to hazards
        """
        # Extract entity states if available
        entity_states = state.get("entity_states", None)
        if entity_states is None or len(entity_states) == 0:
            return 1.0  # No entity data, assume safe

        player_x = state.get("player_x", 0.0)
        player_y = state.get("player_y", 0.0)

        nearest_hazard_dist, hazard_threat = compute_hazard_from_entity_states(
            entity_states, player_x, player_y
        )

        # Return potential based on hazard proximity
        # When far from hazards: high potential (1.0)
        # When close to hazards: low potential (approaches 0.0)
        # This encourages maintaining safe distance
        return 1.0 - (hazard_threat * PBRS_HAZARD_WEIGHT)

    @staticmethod
    def impact_risk_potential(state: Dict[str, Any]) -> float:
        """Potential penalty based on impact risk.

        Uses velocity and surface normal information to estimate collision risk.
        Higher downward velocity near surfaces increases risk.

        Args:
            state: Game state dictionary

        Returns:
            float: Potential in range [0.0, 1.0], lower when high impact risk
        """
        # Extract velocity components from game state
        game_state = state.get("game_state", [])
        if len(game_state) < 4:  # Need at least velocity components
            return 1.0  # No velocity info, assume safe

        # Assuming velocity is in positions 2, 3 of game state (vx, vy)
        vx = game_state[2] if len(game_state) > 2 else 0.0
        vy = game_state[3] if len(game_state) > 3 else 0.0

        # Calculate velocity magnitude
        velocity_magnitude = np.sqrt(vx**2 + vy**2)

        # Focus on downward velocity as primary risk factor
        downward_velocity = max(0.0, vy)  # Positive vy is downward

        # Extract surface normal information if available (from rich features)
        floor_normal_y = 0.0
        if len(game_state) >= 24:  # Rich features available
            # Assuming floor normal y is at position ~20 in rich features
            floor_normal_y = abs(game_state[20]) if len(game_state) > 20 else 0.0

        # Calculate impact risk based on downward velocity and surface proximity
        # Higher risk when moving fast downward near surfaces
        velocity_risk = min(1.0, velocity_magnitude / PBRS_MAX_VELOCITY)
        downward_risk = min(1.0, downward_velocity / PBRS_MAX_VELOCITY)
        surface_proximity = floor_normal_y  # Higher when near floor

        # Combine factors: higher risk when fast downward movement near surfaces
        impact_risk = velocity_risk * downward_risk * (0.5 + 0.5 * surface_proximity)

        # Return inverted risk as potential (lower risk = higher potential)
        return 1.0 - (impact_risk * PBRS_IMPACT_WEIGHT)

    @staticmethod
    def exploration_potential(
        state: Dict[str, Any], visited_positions: List[Tuple[float, float]]
    ) -> float:
        """Potential based on exploration of new areas.

        Encourages visiting unexplored regions of the level.

        Args:
            state: Game state dictionary
            visited_positions: List of previously visited (x, y) positions

        Returns:
            float: Potential in range [0.0, 1.0], higher in unexplored areas
        """
        player_x, player_y = state["player_x"], state["player_y"]

        if not visited_positions:
            return 1.0  # First position, maximum exploration potential

        # Find minimum distance to any previously visited position
        min_distance = float("inf")
        for prev_x, prev_y in visited_positions:
            distance = calculate_distance(player_x, player_y, prev_x, prev_y)
            min_distance = min(min_distance, distance)

        # Exploration radius - positions within this distance are considered "visited"
        exploration_radius = PBRS_EXPLORATION_RADIUS

        # Return potential based on distance from visited areas
        if min_distance > exploration_radius:
            return 1.0  # Far from visited areas, high exploration potential
        else:
            # Linear decay within exploration radius
            return 1.0 - (min_distance / exploration_radius * PBRS_EXPLORATION_WEIGHT)


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
        """Initialize PBRS calculator for completion-focused training.

        Args:
            objective_weight: Weight for objective distance potential (switch/exit)
            hazard_weight: Weight for hazard proximity potential (0.0 = disabled)
            impact_weight: Weight for impact risk potential (0.0 = disabled)
            exploration_weight: Weight for exploration potential (0.0 = disabled)
            path_calculator: CachedPathDistanceCalculator instance for path-aware distances
        """
        # Initialize path distance calculator for path-aware reward shaping
        self.path_calculator = path_calculator

        # Cache for surface area per level
        self._cached_surface_area: Optional[float] = None
        self._cached_level_id: Optional[str] = None

        # Track visited positions for exploration potential (minimal usage)
        self.visited_positions: List[Tuple[float, float]] = []
        self.visit_threshold = PBRS_EXPLORATION_VISIT_THRESHOLD

    def _compute_reachable_surface_area(
        self,
        adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
        level_data: Any,
    ) -> float:
        """Compute total reachable surface area as number of sub-nodes.

        STRICT - NO FALLBACKS: Throws error if adjacency is empty or None.

        The adjacency graph is already filtered to only reachable nodes from
        player spawn (done by GraphBuilder), so we just count total nodes.
        This provides a natural measure of level complexity and navigable space.

        Surface area scaling rationale:
        - Small confined levels: fewer nodes → less normalization → stronger gradients
        - Large open levels: more nodes → more normalization → consistent gradients
        - Scale-invariant: gradient strength proportional to level complexity

        Args:
            adjacency: Graph adjacency structure (filtered to reachable nodes only)
            level_data: Level data object

        Returns:
            Total number of reachable sub-nodes (surface area metric)

        Raises:
            RuntimeError: If adjacency is None or empty (graph building failed)
        """
        if not adjacency:
            raise RuntimeError(
                "PBRS surface area calculation failed: adjacency graph is empty or None.\n"
                "PBRS requires valid graph data to compute surface-area-based normalization.\n"
                "This typically means:\n"
                "  1. Graph building is not enabled in environment config\n"
                "  2. Graph builder failed to create reachable nodes\n"
                "  3. Level has no traversable space\n"
                "Fix: Ensure 'enable_graph_for_pbrs: true' in environment config and verify level is valid."
            )

        # Count total reachable nodes
        total_reachable_nodes = len(adjacency)

        if total_reachable_nodes == 0:
            raise RuntimeError(
                "PBRS surface area calculation failed: adjacency graph has zero nodes.\n"
                "This means the level has no reachable space from player spawn.\n"
                "Verify that:\n"
                "  1. Level geometry allows player movement\n"
                "  2. Player spawn position is valid\n"
                "  3. Graph builder successfully found start position"
            )

        return float(total_reachable_nodes)

    def calculate_combined_potential(
        self,
        state: Dict[str, Any],
        adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
        level_data: Any,
        graph_data: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Calculate completion-focused potential using surface area scaling.

        STRICT REQUIREMENTS:
        - adjacency and level_data are REQUIRED (no Optional)
        - path_calculator MUST be initialized (always is)
        - Throws descriptive errors if any required data is missing

        Args:
            state: Game state dictionary
            adjacency: Graph adjacency structure (REQUIRED)
            level_data: Level data object (REQUIRED)
            graph_data: Graph data dict with spatial_hash for O(1) node lookup (optional)

        Returns:
            float: Combined potential value

        Raises:
            RuntimeError: If required data is missing or invalid
        """
        # Strict validation - no fallbacks
        if adjacency is None:
            raise RuntimeError(
                "PBRS calculate_combined_potential requires adjacency graph.\n"
                "Adjacency is None, which means graph building failed or is disabled.\n"
                "Fix: Ensure graph building is properly configured in environment."
            )

        if level_data is None:
            raise RuntimeError(
                "PBRS calculate_combined_potential requires level_data.\n"
                "Level data is None, which should never happen.\n"
                "This indicates a serious configuration error in the environment."
            )

        if self.path_calculator is None:
            raise RuntimeError(
                "PBRS calculator has no path_calculator initialized.\n"
                "This should never happen - path_calculator is required for PBRS.\n"
                "Check PBRSCalculator initialization in RewardCalculator."
            )

        # Get or compute level ID for caching
        level_id = getattr(level_data, "level_id", None)
        if level_id is None:
            level_id = str(getattr(level_data, "start_position", "unknown"))

        # Compute and cache surface area per level
        if self._cached_level_id != level_id or self._cached_surface_area is None:
            self._cached_surface_area = self._compute_reachable_surface_area(
                adjacency, level_data
            )
            self._cached_level_id = level_id

        surface_area = self._cached_surface_area

        # Add surface area to state for potential calculation
        state_with_metrics = dict(state)
        state_with_metrics["_pbrs_surface_area"] = surface_area

        # Calculate objective distance potential with surface area scaling
        # All parameters are required - will throw descriptive errors if missing
        objective_pot = PBRSPotentials.objective_distance_potential(
            state_with_metrics,
            adjacency=adjacency,
            level_data=level_data,
            path_calculator=self.path_calculator,
            graph_data=graph_data,
        )

        # Apply phase-specific scaling (switch vs exit)
        if not state.get("switch_activated", False):
            combined_potential = (
                PBRS_SWITCH_DISTANCE_SCALE * objective_pot * PBRS_OBJECTIVE_WEIGHT
            )
        else:
            combined_potential = (
                PBRS_EXIT_DISTANCE_SCALE * objective_pot * PBRS_OBJECTIVE_WEIGHT
            )

        max_potential = max(PBRS_SWITCH_DISTANCE_SCALE, PBRS_EXIT_DISTANCE_SCALE)
        return max(0.0, min(max_potential, combined_potential))

    def _update_visited_positions(self, x: float, y: float) -> None:
        """Update the list of visited positions."""
        # Check if this position is significantly different from previous ones
        for prev_x, prev_y in self.visited_positions:
            if calculate_distance(x, y, prev_x, prev_y) < self.visit_threshold:
                return  # Too close to existing position

        # Add new position
        self.visited_positions.append((x, y))

        # Limit memory to prevent unbounded growth
        max_positions = 100
        if len(self.visited_positions) > max_positions:
            self.visited_positions = self.visited_positions[-max_positions:]

    def get_potential_components(
        self,
        state: Dict[str, Any],
        adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
        level_data: Any,
        graph_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """Get individual potential components for debugging/logging.

        Args:
            state: Game state dictionary
            adjacency: Graph adjacency structure (REQUIRED)
            level_data: Level data object (REQUIRED)
            graph_data: Graph data dict with spatial_hash for O(1) node lookup (optional)

        Returns:
            dict: Dictionary of potential component values (completion-focused)
        """
        # Compute surface area if needed
        if (
            self._cached_level_id != getattr(level_data, "level_id", None)
            or self._cached_surface_area is None
        ):
            level_id = getattr(level_data, "level_id", None)
            if level_id is None:
                level_id = str(getattr(level_data, "start_position", "unknown"))

            if self._cached_level_id != level_id or self._cached_surface_area is None:
                self._cached_surface_area = self._compute_reachable_surface_area(
                    adjacency, level_data
                )
                self._cached_level_id = level_id

        surface_area = self._cached_surface_area

        state_with_metrics = dict(state)
        state_with_metrics["_pbrs_surface_area"] = surface_area

        objective_pot = PBRSPotentials.objective_distance_potential(
            state_with_metrics,
            adjacency=adjacency,
            level_data=level_data,
            path_calculator=self.path_calculator,
            graph_data=graph_data,
        )
        return {
            "objective": objective_pot,
            "switch_distance_potential": PBRS_SWITCH_DISTANCE_SCALE * objective_pot
            if not state.get("switch_activated", False)
            else 0.0,
            "exit_distance_potential": PBRS_EXIT_DISTANCE_SCALE * objective_pot
            if state.get("switch_activated", False)
            else 0.0,
            "surface_area": surface_area,
            "sqrt_surface_area": np.sqrt(surface_area),
            "area_scale_px": np.sqrt(surface_area) * SUB_NODE_SIZE,
            "hazard": 0.0,  # Disabled for completion focus
            "impact": 0.0,  # Disabled for completion focus
            "exploration": 0.0,  # Disabled for completion focus
        }

    def reset(self) -> None:
        """Reset calculator state for new episode."""
        self.visited_positions.clear()
        # Keep surface area cache - it's per level, not per episode
        # Will be invalidated when level_data changes
