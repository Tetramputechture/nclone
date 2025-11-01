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
from ..constants import LEVEL_DIAGONAL
from ..util.util import calculate_distance
from .reward_constants import (
    PBRS_MAX_VELOCITY,
    PBRS_EXPLORATION_VISIT_THRESHOLD,
    PBRS_EXPLORATION_RADIUS,
    PBRS_SWITCH_DISTANCE_SCALE,
    PBRS_EXIT_DISTANCE_SCALE,
    PBRS_FALLBACK_DISTANCE_SCALE,
)


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
        adjacency: Optional[
            Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]]
        ] = None,
        level_data: Optional[Any] = None,
        path_calculator: Optional[Any] = None,
        graph_data: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Potential based on shortest path distance to nearest objective.

        Returns higher potential when closer to the current objective:
        - Switch when inactive
        - Exit when switch is active

        Uses shortest path distance instead of Euclidean to respect level geometry.
        REQUIRES: adjacency graph and level_data must be provided when path_calculator is available.

        Guaranteed to return value in [0.0, 1.0] range with defensive clamping.

        Args:
            state: Game state dictionary
            adjacency: Graph adjacency structure (REQUIRED if path_calculator provided)
            level_data: Level data object (REQUIRED if path_calculator provided)
            path_calculator: CachedPathDistanceCalculator instance (optional)
            graph_data: Graph data dict with spatial_hash for O(1) node lookup (optional)

        Returns:
            float: Potential in range [0.0, 1.0], higher when closer to objective

        Raises:
            RuntimeError: If path_calculator provided but adjacency or level_data missing
        """
        # Determine goal position
        if not state["switch_activated"]:
            # Phase 1: Navigate to switch
            goal_pos = (int(state["switch_x"]), int(state["switch_y"]))
            cache_key = "switch"
        else:
            # Phase 2: Navigate to exit
            goal_pos = (int(state["exit_door_x"]), int(state["exit_door_y"]))
            cache_key = "exit"

        player_pos = (int(state["player_x"]), int(state["player_y"]))

        # Use path distance if calculator available
        if path_calculator is not None:
            # STRICT: Validate required data
            if adjacency is None:
                raise RuntimeError(
                    "PBRS path distance calculation requires adjacency graph. "
                    "Graph data must be available when PBRS is enabled."
                )
            if level_data is None:
                raise RuntimeError(
                    "PBRS path distance calculation requires level_data. "
                    "Level data must be available when PBRS is enabled."
                )

            try:
                # Calculate shortest path distance with graph_data for spatial indexing
                distance = path_calculator.get_distance(
                    player_pos,
                    goal_pos,
                    adjacency,
                    cache_key=cache_key,
                    level_data=level_data,
                    graph_data=graph_data,
                )

                # Get adaptive scale for normalization
                # This will be computed and cached per level by PBRSCalculator
                adaptive_scale = state.get(
                    "_pbrs_adaptive_scale", PBRS_FALLBACK_DISTANCE_SCALE
                )

                # Handle unreachable goals (returns inf)
                if distance == float("inf"):
                    # Use adaptive scale for normalization (sets potential to 0.0)
                    normalized_distance = 1.0
                else:
                    # Normalize path distance to [0, 1] using adaptive scale
                    normalized_distance = min(1.0, distance / adaptive_scale)
            except Exception as e:
                raise RuntimeError(
                    f"PBRS path distance calculation failed: {e}. "
                    "Ensure graph data is properly initialized and accessible."
                ) from e
        else:
            # Fallback to Euclidean (should not happen in production)
            # This path exists only for backward compatibility during transition
            if not state["switch_activated"]:
                distance = calculate_distance(
                    state["player_x"],
                    state["player_y"],
                    state["switch_x"],
                    state["switch_y"],
                )
            else:
                distance = calculate_distance(
                    state["player_x"],
                    state["player_y"],
                    state["exit_door_x"],
                    state["exit_door_y"],
                )
            # Normalize Euclidean distance to [0, 1]
            normalized_distance = min(1.0, distance / LEVEL_DIAGONAL)

        potential = 1.0 - normalized_distance

        # Defensive: explicit bounds checking
        # Should always be in range, but defensive against floating point edge cases
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

        # Import the helper function to compute hazard proximity
        from ..observation_processor import compute_hazard_from_entity_states

        nearest_hazard_dist, hazard_threat = compute_hazard_from_entity_states(
            entity_states, player_x, player_y
        )

        # Return potential based on hazard proximity
        # When far from hazards: high potential (1.0)
        # When close to hazards: low potential (approaches 0.0)
        # This encourages maintaining safe distance
        return 1.0 - hazard_threat

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
        return 1.0 - impact_risk

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
            return min_distance / exploration_radius


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

    # Import PBRS scaling constants from centralized module
    PBRS_SWITCH_DISTANCE = PBRS_SWITCH_DISTANCE_SCALE
    PBRS_EXIT_DISTANCE = PBRS_EXIT_DISTANCE_SCALE

    def __init__(
        self,
        objective_weight: float = 1.0,
        hazard_weight: float = 0.0,  # Disabled for completion focus
        impact_weight: float = 0.0,  # Disabled for completion focus
        exploration_weight: float = 0.0,  # Disabled for completion focus
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
        self.objective_weight = objective_weight
        self.hazard_weight = hazard_weight
        self.impact_weight = impact_weight
        self.exploration_weight = exploration_weight

        # Initialize path distance calculator for path-aware reward shaping
        self.path_calculator = path_calculator

        # Cache for adaptive scaling per level
        self._cached_scale: Optional[float] = None
        self._cached_level_id: Optional[str] = None

        # Track visited positions for exploration potential (minimal usage)
        self.visited_positions: List[Tuple[float, float]] = []
        self.visit_threshold = PBRS_EXPLORATION_VISIT_THRESHOLD

    def _compute_max_reachable_distance(
        self,
        adjacency: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
        level_data: Any,
        graph_data: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Compute maximum reachable distance for adaptive scaling.

        Uses BFS flood fill from start position to find maximum distance
        to any reachable node. This provides adaptive normalization per level.

        Args:
            adjacency: Graph adjacency structure
            level_data: Level data object with start_position
            graph_data: Optional graph data dict with spatial_hash for fast lookup

        Returns:
            Maximum reachable distance, or fallback scale if computation fails
        """
        if not adjacency:
            return PBRS_FALLBACK_DISTANCE_SCALE

        # Get start position from level data
        start_pos = level_data.start_position
        if start_pos is None:
            return PBRS_FALLBACK_DISTANCE_SCALE

        # Find closest node to start position
        from ...graph.reachability.pathfinding_utils import (
            find_closest_node_to_position,
            extract_spatial_lookups_from_graph_data,
        )

        # Extract spatial hash and subcell lookup from graph_data if available
        spatial_hash, subcell_lookup = extract_spatial_lookups_from_graph_data(
            graph_data
        )

        start_node = find_closest_node_to_position(
            start_pos,
            adjacency,
            threshold=50.0,
            spatial_hash=spatial_hash,
            subcell_lookup=subcell_lookup,
        )
        if start_node is None or start_node not in adjacency:
            return PBRS_FALLBACK_DISTANCE_SCALE

        # Use BFS to compute distances to all reachable nodes
        from ...graph.reachability.pathfinding_utils import bfs_distance_from_start

        distances, _ = bfs_distance_from_start(start_node, None, adjacency)

        if not distances:
            return PBRS_FALLBACK_DISTANCE_SCALE

        # Find maximum distance
        max_distance = max(distances.values())

        # Use max distance or fallback, whichever is larger
        # This ensures scale is at least LEVEL_DIAGONAL for consistency
        return max(max_distance, PBRS_FALLBACK_DISTANCE_SCALE)

    def calculate_combined_potential(
        self,
        state: Dict[str, Any],
        adjacency: Optional[
            Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]]
        ] = None,
        level_data: Optional[Any] = None,
        graph_data: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Calculate completion-focused potential from switch/exit distance only.

        Computes adaptive scaling per level and uses it for path distance normalization.
        Caches scale per level to avoid recomputation.

        Args:
            state: Game state dictionary
            adjacency: Graph adjacency structure (REQUIRED if path_calculator provided)
            level_data: Level data object (REQUIRED if path_calculator provided)
            graph_data: Graph data dict with spatial_hash for O(1) node lookup (optional)

        Returns:
            float: Combined potential value focused on completion objectives

        Raises:
            RuntimeError: If path_calculator provided but adjacency or level_data missing
        """
        # Compute adaptive scale if path calculator is available
        adaptive_scale = PBRS_FALLBACK_DISTANCE_SCALE

        if self.path_calculator is not None:
            # STRICT: Validate required data
            if adjacency is None:
                raise RuntimeError(
                    "PBRS path distance calculation requires adjacency graph. "
                    "Graph data must be available when PBRS is enabled."
                )
            if level_data is None:
                raise RuntimeError(
                    "PBRS path distance calculation requires level_data. "
                    "Level data must be available when PBRS is enabled."
                )

            # Check if we need to recompute adaptive scale
            # Use level_id if available, otherwise use start_position as identifier
            level_id = getattr(level_data, "level_id", None)
            if level_id is None:
                level_id = str(getattr(level_data, "start_position", "unknown"))

            if self._cached_level_id != level_id or self._cached_scale is None:
                # Compute and cache adaptive scale
                self._cached_scale = self._compute_max_reachable_distance(
                    adjacency, level_data, graph_data
                )
                self._cached_level_id = level_id

            adaptive_scale = self._cached_scale

        # Add adaptive scale to state for use in objective_distance_potential
        state_with_scale = dict(state)
        state_with_scale["_pbrs_adaptive_scale"] = adaptive_scale

        # Calculate objective distance potential with adaptive scaling
        objective_pot = PBRSPotentials.objective_distance_potential(
            state_with_scale,
            adjacency=adjacency,
            level_data=level_data,
            path_calculator=self.path_calculator,
            graph_data=graph_data,
        )

        # Apply completion-focused scaling
        if not state.get("switch_activated", False):
            # Focus on switch distance
            combined_potential = self.PBRS_SWITCH_DISTANCE * objective_pot
        else:
            # Focus on exit distance
            combined_potential = self.PBRS_EXIT_DISTANCE * objective_pot

        return combined_potential

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
        adjacency: Optional[
            Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]]
        ] = None,
        level_data: Optional[Any] = None,
        graph_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """Get individual potential components for debugging/logging.

        Args:
            state: Game state dictionary
            adjacency: Graph adjacency structure (REQUIRED if path_calculator provided)
            level_data: Level data object (REQUIRED if path_calculator provided)
            graph_data: Graph data dict with spatial_hash for O(1) node lookup (optional)

        Returns:
            dict: Dictionary of potential component values (completion-focused)
        """
        # Compute adaptive scale if needed
        adaptive_scale = PBRS_FALLBACK_DISTANCE_SCALE
        if (
            self.path_calculator is not None
            and adjacency is not None
            and level_data is not None
        ):
            level_id = getattr(level_data, "level_id", None)
            if level_id is None:
                level_id = str(getattr(level_data, "start_position", "unknown"))

            if self._cached_level_id != level_id or self._cached_scale is None:
                self._cached_scale = self._compute_max_reachable_distance(
                    adjacency, level_data, graph_data
                )
                self._cached_level_id = level_id

            adaptive_scale = self._cached_scale

        state_with_scale = dict(state)
        state_with_scale["_pbrs_adaptive_scale"] = adaptive_scale

        objective_pot = PBRSPotentials.objective_distance_potential(
            state_with_scale,
            adjacency=adjacency,
            level_data=level_data,
            path_calculator=self.path_calculator,
            graph_data=graph_data,
        )
        return {
            "objective": objective_pot,
            "switch_distance_potential": self.PBRS_SWITCH_DISTANCE * objective_pot
            if not state.get("switch_activated", False)
            else 0.0,
            "exit_distance_potential": self.PBRS_EXIT_DISTANCE * objective_pot
            if state.get("switch_activated", False)
            else 0.0,
            "adaptive_scale": adaptive_scale,
            "hazard": 0.0,  # Disabled for completion focus
            "impact": 0.0,  # Disabled for completion focus
            "exploration": 0.0,  # Disabled for completion focus
        }

    def reset(self) -> None:
        """Reset calculator state for new episode."""
        self.visited_positions.clear()
        # Keep adaptive scale cache - it's per level, not per episode
        # Will be invalidated when level_data changes
