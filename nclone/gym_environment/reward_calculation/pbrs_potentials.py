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
from typing import Dict, Any, List, Tuple
from ..constants import LEVEL_DIAGONAL
from ..util.util import calculate_distance
from .reward_constants import (
    PBRS_MAX_VELOCITY,
    PBRS_EXPLORATION_VISIT_THRESHOLD,
    PBRS_EXPLORATION_RADIUS,
    PBRS_SWITCH_DISTANCE_SCALE,
    PBRS_EXIT_DISTANCE_SCALE,
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
    def objective_distance_potential(state: Dict[str, Any]) -> float:
        """Potential based on distance to nearest objective.

        Returns higher potential when closer to the current objective:
        - Switch when inactive
        - Exit when switch is active

        Guaranteed to return value in [0.0, 1.0] range with defensive clamping.

        Args:
            state: Game state dictionary

        Returns:
            float: Potential in range [0.0, 1.0], higher when closer to objective
        """
        if not state["switch_activated"]:
            # Phase 1: Navigate to switch
            distance = calculate_distance(
                state["player_x"],
                state["player_y"],
                state["switch_x"],
                state["switch_y"],
            )
        else:
            # Phase 2: Navigate to exit
            distance = calculate_distance(
                state["player_x"],
                state["player_y"],
                state["exit_door_x"],
                state["exit_door_y"],
            )

        # Normalize distance to [0, 1], clamping to handle edge cases
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
    ):
        """Initialize PBRS calculator for completion-focused training.

        Args:
            objective_weight: Weight for objective distance potential (switch/exit)
            hazard_weight: Weight for hazard proximity potential (0.0 = disabled)
            impact_weight: Weight for impact risk potential (0.0 = disabled)
            exploration_weight: Weight for exploration potential (0.0 = disabled)
        """
        self.objective_weight = objective_weight
        self.hazard_weight = hazard_weight
        self.impact_weight = impact_weight
        self.exploration_weight = exploration_weight

        # Track visited positions for exploration potential (minimal usage)
        self.visited_positions: List[Tuple[float, float]] = []
        self.visit_threshold = PBRS_EXPLORATION_VISIT_THRESHOLD

    def calculate_combined_potential(self, state: Dict[str, Any]) -> float:
        """Calculate completion-focused potential from switch/exit distance only.

        Args:
            state: Game state dictionary

        Returns:
            float: Combined potential value focused on completion objectives
        """
        # Calculate only objective distance potential (switch/exit focus)
        objective_pot = PBRSPotentials.objective_distance_potential(state)

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

    def get_potential_components(self, state: Dict[str, Any]) -> Dict[str, float]:
        """Get individual potential components for debugging/logging.

        Args:
            state: Game state dictionary

        Returns:
            dict: Dictionary of potential component values (completion-focused)
        """
        objective_pot = PBRSPotentials.objective_distance_potential(state)
        return {
            "objective": objective_pot,
            "switch_distance_potential": self.PBRS_SWITCH_DISTANCE * objective_pot
            if not state.get("switch_activated", False)
            else 0.0,
            "exit_distance_potential": self.PBRS_EXIT_DISTANCE * objective_pot
            if state.get("switch_activated", False)
            else 0.0,
            "hazard": 0.0,  # Disabled for completion focus
            "impact": 0.0,  # Disabled for completion focus
            "exploration": 0.0,  # Disabled for completion focus
        }

    def reset(self) -> None:
        """Reset calculator state for new episode."""
        self.visited_positions.clear()
