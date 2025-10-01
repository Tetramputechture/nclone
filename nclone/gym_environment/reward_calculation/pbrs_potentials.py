"""Potential-Based Reward Shaping (PBRS) potential functions.

This module implements reusable potential functions Φ(s) for reward shaping
without changing the optimal policy. Each potential is normalized to comparable
scales and documented with units and bounds.
"""

import numpy as np
from typing import Dict, Any, List, Tuple
from ..constants import LEVEL_WIDTH, LEVEL_HEIGHT
from ..util.util import calculate_distance


class PBRSPotentials:
    """Collection of potential functions for reward shaping."""

    # Normalization constants
    LEVEL_DIAGONAL = np.sqrt(LEVEL_WIDTH**2 + LEVEL_HEIGHT**2)
    MAX_VELOCITY = 10.0  # Approximate max ninja velocity
    HAZARD_DANGER_RADIUS = 50.0  # Radius within which hazards are considered dangerous

    @staticmethod
    def objective_distance_potential(state: Dict[str, Any]) -> float:
        """Potential based on distance to nearest objective.

        Returns higher potential when closer to the current objective:
        - Switch when inactive
        - Exit when switch is active

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

        # Normalize distance to [0, 1] range, inverted so closer = higher potential
        normalized_distance = min(1.0, distance / PBRSPotentials.LEVEL_DIAGONAL)
        return 1.0 - normalized_distance

    @staticmethod
    def hazard_proximity_potential(state: Dict[str, Any]) -> float:
        """Potential penalty based on proximity to active hazards.

        Currently simplified - returns neutral potential since entity parsing
        from flattened array is complex. Could be enhanced in future versions.

        Args:
            state: Game state dictionary

        Returns:
            float: Potential in range [0.0, 1.0], currently returns 1.0 (neutral)
        """
        # TODO: Implement hazard detection from flattened entity_states array
        # For now, return neutral potential to avoid errors
        return 1.0

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
        velocity_risk = min(1.0, velocity_magnitude / PBRSPotentials.MAX_VELOCITY)
        downward_risk = min(1.0, downward_velocity / PBRSPotentials.MAX_VELOCITY)
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
        exploration_radius = 30.0

        # Return potential based on distance from visited areas
        if min_distance > exploration_radius:
            return 1.0  # Far from visited areas, high exploration potential
        else:
            # Linear decay within exploration radius
            return min_distance / exploration_radius


class PBRSCalculator:
    """Calculator for completion-focused potential functions."""

    # Completion-focused PBRS scaling
    PBRS_SWITCH_DISTANCE = 0.05  # Distance-based shaping to switches
    PBRS_EXIT_DISTANCE = 0.05    # Distance-based shaping to exit

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
            hazard_weight: Disabled for completion focus
            impact_weight: Disabled for completion focus
            exploration_weight: Disabled for completion focus
        """
        self.objective_weight = objective_weight
        self.hazard_weight = hazard_weight
        self.impact_weight = impact_weight
        self.exploration_weight = exploration_weight

        # Track visited positions for exploration potential (minimal usage)
        self.visited_positions: List[Tuple[float, float]] = []
        self.visit_threshold = (
            25.0  # Distance threshold for considering a position "new"
        )

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
            "switch_distance_potential": self.PBRS_SWITCH_DISTANCE * objective_pot if not state.get("switch_activated", False) else 0.0,
            "exit_distance_potential": self.PBRS_EXIT_DISTANCE * objective_pot if state.get("switch_activated", False) else 0.0,
            "hazard": 0.0,  # Disabled for completion focus
            "impact": 0.0,  # Disabled for completion focus
            "exploration": 0.0,  # Disabled for completion focus
        }

    def reset(self) -> None:
        """Reset calculator state for new episode."""
        self.visited_positions.clear()
