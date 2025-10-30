"""Navigation reward calculator for evaluating objective-based movement and progress."""

from typing import Dict, Any, Tuple
from ..constants import LEVEL_DIAGONAL
from ..util.util import calculate_distance
from .reward_constants import (
    NAVIGATION_DISTANCE_IMPROVEMENT_SCALE,
    NAVIGATION_POTENTIAL_SCALE,
    NAVIGATION_MIN_DISTANCE_THRESHOLD,
)

# Use full level diagonal for normalization - handles all valid distances within level
# LEVEL_DIAGONAL = 1214.55, ensures any position within [0,1056]x[0,600] normalizes correctly
# Previous value (LEVEL_DIAGONAL / 4 = 303.64) was too small, causing negative potentials
# when distance > 303.64 pixels, which is common in mazes and large levels
PBRS_DISTANCE_SCALE = LEVEL_DIAGONAL


class NavigationRewardCalculator:
    """Handles calculation of completion-focused navigation rewards.

    Provides distance-based shaping rewards for switch and exit objectives
    using potential-based reward shaping (PBRS) theory from Ng et al. (1999).

    Key features:
    - Distance-based shaping provides dense gradient for learning
    - Potential-based formulation ensures policy invariance
    - Rewards progress toward current objective (switch → exit)
    - Switch activation milestone handled by main calculator

    All constants defined in reward_constants.py with full documentation.
    """

    def __init__(self):
        """Initialize navigation reward calculator."""
        super().__init__()
        self.closest_distance_to_switch = float("inf")
        self.closest_distance_to_exit = float("inf")
        self.prev_potential = None
        self.episode_start_switch_distance = None
        self.episode_start_exit_distance = None

    def get_progress_estimate(self, state: Dict[str, Any]) -> float:
        """Estimate progress through the level (0.0 to 1.0)."""
        if not state["switch_activated"]:
            if self.episode_start_switch_distance is None:
                return 0.0

            curr_distance = calculate_distance(
                state["player_x"],
                state["player_y"],
                state["switch_x"],
                state["switch_y"],
            )
            # Progress is how close we are to switch relative to starting distance
            progress = 1.0 - (curr_distance / self.episode_start_switch_distance)
            # Scale to 0.0-0.5 range since this is first half of level
            return max(0.0, min(0.5, progress * 0.5))
        else:
            if self.episode_start_exit_distance is None:
                return 0.5  # Just activated switch

            curr_distance = calculate_distance(
                state["player_x"],
                state["player_y"],
                state["exit_door_x"],
                state["exit_door_y"],
            )
            # Progress is how close we are to exit relative to starting distance
            progress = 1.0 - (curr_distance / self.episode_start_exit_distance)
            # Scale to 0.5-1.0 range since this is second half of level
            return 0.5 + max(0.0, min(0.5, progress * 0.5))

    def calculate_potential(self, state: Dict[str, Any]) -> float:
        """Calculate state potential with proper normalization.

        Uses full LEVEL_DIAGONAL normalization to guarantee non-negative potentials
        for all valid positions within level bounds.

        Returns:
            float: Potential value, guaranteed to be >= 0.0
        """
        # Determine objective position based on switch state
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

        # Normalize distance using FULL level diagonal (not /4)
        # This ensures distance/scale <= 1.0 for all valid positions
        normalized_dist = min(1.0, distance / PBRS_DISTANCE_SCALE)

        # Base potential: higher when closer (always non-negative now)
        base_potential = NAVIGATION_POTENTIAL_SCALE * (1.0 - normalized_dist)

        # Proximity bonus for being very close to objective
        proximity_bonus = 0.0
        if distance < NAVIGATION_MIN_DISTANCE_THRESHOLD:
            proximity_bonus = NAVIGATION_POTENTIAL_SCALE * 0.5

        # Total potential (guaranteed >= 0 with correct normalization)
        total_potential = base_potential + proximity_bonus

        # Defensive: ensure non-negative (should always be true with correct scale)
        return max(0.0, total_potential)

    def calculate_navigation_reward(
        self, curr_state: Dict[str, Any], prev_state: Dict[str, Any]
    ) -> Tuple[float, bool]:
        """Calculate navigation reward based on movement towards/away from goals."""
        reward = 0.0
        switch_active_changed = False

        # Calculate current distances
        curr_distance_to_switch = calculate_distance(
            curr_state["player_x"],
            curr_state["player_y"],
            curr_state["switch_x"],
            curr_state["switch_y"],
        )
        curr_distance_to_exit = calculate_distance(
            curr_state["player_x"],
            curr_state["player_y"],
            curr_state["exit_door_x"],
            curr_state["exit_door_y"],
        )

        # Initialize episode start distances
        if self.episode_start_switch_distance is None:
            self.episode_start_switch_distance = curr_distance_to_switch
            self.closest_distance_to_switch = curr_distance_to_switch
        if curr_state["switch_activated"] and self.episode_start_exit_distance is None:
            self.episode_start_exit_distance = curr_distance_to_exit
            self.closest_distance_to_exit = curr_distance_to_exit

        if not curr_state["switch_activated"]:
            # Only reward if we've reached a new closest distance to switch
            if curr_distance_to_switch < self.closest_distance_to_switch:
                distance_improvement = (
                    self.closest_distance_to_switch - curr_distance_to_switch
                )
                reward += distance_improvement * NAVIGATION_DISTANCE_IMPROVEMENT_SCALE
                self.closest_distance_to_switch = curr_distance_to_switch
        else:
            # Switch activated
            if not prev_state["switch_activated"]:
                self.closest_distance_to_exit = curr_distance_to_exit
                switch_active_changed = True
            else:
                # Only reward if we've reached a new closest distance to exit
                if curr_distance_to_exit < self.closest_distance_to_exit:
                    distance_improvement = (
                        self.closest_distance_to_exit - curr_distance_to_exit
                    )
                    reward += (
                        distance_improvement * NAVIGATION_DISTANCE_IMPROVEMENT_SCALE
                    )
                    self.closest_distance_to_exit = curr_distance_to_exit

        # Calculate potential-based shaping reward
        current_potential = self.calculate_potential(curr_state)
        if self.prev_potential is not None:
            shaping_reward = current_potential - self.prev_potential
            reward += shaping_reward
        self.prev_potential = current_potential

        return reward, switch_active_changed

    def reset(self):
        """Reset internal state for new episode."""
        self.closest_distance_to_switch = float("inf")
        self.closest_distance_to_exit = float("inf")
        self.prev_potential = None
        self.episode_start_switch_distance = None
        self.episode_start_exit_distance = None
