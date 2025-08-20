"""Navigation reward calculator for evaluating objective-based movement and progress."""
from typing import Dict, Any, Tuple
import numpy as np
from ..constants import LEVEL_WIDTH, LEVEL_HEIGHT
from ...util.util import calculate_distance


class NavigationRewardCalculator:
    """Handles calculation of navigation and objective-based rewards."""

    # Navigation constants
    # Scaled to be meaningful compared to -0.001 per step
    DISTANCE_IMPROVEMENT_SCALE = 0.0001
    MIN_DISTANCE_THRESHOLD = 20.0
    SWITCH_ACTIVATION_REWARD = 0.5
    # Scale potential-based shaping to be smaller than main rewards
    POTENTIAL_SCALE = 0.0005

    def __init__(self):
        """Initialize navigation reward calculator."""
        super().__init__()
        self.closest_distance_to_switch = float('inf')
        self.closest_distance_to_exit = float('inf')
        self.prev_potential = None
        self.episode_start_switch_distance = None
        self.episode_start_exit_distance = None

    def get_progress_estimate(self, state: Dict[str, Any]) -> float:
        """Estimate progress through the level (0.0 to 1.0)."""
        if not state['switch_activated']:
            if self.episode_start_switch_distance is None:
                return 0.0

            curr_distance = calculate_distance(
                state['player_x'], state['player_y'],
                state['switch_x'], state['switch_y']
            )
            # Progress is how close we are to switch relative to starting distance
            progress = 1.0 - (curr_distance /
                              self.episode_start_switch_distance)
            # Scale to 0.0-0.5 range since this is first half of level
            return max(0.0, min(0.5, progress * 0.5))
        else:
            if self.episode_start_exit_distance is None:
                return 0.5  # Just activated switch

            curr_distance = calculate_distance(
                state['player_x'], state['player_y'],
                state['exit_door_x'], state['exit_door_y']
            )
            # Progress is how close we are to exit relative to starting distance
            progress = 1.0 - (curr_distance / self.episode_start_exit_distance)
            # Scale to 0.5-1.0 range since this is second half of level
            return 0.5 + max(0.0, min(0.5, progress * 0.5))

    def calculate_potential(self, state: Dict[str, Any]) -> float:
        """Calculate state potential for reward shaping."""
        # Scale based on level size
        level_diagonal = np.sqrt(LEVEL_WIDTH**2 + LEVEL_HEIGHT**2)
        distance_scale = level_diagonal / 4

        if not state['switch_activated']:
            distance_to_switch = calculate_distance(
                state['player_x'], state['player_y'],
                state['switch_x'], state['switch_y']
            )
            # Potential based on normalized distance to switch
            potential = self.POTENTIAL_SCALE * \
                (1.0 - distance_to_switch / distance_scale)

            # Small bonus for being very close to switch
            if distance_to_switch < self.MIN_DISTANCE_THRESHOLD:
                potential += self.POTENTIAL_SCALE * 0.5

            return potential
        else:
            distance_to_exit = calculate_distance(
                state['player_x'], state['player_y'],
                state['exit_door_x'], state['exit_door_y']
            )
            # Potential based on normalized distance to exit
            potential = self.POTENTIAL_SCALE * \
                (1.0 - distance_to_exit / distance_scale)

            # Small bonus for being very close to exit
            if distance_to_exit < self.MIN_DISTANCE_THRESHOLD:
                potential += self.POTENTIAL_SCALE * 0.5

            return potential

    def calculate_navigation_reward(self,
                                    curr_state: Dict[str, Any],
                                    prev_state: Dict[str, Any]) -> Tuple[float, bool]:
        """Calculate navigation reward based on movement towards/away from goals."""
        reward = 0.0
        switch_active_changed = False

        # Calculate current distances
        curr_distance_to_switch = calculate_distance(
            curr_state['player_x'], curr_state['player_y'],
            curr_state['switch_x'], curr_state['switch_y']
        )
        curr_distance_to_exit = calculate_distance(
            curr_state['player_x'], curr_state['player_y'],
            curr_state['exit_door_x'], curr_state['exit_door_y']
        )

        # Initialize episode start distances
        if self.episode_start_switch_distance is None:
            self.episode_start_switch_distance = curr_distance_to_switch
            self.closest_distance_to_switch = curr_distance_to_switch
        if curr_state['switch_activated'] and self.episode_start_exit_distance is None:
            self.episode_start_exit_distance = curr_distance_to_exit
            self.closest_distance_to_exit = curr_distance_to_exit

        if not curr_state['switch_activated']:
            # Only reward if we've reached a new closest distance to switch
            if curr_distance_to_switch < self.closest_distance_to_switch:
                distance_improvement = self.closest_distance_to_switch - curr_distance_to_switch
                reward += distance_improvement * self.DISTANCE_IMPROVEMENT_SCALE
                self.closest_distance_to_switch = curr_distance_to_switch
        else:
            # Reward for activating switch
            if not prev_state['switch_activated']:
                reward += self.SWITCH_ACTIVATION_REWARD
                self.closest_distance_to_exit = curr_distance_to_exit
                switch_active_changed = True
            else:
                # Only reward if we've reached a new closest distance to exit
                if curr_distance_to_exit < self.closest_distance_to_exit:
                    distance_improvement = self.closest_distance_to_exit - curr_distance_to_exit
                    reward += distance_improvement * self.DISTANCE_IMPROVEMENT_SCALE
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
        self.closest_distance_to_switch = float('inf')
        self.closest_distance_to_exit = float('inf')
        self.prev_potential = None
        self.episode_start_switch_distance = None
        self.episode_start_exit_distance = None
