"""Main reward calculator that orchestrates all reward components."""
from typing import Dict, Any
from nclone_environments.basic_level_no_gold.reward_calculation.navigation_reward_calculator import NavigationRewardCalculator
from nclone_environments.basic_level_no_gold.reward_calculation.exploration_reward_calculator import ExplorationRewardCalculator


class RewardCalculator:
    """Main reward calculator."""
    BASE_TERMINAL_REWARD = 1.0
    DEATH_PENALTY = -0.5
    GOLD_REWARD = 0.01
    DOOR_OPEN_REWARD = 0.01

    def __init__(self):
        """Initialize reward calculator with all components."""
        self.navigation_calculator = NavigationRewardCalculator()
        self.exploration_calculator = ExplorationRewardCalculator()
        self.steps_taken = 0
        self.total_gold_available = 0
        self.gold_collected = 0

    def calculate_reward(self, obs: Dict[str, Any], prev_obs: Dict[str, Any]) -> float:
        """Calculate reward.

        Args:
            obs: Current game state
            prev_obs: Previous game state

        Returns:
            float: Total reward for the transition
        """
        self.steps_taken += 1

        # Termination penalties
        if obs.get('player_dead', False):
            return self.DEATH_PENALTY

        # Initialize reward
        reward = 0.0

        # Add gold reward for the difference in gold collected
        gold_diff = obs.get('gold_collected', 0) - \
            prev_obs.get('gold_collected', 0)
        reward += self.GOLD_REWARD * gold_diff

        # Add door open reward for the difference in doors opened
        door_diff = obs.get('doors_opened', 0) - \
            prev_obs.get('doors_opened', 0)
        reward += self.DOOR_OPEN_REWARD * door_diff

        # Navigation reward with progressive scaling
        navigation_reward = self.navigation_calculator.calculate_navigation_reward(
            obs, prev_obs
        )
        reward += navigation_reward

        # Exploration reward
        exploration_reward = self.exploration_calculator.calculate_exploration_reward(
            obs['player_x'], obs['player_y']
        )
        reward += exploration_reward

        # Win condition
        if obs.get('player_won', False):
            reward += self.BASE_TERMINAL_REWARD

        return reward

    def reset(self):
        """Reset all components for new episode."""
        self.navigation_calculator.reset()
        self.exploration_calculator.reset()
        self.steps_taken = 0
