"""Main reward calculator that orchestrates all reward components."""
from typing import Dict, Any
from nclone_environments.basic_level_no_gold.reward_calculation.navigation_reward_calculator import NavigationRewardCalculator


class RewardCalculator:
    """Main reward calculator."""
    BASE_TERMINAL_REWARD = 1.0
    DEATH_PENALTY = -0.1
    BASE_TIME_PENALTY = -0.001
    GOLD_REWARD = 0.01

    def __init__(self):
        """Initialize reward calculator with all components."""
        self.navigation_calculator = NavigationRewardCalculator()
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

        # Add time pressure - scales with progress to encourage efficiency
        progress = self.navigation_calculator.get_progress_estimate(obs)

        # Higher penalty when closer to goal
        reward += self.BASE_TIME_PENALTY * (1.0 + progress)

        # Add gold reward for the difference in gold collected
        gold_diff = obs.get('gold_collected', 0) - \
            prev_obs.get('gold_collected', 0)
        reward += self.GOLD_REWARD * gold_diff

        # Navigation reward with progressive scaling
        navigation_reward = self.navigation_calculator.calculate_navigation_reward(
            obs, prev_obs
        )
        reward += navigation_reward

        # Win condition
        if obs.get('player_won', False):
            # Base completion reward
            reward += self.BASE_TERMINAL_REWARD

        return reward

    def reset(self):
        """Reset all components for new episode."""
        self.navigation_calculator.reset()
        self.steps_taken = 0
