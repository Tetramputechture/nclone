"""Main reward calculator that orchestrates all reward components."""
from typing import Dict, Any
from nclone_environments.basic_level_no_gold.reward_calculation.navigation_reward_calculator import NavigationRewardCalculator


class RewardCalculator:
    """
    A curriculum-based reward calculator for the N++ environment that progressively
    adapts rewards based on the agent's demonstrated capabilities and learning stage.

    The calculator implements three main learning stages:
    2. Navigation: Efficient path-finding and objective targeting

    Each stage builds upon the skills learned in previous stages, with rewards
    automatically adjusting based on the agent's demonstrated competence.
    """
    TERMINAL_REWARD = 1.0
    DEATH_PENALTY = -0.1
    BASE_TIME_PENALTY = -0.001
    GOLD_REWARD = 0.01

    def __init__(self):
        """Initialize reward calculator with all components."""
        self.navigation_calculator = NavigationRewardCalculator()

    def calculate_reward(self, obs: Dict[str, Any], prev_obs: Dict[str, Any]) -> float:
        """Calculate reward.

        Args:
            obs: Current game state

        Returns:
            float: Total reward for the transition
        """
        # Termination penalties
        if obs.get('player_dead', False):
            return self.DEATH_PENALTY

        # Win condition
        if obs.get('player_won', False):
            return self.TERMINAL_REWARD

        # Initialize reward
        reward = 0.0

        # Add time pressure
        reward += self.BASE_TIME_PENALTY

        # Add gold reward for the difference in gold collected
        gold_diff = obs.get('gold_collected', 0) - \
            prev_obs.get('gold_collected', 0)
        if gold_diff > 0:
            print(f'gold collected: {gold_diff}')
        reward += self.GOLD_REWARD * gold_diff

        # Navigation reward with progressive scaling
        navigation_reward = self.navigation_calculator.calculate_navigation_reward(
            obs, prev_obs
        )
        reward += navigation_reward

        return reward

    def reset(self):
        """Reset all components for new episode."""
        self.navigation_calculator.reset()
