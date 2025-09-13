"""Main reward calculator that orchestrates all reward components."""
from typing import Dict, Any, Optional
from .navigation_reward_calculator import NavigationRewardCalculator
from .exploration_reward_calculator import ExplorationRewardCalculator
from .pbrs_potentials import PBRSCalculator


class RewardCalculator:
    """Main reward calculator."""
    BASE_TERMINAL_REWARD = 1.0
    DEATH_PENALTY = -0.5
    GOLD_REWARD = 0.00  # No gold reward for now. Lets just focus on completion
    DOOR_OPEN_REWARD = 0.01

    def __init__(self, 
                 enable_pbrs: bool = True,
                 pbrs_weights: Optional[Dict[str, float]] = None,
                 pbrs_gamma: float = 0.99):
        """Initialize reward calculator with all components.
        
        Args:
            enable_pbrs: Whether to enable potential-based reward shaping
            pbrs_weights: Weights for PBRS components (objective, hazard, impact, exploration)
            pbrs_gamma: Discount factor for PBRS (γ in r_shaped = r_env + γ * Φ(s') - Φ(s))
        """
        self.navigation_calculator = NavigationRewardCalculator()
        self.exploration_calculator = ExplorationRewardCalculator()
        self.steps_taken = 0
        self.total_gold_available = 0
        self.gold_collected = 0
        
        # PBRS configuration
        self.enable_pbrs = enable_pbrs
        self.pbrs_gamma = pbrs_gamma
        self.prev_potential = None
        
        # Initialize PBRS calculator with weights
        if pbrs_weights is None:
            pbrs_weights = {
                'objective_weight': 1.0,
                'hazard_weight': 0.5,
                'impact_weight': 0.3,
                'exploration_weight': 0.2
            }
        
        self.pbrs_calculator = PBRSCalculator(**pbrs_weights) if enable_pbrs else None

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
        navigation_reward, switch_active_changed = self.navigation_calculator.calculate_navigation_reward(
            obs, prev_obs
        )
        reward += navigation_reward

        # If our switch was activated or a door was opened, reset our exploration reward calculator
        # This is so that the agent is curious about areas its already been to when it activates the switch,
        # since the exit could be in any of the areas it has already explored
        if switch_active_changed or door_diff > 0:
            self.exploration_calculator.reset()

        # Exploration reward
        exploration_reward = self.exploration_calculator.calculate_exploration_reward(
            obs['player_x'], obs['player_y']
        )
        reward += exploration_reward

        # Win condition
        if obs.get('player_won', False):
            reward += self.BASE_TERMINAL_REWARD

        # Add PBRS shaping reward if enabled
        pbrs_reward = 0.0
        pbrs_components = {}
        if self.enable_pbrs and self.pbrs_calculator is not None:
            current_potential = self.pbrs_calculator.calculate_combined_potential(obs)
            
            # Get individual potential components for logging
            pbrs_components = self.pbrs_calculator.get_potential_components(obs)
            
            if self.prev_potential is not None:
                # r_shaped = r_env + γ * Φ(s') - Φ(s)
                pbrs_reward = self.pbrs_gamma * current_potential - self.prev_potential
                reward += pbrs_reward
            
            self.prev_potential = current_potential

        # Store component rewards for episode info
        self.last_pbrs_components = {
            'navigation_reward': navigation_reward,
            'exploration_reward': exploration_reward,
            'pbrs_reward': pbrs_reward,
            'pbrs_components': pbrs_components,
            'total_reward': reward
        }

        return reward

    def reset(self):
        """Reset all components for new episode."""
        self.navigation_calculator.reset()
        self.exploration_calculator.reset()
        self.steps_taken = 0
        
        # Reset PBRS state
        self.prev_potential = None
        if self.pbrs_calculator is not None:
            self.pbrs_calculator.reset()
    
    def get_reward_components(self, obs: Dict[str, Any]) -> Dict[str, float]:
        """Get individual reward components for debugging/logging.
        
        Args:
            obs: Current game state
            
        Returns:
            dict: Dictionary of reward component values
        """
        components = {}
        
        if self.enable_pbrs and self.pbrs_calculator is not None:
            components.update(self.pbrs_calculator.get_potential_components(obs))
            components['combined_potential'] = self.pbrs_calculator.calculate_combined_potential(obs)
        
        return components
