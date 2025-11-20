"""Simplified reward calculator with curriculum-aware component lifecycle.

Focused reward system with clear hierarchy:
1. Terminal rewards (completion, death) - Define task success/failure
2. PBRS objective potential - Policy-invariant guidance to goal
3. Time penalty - Efficiency pressure (curriculum-controlled)

All redundant components removed for clarity.
"""

import logging
from typing import Dict, Any, Optional
from .reward_config import RewardConfig
from .pbrs_potentials import PBRSCalculator
from .reward_constants import (
    LEVEL_COMPLETION_REWARD,
    DEATH_PENALTY,
    MINE_DEATH_PENALTY,
    SWITCH_ACTIVATION_REWARD,
    PBRS_GAMMA,
)
from ...graph.reachability.path_distance_calculator import (
    CachedPathDistanceCalculator,
)


logger = logging.getLogger(__name__)


class RewardCalculator:
    """Simplified reward calculator with curriculum-aware lifecycle.

    Components:
    - Terminal rewards: Always active (completion, death, switch milestone)
    - PBRS objective potential: Always active (curriculum-scaled weight)
    - Time penalty: Conditionally active (curriculum-controlled)

    Removed components (redundant/confusing):
    - Exploration rewards (PBRS provides via potential gradients)
    - Progress bonuses (PBRS already rewards progress)
    - Backtrack penalties (confusing, PBRS handles naturally)
    - NOOP penalties (let PBRS guide, don't punish stillness)
    - Buffer bonuses (gameplay mechanic, not learning signal)
    - Hazard/impact PBRS (death penalty is clearer signal)
    """

    def __init__(
        self,
        reward_config: Optional[RewardConfig] = None,
        pbrs_gamma: float = PBRS_GAMMA,
    ):
        """Initialize simplified reward calculator.

        Args:
            reward_config: RewardConfig instance managing curriculum-aware component lifecycle
            pbrs_gamma: Discount factor for PBRS (γ in F(s,s') = γ * Φ(s') - Φ(s))
        """
        # Single config object manages all curriculum logic
        self.config = reward_config or RewardConfig()

        # PBRS configuration
        self.pbrs_gamma = pbrs_gamma
        self.prev_potential = None

        # Create path calculator for PBRS
        path_calculator = CachedPathDistanceCalculator(
            max_cache_size=200, use_astar=True
        )
        self.pbrs_calculator = PBRSCalculator(path_calculator=path_calculator)

        # Physics discovery rewards (NEW)
        if self.config.enable_physics_discovery:
            from .physics_discovery_rewards import PhysicsDiscoveryRewards

            self.physics_discovery = PhysicsDiscoveryRewards()
        else:
            self.physics_discovery = None

        self.steps_taken = 0

        # Track closest distances for diagnostic metrics only
        self.closest_distance_to_switch = float("inf")
        self.closest_distance_to_exit = float("inf")

    def calculate_reward(
        self,
        obs: Dict[str, Any],
        prev_obs: Dict[str, Any],
        action: Optional[int] = None,
    ) -> float:
        """Calculate simplified, curriculum-aware reward.

        Components:
        1. Terminal rewards (always active)
        2. Time penalty (curriculum-controlled via config)
        3. PBRS objective potential (curriculum-scaled via config)

        Args:
            obs: Current game state
            prev_obs: Previous game state
            action: Action taken (optional, unused in simplified version)

        Returns:
            float: Total reward for the transition
        """
        self.steps_taken += 1

        # === TERMINAL REWARDS (always active) ===

        # Death penalties
        if obs.get("player_dead", False):
            death_cause = obs.get("death_cause", None)
            return MINE_DEATH_PENALTY if death_cause == "mine" else DEATH_PENALTY

        # Completion reward
        if obs.get("player_won", False):
            return LEVEL_COMPLETION_REWARD

        # === TIME PENALTY (curriculum-controlled) ===
        # Config determines if active and magnitude based on training phase
        reward = self.config.time_penalty_per_step

        # === MILESTONE REWARD ===
        switch_just_activated = obs.get("switch_activated", False) and not prev_obs.get(
            "switch_activated", False
        )
        if switch_just_activated:
            reward += SWITCH_ACTIVATION_REWARD

        # === PBRS SHAPING (curriculum-scaled) ===
        adjacency = obs.get("_adjacency_graph")
        level_data = obs.get("level_data")
        graph_data = obs.get("_graph_data")

        # Validate required data for PBRS
        if not adjacency or not level_data:
            raise ValueError(
                "PBRS requires adjacency graph and level_data in observation. "
                "Ensure graph building is enabled in environment config."
            )

        # Calculate potential with curriculum-scaled weight and normalization
        current_potential = self.pbrs_calculator.calculate_combined_potential(
            obs,
            adjacency=adjacency,
            level_data=level_data,
            graph_data=graph_data,
            objective_weight=self.config.pbrs_objective_weight,
            scale_factor=self.config.pbrs_normalization_scale,
        )

        # Apply PBRS shaping: F(s,s') = γ * Φ(s') - Φ(s)
        if self.prev_potential is not None:
            if switch_just_activated:
                # Prevent discontinuity at switch activation
                # Switch milestone reward (+2.0) is sufficient signal
                pbrs_reward = 0.0
            else:
                # Normal PBRS: positive when moving closer, negative when moving away
                pbrs_reward = self.pbrs_gamma * current_potential - self.prev_potential
            reward += pbrs_reward

        # Update state for next step
        self.prev_potential = current_potential

        # Track diagnostic metrics (Euclidean distance)
        from ..util.util import calculate_distance

        distance_to_switch = calculate_distance(
            obs["player_x"], obs["player_y"], obs["switch_x"], obs["switch_y"]
        )
        if distance_to_switch < self.closest_distance_to_switch:
            self.closest_distance_to_switch = distance_to_switch

        if obs.get("switch_activated", False):
            distance_to_exit = calculate_distance(
                obs["player_x"], obs["player_y"], obs["exit_door_x"], obs["exit_door_y"]
            )
            if distance_to_exit < self.closest_distance_to_exit:
                self.closest_distance_to_exit = distance_to_exit

        # === PHYSICS DISCOVERY REWARDS (curriculum-controlled) ===
        if self.config.enable_physics_discovery and self.physics_discovery is not None:
            physics_rewards = self.physics_discovery.calculate_physics_rewards(
                obs, prev_obs, action
            )
            total_physics_reward = (
                sum(physics_rewards.values()) * self.config.physics_discovery_weight
            )
            reward += total_physics_reward

        return reward

    def reset(self):
        """Reset episode state for new episode."""
        self.steps_taken = 0
        self.prev_potential = None

        # Reset diagnostic tracking
        self.closest_distance_to_switch = float("inf")
        self.closest_distance_to_exit = float("inf")

        # Reset PBRS calculator state
        if self.pbrs_calculator is not None:
            self.pbrs_calculator.reset()

        # Reset physics discovery rewards
        if self.physics_discovery is not None:
            self.physics_discovery.reset()

    def update_config(self, timesteps: int, success_rate: float) -> None:
        """Update reward configuration based on training progress.

        Called by trainer at evaluation checkpoints to trigger curriculum transitions.

        Args:
            timesteps: Total timesteps trained so far
            success_rate: Recent evaluation success rate (0.0-1.0)
        """
        old_phase = self.config.training_phase
        self.config.update(timesteps, success_rate)

        # Log phase transitions
        if self.config.training_phase != old_phase:
            logger.info(
                f"\n{'=' * 60}\n"
                f"REWARD PHASE TRANSITION: {old_phase} → {self.config.training_phase}\n"
                f"Timesteps: {timesteps:,}\n"
                f"Success Rate: {success_rate:.1%}\n"
                f"Active Components:\n"
                f"  PBRS Weight: {self.config.pbrs_objective_weight:.2f}\n"
                f"  Time Penalty: {self.config.time_penalty_per_step:.4f}/step\n"
                f"  Normalization Scale: {self.config.pbrs_normalization_scale:.2f}\n"
                f"{'=' * 60}\n"
            )

    def get_config_state(self) -> Dict[str, Any]:
        """Get current reward configuration state for logging.

        Returns:
            Dictionary with current configuration values
        """
        return self.config.get_active_components()
