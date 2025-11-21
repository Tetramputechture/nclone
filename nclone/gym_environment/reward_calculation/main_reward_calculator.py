"""Simplified reward calculator with curriculum-aware component lifecycle.

Focused reward system with clear hierarchy:
1. Terminal rewards (completion, death) - Define task success/failure
2. PBRS objective potential - Policy-invariant guidance to goal
3. Time penalty - Efficiency pressure (curriculum-controlled)

All redundant components removed for clarity.
"""

import logging
from typing import Dict, Any, Optional, Tuple
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

        # PBRS configuration (kept for backwards compatibility with path calculator)
        self.pbrs_gamma = pbrs_gamma

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

        # Track best path distances for discrete achievement bonuses
        self.best_path_distance_to_switch = float("inf")
        self.best_path_distance_to_exit = float("inf")

        # Track total distance bonuses awarded this episode
        self.total_distance_bonuses_awarded = 0.0

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

        # === DISCRETE DISTANCE ACHIEVEMENT BONUSES (curriculum-scaled) ===
        adjacency = obs.get("_adjacency_graph")
        level_data = obs.get("level_data")
        graph_data = obs.get("_graph_data")

        # Validate required data for distance calculation
        if not adjacency or not level_data:
            raise ValueError(
                "Distance calculation requires adjacency graph and level_data in observation. "
                "Ensure graph building is enabled in environment config."
            )

        # Calculate discrete distance achievement bonuses
        distance_bonus = self._calculate_distance_achievement_bonus(
            obs,
            adjacency=adjacency,
            level_data=level_data,
            graph_data=graph_data,
            switch_just_activated=switch_just_activated,
        )
        reward += distance_bonus

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

    def _calculate_distance_achievement_bonus(
        self,
        obs: Dict[str, Any],
        adjacency: Dict[Tuple[int, int], list],
        level_data: Any,
        graph_data: Optional[Dict[str, Any]] = None,
        switch_just_activated: bool = False,
    ) -> float:
        """Calculate discrete distance achievement bonuses for new closest distances.

        Uses identical path calculation logic as original PBRS system but only awards
        bonuses when agent achieves new closest path distance to current objective.

        Args:
            obs: Current game state
            adjacency: Graph adjacency structure
            level_data: Level data object
            graph_data: Graph data dict with spatial_hash for optimization
            switch_just_activated: Whether switch was just activated this step

        Returns:
            Distance achievement bonus (0.0 if no improvement)
        """
        from ...constants.physics_constants import (
            EXIT_SWITCH_RADIUS,
            EXIT_DOOR_RADIUS,
            NINJA_RADIUS,
        )

        # No bonus during switch activation transition (consistent with original PBRS)
        if switch_just_activated:
            return 0.0

        # Determine current objective and goal position
        if not obs.get("switch_activated", False):
            goal_pos = (int(obs["switch_x"]), int(obs["switch_y"]))
            cache_key = "switch"
            entity_radius = EXIT_SWITCH_RADIUS
            current_best = self.best_path_distance_to_switch
        else:
            goal_pos = (int(obs["exit_door_x"]), int(obs["exit_door_y"]))
            cache_key = "exit"
            entity_radius = EXIT_DOOR_RADIUS
            current_best = self.best_path_distance_to_exit

        player_pos = (int(obs["player_x"]), int(obs["player_y"]))

        # Calculate current path distance using same logic as PBRS
        try:
            current_distance = self.pbrs_calculator.path_calculator.get_distance(
                player_pos,
                goal_pos,
                adjacency,
                cache_key=cache_key,
                level_data=level_data,
                graph_data=graph_data,
                entity_radius=entity_radius,
                ninja_radius=NINJA_RADIUS,
            )
        except Exception as e:
            # Fallback: no bonus if distance calculation fails
            logger.warning(f"Distance calculation failed: {e}")
            return 0.0

        # Get surface area for level-adaptive thresholds
        surface_area = obs.get("_pbrs_surface_area", 200.0)  # Default fallback

        # Estimate max path distance as 4x sqrt(surface_area) as suggested
        import math

        estimated_max_path_distance = 4.0 * math.sqrt(surface_area)

        # Set meaningful improvement threshold proportional to level size
        min_improvement_threshold = max(
            5.0, estimated_max_path_distance * 0.02
        )  # 2% of max distance

        if current_distance < current_best:
            # Calculate improvement amount
            if current_best == float("inf"):
                # First distance measurement - baseline proportional to level size
                distance_improvement = (
                    estimated_max_path_distance * 0.1
                )  # 10% of estimated max
            else:
                distance_improvement = current_best - current_distance

                # Only award bonus for meaningful improvements relative to level size
                if distance_improvement < min_improvement_threshold:
                    return 0.0

            # Apply curriculum scaling (same as original PBRS)
            curriculum_weight = self.config.pbrs_objective_weight
            scale_factor = self.config.pbrs_normalization_scale

            # Normalize improvement by estimated max distance for level-adaptive scaling
            normalized_improvement = distance_improvement / estimated_max_path_distance

            # Calculate base bonus proportional to normalized improvement
            # This ensures similar bonus magnitudes across different level sizes
            base_bonus = (
                normalized_improvement * curriculum_weight * scale_factor * 2.0
            )  # Scale factor for reasonable magnitudes

            # Apply episode length normalization to discourage inefficient meandering
            bonus = self._apply_episode_length_normalization(base_bonus, obs)

            # Cap bonus to prevent excessive rewards (max ~0.2 per achievement)
            bonus = min(bonus, 0.2)

            # Check episode total bonus cap to prevent accumulation exploitation
            max_total_bonuses_per_episode = (
                1.0  # Very conservative maximum total bonuses per episode
            )
            if (
                self.total_distance_bonuses_awarded + bonus
                > max_total_bonuses_per_episode
            ):
                # Reduce bonus to stay within episode limit
                bonus = max(
                    0.0,
                    max_total_bonuses_per_episode - self.total_distance_bonuses_awarded,
                )

            # Track total bonuses awarded this episode
            self.total_distance_bonuses_awarded += bonus

            # Update best distance tracking
            if not obs.get("switch_activated", False):
                self.best_path_distance_to_switch = current_distance
            else:
                self.best_path_distance_to_exit = current_distance

            logger.debug(
                f"Distance achievement bonus: {bonus:.3f} "
                f"(improvement: {distance_improvement:.1f}, "
                f"objective: {cache_key}, distance: {current_distance:.1f})"
            )

            return bonus

        # No improvement - no bonus
        return 0.0

    def _apply_episode_length_normalization(
        self, base_bonus: float, obs: Dict[str, Any]
    ) -> float:
        """Apply episode length normalization to discourage inefficient meandering.

        Reduces bonuses for episodes that are taking much longer than optimal,
        while preserving rewards for complex navigation in large levels.

        Args:
            base_bonus: Unnormalized distance achievement bonus
            obs: Current game state (contains surface area info)

        Returns:
            Normalized bonus scaled by episode efficiency
        """
        # Get surface area for optimal length estimation
        surface_area = obs.get("_pbrs_surface_area")
        if not surface_area:
            # No normalization if surface area unavailable
            return base_bonus

        # Estimate optimal episode length based on level complexity
        # Formula: base time for navigation + complexity scaling
        # Larger levels get proportionally more time
        import math

        base_optimal_steps = 150  # Base time for simple levels (reduced)
        complexity_factor = (
            math.sqrt(surface_area) * 1.5
        )  # Scale with sqrt(area) (reduced)
        optimal_episode_length = base_optimal_steps + complexity_factor

        # Current episode length
        current_length = self.steps_taken

        # Apply normalization: longer episodes get reduced bonuses
        # Formula: min(1.0, optimal_length / current_length)
        # - Episodes at or below optimal length: full bonus (ratio >= 1.0)
        # - Episodes above optimal length: reduced bonus proportional to efficiency
        if current_length <= optimal_episode_length:
            efficiency_factor = 1.0  # Full bonus for efficient episodes
        else:
            efficiency_factor = optimal_episode_length / current_length
            # Don't penalize too harshly - minimum 0.1x multiplier
            efficiency_factor = max(efficiency_factor, 0.1)

        normalized_bonus = base_bonus * efficiency_factor

        # Log normalization for debugging
        if efficiency_factor < 1.0:
            logger.debug(
                f"Episode length normalization: {efficiency_factor:.3f} "
                f"(steps: {current_length}, optimal: {optimal_episode_length:.0f}, "
                f"surface_area: {surface_area:.0f})"
            )

        return normalized_bonus

    def reset(self):
        """Reset episode state for new episode."""
        self.steps_taken = 0

        # Reset diagnostic tracking
        self.closest_distance_to_switch = float("inf")
        self.closest_distance_to_exit = float("inf")

        # Reset distance achievement tracking
        self.best_path_distance_to_switch = float("inf")
        self.best_path_distance_to_exit = float("inf")
        self.total_distance_bonuses_awarded = 0.0

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
