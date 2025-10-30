"""Main reward calculator that orchestrates all reward components."""

from typing import Dict, Any, Optional
from .exploration_reward_calculator import ExplorationRewardCalculator
from .pbrs_potentials import PBRSCalculator
from .reward_constants import (
    LEVEL_COMPLETION_REWARD,
    DEATH_PENALTY,
    SWITCH_ACTIVATION_REWARD,
    TIME_PENALTY_PER_STEP,
    TIME_PENALTY_EARLY,
    TIME_PENALTY_MIDDLE,
    TIME_PENALTY_LATE,
    TIME_PENALTY_EARLY_THRESHOLD,
    TIME_PENALTY_LATE_THRESHOLD,
    COMPLETION_TIME_BONUS_MAX,
    COMPLETION_TIME_TARGET,
    PBRS_GAMMA,
    NOOP_ACTION_PENALTY,
)
from ...graph.reachability.path_distance_calculator import (
    CachedPathDistanceCalculator,
)

from ..constants import LEVEL_DIAGONAL as PBRS_DISTANCE_SCALE


class RewardCalculator:
    """Main reward calculator for completion-focused training.

        Orchestrates multiple reward components:
        - Terminal rewards (completion, death)
        - Milestone rewards (switch activation)
        - Time-based penalties (efficiency)
        - Navigation shaping (PBRS-based distance rewards)
    - Exploration rewards (multi-scale spatial coverage)
        - PBRS potentials (policy-invariant shaping)

        All constants are defined in reward_constants.py to eliminate magic numbers
        and provide clear documentation of reward design decisions.
    """

    def __init__(
        self,
        reward_config: Optional[Dict[str, Any]] = None,
        pbrs_gamma: float = PBRS_GAMMA,
        time_penalty_mode: str = "fixed",
        time_penalty_early: float = TIME_PENALTY_EARLY,
        time_penalty_middle: float = TIME_PENALTY_MIDDLE,
        time_penalty_late: float = TIME_PENALTY_LATE,
        enable_completion_bonus: bool = False,
        completion_bonus_max: float = COMPLETION_TIME_BONUS_MAX,
        completion_bonus_target: int = COMPLETION_TIME_TARGET,
        max_episode_steps: int = 20000,
    ):
        """Initialize reward calculator with all components.

        Args:
            reward_config: Complete reward configuration dict (overrides individual params if provided)
            pbrs_gamma: Discount factor for PBRS (γ in r_shaped = r_env + γ * Φ(s') - Φ(s))
            time_penalty_mode: "fixed" or "progressive" time penalty
            time_penalty_early: Early phase penalty (for progressive mode)
            time_penalty_middle: Middle phase penalty (for progressive mode)
            time_penalty_late: Late phase penalty (for progressive mode)
            enable_completion_bonus: Whether to give bonus for fast completion
            completion_bonus_max: Maximum completion time bonus
            completion_bonus_target: Target steps for full bonus
            max_episode_steps: Maximum episode length (for progressive penalty phases)
        """
        # If reward_config provided, extract parameters from it
        if reward_config is not None:
            pbrs_gamma = reward_config.get("pbrs_gamma", pbrs_gamma)
            time_penalty_mode = reward_config.get(
                "time_penalty_mode", time_penalty_mode
            )
            time_penalty_early = reward_config.get(
                "time_penalty_early", time_penalty_early
            )
            time_penalty_middle = reward_config.get(
                "time_penalty_middle", time_penalty_middle
            )
            time_penalty_late = reward_config.get(
                "time_penalty_late", time_penalty_late
            )
            enable_completion_bonus = reward_config.get(
                "enable_completion_bonus", enable_completion_bonus
            )
            completion_bonus_max = reward_config.get(
                "completion_bonus_max", completion_bonus_max
            )
            completion_bonus_target = reward_config.get(
                "completion_bonus_target", completion_bonus_target
            )
            max_episode_steps = reward_config.get(
                "max_episode_steps", max_episode_steps
            )

        self.exploration_calculator = ExplorationRewardCalculator()
        self.steps_taken = 0

        # Track closest distances for diagnostic metrics
        # PBRS handles all reward shaping, but we track progress for diagnostics
        self.closest_distance_to_switch = float("inf")
        self.closest_distance_to_exit = float("inf")

        # Time penalty configuration
        self.time_penalty_mode = time_penalty_mode
        self.time_penalty_early = time_penalty_early
        self.time_penalty_middle = time_penalty_middle
        self.time_penalty_late = time_penalty_late
        self.max_episode_steps = max_episode_steps

        # Completion bonus configuration
        self.enable_completion_bonus = enable_completion_bonus
        self.completion_bonus_max = completion_bonus_max
        self.completion_bonus_target = completion_bonus_target

        # PBRS configuration
        self.pbrs_gamma = pbrs_gamma if pbrs_gamma is not None else PBRS_GAMMA
        self.prev_potential = None

        path_calculator = CachedPathDistanceCalculator(
            max_cache_size=200, use_astar=True
        )

        self.pbrs_calculator = PBRSCalculator(path_calculator=path_calculator)

    def calculate_reward(
        self,
        obs: Dict[str, Any],
        prev_obs: Dict[str, Any],
        action: Optional[int] = None,
    ) -> float:
        """Calculate completion-focused reward.

        Args:
            obs: Current game state
            prev_obs: Previous game state
            action: Action taken (0=NOOP, 1=Left, 2=Right, 3=Jump, 4=Jump+Left, 5=Jump+Right)

        Returns:
            float: Total reward for the transition
        """
        self.steps_taken += 1

        # Death penalty (terminal)
        if obs.get("player_dead", False):
            return DEATH_PENALTY

        # Initialize reward with time penalty to encourage efficiency
        reward = self._calculate_time_penalty()

        # NOOP action penalty (discourage standing still)
        noop_penalty = 0.0
        if action is not None and action == 0:
            noop_penalty = NOOP_ACTION_PENALTY
            reward += noop_penalty

        # Switch activation reward
        if obs.get("switch_activated", False) and not prev_obs.get(
            "switch_activated", False
        ):
            reward += SWITCH_ACTIVATION_REWARD

        # Exit completion reward (terminal)
        if obs.get("player_won", False):
            reward += LEVEL_COMPLETION_REWARD

            # Add completion time bonus if enabled
            if self.enable_completion_bonus:
                bonus = self._calculate_completion_bonus(self.steps_taken)
                reward += bonus

        # Track closest distances for diagnostic metrics
        from ..util.util import calculate_distance

        distance_to_switch = calculate_distance(
            obs["player_x"], obs["player_y"], obs["switch_x"], obs["switch_y"]
        )
        distance_to_exit = calculate_distance(
            obs["player_x"], obs["player_y"], obs["exit_door_x"], obs["exit_door_y"]
        )

        if distance_to_switch < self.closest_distance_to_switch:
            self.closest_distance_to_switch = distance_to_switch

        # Detect switch activation and reset exploration when switch is activated
        switch_active_changed = obs.get("switch_activated", False) and not prev_obs.get(
            "switch_activated", False
        )

        if switch_active_changed:
            self.exploration_calculator.reset()
            # Reset closest distance tracking for exit phase
            self.closest_distance_to_exit = distance_to_exit
        elif obs.get("switch_activated", False):
            # Track closest distance to exit during exit phase
            if distance_to_exit < self.closest_distance_to_exit:
                self.closest_distance_to_exit = distance_to_exit

        # Exploration reward (focused on switch/exit discovery)
        exploration_reward = self.exploration_calculator.calculate_exploration_reward(
            obs["player_x"], obs["player_y"]
        )
        reward += exploration_reward

        # Add PBRS shaping reward if enabled (focused on switch/exit objectives)
        pbrs_reward = 0.0
        pbrs_components = {}
        # Extract adjacency graph, level_data, and graph_data from observation
        # These are required for path-aware PBRS calculations with spatial indexing
        adjacency = obs.get("_adjacency_graph")
        level_data = obs.get("level_data")
        graph_data = obs.get("_graph_data")  # Contains spatial_hash for O(1) lookup

        # STRICT: Validate required data
        if adjacency is None:
            raise ValueError(
                "PBRS enabled but adjacency graph not found in observation. "
                "Ensure graph updates are enabled and adjacency is provided."
            )
        if level_data is None:
            raise ValueError(
                "PBRS enabled but level_data not found in observation. "
                "Ensure level_data is included in observation dict."
            )

        current_potential = self.pbrs_calculator.calculate_combined_potential(
            obs, adjacency=adjacency, level_data=level_data, graph_data=graph_data
        )

        # Get individual potential components for logging
        pbrs_components = self.pbrs_calculator.get_potential_components(
            obs, adjacency=adjacency, level_data=level_data, graph_data=graph_data
        )

        if self.prev_potential is not None:
            # PBRS formula: F(s,s') = γ * Φ(s') - Φ(s)
            # This ensures policy invariance (Ng et al., 1999) while providing
            # dense reward signal. Positive reward when moving closer (increasing potential),
            # negative when moving away (decreasing potential).
            #
            # When switch activates, potential switches from switch to exit calculation.
            # This is handled naturally by the formula - the transition reward reflects
            # the change in potential, which appropriately rewards reaching the switch.
            pbrs_reward = self.pbrs_gamma * current_potential - self.prev_potential
            reward += pbrs_reward
        else:
            # First step: initialize potential but don't add reward (no previous state)
            # This ensures PBRS starts correctly on second step
            pass

        # Update previous potential for next step
        # Note: Even if switch just activated, we update to exit potential smoothly
        # The PBRS formula handles the transition correctly
        self.prev_potential = current_potential

        # Store component rewards for episode info
        self.last_pbrs_components = {
            "exploration_reward": exploration_reward,
            "pbrs_reward": pbrs_reward,
            "noop_penalty": noop_penalty,
            "pbrs_components": pbrs_components,
            "total_reward": reward,
        }

        return reward

    def reset(self):
        """Reset all components for new episode."""
        self.exploration_calculator.reset()
        self.steps_taken = 0

        # Reset progress tracking for diagnostics
        self.closest_distance_to_switch = float("inf")
        self.closest_distance_to_exit = float("inf")

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

        adjacency = obs.get("_adjacency_graph")
        level_data = obs.get("level_data")
        graph_data = obs.get("_graph_data")
        components.update(
            self.pbrs_calculator.get_potential_components(
                obs,
                adjacency=adjacency,
                level_data=level_data,
                graph_data=graph_data,
            )
        )
        components["combined_potential"] = (
            self.pbrs_calculator.calculate_combined_potential(
                obs,
                adjacency=adjacency,
                level_data=level_data,
                graph_data=graph_data,
            )
        )

        return components

    def get_diagnostic_metrics(self, obs: Dict[str, Any]) -> Dict[str, float]:
        """Generate comprehensive diagnostic metrics for TensorBoard.

        Returns detailed breakdowns of distances, potentials, and reward components
        to enable deep diagnosis of training issues.

        Args:
            obs: Current game state dictionary

        Returns:
            dict: Comprehensive metrics for TensorBoard logging
        """
        from ..util.util import calculate_distance
        from ..constants import LEVEL_DIAGONAL

        metrics = {}

        # === DISTANCE METRICS ===
        # Track raw distances to objectives
        distance_to_switch = calculate_distance(
            obs["player_x"], obs["player_y"], obs["switch_x"], obs["switch_y"]
        )
        distance_to_exit = calculate_distance(
            obs["player_x"], obs["player_y"], obs["exit_door_x"], obs["exit_door_y"]
        )

        # Current objective distance
        if not obs.get("switch_activated", False):
            metrics["distance/to_current_objective"] = distance_to_switch
            metrics["distance/to_switch"] = distance_to_switch
        else:
            metrics["distance/to_current_objective"] = distance_to_exit
            metrics["distance/to_exit"] = distance_to_exit

        # Normalized distances (for understanding potential calculations)
        metrics["distance/normalized_to_objective"] = min(
            1.0, metrics["distance/to_current_objective"] / LEVEL_DIAGONAL
        )

        metrics["distance/relative_to_scale"] = (
            metrics["distance/to_current_objective"] / PBRS_DISTANCE_SCALE
        )

        # === POTENTIAL METRICS ===
        # Extract adjacency graph, level_data, and graph_data for PBRS calculations
        adjacency = obs.get("_adjacency_graph")
        level_data = obs.get("level_data")
        graph_data = obs.get("_graph_data")

        # Calculate current potential with required parameters
        current_potential = self.pbrs_calculator.calculate_combined_potential(
            obs, adjacency=adjacency, level_data=level_data, graph_data=graph_data
        )
        metrics["pbrs/current_potential"] = current_potential

        if self.prev_potential is not None:
            metrics["pbrs/prev_potential"] = self.prev_potential
            potential_delta = current_potential - self.prev_potential
            metrics["pbrs/potential_delta"] = potential_delta

            # PBRS reward (before and after gamma)
            pbrs_reward_before_gamma = potential_delta
            pbrs_reward_after_gamma = (
                self.pbrs_gamma * current_potential - self.prev_potential
            )
            metrics["pbrs/reward_before_gamma"] = pbrs_reward_before_gamma
            metrics["pbrs/reward_after_gamma"] = pbrs_reward_after_gamma

            # Flag potential issues
            metrics["pbrs/is_negative"] = 1.0 if pbrs_reward_after_gamma < 0 else 0.0
            metrics["pbrs/is_positive"] = 1.0 if pbrs_reward_after_gamma > 0 else 0.0

        # Component breakdown
        pbrs_components = self.pbrs_calculator.get_potential_components(
            obs, adjacency=adjacency, level_data=level_data, graph_data=graph_data
        )
        for comp_name, comp_value in pbrs_components.items():
            metrics[f"pbrs/component_{comp_name}"] = comp_value

        # Track closest distances achieved (progress tracking)
        metrics["navigation/closest_to_switch"] = (
            self.closest_distance_to_switch
            if self.closest_distance_to_switch != float("inf")
            else distance_to_switch
        )
        metrics["navigation/closest_to_exit"] = (
            self.closest_distance_to_exit
            if self.closest_distance_to_exit != float("inf")
            else distance_to_exit
        )

        # === REWARD COMPONENT BREAKDOWN ===
        if hasattr(self, "last_pbrs_components"):
            components = self.last_pbrs_components
            for comp_name, comp_value in components.items():
                if isinstance(comp_value, (int, float)):
                    metrics[f"reward_components/{comp_name}"] = comp_value

        # === EFFICIENCY METRICS ===
        metrics["episode/steps_taken"] = self.steps_taken
        metrics["episode/switch_activated"] = (
            1.0 if obs.get("switch_activated") else 0.0
        )

        return metrics

    def _calculate_time_penalty(self) -> float:
        """Calculate time penalty based on configured mode.

        Returns:
            float: Time penalty for current step
        """
        if self.time_penalty_mode == "progressive":
            # Progressive penalty increases pressure over episode duration
            progress = self.steps_taken / self.max_episode_steps

            if progress < TIME_PENALTY_EARLY_THRESHOLD:
                return self.time_penalty_early
            elif progress < TIME_PENALTY_LATE_THRESHOLD:
                return self.time_penalty_middle
            else:
                return self.time_penalty_late
        else:
            # Fixed penalty mode (default)
            return TIME_PENALTY_PER_STEP

    def _calculate_completion_bonus(self, completion_steps: int) -> float:
        """Calculate bonus reward for fast completion.

        Bonus linearly decreases from max to zero as completion time increases.

        Args:
            completion_steps: Number of steps taken to complete level

        Returns:
            float: Completion time bonus (0.0 to completion_bonus_max)
        """
        if completion_steps <= self.completion_bonus_target:
            # Linear interpolation: full bonus at 0 steps, zero bonus at target
            progress = completion_steps / self.completion_bonus_target
            return self.completion_bonus_max * (1.0 - progress)
        else:
            # No bonus for completion slower than target
            return 0.0
