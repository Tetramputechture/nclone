"""Main reward calculator that orchestrates all reward components."""

import logging
import numpy as np
from typing import Dict, Any, Optional
from .exploration_reward_calculator import ExplorationRewardCalculator
from .pbrs_potentials import PBRSCalculator
from .reward_constants import (
    LEVEL_COMPLETION_REWARD,
    DEATH_PENALTY,
    MINE_DEATH_PENALTY,
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
    MASKED_ACTION_PENALTY,
    BUFFER_USAGE_BONUS,
    DIRECTIONAL_MOMENTUM_BONUS_PER_STEP,
    BACKWARD_VELOCITY_PENALTY,
    DIRECTIONAL_MOMENTUM_UPDATE_INTERVAL,
    BACKTRACK_THRESHOLD_DISTANCE,
    BACKTRACK_PENALTY_SCALE,
    PROGRESS_BONUS_SCALE,
    STAGNATION_THRESHOLD,
    STAGNATION_PENALTY_PER_FRAME,
    PROGRESS_CHECK_THRESHOLD,
    EXPLORATION_GRID_WIDTH,
    EXPLORATION_GRID_HEIGHT,
)
from ...graph.reachability.path_distance_calculator import (
    CachedPathDistanceCalculator,
)

from ..constants import LEVEL_DIAGONAL as PBRS_DISTANCE_SCALE
from ...constants.physics_constants import (
    EXIT_SWITCH_RADIUS,
    EXIT_DOOR_RADIUS,
    NINJA_RADIUS,
)

logger = logging.getLogger(__name__)


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
        enable_completion_bonus: bool = True,
        completion_bonus_max: float = COMPLETION_TIME_BONUS_MAX,
        completion_bonus_target: int = COMPLETION_TIME_TARGET,
        max_episode_steps: int = 5000,
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

        # Tier 1 path efficiency: Directional momentum tracking
        self.frame_count = 0
        self.cached_progress_direction = 0  # +1 forward, -1 backward, 0 neutral

        # Tier 1 path efficiency: Progress tracking
        self.best_path_distance_to_switch = float("inf")
        self.best_path_distance_to_exit = float("inf")
        self.frames_since_progress = 0
        self.backtrack_events_total = 0
        self.progress_bonus_total = 0.0
        self.backtrack_penalty_total = 0.0
        self.stagnation_penalty_total = 0.0

        # Store path calculator for directional momentum and progress tracking
        self.path_calculator = path_calculator

        # Track switch activation for PBRS continuity
        self._prev_switch_activated = False

    def calculate_directional_momentum_bonus(self, state: Dict[str, Any]) -> float:
        """
        Reward velocity component toward current objective only.
        Uses GRAPH-BASED shortest path to determine forward direction.

        Key insight: "Forward progress" means reducing shortest PATH distance,
        not Euclidean distance. This respects level geometry and obstacles.

        Args:
            state: Current game state dictionary

        Returns:
            float: Directional momentum reward (positive for forward, negative for backward)
        """
        # Get velocity
        vel_x = state.get("player_xspeed", 0.0)
        vel_y = state.get("player_yspeed", 0.0)
        velocity = np.array([vel_x, vel_y])
        velocity_magnitude = np.linalg.norm(velocity)

        if velocity_magnitude < 0.1:
            return 0.0

        # Get required graph data
        adjacency = state.get("_adjacency_graph")
        level_data = state.get("level_data")
        graph_data = state.get("_graph_data")

        if adjacency is None or level_data is None:
            # Fallback to non-directional momentum if graph data unavailable
            return 0.0

        # Current position
        ninja_pos = (state["player_x"], state["player_y"])

        if not state.get("switch_activated", False):
            objective_pos = (state["switch_x"], state["switch_y"])
            entity_radius = EXIT_SWITCH_RADIUS
        else:
            objective_pos = (state["exit_door_x"], state["exit_door_y"])
            entity_radius = EXIT_DOOR_RADIUS

        # Update cached direction periodically (amortize pathfinding cost)
        self.frame_count += 1
        if self.frame_count % DIRECTIONAL_MOMENTUM_UPDATE_INTERVAL == 0:
            # Get GRAPH-BASED shortest path distance from current position to objective
            current_distance = self.path_calculator.get_distance(
                ninja_pos,
                objective_pos,
                adjacency,
                cache_key="objective",
                level_data=level_data,
                graph_data=graph_data,
                entity_radius=entity_radius,
                ninja_radius=NINJA_RADIUS,
            )

            # Handle unreachable objectives
            if current_distance == float("inf"):
                self.cached_progress_direction = 0
            else:
                # Sample nearby positions in velocity direction to estimate gradient
                # Try position slightly ahead in movement direction
                sample_distance = 20.0  # pixels ahead
                sample_direction = velocity / velocity_magnitude
                sample_pos = (
                    ninja_pos[0] + sample_direction[0] * sample_distance,
                    ninja_pos[1] + sample_direction[1] * sample_distance,
                )

                # Get path distance from sample position to objective
                sample_path_distance = self.path_calculator.get_distance(
                    sample_pos,
                    objective_pos,
                    adjacency,
                    cache_key="objective",
                    level_data=level_data,
                    graph_data=graph_data,
                    entity_radius=entity_radius,
                    ninja_radius=NINJA_RADIUS,
                )

                # Determine if velocity direction REDUCES path distance
                # Positive gradient = moving toward (reducing distance)
                # Negative gradient = moving away (increasing distance)
                if sample_path_distance == float("inf"):
                    path_gradient = 0.0
                else:
                    path_gradient = current_distance - sample_path_distance

                # Cache the gradient sign for use in intermediate frames
                self.cached_progress_direction = np.sign(path_gradient)

        # Use cached direction for current frame
        progress_direction = self.cached_progress_direction

        # Reward velocity when it reduces path distance, penalize when it increases
        if progress_direction > 0:
            # Moving toward objective (reducing path distance)
            reward = velocity_magnitude * DIRECTIONAL_MOMENTUM_BONUS_PER_STEP
        elif progress_direction < 0:
            # Moving away from objective (increasing path distance)
            reward = -velocity_magnitude * BACKWARD_VELOCITY_PENALTY
        else:
            # Neutral (no clear progress)
            reward = 0.0

        return reward

    def calculate_progress_rewards(self, state: Dict[str, Any]) -> float:
        """
        Track best PATH distance achieved and penalize significant backtracking.
        Uses GRAPH-BASED shortest path distances, not Euclidean.

        Key insight: Progress means reducing shortest PATH distance through
        the adjacency graph. This respects level geometry and obstacles.

        Args:
            state: Current game state dictionary

        Returns:
            float: Progress reward (positive for progress, negative for backtracking/stagnation)
        """
        # Get required graph data
        adjacency = state.get("_adjacency_graph")
        level_data = state.get("level_data")
        graph_data = state.get("_graph_data")

        if adjacency is None or level_data is None:
            raise ValueError("Adjacency graph or level data not found in state")

        if not state.get("switch_activated", False):
            objective_pos = (state["switch_x"], state["switch_y"])
            best_path_distance = self.best_path_distance_to_switch
            objective_type = "switch"
            entity_radius = EXIT_SWITCH_RADIUS
        else:
            objective_pos = (state["exit_door_x"], state["exit_door_y"])
            best_path_distance = self.best_path_distance_to_exit
            objective_type = "exit"
            entity_radius = EXIT_DOOR_RADIUS

        # Get current GRAPH-BASED path distance to objective
        ninja_pos = (state["player_x"], state["player_y"])
        current_path_distance = self.path_calculator.get_distance(
            ninja_pos,
            objective_pos,
            adjacency,
            cache_key="objective",
            level_data=level_data,
            graph_data=graph_data,
            entity_radius=entity_radius,
            ninja_radius=NINJA_RADIUS,
        )

        # Handle unreachable objectives
        if current_path_distance == float("inf"):
            # Objective unreachable from current position
            # This shouldn't happen often but handle gracefully
            return -0.001  # Small penalty for being in unreachable area

        # Check if this is the first valid measurement (baseline initialization)
        if best_path_distance == float("inf"):
            # Initialize baseline silently without reward
            if objective_type == "switch":
                self.best_path_distance_to_switch = current_path_distance
            else:
                self.best_path_distance_to_exit = current_path_distance
            self.frames_since_progress = 0
            return 0.0  # No reward for establishing baseline

        # Check for progress (measurable improvement in PATH distance)
        progress_reward = 0.0
        backtrack_penalty = 0.0
        progress_bonus = 0.0

        if current_path_distance < best_path_distance - PROGRESS_CHECK_THRESHOLD:
            # Significant progress made!
            progress_made = best_path_distance - current_path_distance
            progress_bonus = progress_made * PROGRESS_BONUS_SCALE
            progress_reward = progress_bonus
            self.progress_bonus_total += progress_bonus

            # Update best path distance and reset stagnation counter
            if objective_type == "switch":
                self.best_path_distance_to_switch = current_path_distance
            else:
                self.best_path_distance_to_exit = current_path_distance
            self.frames_since_progress = 0

        else:
            # No progress or backtracking
            self.frames_since_progress += 1

            # Check for significant backtracking (PATH distance increased)
            backtrack_distance = current_path_distance - best_path_distance
            if backtrack_distance > BACKTRACK_THRESHOLD_DISTANCE:
                # Penalize significant regression in path distance
                # This means agent moved to location with LONGER optimal path
                backtrack_penalty = backtrack_distance * BACKTRACK_PENALTY_SCALE
                self.backtrack_penalty_total += backtrack_penalty
                self.backtrack_events_total += 1
                progress_reward = -backtrack_penalty

        # Stagnation penalty (gradual increase)
        stagnation_penalty = 0.0
        if self.frames_since_progress > STAGNATION_THRESHOLD:
            excess_frames = self.frames_since_progress - STAGNATION_THRESHOLD
            stagnation_penalty = min(
                excess_frames * STAGNATION_PENALTY_PER_FRAME,
                0.005,  # Cap at 0.005 per step
            )
            self.stagnation_penalty_total += stagnation_penalty
            progress_reward -= stagnation_penalty

        return progress_reward

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
            # Apply stronger penalty for mine deaths
            death_cause = obs.get("death_cause", None)
            if death_cause == "mine":
                return MINE_DEATH_PENALTY
            else:
                return DEATH_PENALTY

        # Initialize reward with time penalty to encourage efficiency
        reward = self._calculate_time_penalty()
        if np.isnan(reward):
            print(
                f"[REWARD_NAN] NaN detected after time_penalty: {reward}, "
                f"steps_taken={self.steps_taken}"
            )

        # NOOP action penalty (discourage standing still)
        noop_penalty = 0.0
        if action is not None and action == 0:
            noop_penalty = NOOP_ACTION_PENALTY
            reward += noop_penalty
            if np.isnan(reward):
                print(
                    f"[REWARD_NAN] NaN detected after noop_penalty: {noop_penalty}, "
                    f"reward={reward}"
                )

        # Masked action penalty (should never trigger during learning if masking works correctly)
        masked_action_penalty = 0.0
        if action is not None:
            action_mask = prev_obs.get("action_mask")
            if action_mask is None:
                raise ValueError("Action mask not found in previous observation")
            if not action_mask[action]:
                # Agent selected a masked action - this is a bug if it happens
                masked_action_penalty = MASKED_ACTION_PENALTY
                reward += masked_action_penalty
                print(
                    f"Masked action {action} was selected! "
                    f"Mask: {action_mask}, Action: {action}"
                )

        # Momentum preservation reward
        momentum_bonus = self.calculate_directional_momentum_bonus(obs)
        reward += momentum_bonus
        if np.isnan(reward) or np.isnan(momentum_bonus):
            print(
                f"[REWARD_NAN] NaN detected after momentum_bonus: {momentum_bonus}, "
                f"reward={reward}, player_pos=({obs.get('player_x')}, {obs.get('player_y')})"
            )

        # Buffer utilization reward
        buffer_bonus = 0.0
        if obs.get("buffered_jump_executed", False):
            buffer_bonus = BUFFER_USAGE_BONUS
            reward += buffer_bonus
            if np.isnan(reward):
                print(
                    f"[REWARD_NAN] NaN detected after buffer_bonus: {buffer_bonus}, "
                    f"reward={reward}"
                )

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
        if np.isnan(reward) or np.isnan(exploration_reward):
            print(
                f"[REWARD_NAN] NaN detected after exploration_reward: {exploration_reward}, "
                f"reward={reward}, player_pos=({obs.get('player_x')}, {obs.get('player_y')})"
            )

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
        if np.isnan(current_potential):
            print(
                f"[REWARD_NAN] NaN detected in current_potential: {current_potential}, "
                f"player_pos=({obs.get('player_x')}, {obs.get('player_y')})"
            )

        # Get individual potential components for logging
        pbrs_components = self.pbrs_calculator.get_potential_components(
            obs, adjacency=adjacency, level_data=level_data, graph_data=graph_data
        )

        # Detect switch activation transition for PBRS continuity
        switch_just_activated = (
            obs.get("switch_activated", False) and not self._prev_switch_activated
        )

        if self.prev_potential is not None:
            # Maintain PBRS continuity at switch activation
            if switch_just_activated:
                # Set prev_potential = current_potential to ensure zero PBRS reward
                # This eliminates discontinuity from switching objective (switch→exit)
                # The switch activation reward (+2.0) is preserved without PBRS interference
                # Note: We update prev_potential here, then again after this block for next step
                # This ensures continuity: F(s,s') = γ * Φ(s') - Φ(s) = γ * Φ(s') - Φ(s') = 0
                pbrs_reward = 0.0
            else:
                # Normal PBRS calculation
                # PBRS formula: F(s,s') = γ * Φ(s') - Φ(s)
                # This ensures policy invariance (Ng et al., 1999) while providing
                # dense reward signal. Positive reward when moving closer (increasing potential),
                # negative when moving away (decreasing potential).
                pbrs_reward = self.pbrs_gamma * current_potential - self.prev_potential

            # Check for NaN or Inf in PBRS components
            if (
                np.isnan(pbrs_reward)
                or np.isinf(pbrs_reward)
                or np.isnan(current_potential)
                or np.isinf(current_potential)
                or np.isnan(self.prev_potential)
                or np.isinf(self.prev_potential)
            ):
                logger.error(
                    f"[REWARD_NAN] Invalid PBRS components detected: "
                    f"pbrs_reward={pbrs_reward}, "
                    f"current_potential={current_potential}, prev_potential={self.prev_potential}, "
                    f"pbrs_gamma={self.pbrs_gamma}, "
                    f"player_pos=({obs.get('player_x')}, {obs.get('player_y')})"
                )
                # Use fallback: don't add PBRS reward if it's invalid
                pbrs_reward = 0.0

            reward += pbrs_reward

            # Check reward after adding PBRS
            if np.isnan(reward) or np.isinf(reward):
                logger.error(
                    f"[REWARD_NAN] Invalid reward after PBRS: {reward}, "
                    f"pbrs_reward={pbrs_reward}, pbrs_components={pbrs_components}"
                )
        else:
            # First step: initialize potential but don't add reward (no previous state)
            # This ensures PBRS starts correctly on second step
            pbrs_reward = 0.0

        # Update previous potential for next step
        # If switch just activated, we maintain continuity by using current_potential
        # Otherwise, we use current_potential normally
        self.prev_potential = current_potential

        # Update switch tracking for next step
        self._prev_switch_activated = obs.get("switch_activated", False)

        # Calculate progress rewards - DISABLED (redundant with PBRS objective potential)
        # Progress bonus now set to 0.0 in reward_constants.py
        # Keeping the tracking for diagnostic metrics only
        progress_reward = self.calculate_progress_rewards(obs)
        if PROGRESS_BONUS_SCALE > 0:  # Only add if enabled
            reward += progress_reward
        if np.isnan(reward) or np.isnan(progress_reward):
            print(
                f"[REWARD_NAN] NaN detected after progress_reward: {progress_reward}, "
                f"reward={reward}, player_pos=({obs.get('player_x')}, {obs.get('player_y')})"
            )

        # Store component rewards for episode info
        self.last_pbrs_components = {
            "exploration_reward": exploration_reward,
            "pbrs_reward": pbrs_reward,
            "noop_penalty": noop_penalty,
            "masked_action_penalty": masked_action_penalty,
            "momentum_bonus": momentum_bonus,
            "buffer_bonus": buffer_bonus,
            "progress_reward": progress_reward,
            "pbrs_components": pbrs_components,
            "total_reward": reward,
        }

        # Final check before returning
        if np.isnan(reward) or np.isinf(reward):
            logger.error(
                f"[REWARD_NAN] FINAL Invalid reward: {reward} (NaN or Inf), "
                f"components: exploration={exploration_reward}, pbrs={pbrs_reward}, "
                f"momentum={momentum_bonus}, progress={progress_reward}, "
                f"noop={noop_penalty}, masked_action={masked_action_penalty}, "
                f"buffer_bonus={buffer_bonus}, "
                f"player_pos=({obs.get('player_x')}, {obs.get('player_y')})"
            )
            # Return 0.0 instead of inf/nan to avoid breaking training
            return 0.0

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

        # Reset switch tracking
        self._prev_switch_activated = False

        # Tier 1 path efficiency: Reset directional momentum tracking
        self.frame_count = 0
        self.cached_progress_direction = 0

        # Tier 1 path efficiency: Reset progress tracking
        self.best_path_distance_to_switch = float("inf")
        self.best_path_distance_to_exit = float("inf")
        self.frames_since_progress = 0
        self.backtrack_events_total = 0
        self.progress_bonus_total = 0.0
        self.backtrack_penalty_total = 0.0
        self.stagnation_penalty_total = 0.0

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

            # Detect switch activation transition for accurate PBRS reward calculation
            switch_just_activated = (
                obs.get("switch_activated", False) and not self._prev_switch_activated
            )

            # PBRS reward calculation (matches actual reward calculation logic)
            if switch_just_activated:
                # Switch activation: zero PBRS reward for continuity
                pbrs_reward_after_gamma = 0.0
                pbrs_reward_before_gamma = 0.0
            else:
                # Normal PBRS calculation: F(s,s') = γ * Φ(s') - Φ(s)
                pbrs_reward_before_gamma = potential_delta
                pbrs_reward_after_gamma = (
                    self.pbrs_gamma * current_potential - self.prev_potential
                )

            metrics["pbrs/reward_before_gamma"] = pbrs_reward_before_gamma
            metrics["pbrs/reward_after_gamma"] = pbrs_reward_after_gamma

            # Flag potential issues
            metrics["pbrs/is_negative"] = 1.0 if pbrs_reward_after_gamma < 0 else 0.0
            metrics["pbrs/is_positive"] = 1.0 if pbrs_reward_after_gamma > 0 else 0.0
            metrics["pbrs/switch_transition"] = 1.0 if switch_just_activated else 0.0

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

        # === TIER 1 PATH EFFICIENCY METRICS ===
        # Directional momentum metrics
        vel_x = obs.get("player_xspeed", 0.0)
        vel_y = obs.get("player_yspeed", 0.0)
        velocity = np.array([vel_x, vel_y])
        velocity_magnitude = np.linalg.norm(velocity)

        if self.cached_progress_direction > 0:
            metrics["movement/forward_velocity_avg"] = velocity_magnitude
            metrics["movement/backward_velocity_avg"] = 0.0
        elif self.cached_progress_direction < 0:
            metrics["movement/forward_velocity_avg"] = 0.0
            metrics["movement/backward_velocity_avg"] = velocity_magnitude
        else:
            metrics["movement/forward_velocity_avg"] = 0.0
            metrics["movement/backward_velocity_avg"] = 0.0

        # Progress tracking metrics
        metrics["progress/best_distance_to_switch"] = (
            self.best_path_distance_to_switch
            if self.best_path_distance_to_switch != float("inf")
            else 0.0
        )
        metrics["progress/best_distance_to_exit"] = (
            self.best_path_distance_to_exit
            if self.best_path_distance_to_exit != float("inf")
            else 0.0
        )
        metrics["progress/frames_since_progress"] = float(self.frames_since_progress)
        metrics["progress/backtrack_events_total"] = float(self.backtrack_events_total)

        # Reward component breakdown for Tier 1
        if hasattr(self, "last_pbrs_components"):
            components = self.last_pbrs_components
            if "progress_reward" in components:
                metrics["reward/progress_bonus_total"] = self.progress_bonus_total
                metrics["reward/backtrack_penalty_total"] = self.backtrack_penalty_total
                metrics["reward/stagnation_penalty_total"] = (
                    self.stagnation_penalty_total
                )

            if "momentum_bonus" in components:
                metrics["reward/directional_momentum_total"] = components[
                    "momentum_bonus"
                ]

        # Exploration coverage metric
        if hasattr(self, "exploration_calculator"):
            total_cells = EXPLORATION_GRID_WIDTH * EXPLORATION_GRID_HEIGHT
            visited_cells = np.sum(self.exploration_calculator.visited_cells)
            metrics["exploration/area_coverage_fraction"] = (
                visited_cells / total_cells if total_cells > 0 else 0.0
            )

        return metrics

    def get_reward_statistics(self) -> Dict[str, float]:
        """Get statistics about reward components for validation.

        Returns:
            Dictionary containing reward component statistics
        """
        return {
            "steps_taken": self.steps_taken,
            "backtrack_events": self.backtrack_events_total,
            "progress_bonus_total": self.progress_bonus_total,
            "backtrack_penalty_total": self.backtrack_penalty_total,
            "stagnation_penalty_total": self.stagnation_penalty_total,
        }

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
