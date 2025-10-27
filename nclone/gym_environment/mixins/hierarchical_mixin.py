"""
Hierarchical RL functionality mixin for N++ environment.

This module contains hierarchical RL functionality including completion planner
integration, subtask management, and reward shaping for strategic gameplay.
"""

import logging
import time
import numpy as np
from typing import Dict, Any, Optional, Tuple
from enum import Enum

from ...planning.completion_planner import LevelCompletionPlanner
from ..config import HierarchicalConfig
from ..constants import LEVEL_DIAGONAL


class Subtask(Enum):
    """Enumeration of available subtasks for hierarchical control."""

    NAVIGATE_TO_EXIT_SWITCH = 0
    NAVIGATE_TO_LOCKED_DOOR_SWITCH = 1
    NAVIGATE_TO_EXIT_DOOR = 2
    AVOID_MINE = 3


class HierarchicalMixin:
    """
    Mixin class providing hierarchical RL functionality for N++ environment.

    This mixin handles:
    - Completion planner integration for strategic subtask selection
    - Subtask state management and transitions
    - Subtask-specific reward shaping
    - Performance tracking and logging
    """

    def _init_hierarchical_system(self, config: HierarchicalConfig):
        """Initialize the hierarchical system components."""
        self.hierarchical_config = config
        self.enable_hierarchical = config.enable_hierarchical
        self.enable_subtask_rewards = config.enable_subtask_rewards
        self.subtask_reward_scale = config.subtask_reward_scale
        self.max_subtask_steps = config.max_subtask_steps
        self.debug = config.debug

        # Validate dependencies
        if self.enable_hierarchical:
            if not hasattr(self, "enable_reachability") or not self.enable_reachability:
                raise ValueError(
                    "Hierarchical RL requires reachability analysis to be enabled. "
                    "Set enable_reachability=True in your environment configuration."
                )

        # Initialize completion planner
        self.completion_planner = config.completion_planner or LevelCompletionPlanner()

        # Hierarchical state tracking
        self.current_subtask = Subtask.NAVIGATE_TO_EXIT_SWITCH
        self.subtask_start_time = 0
        self.subtask_step_count = 0

        # Subtask transition history for logging
        self.subtask_history = []
        self.last_switch_states = {}
        self.last_ninja_pos = None

        # Performance tracking
        self.hierarchical_times = []
        self.max_time_samples = 100

        if self.debug:
            logging.info("Hierarchical system initialized with reachability dependency")

    def _get_current_subtask(
        self, obs: Dict[str, Any], info: Dict[str, Any]
    ) -> Subtask:
        """
        Use completion planner to determine current subtask.

        Args:
            obs: Environment observation containing multimodal data
            info: Environment info containing game state

        Returns:
            Current subtask based on completion planner analysis
        """
        if not self.enable_hierarchical:
            return self.current_subtask

        start_time = time.time()

        try:
            # Extract game state information
            ninja_pos = self._extract_ninja_position(obs, info)
            level_data = self._extract_level_data(obs, info)
            switch_states = self._extract_switch_states(obs, info)
            reachability_features = self._extract_reachability_features(obs)

            # Check if we should switch subtasks based on completion planner
            if self._should_switch_subtask(obs, info):
                new_subtask = self._determine_next_subtask(
                    ninja_pos, level_data, switch_states, reachability_features
                )

                if new_subtask != self.current_subtask:
                    self._transition_to_subtask(new_subtask)

        except Exception as e:
            if self.debug:
                logging.warning(f"Hierarchical subtask selection failed: {e}")

        # Track performance
        elapsed_time = time.time() - start_time
        self.hierarchical_times.append(elapsed_time)
        if len(self.hierarchical_times) > self.max_time_samples:
            self.hierarchical_times.pop(0)

        return self.current_subtask

    def _should_switch_subtask(self, obs: Dict[str, Any], info: Dict[str, Any]) -> bool:
        """
        Determine if subtask should change based on completion planner.

        Args:
            obs: Environment observation
            info: Environment info

        Returns:
            True if subtask should switch, False otherwise
        """
        # Check for forced transition due to step limit
        if self.subtask_step_count >= self.max_subtask_steps:
            return True

        # Check for completion of current subtask
        switch_states = self._extract_switch_states(obs, info)
        ninja_pos = self._extract_ninja_position(obs, info)

        # Detect switch state changes (subtask completion)
        if self.last_switch_states != switch_states:
            return True

        # Check for specific subtask completion conditions
        if self.current_subtask == Subtask.NAVIGATE_TO_EXIT_SWITCH:
            # Check if exit switch was activated
            exit_switch_id = self._find_exit_switch_id(obs, info)
            if exit_switch_id and switch_states.get(exit_switch_id, False):
                return True

        elif self.current_subtask == Subtask.NAVIGATE_TO_LOCKED_DOOR_SWITCH:
            # Check if any locked door switch was activated
            for switch_id, activated in switch_states.items():
                if activated and not self.last_switch_states.get(switch_id, False):
                    return True

        elif self.current_subtask == Subtask.NAVIGATE_TO_EXIT_DOOR:
            # Check if ninja reached exit (level completion)
            if info.get("level_complete", False):
                return True

        # Check for significant position change (potential mine avoidance completion)
        if self.current_subtask == Subtask.AVOID_MINE:
            if self.last_ninja_pos and ninja_pos:
                distance_moved = np.linalg.norm(
                    np.array(ninja_pos) - np.array(self.last_ninja_pos)
                )
                if (
                    distance_moved
                    > self.hierarchical_config.SIGNIFICANT_MOVEMENT_THRESHOLD
                ):
                    return True

        return False

    def _determine_next_subtask(
        self,
        ninja_pos: Tuple[int, int],
        level_data: Dict[str, Any],
        switch_states: Dict[str, bool],
        reachability_features: np.ndarray,
    ) -> Subtask:
        """
        Determine the next subtask using completion planner logic.

        Args:
            ninja_pos: Current ninja position
            level_data: Level layout data
            switch_states: Current switch activation states
            reachability_features: 8D reachability features

        Returns:
            Next subtask to execute
        """
        # Validate inputs before proceeding
        if not ninja_pos or len(ninja_pos) != 2:
            if self.debug:
                logging.warning(f"Invalid ninja position: {ninja_pos}")
            return self._fallback_subtask_selection(
                switch_states, reachability_features
            )

        try:
            # Use actual reachability system from ReachabilityMixin
            reachability_analysis = self._get_reachability_analysis_for_planner(
                ninja_pos, level_data, reachability_features
            )

            completion_strategy = self.completion_planner.plan_completion(
                ninja_pos, level_data, switch_states, reachability_analysis
            )

            if (
                completion_strategy
                and hasattr(completion_strategy, "steps")
                and completion_strategy.steps
            ):
                # Map completion step to subtask
                first_step = completion_strategy.steps[0]
                return self._map_completion_step_to_subtask(first_step)

        except Exception as e:
            if self.debug:
                logging.warning(f"Completion planner failed: {e}")

        # Fallback logic based on reachability features and game state
        return self._fallback_subtask_selection(switch_states, reachability_features)

    def _map_completion_step_to_subtask(self, completion_step) -> Subtask:
        """Map completion planner step to subtask enum."""
        try:
            action_type = getattr(completion_step, "action_type", None)

            if action_type == "navigate_and_activate":
                # Determine if it's exit switch or locked door switch
                description = getattr(completion_step, "description", "").lower()
                if "exit" in description:
                    return Subtask.NAVIGATE_TO_EXIT_SWITCH
                else:
                    return Subtask.NAVIGATE_TO_LOCKED_DOOR_SWITCH
            elif action_type == "navigate_to_exit":
                return Subtask.NAVIGATE_TO_EXIT_DOOR
            else:
                # Default to exit switch navigation
                return Subtask.NAVIGATE_TO_EXIT_SWITCH
        except Exception as e:
            if self.debug:
                logging.warning(f"Failed to map completion step to subtask: {e}")
            return Subtask.NAVIGATE_TO_EXIT_SWITCH

    def _fallback_subtask_selection(
        self, switch_states: Dict[str, bool], reachability_features: np.ndarray
    ) -> Subtask:
        """Fallback subtask selection when completion planner fails."""
        # Simple heuristic: if exit switch not activated, go for it
        # Otherwise, go for exit door
        exit_switch_activated = any(switch_states.values())

        if not exit_switch_activated:
            return Subtask.NAVIGATE_TO_EXIT_SWITCH
        else:
            return Subtask.NAVIGATE_TO_EXIT_DOOR

    def _transition_to_subtask(self, new_subtask: Subtask):
        """Transition to a new subtask with logging."""
        old_subtask = self.current_subtask
        self.current_subtask = new_subtask
        self.subtask_start_time = time.time()
        self.subtask_step_count = 0

        # Log transition
        transition = {
            "timestamp": time.time(),
            "from_subtask": old_subtask.name,
            "to_subtask": new_subtask.name,
            "step_count": self.subtask_step_count,
        }
        self.subtask_history.append(transition)

        if self.debug:
            logging.info(
                f"Subtask transition: {old_subtask.name} -> {new_subtask.name}"
            )

    def _calculate_subtask_reward(
        self,
        current_subtask: Subtask,
        obs: Dict[str, Any],
        info: Dict[str, Any],
        terminated: bool,
    ) -> float:
        """
        Calculate subtask-specific reward shaping.

        Args:
            current_subtask: Current subtask enum
            obs: Observation dictionary
            info: Environment info
            terminated: Whether episode terminated

        Returns:
            Subtask-specific reward
        """
        if not self.enable_subtask_rewards:
            return 0.0

        reward = 0.0

        # Reward based on subtask progress
        if current_subtask == Subtask.NAVIGATE_TO_EXIT_SWITCH:
            # Reward getting closer to exit switch
            if "switch_distance" in info:
                # Negative distance as reward (closer = higher reward)
                reward += (
                    -info["switch_distance"]
                    * self.hierarchical_config.DISTANCE_REWARD_SCALE
                )

        elif current_subtask == Subtask.NAVIGATE_TO_LOCKED_DOOR_SWITCH:
            # Reward getting closer to locked door switches
            if "locked_door_distance" in info:
                reward += (
                    -info["locked_door_distance"]
                    * self.hierarchical_config.DISTANCE_REWARD_SCALE
                )

        elif current_subtask == Subtask.NAVIGATE_TO_EXIT_DOOR:
            # Reward getting closer to exit door
            if "exit_distance" in info:
                reward += (
                    -info["exit_distance"]
                    * self.hierarchical_config.DISTANCE_REWARD_SCALE
                )

        elif current_subtask == Subtask.AVOID_MINE:
            # Reward staying away from mines
            if "mine_distance" in info:
                reward += (
                    info["mine_distance"]
                    * self.hierarchical_config.MINE_AVOIDANCE_REWARD_SCALE
                )

        # Bonus for subtask completion
        if "subtask_transition" in info:
            reward += self.hierarchical_config.SUBTASK_COMPLETION_BONUS

        # Penalty for taking too long on a subtask
        if self.subtask_step_count > self.hierarchical_config.SUBTASK_TIMEOUT_THRESHOLD:
            reward -= self.hierarchical_config.SUBTASK_TIMEOUT_PENALTY

        return reward

    def _update_hierarchical_state(self, obs: Dict[str, Any], info: Dict[str, Any]):
        """Update hierarchical state after environment step."""
        if not self.enable_hierarchical:
            return

        self.subtask_step_count += 1
        self.last_switch_states = self._extract_switch_states(obs, info)
        self.last_ninja_pos = self._extract_ninja_position(obs, info)

    def _reset_hierarchical_state(self):
        """Reset hierarchical state for new episode."""
        if not self.enable_hierarchical:
            return

        self.current_subtask = Subtask.NAVIGATE_TO_EXIT_SWITCH
        self.subtask_start_time = time.time()
        self.subtask_step_count = 0
        self.last_switch_states = {}
        self.last_ninja_pos = None

    def _get_subtask_features(self) -> np.ndarray:
        """
        Get current subtask as one-hot encoded features.

        Returns:
            4-dimensional one-hot vector representing current subtask
        """
        if not self.enable_hierarchical:
            return np.zeros(4, dtype=np.float32)

        features = np.zeros(4, dtype=np.float32)
        features[self.current_subtask.value] = 1.0
        return features

    def _get_hierarchical_info(self) -> Dict[str, Any]:
        """Get hierarchical information for environment info."""
        if not self.enable_hierarchical:
            return {}

        return {
            "current_subtask": self.current_subtask.name,
            "subtask_features": self._get_subtask_features(),
            "subtask_step_count": self.subtask_step_count,
            "subtask_duration": time.time() - self.subtask_start_time,
            "total_transitions": len(self.subtask_history),
            "recent_transitions": self.subtask_history[-5:]
            if self.subtask_history
            else [],
        }

    def _get_hierarchical_performance_stats(self) -> Dict[str, float]:
        """Get hierarchical performance statistics."""
        if not self.hierarchical_times:
            return {}

        return {
            "avg_hierarchical_time": np.mean(self.hierarchical_times),
            "max_hierarchical_time": np.max(self.hierarchical_times),
            "min_hierarchical_time": np.min(self.hierarchical_times),
            "hierarchical_time_std": np.std(self.hierarchical_times),
        }

    # Helper methods for extracting information from observations
    def _extract_ninja_position(
        self, obs: Dict[str, Any], info: Dict[str, Any]
    ) -> Tuple[int, int]:
        """Extract ninja position from observation/info."""
        # Try to get from info first
        if "ninja_pos" in info:
            pos = info["ninja_pos"]
            if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                return (int(pos[0]), int(pos[1]))

        # Try to get from game_state in observation
        if "game_state" in obs and obs["game_state"] is not None:
            game_state = obs["game_state"]
            if hasattr(game_state, "ninja_pos") and game_state.ninja_pos is not None:
                pos = game_state.ninja_pos
                return (int(pos[0]), int(pos[1]))

        # Fallback: log warning and return default
        if self.debug:
            logging.warning("Could not extract ninja position, using default (0, 0)")
        return (0, 0)

    def _extract_level_data(
        self, obs: Dict[str, Any], info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract level data from observation/info."""
        # Try to get from info first
        if "level_data" in info and info["level_data"]:
            return info["level_data"]

        # Try to get from game_state
        if "game_state" in obs and obs["game_state"] is not None:
            game_state = obs["game_state"]
            if hasattr(game_state, "level_data"):
                return game_state.level_data

        # Fallback: return empty dict with warning
        if self.debug:
            logging.warning("Could not extract level data, using empty dict")
        return {}

    def _extract_switch_states(
        self, obs: Dict[str, Any], info: Dict[str, Any]
    ) -> Dict[str, bool]:
        """Extract switch states from observation/info."""
        # Try to get from info first
        if "switch_states" in info and info["switch_states"]:
            return info["switch_states"]

        # Try to get from game_state
        if "game_state" in obs and obs["game_state"] is not None:
            game_state = obs["game_state"]
            if hasattr(game_state, "switch_states"):
                return game_state.switch_states

        # Fallback: return empty dict
        if self.debug:
            logging.warning("Could not extract switch states, using empty dict")
        return {}

    def _extract_reachability_features(self, obs: Dict[str, Any]) -> np.ndarray:
        """Extract 8D reachability features from observation."""
        # Try to get from observation if available
        if "reachability_features" in obs:
            return obs["reachability_features"]

        # Try to compute using ReachabilityMixin if available
        if hasattr(self, "_get_reachability_features") and self.enable_reachability:
            try:
                return self._get_reachability_features()
            except Exception as e:
                if self.debug:
                    logging.warning(
                        f"Failed to compute reachability features, using fallback: {e}"
                    )

        # Fallback: compute simplified reachability features from basic game state
        # This fallback is only used when reachability system is disabled or fails
        try:
            ninja_x, ninja_y = self.nplay_headless.ninja_position()
            features = np.zeros(8, dtype=np.float32)

            # At minimum, provide distance-based features
            switch_x, switch_y = self.nplay_headless.exit_switch_position()
            exit_x, exit_y = self.nplay_headless.exit_door_position()

            dist_to_switch = np.sqrt(
                (ninja_x - switch_x) ** 2 + (ninja_y - switch_y) ** 2
            )
            dist_to_exit = np.sqrt((ninja_x - exit_x) ** 2 + (ninja_y - exit_y) ** 2)

            features[1] = 1.0 - min(
                dist_to_switch / LEVEL_DIAGONAL, 1.0
            )  # Distance to switch
            features[2] = 1.0 - min(
                dist_to_exit / LEVEL_DIAGONAL, 1.0
            )  # Distance to exit
            features[6] = (
                1.0 if self.nplay_headless.exit_switch_activated() else 0.0
            )  # Exit reachable
            features[7] = features[6]  # Path exists (simplified)

            return features
        except Exception as e:
            if self.debug:
                logging.warning(f"Fallback reachability computation failed: {e}")
            return np.zeros(8, dtype=np.float32)

    def _find_exit_switch_id(
        self, obs: Dict[str, Any], info: Dict[str, Any]
    ) -> Optional[str]:
        """Find the exit switch ID from level data."""
        level_data = self._extract_level_data(obs, info)

        # Look for exit switch in level data
        if "switches" in level_data:
            for switch_id, switch_data in level_data["switches"].items():
                if switch_data.get("type") == "exit" or switch_data.get(
                    "is_exit", False
                ):
                    return switch_id

        # Try to find from switch states (exit switches are typically the main ones)
        switch_states = self._extract_switch_states(obs, info)
        if switch_states:
            # Return the first switch ID as a fallback
            return next(iter(switch_states.keys()), None)

        if self.debug:
            logging.warning("Could not find exit switch ID")
        return None

    def _get_reachability_analysis_for_planner(
        self,
        ninja_pos: Tuple[int, int],
        level_data: Dict[str, Any],
        reachability_features: np.ndarray,
    ):
        """Get reachability analysis using the actual ReachabilityMixin system."""
        try:
            # Use the actual reachability system from ReachabilityMixin
            if hasattr(self, "_reachability_system") and self._reachability_system:
                # Get reachable positions using the real flood fill analysis
                reachable_positions = self._flood_fill_reachability(
                    ninja_pos, level_data
                )

                # Create analysis result compatible with completion planner
                analysis_result = {
                    "reachable_positions": reachable_positions,
                    "features": reachability_features,
                    "ninja_pos": ninja_pos,
                    "level_data": level_data,
                }

                # Add reachability system methods for planner compatibility
                class ReachabilityAnalysis:
                    def __init__(self, result):
                        self.result = result

                    def analyze_reachability(
                        self, level_data, ninja_pos, switch_states
                    ):
                        return self.result

                    def get_reachable_positions(self):
                        return self.result["reachable_positions"]

                    def get_features(self):
                        return self.result["features"]

                return ReachabilityAnalysis(analysis_result)
            else:
                # Fallback if reachability system not available
                if self.debug:
                    logging.warning("Reachability system not available, using fallback")
                return self._create_fallback_reachability_analysis(
                    reachability_features
                )

        except Exception as e:
            if self.debug:
                logging.warning(f"Failed to get reachability analysis: {e}")
            return self._create_fallback_reachability_analysis(reachability_features)

    def _create_fallback_reachability_analysis(self, reachability_features: np.ndarray):
        """Create fallback reachability analysis when real system unavailable."""

        class FallbackReachabilityAnalysis:
            def __init__(self, features):
                self.features = features

            def analyze_reachability(self, level_data, ninja_pos, switch_states):
                return {"reachable_positions": set(), "features": self.features}

            def get_reachable_positions(self):
                return set()

            def get_features(self):
                return self.features

        return FallbackReachabilityAnalysis(reachability_features)
