"""
Refactored N++ environment class using mixins.

This environment provides a Gym interface for the N++ game, allowing reinforcement
learning agents to learn to play levels. The functionality is now organized using
mixins for better code organization and maintainability.
"""

import logging
import numpy as np
from gymnasium.spaces import box, Dict as SpacesDict
from typing import Dict, Any
import time

from ..graph.common import N_MAX_NODES, E_MAX_EDGES

from .base_environment import BaseNppEnvironment
from .mixins import GraphMixin, ReachabilityMixin, DebugMixin, HierarchicalMixin
from .config import EnvironmentConfig


class NppEnvironment(
    BaseNppEnvironment, GraphMixin, ReachabilityMixin, DebugMixin, HierarchicalMixin
):
    """
    Consolidated N++ environment class using mixins.

    This environment provides a Gym interface for the N++ game, allowing reinforcement
    learning agents to learn to play levels. We use a headless version of the game
    to speed up training.

    Features:
    - Multiple observation profiles (minimal/rich)
    - Potential-based reward shaping (PBRS)
    - Frame stacking support
    - Graph-based planning and visualization
    - Episode truncation based on progress
    - Reachability analysis
    - Debug overlays and visualization
    """

    def __init__(self, config: EnvironmentConfig):
        """
        Initialize the N++ environment.

        Args:
            config: Environment configuration object
        """
        self.config = config

        # Validate configuration
        from .config import validate_config

        validate_config(config)

        # Initialize base environment using config
        super().__init__(
            render_mode=self.config.render.render_mode,
            enable_animation=self.config.render.enable_animation,
            enable_logging=self.config.enable_logging,
            enable_debug_overlay=self.config.render.enable_debug_overlay,
            enable_short_episode_truncation=self.config.enable_short_episode_truncation,
            seed=self.config.seed,
            eval_mode=self.config.eval_mode,
            enable_pbrs=self.config.pbrs.enable_pbrs,
            pbrs_weights=self.config.pbrs.pbrs_weights,
            pbrs_gamma=self.config.pbrs.pbrs_gamma,
            custom_map_path=self.config.custom_map_path,
            enable_augmentation=self.config.augmentation.enable_augmentation,
            augmentation_config={
                "p": self.config.augmentation.p,
                "intensity": self.config.augmentation.intensity,
                "disable_validation": self.config.augmentation.disable_validation,
            },
        )

        # Initialize mixin systems using config
        self._init_graph_system(
            self.config.graph.enable_graph_updates, self.config.graph.debug
        )
        self._init_reachability_system(
            self.config.reachability.enable_reachability, self.config.reachability.debug
        )
        self._init_debug_system(self.config.render.enable_debug_overlay)
        self._init_hierarchical_system(self.config.hierarchical)

        # Update configuration flags with new options
        self.config_flags.update(
            {
                "enable_graph_updates": self.config.graph.enable_graph_updates,
                "enable_reachability": self.config.reachability.enable_reachability,
                "enable_hierarchical": self.config.hierarchical.enable_hierarchical,
                "debug": self.config.graph.debug
                or self.config.reachability.debug
                or self.config.hierarchical.debug,
            }
        )

        # Extend observation space with graph, reachability, and hierarchical features
        self.observation_space = self._build_extended_observation_space(
            self.config.graph.enable_graph_updates,
            self.config.reachability.enable_reachability,
            self.config.hierarchical.enable_hierarchical,
        )

        # Initialize graph state if enabled
        if self.config.graph.enable_graph_updates:
            self._update_graph_from_env_state()

    def _build_extended_observation_space(
        self,
        enable_graph_updates: bool,
        enable_reachability: bool,
        enable_hierarchical: bool,
    ) -> SpacesDict:
        """Build the extended observation space with graph and reachability features."""
        obs_spaces = dict(self.observation_space.spaces)

        # Add entity positions (always available)
        # Format: [ninja_x, ninja_y, switch_x, switch_y, exit_x, exit_y]
        obs_spaces["entity_positions"] = box.Box(
            low=0.0, high=1.0, shape=(6,), dtype=np.float32
        )

        # Add reachability features
        if enable_reachability:
            obs_spaces["reachability_features"] = box.Box(
                low=0.0, high=1.0, shape=(8,), dtype=np.float32
            )

        # Add hierarchical features
        if enable_hierarchical:
            obs_spaces["subtask_features"] = box.Box(
                low=0.0, high=1.0, shape=(4,), dtype=np.float32
            )

        # Add graph observation spaces if graph updates are enabled
        if enable_graph_updates:
            from ..graph.common import NODE_FEATURE_DIM, EDGE_FEATURE_DIM

            # Graph node features: comprehensive features from graph builder
            obs_spaces["graph_node_feats"] = box.Box(
                low=-np.inf,
                high=np.inf,
                shape=(N_MAX_NODES, NODE_FEATURE_DIM),
                dtype=np.float32,
            )
            # Graph edge index: [2, max_edges] connectivity matrix
            obs_spaces["graph_edge_index"] = box.Box(
                low=0, high=N_MAX_NODES - 1, shape=(2, E_MAX_EDGES), dtype=np.int32
            )
            # Graph edge features: comprehensive features from graph builder
            obs_spaces["graph_edge_feats"] = box.Box(
                low=-np.inf,
                high=np.inf,
                shape=(E_MAX_EDGES, EDGE_FEATURE_DIM),
                dtype=np.float32,
            )
            # Graph masks for variable-size graphs
            obs_spaces["graph_node_mask"] = box.Box(
                low=0, high=1, shape=(N_MAX_NODES,), dtype=np.int32
            )
            obs_spaces["graph_edge_mask"] = box.Box(
                low=0, high=1, shape=(E_MAX_EDGES,), dtype=np.int32
            )
            # Graph node and edge types
            obs_spaces["graph_node_types"] = box.Box(
                low=0, high=10, shape=(N_MAX_NODES,), dtype=np.int32
            )
            obs_spaces["graph_edge_types"] = box.Box(
                low=0, high=10, shape=(E_MAX_EDGES,), dtype=np.int32
            )

        return SpacesDict(obs_spaces)

    def step(self, action: int):
        """Execute one environment step with enhanced episode info."""
        # Get previous observation
        prev_obs = self._get_observation()

        # Execute action
        action_hoz, action_jump = self._actions_to_execute(action)
        self.nplay_headless.tick(action_hoz, action_jump)

        # Update graph if needed
        if self.enable_graph_updates and self._should_update_graph():
            start_time = time.time()
            self._update_graph_from_env_state()
            update_time = (time.time() - start_time) * 1000  # Convert to ms

            # Update performance stats
            self._update_graph_performance_stats(update_time)

            if self.debug:
                self.logger.debug(f"Graph updated in {update_time:.2f}ms")

        # Get current observation
        curr_obs = self._get_observation()
        terminated, truncated, player_won = self._check_termination()

        # Build initial episode info
        info = {"is_success": player_won}

        # Update hierarchical state and get current subtask
        current_subtask = None
        if self.enable_hierarchical:
            current_subtask = self._get_current_subtask(curr_obs, info)
            self._update_hierarchical_state(curr_obs, info)

        # Calculate reward
        reward = self._calculate_reward(curr_obs, prev_obs)

        # Add hierarchical reward shaping if enabled
        if self.enable_hierarchical and current_subtask is not None:
            hierarchical_reward = self._calculate_subtask_reward(
                current_subtask, curr_obs, info, terminated
            )
            reward += (
                hierarchical_reward * self.config.hierarchical.subtask_reward_scale
            )

        self.current_ep_reward += reward

        # Process observation for training
        processed_obs = self._process_observation(curr_obs)

        # Add configuration flags to episode info
        info.update(
            {
                "config_flags": self.config_flags.copy(),
                "pbrs_enabled": self.config_flags["enable_pbrs"],
            }
        )

        # Add PBRS component rewards if available
        if hasattr(self.reward_calculator, "last_pbrs_components"):
            info["pbrs_components"] = self.reward_calculator.last_pbrs_components.copy()

        # Add reachability performance info if enabled
        if self.enable_reachability and self.reachability_times:
            avg_time = np.mean(self.reachability_times[-10:])  # Last 10 samples
            info["reachability_time_ms"] = avg_time * 1000

        # Add hierarchical info if enabled
        if self.enable_hierarchical:
            info["hierarchical"] = self._get_hierarchical_info()

        return processed_obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """Reset the environment with planning components and visualization."""
        # Call base reset
        result = super().reset(seed, options)

        # Reset mixin states
        self._reset_graph_state()
        self._reset_reachability_state()
        if self.enable_hierarchical:
            self._reset_hierarchical_state()

        # Build initial graph if enabled
        if self.enable_graph_updates:
            self._update_graph_from_env_state()

        return result

    def _get_observation(self) -> Dict[str, Any]:
        """Get the current observation from the game state."""
        # Get base observation
        obs = super()._get_observation()

        # Add reachability features if enabled
        if self.enable_reachability:
            obs["reachability_features"] = self._get_reachability_features()
        else:
            obs["reachability_features"] = np.zeros(8, dtype=np.float32)

        # Add graph observations if enabled
        if self.enable_graph_updates:
            obs.update(self._get_graph_observations())

        # Add hierarchical features if enabled
        if self.enable_hierarchical:
            obs["subtask_features"] = self._get_subtask_features()

        return obs

    def _process_observation(self, obs):
        """Process the observation from the environment."""
        processed_obs = super()._process_observation(obs)

        # Add reachability features if enabled and not already added
        if self.enable_reachability and "reachability_features" not in processed_obs:
            processed_obs["reachability_features"] = obs.get(
                "reachability_features", np.zeros(8, dtype=np.float32)
            )

        # Add graph observations if enabled and not already added
        if self.enable_graph_updates:
            graph_obs = self._get_graph_observations()
            for key, value in graph_obs.items():
                if key not in processed_obs:
                    processed_obs[key] = value

        return processed_obs

    def __getstate__(self):
        """Custom pickle method to handle non-picklable pygame objects and support vectorization."""
        state = super().__getstate__()

        # Remove non-picklable objects that will be recreated by mixins
        non_picklable_attrs = [
            "graph_builder",
            "_reachability_system",
            "_reachability_extractor",
            "logger",
        ]

        for attr in non_picklable_attrs:
            if attr in state:
                del state[attr]

        return state

    def __setstate__(self, state):
        """Custom unpickle method to restore the environment and support vectorization."""
        super().__setstate__(state)

        # Reinitialize mixin systems after unpickling
        debug = getattr(self, "debug", False)

        self._reinit_graph_system_after_unpickling(debug)
        self._reinit_reachability_system_after_unpickling()

        # Recreate logger if debug is enabled
        if debug and not hasattr(self, "logger"):
            logging.basicConfig(level=logging.DEBUG)
            self.logger = logging.getLogger(__name__)

        # Mark that we need to reinitialize on next reset
        self._needs_reinit = True
