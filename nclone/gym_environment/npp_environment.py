"""
Refactored N++ environment class using mixins.

This environment provides a Gym interface for the N++ game, allowing reinforcement
learning agents to learn to play levels. The functionality is now organized using
mixins for better code organization and maintainability.
"""

import logging
import numpy as np
from gymnasium.spaces import box, Dict as SpacesDict
from typing import Optional, Dict, Any
import time

from ..graph.common import N_MAX_NODES, E_MAX_EDGES

from .base_environment import BaseNppEnvironment
from .mixins import GraphMixin, ReachabilityMixin, DebugMixin


class NppEnvironment(BaseNppEnvironment, GraphMixin, ReachabilityMixin, DebugMixin):
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

    def __init__(
        self,
        render_mode: str = "rgb_array",
        enable_animation: bool = False,
        enable_logging: bool = False,
        enable_debug_overlay: bool = False,
        enable_short_episode_truncation: bool = False,
        seed: Optional[int] = None,
        eval_mode: bool = False,
        enable_pbrs: bool = True,
        pbrs_weights: Optional[dict] = None,
        pbrs_gamma: float = 0.99,
        custom_map_path: Optional[str] = None,
        enable_graph_updates: bool = True,
        enable_reachability: bool = True,
        debug: bool = False,
    ):
        """
        Initialize the N++ environment.

        Args:
            render_mode: Rendering mode ("human" or "rgb_array")
            enable_animation: Enable animation in rendering
            enable_logging: Enable debug logging
            enable_debug_overlay: Enable debug overlay visualization
            enable_short_episode_truncation: Enable episode truncation on lack of progress
            seed: Random seed for reproducibility
            eval_mode: Use evaluation maps instead of training maps
            enable_pbrs: Enable potential-based reward shaping
            pbrs_weights: PBRS component weights dictionary
            pbrs_gamma: PBRS discount factor
            custom_map_path: Path to custom map file
            enable_graph_updates: Enable dynamic graph updates
            enable_reachability: Enable reachability analysis
            debug: Enable debug logging for graph operations
        """
        # Initialize base environment
        super().__init__(
            render_mode=render_mode,
            enable_animation=enable_animation,
            enable_logging=enable_logging,
            enable_debug_overlay=enable_debug_overlay,
            enable_short_episode_truncation=enable_short_episode_truncation,
            seed=seed,
            eval_mode=eval_mode,
            enable_pbrs=enable_pbrs,
            pbrs_weights=pbrs_weights,
            pbrs_gamma=pbrs_gamma,
            custom_map_path=custom_map_path,
        )

        # Initialize mixin systems
        self._init_graph_system(enable_graph_updates, debug)
        self._init_reachability_system(enable_reachability, debug)
        self._init_debug_system(enable_debug_overlay)

        # Update configuration flags with new options
        self.config_flags.update(
            {
                "enable_graph_updates": enable_graph_updates,
                "enable_reachability": enable_reachability,
                "debug": debug,
            }
        )

        # Extend observation space with graph and reachability features
        self.observation_space = self._build_extended_observation_space(
            enable_graph_updates, enable_reachability
        )

        # Initialize graph state if enabled
        if enable_graph_updates:
            self._update_graph_from_env_state()

    def _build_extended_observation_space(
        self, enable_graph_updates: bool, enable_reachability: bool
    ) -> SpacesDict:
        """Build the extended observation space with graph and reachability features."""
        obs_spaces = dict(self.observation_space.spaces)

        # Add reachability features
        if enable_reachability:
            obs_spaces["reachability_features"] = box.Box(
                low=0.0, high=1.0, shape=(8,), dtype=np.float32
            )

        # Add graph observation spaces if graph updates are enabled
        if enable_graph_updates:
            # Graph node features: [x, y, node_type] for each node
            obs_spaces["graph_node_feats"] = box.Box(
                low=-np.inf, high=np.inf, shape=(N_MAX_NODES, 3), dtype=np.float32
            )
            # Graph edge index: [2, max_edges] connectivity matrix
            obs_spaces["graph_edge_index"] = box.Box(
                low=0, high=N_MAX_NODES - 1, shape=(2, E_MAX_EDGES), dtype=np.int32
            )
            # Graph edge features: [weight] for each edge
            obs_spaces["graph_edge_feats"] = box.Box(
                low=0.0, high=np.inf, shape=(E_MAX_EDGES, 1), dtype=np.float32
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

        # Calculate reward
        reward = self._calculate_reward(curr_obs, prev_obs)
        self.current_ep_reward += reward

        # Process observation for training
        processed_obs = self._process_observation(curr_obs)

        # Build episode info
        info = {"is_success": player_won}

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

        return processed_obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """Reset the environment with planning components and visualization."""
        # Call base reset
        result = super().reset(seed, options)

        # Reset mixin states
        self._reset_graph_state()
        self._reset_reachability_state()

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
