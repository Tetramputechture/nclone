"""
Refactored N++ environment class using mixins.

This environment provides a Gym interface for the N++ game, allowing reinforcement
learning agents to learn to play levels. The functionality is now organized using
mixins for better code organization and maintainability.
"""

import logging
import numpy as np
from gymnasium.spaces import box, Dict as SpacesDict
from typing import Dict, Any, Optional
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
            test_dataset_path=self.config.test_dataset_path,
            enable_augmentation=self.config.augmentation.enable_augmentation,
            augmentation_config={
                "p": self.config.augmentation.p,
                "intensity": self.config.augmentation.intensity,
                "disable_validation": self.config.augmentation.disable_validation,
            },
        )

        # STRICT: Validate that PBRS requires graph for PBRS
        if self.config.pbrs.enable_pbrs and not self.config.graph.enable_graph_for_pbrs:
            raise ValueError(
                "PBRS requires graph for PBRS to be enabled. "
                "Set config.graph.enable_graph_for_pbrs=True when using PBRS."
            )

        # Initialize mixin systems using config
        self._init_graph_system(
            enable_graph_for_pbrs=self.config.graph.enable_graph_for_pbrs,
            enable_graph_for_observations=self.config.graph.enable_graph_for_observations,
            debug=self.config.graph.debug,
        )
        self._init_reachability_system(
            self.config.reachability.enable_reachability, self.config.reachability.debug
        )
        self._init_debug_system(self.config.render.enable_debug_overlay)
        self._init_hierarchical_system(self.config.hierarchical)

        # Update configuration flags with new options
        self.config_flags.update(
            {
                "enable_graph_for_pbrs": self.config.graph.enable_graph_for_pbrs,
                "enable_graph_for_observations": self.config.graph.enable_graph_for_observations,
                "enable_reachability": self.config.reachability.enable_reachability,
                "enable_hierarchical": self.config.hierarchical.enable_hierarchical,
                "debug": self.config.graph.debug
                or self.config.reachability.debug
                or self.config.hierarchical.debug,
            }
        )

        # Extend observation space with graph, reachability, and hierarchical features
        self.observation_space = self._build_extended_observation_space(
            self.config.graph.enable_graph_for_observations,
            self.config.reachability.enable_reachability,
            self.config.hierarchical.enable_hierarchical,
        )

        # Initialize graph state if graph building is needed (either flag True)
        if (
            self.config.graph.enable_graph_for_pbrs
            or self.config.graph.enable_graph_for_observations
        ):
            self._update_graph_from_env_state()

    def _build_extended_observation_space(
        self,
        enable_graph_for_observations: bool,
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

        # Add switch states (always available for hierarchical and ICM systems)
        # Format: Fixed array with per-door information for up to 5 locked doors
        # Per door: [switch_x_norm, switch_y_norm, door_x_norm, door_y_norm, collected]
        # Note: collected state represents both switch collection AND door open (they're always synchronized)
        # Total: 5 features * 5 doors = 25 features
        obs_spaces["switch_states"] = box.Box(
            low=0.0, high=1.0, shape=(25,), dtype=np.float32
        )

        # Add reachability features (always available - zeros if reachability disabled)
        # The feature extractor expects this key to always be present
        obs_spaces["reachability_features"] = box.Box(
            low=0.0, high=1.0, shape=(8,), dtype=np.float32
        )

        # Add hierarchical features
        if enable_hierarchical:
            obs_spaces["subtask_features"] = box.Box(
                low=0.0, high=1.0, shape=(4,), dtype=np.float32
            )

        # Add graph observation spaces if graph observations are enabled
        if enable_graph_for_observations:
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

    def _post_action_hook(self):
        """Update graph after action execution if needed."""
        # Graph building happens if either flag is True
        if (
            self.enable_graph_for_pbrs or self.enable_graph_for_observations
        ) and self._should_update_graph():
            start_time = time.time()
            self._update_graph_from_env_state()
            update_time = (time.time() - start_time) * 1000  # Convert to ms

            # Update performance stats
            self._update_graph_performance_stats(update_time)

            if self.debug:
                self.logger.debug(f"Graph updated in {update_time:.2f}ms")

    def _pre_reward_hook(self, curr_obs: Dict[str, Any], player_won: bool):
        """Update hierarchical state before reward calculation."""
        if self.enable_hierarchical:
            # Store current subtask for reward modification
            self._current_subtask = self._get_current_subtask(
                curr_obs, {"is_success": player_won}
            )
            self._update_hierarchical_state(curr_obs, {"is_success": player_won})
        else:
            self._current_subtask = None

    def _modify_reward_hook(
        self,
        reward: float,
        curr_obs: Dict[str, Any],
        player_won: bool,
        terminated: bool,
    ) -> float:
        """Add hierarchical reward shaping if enabled."""
        if self.enable_hierarchical and self._current_subtask is not None:
            hierarchical_reward = self._calculate_subtask_reward(
                self._current_subtask, curr_obs, {"is_success": player_won}, terminated
            )
            reward += (
                hierarchical_reward * self.config.hierarchical.subtask_reward_scale
            )

        return reward

    def _extend_info_hook(self, info: Dict[str, Any]):
        """Add NppEnvironment-specific info fields."""
        # Add raw position fields for wrappers that need them (e.g., HierarchicalRewardWrapper)
        # These fields are removed during observation processing but needed for reward calculation
        info["player_x"] = self.nplay_headless.ninja_position()[0]
        info["player_y"] = self.nplay_headless.ninja_position()[1]
        info["switch_x"] = self.nplay_headless.exit_switch_position()[0]
        info["switch_y"] = self.nplay_headless.exit_switch_position()[1]
        info["switch_activated"] = self.nplay_headless.exit_switch_activated()
        info["exit_door_x"] = self.nplay_headless.exit_door_position()[0]
        info["exit_door_y"] = self.nplay_headless.exit_door_position()[1]
        info["player_dead"] = self.nplay_headless.ninja_has_died()
        info["player_won"] = self.nplay_headless.ninja_has_won()

        # Add reachability performance info if enabled
        if self.enable_reachability and self.reachability_times:
            avg_time = np.mean(self.reachability_times[-10:])  # Last 10 samples
            info["reachability_time_ms"] = avg_time * 1000

        # Add hierarchical info if enabled
        if self.enable_hierarchical:
            info["hierarchical"] = self._get_hierarchical_info()

    def reset(self, seed=None, options=None):
        """Reset the environment with planning components and visualization."""
        # Call base reset
        result = super().reset(seed, options)

        # Reset mixin states
        self._reset_graph_state()
        self._reset_reachability_state()
        if self.enable_hierarchical:
            self._reset_hierarchical_state()

        # Build initial graph if graph building is needed (either flag True)
        if self.enable_graph_for_pbrs or self.enable_graph_for_observations:
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
        if self.enable_graph_for_observations:
            obs.update(self._get_graph_observations())

        # Add hierarchical features if enabled
        if self.enable_hierarchical:
            obs["subtask_features"] = self._get_subtask_features()

        # Add switch states for hierarchical PPO and ICM
        # Extract locked door switch states from environment
        switch_states_dict = self._get_switch_states_from_env()

        # Store dict version for ICM and reachability systems
        obs["switch_states_dict"] = switch_states_dict

        # Convert to numpy array for hierarchical policy (max 5 doors)
        # Format: [switch_x, switch_y, door_x, door_y, switch_collected, door_open] per door
        # Total: 6 features * 5 doors = 30 features
        switch_states_array = self._build_switch_states_array(obs)
        obs["switch_states"] = switch_states_array

        # Add level data for reachability analysis and hierarchical planning
        # This is needed by ICM and reachability-aware exploration
        obs["level_data"] = self._extract_level_data()

        # Add adjacency graph and full graph data for PBRS path-aware reward shaping
        # STRICT: Required when PBRS is enabled
        if self.config.pbrs.enable_pbrs and self.config.graph.enable_graph_for_pbrs:
            adjacency = self._get_adjacency_for_rewards()
            if adjacency is None:
                raise RuntimeError(
                    "PBRS enabled but adjacency graph not available. "
                    "Ensure graph for PBRS is enabled and graph has been built."
                )
            obs["_adjacency_graph"] = adjacency
            # Include full graph_data for spatial indexing (contains spatial_hash)
            obs["_graph_data"] = self.current_graph_data

        return obs

    def _get_adjacency_for_rewards(self) -> Optional[Dict]:
        """Get adjacency graph for reward calculation.

        Returns adjacency graph from current_graph_data if available.
        This is used by PBRS path-aware reward shaping.

        Returns:
            Adjacency dictionary, or None if not available
        """
        if hasattr(self, "current_graph_data") and self.current_graph_data:
            return self.current_graph_data.get("adjacency")
        return None

    def _build_switch_states_array(self, obs: Dict[str, Any]) -> np.ndarray:
        """
        Build switch states array with detailed locked door information.

        Format per door (5 features):
        - switch_x_norm: Normalized X position of switch (0-1)
        - switch_y_norm: Normalized Y position of switch (0-1)
        - door_x_norm: Normalized X position of door (0-1)
        - door_y_norm: Normalized Y position of door (0-1)
        - collected: 1.0 if switch collected (door open), 0.0 otherwise (door closed)

        Note: collected state represents both switch collection AND door open state
        since they are always synchronized (collected switch = open door).

        Returns:
            Array of shape (25,) for up to 5 locked doors
        """
        from ..constants import LEVEL_WIDTH_PX, LEVEL_HEIGHT_PX

        switch_states_array = np.zeros(25, dtype=np.float32)

        # Get locked door entities
        locked_doors = obs.get("locked_doors", [])

        for i, locked_door in enumerate(locked_doors[:5]):  # Max 5 doors
            base_idx = i * 5

            # Get door segment position (stored before position is updated to switch position)
            # The door entity stores the door segment coordinates
            # Note: entity.segment contains the door position, entity.xpos/ypos is switch position
            door_segment = getattr(locked_door, "segment", None)
            if door_segment and hasattr(door_segment, "p1"):
                # Door segment center
                door_x = (door_segment.p1[0] + door_segment.p2[0]) / 2.0
                door_y = (door_segment.p1[1] + door_segment.p2[1]) / 2.0
            else:
                # Fallback: use entity position (which is actually switch position after init)
                door_x = getattr(locked_door, "xpos", 0.0)
                door_y = getattr(locked_door, "ypos", 0.0)

            # Switch position (stored in sw_xpos, sw_ypos OR in xpos, ypos)
            switch_x = getattr(
                locked_door, "sw_xpos", getattr(locked_door, "xpos", 0.0)
            )
            switch_y = getattr(
                locked_door, "sw_ypos", getattr(locked_door, "ypos", 0.0)
            )

            # Normalize positions to [0, 1]
            switch_x_norm = switch_x / LEVEL_WIDTH_PX
            switch_y_norm = switch_y / LEVEL_HEIGHT_PX
            door_x_norm = door_x / LEVEL_WIDTH_PX
            door_y_norm = door_y / LEVEL_HEIGHT_PX

            # Switch collected state (also represents door open state - they're synchronized)
            # active=True means switch not yet collected (door closed)
            # active=False means collected (door open)
            switch_collected = 0.0 if getattr(locked_door, "active", True) else 1.0

            # Store in array
            switch_states_array[base_idx + 0] = np.clip(switch_x_norm, 0.0, 1.0)
            switch_states_array[base_idx + 1] = np.clip(switch_y_norm, 0.0, 1.0)
            switch_states_array[base_idx + 2] = np.clip(door_x_norm, 0.0, 1.0)
            switch_states_array[base_idx + 3] = np.clip(door_y_norm, 0.0, 1.0)
            switch_states_array[base_idx + 4] = switch_collected

        return switch_states_array

    def _process_observation(self, obs):
        """Process the observation from the environment."""
        processed_obs = super()._process_observation(obs)

        # Add reachability features if not already added (always present in raw obs)
        if "reachability_features" not in processed_obs:
            processed_obs["reachability_features"] = obs.get(
                "reachability_features", np.zeros(8, dtype=np.float32)
            )

        # Add graph observations if enabled and not already added
        if self.enable_graph_for_observations:
            graph_obs = self._get_graph_observations()
            for key, value in graph_obs.items():
                if key not in processed_obs:
                    processed_obs[key] = value

        # Add switch states if not already added (always present in raw obs)
        if "switch_states" not in processed_obs:
            processed_obs["switch_states"] = obs.get(
                "switch_states", np.zeros(30, dtype=np.float32)
            )

        # Add subtask features if enabled and not already added
        if self.enable_hierarchical and "subtask_features" not in processed_obs:
            processed_obs["subtask_features"] = obs.get(
                "subtask_features", np.zeros(4, dtype=np.float32)
            )

        # Add switch states dict if not already added (dict version for ICM - not in obs space)
        if "switch_states_dict" not in processed_obs:
            processed_obs["switch_states_dict"] = obs.get("switch_states_dict", {})

        # Add level data if not already added (for reachability/hierarchical - not in obs space)
        if "level_data" not in processed_obs:
            processed_obs["level_data"] = obs.get("level_data", None)

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
