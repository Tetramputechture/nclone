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


from ..graph.common import N_MAX_NODES, E_MAX_EDGES, NODE_FEATURE_DIM, EDGE_FEATURE_DIM
from ..constants.physics_constants import LEVEL_WIDTH_PX, LEVEL_HEIGHT_PX
from .constants import (
    GAME_STATE_CHANNELS,
    LEVEL_DIAGONAL,
    REACHABILITY_FEATURES_DIM,
    MAX_LOCKED_DOORS,
    FEATURES_PER_DOOR,
    SWITCH_STATES_DIM,
    MAX_LOCKED_DOORS_ATTENTION,
    LOCKED_DOOR_FEATURES_DIM,
)
from ..constants.entity_types import EntityType
from .base_environment import BaseNppEnvironment
from .mixins import GraphMixin, ReachabilityMixin, DebugMixin
from .config import EnvironmentConfig
from .precomputed_door_features import PrecomputedDoorFeatureCache

logger = logging.getLogger(__name__)


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

    def __init__(self, config: EnvironmentConfig):
        """
        Initialize the N++ environment.

        Args:
            config: Environment configuration object
        """
        self.config = config
        super().__init__(
            render_mode=self.config.render.render_mode,
            enable_animation=self.config.render.enable_animation,
            enable_logging=self.config.enable_logging,
            enable_debug_overlay=self.config.render.enable_debug_overlay,
            seed=self.config.seed,
            eval_mode=self.config.eval_mode,
            pbrs_gamma=self.config.pbrs.pbrs_gamma,
            custom_map_path=self.config.custom_map_path,
            test_dataset_path=self.config.test_dataset_path,
            enable_augmentation=self.config.augmentation.enable_augmentation,
            augmentation_config={
                "p": self.config.augmentation.p,
                "intensity": self.config.augmentation.intensity,
                "disable_validation": self.config.augmentation.disable_validation,
            },
            enable_visual_observations=self.config.enable_visual_observations,
        )

        # Initialize mixin systems using config
        self._init_graph_system(
            debug=self.config.graph.debug,
        )
        self._init_reachability_system(self.config.reachability.debug)
        self._init_debug_system(self.config.render.enable_debug_overlay)

        # Update configuration flags with new options
        self.config_flags.update(
            {"debug": self.config.graph.debug or self.config.reachability.debug}
        )

        # Extend observation space with graph, reachability features
        self.observation_space = self._build_extended_observation_space()

        # Pre-allocate observation buffers
        self._game_state_buffer = np.zeros(
            GAME_STATE_CHANNELS, dtype=np.float32
        )  # Now 29 (only ninja physics)

        # Cache for locked door features (keyed by switch state hash + ninja grid cell)
        #   1. Switch states (rarely change)
        #   2. Ninja grid cell (24px quantization to balance cache hits vs accuracy)
        self._locked_door_cache: Dict[str, np.ndarray] = {}
        self._last_switch_state_hash: Optional[str] = None
        self._last_ninja_grid_cell: Optional[tuple] = None

        # Cache invalidation flag (set by entity interactions)
        self._switch_states_changed: bool = True  # Start True to compute initial state

        # Precomputed door feature cache (aggressive optimization)
        # PERFORMANCE: Precomputes ALL path distances at level load for O(1) lookup
        self.door_feature_cache = PrecomputedDoorFeatureCache()

        # Level-specific locked door caches (static for level duration)
        self._has_locked_doors: bool = False  # Flag: does level have locked doors?
        self._cached_locked_doors: list = []  # Cached locked door entities (static)
        self._cached_switch_states: Optional[np.ndarray] = (
            None  # Cached switch states (updated on invalidation)
        )

        # Exit features cache (position-based invalidation)
        # PERFORMANCE: Avoids recomputing path distances when ninja hasn't moved significantly
        self._cached_exit_features: Optional[np.ndarray] = None
        self._last_exit_cache_ninja_pos: Optional[tuple] = None
        self._exit_cache_grid_size: int = (
            24  # Pixels - invalidate when ninja moves to new grid cell
        )

        # Set environment reference in simulator for cache invalidation
        self.nplay_headless.sim.gym_env = self

    def _safe_path_distance(
        self,
        start_pos,
        goal_pos,
        adjacency,
        feature_name,
        level_data=None,
        graph_data=None,
        entity_radius=0.0,
    ):
        """Calculate path distance with inf handling and detailed logging.

        Returns safe max value (LEVEL_DIAGONAL * 2.0) for unreachable paths.
        Logs detailed warnings when paths are unreachable for debugging.

        Args:
            start_pos: Starting position tuple (x, y)
            goal_pos: Goal position tuple (x, y)
            adjacency: Graph adjacency structure
            feature_name: Name of feature for logging (e.g., "exit_switch", "mine")
            level_data: Optional level data for caching
            graph_data: Optional graph data for spatial indexing
            entity_radius: Collision radius of the goal entity (default 0.0)

        Returns:
            Path distance in pixels, or LEVEL_DIAGONAL * 2.0 if unreachable
        """
        from ..constants.physics_constants import NINJA_RADIUS

        distance = self._path_calculator.get_distance(
            start_pos,
            goal_pos,
            adjacency,
            level_data=level_data,
            graph_data=graph_data,
            entity_radius=entity_radius,
            ninja_radius=NINJA_RADIUS,
        )

        if distance == float("inf"):
            logger.warning(
                f"[PATH_DISTANCE] Unreachable {feature_name}: "
                f"start={start_pos}, goal={goal_pos}"
            )
            return LEVEL_DIAGONAL * 2.0  # Safe max value

        return distance

    def _build_extended_observation_space(self) -> SpacesDict:
        """Build the extended observation space with graph and reachability features."""
        obs_spaces = dict(self.observation_space.spaces)

        # Add reachability features (always available)
        obs_spaces["reachability_features"] = box.Box(
            low=0.0, high=1.0, shape=(REACHABILITY_FEATURES_DIM,), dtype=np.float32
        )

        # Add death context features (always available for learning from terminal states)
        obs_spaces["death_context"] = box.Box(
            low=-1.0, high=1.0, shape=(9,), dtype=np.float32
        )

        # Add exit features for objective attention (always available)
        obs_spaces["exit_features"] = box.Box(
            low=-1.0,
            high=1.0,
            shape=(
                7,
            ),  # [switch_x, switch_y, switch_activated, switch_path_dist, door_x, door_y, door_path_dist]
            dtype=np.float32,
        )

        # Add locked door features for objective attention (always available)
        obs_spaces["locked_door_features"] = box.Box(
            low=-1.0,
            high=1.0,
            shape=(MAX_LOCKED_DOORS_ATTENTION, LOCKED_DOOR_FEATURES_DIM),
            dtype=np.float32,
        )
        obs_spaces["num_locked_doors"] = box.Box(
            low=0, high=MAX_LOCKED_DOORS_ATTENTION, shape=(1,), dtype=np.int32
        )

        # Add graph observation spaces
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
        if self._should_update_graph():
            self._update_graph_from_env_state()

    def _extend_info_hook(self, info: Dict[str, Any]):
        """Add NppEnvironment-specific info fields."""
        # Call parent to add diagnostic and observation metrics
        super()._extend_info_hook(info)

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

    def _build_door_feature_cache(self):
        """
        Build precomputed door feature cache.

        Precomputes path distances from all reachable grid cells to all doors/switches.
        This is a one-time cost at level load that eliminates expensive runtime path
        distance calculations.

        PERFORMANCE: Build time ~50-200ms, saves ~25s per 858 steps (26x speedup)
        """
        # CRITICAL: Invalidate cached observation to ensure fresh positions
        # This prevents using stale positions from previous level
        if hasattr(self, "_cached_observation"):
            self._cached_observation = None

        # Get locked doors from current observation
        obs = super()._get_observation()
        locked_doors = obs.get("locked_doors", [])

        if not locked_doors:
            # No doors to cache
            return

        # Ensure graph is built
        if self.current_graph_data is None:
            logger.warning(
                "Graph data not available for door feature cache. "
                "Cache will not be built."
            )
            return

        adjacency = self.current_graph_data.get("adjacency")
        if not adjacency:
            logger.warning("No adjacency graph. Door feature cache will not be built.")
            return

        # Get area scale for normalization
        try:
            area_scale = self._get_reachable_area_scale()
        except RuntimeError as e:
            logger.warning(f"Could not compute area scale: {e}. Using default.")
            area_scale = 1.0

        # Build the cache
        self.door_feature_cache.build_cache(
            locked_doors=locked_doors,
            adjacency=adjacency,
            path_calculator=self._path_calculator,
            level_data=self.level_data,
            graph_data=self.current_graph_data,
            area_scale=area_scale,
            verbose=False,
        )

    def _initialize_locked_door_caches(self):
        """
        Initialize locked door entity and switch state caches.

        Caches static locked door entities and initializes switch collection states.
        Called once per level after map load. Door positions don't change, and
        switches can only be activated (not deactivated) by the player.

        PERFORMANCE: Eliminates per-step loops and observation calls for door features.
        """
        # CRITICAL: Invalidate cached observation to ensure fresh positions
        # This prevents using stale positions from previous level
        if hasattr(self, "_cached_observation"):
            self._cached_observation = None

        # Get locked doors from observation
        obs = super()._get_observation()
        locked_doors = obs.get("locked_doors", [])

        # Cache level door state
        self._has_locked_doors = len(locked_doors) > 0
        self._cached_locked_doors = locked_doors[:MAX_LOCKED_DOORS_ATTENTION]

        if self._has_locked_doors:
            # Initialize all switches as uncollected (0.0)
            self._cached_switch_states = np.zeros(
                len(self._cached_locked_doors), dtype=np.float32
            )
        else:
            self._cached_switch_states = None

        # Mark switch states as fresh (no changes yet)
        self._switch_states_changed = False

    def reset(self, seed=None, options=None):
        """Reset the environment with planning components and visualization."""

        # Handle reinitialization after unpickling
        if hasattr(self, "_needs_reinit") and self._needs_reinit:
            self.observation_processor.reset()
            self.reward_calculator.reset()
            self._needs_reinit = False

        # Reset observation processor
        self.observation_processor.reset()

        # Reset reward calculator
        self.reward_calculator.reset()

        # Reset truncation checker
        self.truncation_checker.reset()

        # Reset episode reward
        self.current_ep_reward = 0

        # Clear all caches for new level
        from nclone.cache_management import clear_all_caches_for_reset

        clear_all_caches_for_reset(self)

        # Check if map loading should be skipped
        skip_map_load = False
        if options is not None and isinstance(options, dict):
            skip_map_load = options.get("skip_map_load", False)

        if not skip_map_load:
            # Load map - this calls sim.load() which calls sim.reset()
            self.map_loader.load_map()
        else:
            # If map loading is skipped, we still need to reset the sim
            self.nplay_headless.reset()

        # CRITICAL: Reset and rebuild graph BEFORE getting observation
        self._reset_graph_state()
        self._reset_reachability_state()

        # Build graph from the newly loaded map
        self._update_graph_from_env_state()

        # Build precomputed door feature cache after graph is ready
        # PERFORMANCE: Precomputes ALL path distances for O(1) runtime lookup
        self._build_door_feature_cache()

        # Initialize locked door caches for this level
        # PERFORMANCE: Cache static door entities and initialize switch states
        self._initialize_locked_door_caches()

        # Set dynamic truncation limit based on level complexity
        # Uses PBRS surface area and reachable mine count (both already cached)
        pbrs_calculator = self.reward_calculator.pbrs_calculator
        if pbrs_calculator._cached_surface_area is not None:
            surface_area = pbrs_calculator._cached_surface_area
            reachable_mine_count = (
                len(pbrs_calculator._cached_reachable_mines)
                if pbrs_calculator._cached_reachable_mines
                else 0
            )

            truncation_limit = self.truncation_checker.set_level_truncation_limit(
                surface_area, reachable_mine_count
            )

            if self.enable_logging:
                logger.info(
                    f"Level truncation limit: {truncation_limit} frames "
                    f"(surface_area={surface_area:.0f}, mines={reachable_mine_count})"
                )

        # NOW get initial observation (with valid graph data)
        initial_obs = self._get_observation()
        processed_obs = self._process_observation(initial_obs)

        return (processed_obs, {})

    def _get_observation(self) -> Dict[str, Any]:
        """Get the current observation from the game state."""
        # Get base observation
        obs = super()._get_observation()

        from .constants import NINJA_STATE_DIM

        # Phase 4: Only keep ninja physics state (29 dims)
        # All other features (path objectives, mines, progress, etc.) are now in the graph
        base_game_state = obs["game_state"]
        self._game_state_buffer[:NINJA_STATE_DIM] = base_game_state[:NINJA_STATE_DIM]

        obs["game_state"] = np.array(self._game_state_buffer, copy=True)

        obs["reachability_features"] = self._get_reachability_features()

        obs.update(self._get_graph_observations())

        # Extract locked door switch states from environment
        switch_states_dict = self._get_switch_states_from_env()

        # Store dict version for ICM and reachability systems
        obs["switch_states_dict"] = switch_states_dict

        switch_states_array = self._build_switch_states_array(obs)
        obs["switch_states"] = switch_states_array

        # Add locked door features for objective attention
        obs["locked_door_features"] = self._compute_locked_door_features()
        obs["num_locked_doors"] = np.array(
            [len(obs.get("locked_doors", []))], dtype=np.int32
        )

        # Add exit features for objective attention
        obs["exit_features"] = self._compute_exit_features()

        obs["level_data"] = self.level_data

        # Add adjacency graph and full graph data for PBRS path-aware reward shaping
        adjacency = self._get_adjacency_for_rewards()
        if adjacency is None:
            raise RuntimeError("Adjacency graph not available. Graph building failed.")
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
        if self.current_graph_data is None:
            return None
        return self.current_graph_data.get("adjacency")

    def _extract_locked_door_positions(self, locked_door):
        """Extract switch and door positions from a locked door entity.

        Args:
            locked_door: Locked door entity

        Returns:
            Tuple of (switch_x, switch_y, door_x, door_y, switch_collected)
        """
        # Door position from segment (actual door location)
        door_segment = getattr(locked_door, "segment", None)
        if door_segment and hasattr(door_segment, "p1"):
            door_x = (door_segment.p1[0] + door_segment.p2[0]) / 2.0
            door_y = (door_segment.p1[1] + door_segment.p2[1]) / 2.0
        else:
            door_x = getattr(locked_door, "xpos", 0.0)
            door_y = getattr(locked_door, "ypos", 0.0)

        # Switch position (stored in sw_xpos, sw_ypos OR in xpos, ypos)
        switch_x = getattr(locked_door, "sw_xpos", getattr(locked_door, "xpos", 0.0))
        switch_y = getattr(locked_door, "sw_ypos", getattr(locked_door, "ypos", 0.0))

        # Switch collected state (active=True means switch not collected, active=False means collected)
        switch_collected = 0.0 if getattr(locked_door, "active", True) else 1.0

        return switch_x, switch_y, door_x, door_y, switch_collected

    def _build_switch_states_array(self, obs: Dict[str, Any]) -> np.ndarray:
        """
        Build switch states array with detailed locked door information.

        Format per door (5 features):
        - switch_x_norm: Normalized X position of switch (0-1)
        - switch_y_norm: Normalized Y position of switch (0-1)
        - door_x_norm: Normalized X position of door (0-1)
        - door_y_norm: Normalized Y position of door (0-1)
        - collected: 1.0 if switch collected (door open), 0.0 otherwise (door closed)

        Returns:
            Array of shape (SWITCH_STATES_DIM,) for up to MAX_LOCKED_DOORS
        """
        switch_states_array = np.zeros(SWITCH_STATES_DIM, dtype=np.float32)
        locked_doors = obs.get("locked_doors", [])

        for i, locked_door in enumerate(locked_doors[:MAX_LOCKED_DOORS]):
            switch_x, switch_y, door_x, door_y, switch_collected = (
                self._extract_locked_door_positions(locked_door)
            )

            # Normalize positions to [0, 1]
            base_idx = i * FEATURES_PER_DOOR
            switch_states_array[base_idx + 0] = np.clip(
                switch_x / LEVEL_WIDTH_PX, 0.0, 1.0
            )
            switch_states_array[base_idx + 1] = np.clip(
                switch_y / LEVEL_HEIGHT_PX, 0.0, 1.0
            )
            switch_states_array[base_idx + 2] = np.clip(
                door_x / LEVEL_WIDTH_PX, 0.0, 1.0
            )
            switch_states_array[base_idx + 3] = np.clip(
                door_y / LEVEL_HEIGHT_PX, 0.0, 1.0
            )
            switch_states_array[base_idx + 4] = switch_collected

        return switch_states_array

    def _compute_locked_door_features(self) -> np.ndarray:
        """
        Compute features for all locked doors (up to 16) for objective attention.

        Uses PRECOMPUTED CACHE + CACHED SWITCH STATES for maximum performance.
        All path distances are precomputed at level load, and switch states are
        only updated when invalidate_switch_cache() is called (rare event).

        PERFORMANCE OPTIMIZATION:
        - Levels with no doors: Immediate return (zero cost)
        - Levels with doors: Cached switch states + O(1) precomputed lookup
        - Switch state updates: Only when _switch_states_changed flag is True

        Returns (16, 8) array where each row contains:
        [switch_x, switch_y, switch_collected, switch_path_dist, door_x, door_y, door_open, door_path_dist]

        Rows beyond actual door count are zero-padded.
        """
        # FAST PATH: Early exit if level has no locked doors
        if not self._has_locked_doors:
            return np.zeros(
                (MAX_LOCKED_DOORS_ATTENTION, LOCKED_DOOR_FEATURES_DIM), dtype=np.float32
            )

        # Update cached switch states only when switches have changed
        if self._switch_states_changed:
            # Extract current switch collection states from cached door entities
            for idx, door in enumerate(self._cached_locked_doors):
                # active=True means switch not collected, active=False means collected
                switch_collected = 0.0 if getattr(door, "active", True) else 1.0
                self._cached_switch_states[idx] = switch_collected

            # Mark states as up-to-date
            self._switch_states_changed = False

        # Get ninja position directly (no full observation needed)
        ninja_x, ninja_y = self.nplay_headless.ninja_position()

        # Use precomputed cache with cached switch states (ultra-fast path)
        if self.door_feature_cache.cache_built:
            # PERFORMANCE: No loops, just O(1) dict lookup + array copy
            features = self.door_feature_cache.get_features(
                ninja_x, ninja_y, self._cached_switch_states
            )
            return features

        # Fallback: Cache not built yet (should not happen in normal operation)
        # Return zeros to avoid crashes
        logger.warning("Door feature cache not built, returning zeros")
        return np.zeros(
            (MAX_LOCKED_DOORS_ATTENTION, LOCKED_DOOR_FEATURES_DIM), dtype=np.float32
        )

    def _hash_switch_states(self, switch_states: Dict[int, bool]) -> str:
        """
        Create hashable key from switch states.

        Args:
            switch_states: Dictionary mapping switch IDs to their states

        Returns:
            String hash of switch states for caching
        """
        if not switch_states:
            return "empty"
        return ",".join(f"{k}:{int(v)}" for k, v in sorted(switch_states.items()))

    def _compute_exit_features(self) -> np.ndarray:
        """
        Compute features for exit switch and exit door for objective attention.

        Returns (7,) array containing:
        [switch_x, switch_y, switch_activated, switch_path_dist, door_x, door_y, door_path_dist]

        All positions are relative to ninja and normalized to [-1, 1].
        Path distances are normalized to [0, 1].

        PERFORMANCE: Uses caching to avoid expensive path distance calculations when
        ninja hasn't moved significantly (>24px grid cell).
        """
        from ..constants.entity_types import EntityType
        from ..constants.physics_constants import (
            EXIT_SWITCH_RADIUS,
            EXIT_DOOR_RADIUS,
            NINJA_RADIUS,
            LEVEL_WIDTH_PX,
            LEVEL_HEIGHT_PX,
        )

        # Get ninja position
        ninja_x, ninja_y = self.nplay_headless.ninja_position()

        # Check if cache is valid (ninja in same grid cell)
        if (
            self._cached_exit_features is not None
            and self._last_exit_cache_ninja_pos is not None
        ):
            last_x, last_y = self._last_exit_cache_ninja_pos
            # Check if ninja moved to a new grid cell
            ninja_grid_x = int(ninja_x // self._exit_cache_grid_size)
            ninja_grid_y = int(ninja_y // self._exit_cache_grid_size)
            last_grid_x = int(last_x // self._exit_cache_grid_size)
            last_grid_y = int(last_y // self._exit_cache_grid_size)

            if ninja_grid_x == last_grid_x and ninja_grid_y == last_grid_y:
                # Cache hit: ninja in same grid cell, return cached features
                return self._cached_exit_features

        features = np.zeros(7, dtype=np.float32)

        # Get current entities
        level_data = self.level_data
        entities = level_data.entities if hasattr(level_data, "entities") else []

        # Find exit switch
        exit_switch = None
        for entity in entities:
            entity_type = getattr(entity, "type", None)
            if entity_type == EntityType.EXIT_SWITCH:
                exit_switch = entity
                break

        # Extract exit switch features
        if exit_switch is not None:
            switch_x = int(getattr(exit_switch, "xpos", 0))
            switch_y = int(getattr(exit_switch, "ypos", 0))
            switch_active = getattr(exit_switch, "active", True)

            # Relative position normalized to [-1, 1]
            rel_switch_x = (switch_x - ninja_x) / (LEVEL_WIDTH_PX / 2)
            rel_switch_y = (switch_y - ninja_y) / (LEVEL_HEIGHT_PX / 2)
            features[0] = np.clip(rel_switch_x, -1.0, 1.0)
            features[1] = np.clip(rel_switch_y, -1.0, 1.0)

            # Switch activation status (0.0 = active/not collected, 1.0 = collected)
            features[2] = 0.0 if switch_active else 1.0

            # Path distance to switch
            if (
                hasattr(self, "path_distance_calculator")
                and self.path_distance_calculator is not None
            ):
                adjacency = self._get_adjacency_for_rewards()
                graph_data = self.current_graph_data

                if adjacency and graph_data:
                    switch_path_dist = self.path_distance_calculator.get_distance(
                        (ninja_x, ninja_y),
                        (switch_x, switch_y),
                        adjacency,
                        level_data=level_data,
                        graph_data=graph_data,
                        entity_radius=EXIT_SWITCH_RADIUS,
                        ninja_radius=NINJA_RADIUS,
                    )

                    if switch_path_dist != float("inf"):
                        # Normalize by area scale (similar to reachability features)
                        from ..gym_environment.constants import LEVEL_DIAGONAL

                        area_scale = LEVEL_DIAGONAL  # Fallback

                        # Try to use actual reachable area scale
                        try:
                            reach_feats = self._get_reachability_features()
                            if reach_feats is not None and len(reach_feats) > 0:
                                reachable_ratio = reach_feats[0]
                                if reachable_ratio > 0:
                                    from ..graph.reachability.subcell_node_lookup import (
                                        SUB_NODE_SIZE,
                                    )
                                    from ..constants.physics_constants import (
                                        MAP_TILE_WIDTH,
                                        MAP_TILE_HEIGHT,
                                    )

                                    total_possible_nodes = (
                                        MAP_TILE_WIDTH * MAP_TILE_HEIGHT
                                    )
                                    reachable_nodes = (
                                        reachable_ratio * total_possible_nodes
                                    )
                                    area_scale = (
                                        np.sqrt(reachable_nodes) * SUB_NODE_SIZE
                                    )
                        except Exception:
                            pass

                        features[3] = np.clip(switch_path_dist / area_scale, 0.0, 1.0)

        # Find exit door
        exit_door = None
        for entity in entities:
            entity_type = getattr(entity, "type", None)
            if entity_type == EntityType.EXIT_DOOR:
                exit_door = entity
                break

        # Extract exit door features
        if exit_door is not None:
            door_x = int(getattr(exit_door, "xpos", 0))
            door_y = int(getattr(exit_door, "ypos", 0))

            # Relative position normalized to [-1, 1]
            rel_door_x = (door_x - ninja_x) / (LEVEL_WIDTH_PX / 2)
            rel_door_y = (door_y - ninja_y) / (LEVEL_HEIGHT_PX / 2)
            features[4] = np.clip(rel_door_x, -1.0, 1.0)
            features[5] = np.clip(rel_door_y, -1.0, 1.0)

            # Path distance to door
            if (
                hasattr(self, "path_distance_calculator")
                and self.path_distance_calculator is not None
            ):
                adjacency = self._get_adjacency_for_rewards()
                graph_data = self.current_graph_data

                if adjacency and graph_data:
                    door_path_dist = self.path_distance_calculator.get_distance(
                        (ninja_x, ninja_y),
                        (door_x, door_y),
                        adjacency,
                        level_data=level_data,
                        graph_data=graph_data,
                        entity_radius=EXIT_DOOR_RADIUS,
                        ninja_radius=NINJA_RADIUS,
                    )

                    if door_path_dist != float("inf"):
                        # Use same area scale as switch
                        from ..gym_environment.constants import LEVEL_DIAGONAL

                        area_scale = LEVEL_DIAGONAL

                        try:
                            reach_feats = self._get_reachability_features()
                            if reach_feats is not None and len(reach_feats) > 0:
                                reachable_ratio = reach_feats[0]
                                if reachable_ratio > 0:
                                    from ..graph.reachability.subcell_node_lookup import (
                                        SUB_NODE_SIZE,
                                    )
                                    from ..constants.physics_constants import (
                                        MAP_TILE_WIDTH,
                                        MAP_TILE_HEIGHT,
                                    )

                                    total_possible_nodes = (
                                        MAP_TILE_WIDTH * MAP_TILE_HEIGHT
                                    )
                                    reachable_nodes = (
                                        reachable_ratio * total_possible_nodes
                                    )
                                    area_scale = (
                                        np.sqrt(reachable_nodes) * SUB_NODE_SIZE
                                    )
                        except Exception:
                            pass

                        features[6] = np.clip(door_path_dist / area_scale, 0.0, 1.0)

        # Cache the computed features and ninja position
        self._cached_exit_features = features.copy()
        self._last_exit_cache_ninja_pos = (ninja_x, ninja_y)

        return features

    def invalidate_switch_cache(self):
        """
        Invalidate switch-dependent caches.

        Called by entity collision methods when switch states change.
        This is more efficient than computing hash keys every frame.

        Clears:
        - Pathfinding visualization cache (debug overlay renderer)
        - Path distance calculator cache (forces rebuild with new goals)
        - Cached level_data (entities may be stale after switch activation)
        - Exit features cache (switch activation changes objectives)
        """
        self._switch_states_changed = True

        # Clear cached level_data so it gets refreshed with fresh entities
        # CRITICAL: level_data.entities may be stale after switch activation
        if hasattr(self, "_cached_level_data"):
            self._cached_level_data = None

        # Clear exit features cache since switch activation changes goals
        # PERFORMANCE: Forces recomputation with updated switch states
        if hasattr(self, "_cached_exit_features"):
            self._cached_exit_features = None
            self._last_exit_cache_ninja_pos = None

        # Clear pathfinding visualization cache in debug overlay renderer
        # This ensures path visualizations update when switch state changes
        if (
            hasattr(self, "nplay_headless")
            and hasattr(self.nplay_headless, "sim_renderer")
            and hasattr(self.nplay_headless.sim_renderer, "debug_overlay_renderer")
        ):
            debug_renderer = self.nplay_headless.sim_renderer.debug_overlay_renderer
            if hasattr(debug_renderer, "clear_pathfinding_cache"):
                debug_renderer.clear_pathfinding_cache()

        # Clear path distance calculator cache
        # This forces rebuild with updated goals (exit switch removed, exit door becomes primary)
        if hasattr(self, "_path_calculator") and self._path_calculator is not None:
            self._path_calculator.clear_cache()

        # Clear reachability cache so it gets recomputed with fresh entities
        # CRITICAL: Reachability features depend on goal positions which change when switches activate
        if hasattr(self, "_clear_reachability_cache"):
            self._clear_reachability_cache()

    def _extract_mine_features(
        self, obs: Dict[str, Any], buffer: np.ndarray
    ) -> np.ndarray:
        """Extract enhanced mine features (8 dims).

        Features:
        [0-1] nearest_mine_rel_x, rel_y (normalized -1 to 1)
        [2] nearest_mine_state (0=deadly, 0.5=toggling, 1=safe, -1=none)
        [3] nearest_mine_path_distance (normalized 0-1)
        [4] deadly_mines_nearby_count (normalized 0-1)
        [5] mine_state_certainty (0=unknown to 1=recently seen)
        [6] safe_mines_nearby_count (normalized 0-1)
        [7] mine_avoidance_difficulty (0-1: spatial complexity)
        """
        NEARBY_RADIUS = 100.0
        MAX_NEARBY = 10.0
        CERTAINTY_RADIUS = 150.0

        features = buffer
        features.fill(0.0)

        # Early exit checks
        if (
            not hasattr(self, "_path_calculator")
            or self._path_calculator is None
            or self.current_graph_data is None
        ):
            features[2] = -1.0
            features[5] = 1.0  # High certainty when no mines
            return features

        adjacency = self.current_graph_data.get("adjacency")
        if not adjacency:
            features[2] = -1.0
            features[5] = 1.0
            return features

        # Extract ninja position
        ninja_x = obs.get("player_x", 0.0)
        ninja_y = obs.get("player_y", 0.0)

        ninja_pos = (int(ninja_x), int(ninja_y))

        # Collect mines
        entities = obs.get("entities", [])
        mines = []
        deadly_nearby = 0
        safe_nearby = 0

        for entity in entities:
            entity_type = getattr(entity, "entity_type", None)
            if entity_type in (EntityType.TOGGLE_MINE, EntityType.TOGGLE_MINE_TOGGLED):
                mx = getattr(entity, "xpos", 0.0)
                my = getattr(entity, "ypos", 0.0)
                mstate = getattr(entity, "state", 1)

                if entity_type == EntityType.TOGGLE_MINE_TOGGLED:
                    mstate = 0

                dist = np.sqrt((mx - ninja_x) ** 2 + (my - ninja_y) ** 2)

                mines.append({"x": mx, "y": my, "state": mstate, "dist": dist})

                if dist < NEARBY_RADIUS:
                    if mstate == 0:
                        deadly_nearby += 1
                    elif mstate == 1:
                        safe_nearby += 1

        if not mines:
            features[2] = -1.0
            features[5] = 1.0
            return features

        # Find nearest mine
        nearest = min(mines, key=lambda m: m["dist"])

        # Features 0-1: Relative position
        area_scale = self._get_reachable_area_scale()
        features[0] = np.clip((nearest["x"] - ninja_x) / area_scale, -1.0, 1.0)
        features[1] = np.clip((nearest["y"] - ninja_y) / area_scale, -1.0, 1.0)

        # Feature 2: State
        if nearest["state"] == 0:
            features[2] = 0.0  # Deadly
        elif nearest["state"] == 2:
            features[2] = 0.5  # Toggling
        else:
            features[2] = 1.0  # Safe

        # Feature 3: Path distance
        try:
            from ..constants.physics_constants import TOGGLE_MINE_RADII

            mine_radius = TOGGLE_MINE_RADII.get(nearest["state"], 4.0)
            mine_pos = (int(nearest["x"]), int(nearest["y"]))
            path_dist = self._safe_path_distance(
                ninja_pos,
                mine_pos,
                adjacency,
                "mine",
                level_data=self.level_data,
                graph_data=self.current_graph_data,
                entity_radius=mine_radius,
            )
            features[3] = min(path_dist / area_scale, 1.0)
        except Exception:
            features[3] = min(nearest["dist"] / area_scale, 1.0)

        # Feature 4: Deadly mines nearby
        features[4] = min(deadly_nearby / MAX_NEARBY, 1.0)

        # Feature 5: State certainty (based on distance)
        features[5] = 1.0 - min(nearest["dist"] / CERTAINTY_RADIUS, 1.0)

        # Feature 6: Safe mines nearby
        features[6] = min(safe_nearby / MAX_NEARBY, 1.0)

        # Feature 7: Avoidance difficulty
        total_nearby = deadly_nearby + safe_nearby
        if total_nearby > 0:
            danger_ratio = deadly_nearby / total_nearby
            density = total_nearby / MAX_NEARBY
            features[7] = min(0.7 * danger_ratio + 0.3 * density, 1.0)

        return features

    def _compute_progress_features(
        self, obs: Dict[str, Any], buffer: np.ndarray
    ) -> np.ndarray:
        """
        Compute progress tracking features.

        Returns PROGRESS_FEATURES_DIM (3) features:
        - current_objective_type (0=switch, 0.33=door, 0.67=exit, normalized 0-1)
        - objectives_completed_ratio (0 to 1)
        - total_path_distance_remaining (normalized)
        """
        MAX_OBJECTIVE_PATHS = 3.0  # switch + door + exit

        features = buffer
        features.fill(0.0)

        if not hasattr(self, "_path_calculator") or self._path_calculator is None:
            return features

        if self.current_graph_data is None:
            return features

        adjacency = self.current_graph_data.get("adjacency")
        if adjacency is None or len(adjacency) == 0:
            return features

        # Extract positions directly from raw observation (not dependent on processing order)
        ninja_x = obs.get("player_x", 0.0)
        ninja_y = obs.get("player_y", 0.0)
        switch_x = obs.get("switch_x", 0.0)
        switch_y = obs.get("switch_y", 0.0)
        exit_door_x = obs.get("exit_door_x", 0.0)
        exit_door_y = obs.get("exit_door_y", 0.0)

        ninja_pos = (
            int(ninja_x),
            int(ninja_y),
        )

        exit_switch_collected = obs.get("switch_activated", False)
        locked_doors = obs.get("locked_doors", [])

        completed = 0
        total = 1

        for door in locked_doors:
            total += 1
            if not getattr(door, "active", True):
                completed += 1

        if not exit_switch_collected:
            current_obj_type = 0.0
        elif any(getattr(door, "active", True) for door in locked_doors):
            current_obj_type = 0.33
        else:
            current_obj_type = 0.67

        total_path_dist = 0.0

        if not exit_switch_collected:
            from ..constants.physics_constants import EXIT_SWITCH_RADIUS

            exit_switch_pos = (
                int(switch_x),
                int(switch_y),
            )
            switch_dist = self._safe_path_distance(
                ninja_pos,
                exit_switch_pos,
                adjacency,
                "progress_exit_switch",
                level_data=self.level_data,
                graph_data=self.current_graph_data,
                entity_radius=EXIT_SWITCH_RADIUS,
            )
            total_path_dist += switch_dist

        for door in locked_doors:
            if getattr(door, "active", True):
                from ..constants.physics_constants import LOCKED_DOOR_SWITCH_RADIUS

                switch_x = getattr(door, "sw_xpos", getattr(door, "xpos", 0.0))
                switch_y = getattr(door, "sw_ypos", getattr(door, "ypos", 0.0))
                switch_pos = (int(switch_x), int(switch_y))

                door_dist = self._safe_path_distance(
                    ninja_pos,
                    switch_pos,
                    adjacency,
                    "progress_locked_door_switch",
                    level_data=self.level_data,
                    graph_data=self.current_graph_data,
                    entity_radius=LOCKED_DOOR_SWITCH_RADIUS,
                )
                total_path_dist += door_dist

        from ..constants.physics_constants import EXIT_DOOR_RADIUS

        exit_door_pos = (
            int(exit_door_x),
            int(exit_door_y),
        )
        exit_dist = self._safe_path_distance(
            ninja_pos,
            exit_door_pos,
            adjacency,
            "progress_exit_door",
            level_data=self.level_data,
            graph_data=self.current_graph_data,
            entity_radius=EXIT_DOOR_RADIUS,
        )
        total_path_dist += exit_dist

        features[0] = current_obj_type
        features[1] = completed / max(total, 1)
        area_scale = self._get_reachable_area_scale()
        features[2] = np.clip(
            total_path_dist / (area_scale * MAX_OBJECTIVE_PATHS), 0.0, 1.0
        )

        return features

    def _process_observation(self, obs):
        """Process the observation from the environment."""
        processed_obs = super()._process_observation(obs)

        # Add reachability features if not already added (always present in raw obs)
        if "reachability_features" not in processed_obs:
            processed_obs["reachability_features"] = obs.get(
                "reachability_features",
                np.zeros(REACHABILITY_FEATURES_DIM, dtype=np.float32),
            )

        # Add graph observations if enabled and not already added
        graph_obs = self._get_graph_observations()
        for key, value in graph_obs.items():
            if key not in processed_obs:
                processed_obs[key] = value

        if "switch_states" not in processed_obs:
            processed_obs["switch_states"] = obs.get(
                "switch_states", np.zeros(SWITCH_STATES_DIM, dtype=np.float32)
            )

        # MEMORY OPTIMIZATION: Remove internal-only observations that are not needed for training
        # These are only used during environment step for PBRS reward computation
        # Removing them prevents storing ~30-40% extra data in rollout buffer
        internal_only_keys = [
            "_adjacency_graph",  # Used internally for PBRS path distance calculation
            "_graph_data",  # Contains spatial_hash and other build artifacts
            "switch_states_dict",  # Dict version, array version is kept as switch_states
            "level_data",  # Only needed during graph building
        ]
        for key in internal_only_keys:
            processed_obs.pop(key, None)

        # Add death_context if not already added (always present in raw obs)
        if "death_context" not in processed_obs:
            processed_obs["death_context"] = obs.get(
                "death_context",
                np.zeros(9, dtype=np.float32),  # Default: all zeros (no death)
            )

        # Add exit features if not already added (for objective attention)
        if "exit_features" not in processed_obs:
            processed_obs["exit_features"] = obs.get(
                "exit_features",
                np.zeros(7, dtype=np.float32),  # Default: all zeros
            )

        # Add locked door features if not already added (for objective attention)
        if "locked_door_features" not in processed_obs:
            from .constants import MAX_LOCKED_DOORS_ATTENTION, LOCKED_DOOR_FEATURES_DIM

            processed_obs["locked_door_features"] = obs.get(
                "locked_door_features",
                np.zeros(
                    (MAX_LOCKED_DOORS_ATTENTION, LOCKED_DOOR_FEATURES_DIM),
                    dtype=np.float32,
                ),
            )
        if "num_locked_doors" not in processed_obs:
            processed_obs["num_locked_doors"] = obs.get(
                "num_locked_doors",
                np.array([0], dtype=np.int32),
            )

        return processed_obs

    def clear_caches(self, verbose: bool = False):
        """
        Clear all caches manually (for testing/debugging).

        This method provides external access to cache clearing,
        useful for manual cache busting in test environments.

        Args:
            verbose: If True, print cache clearing operations
        """
        from nclone.cache_management import clear_all_caches_for_new_level

        clear_all_caches_for_new_level(self, verbose=verbose)

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

        self._reinit_graph_system_after_unpickling(self.config.graph.debug)
        self._reinit_reachability_system_after_unpickling()

        # Mark that we need to reinitialize on next reset
        self._needs_reinit = True
