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


from ..graph.common import N_MAX_NODES, E_MAX_EDGES, NODE_FEATURE_DIM
from ..constants.physics_constants import LEVEL_WIDTH_PX, LEVEL_HEIGHT_PX
from .constants import (
    GAME_STATE_CHANNELS,
    LEVEL_DIAGONAL,
    REACHABILITY_FEATURES_DIM,
    MINE_SDF_FEATURES_DIM,
    MAX_LOCKED_DOORS,
    FEATURES_PER_DOOR,
    SWITCH_STATES_DIM,
    MAX_LOCKED_DOORS_ATTENTION,
    SPATIAL_CONTEXT_DIM,
)
from .spatial_context import compute_spatial_context
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
        _logger = logging.getLogger(__name__)

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
            enable_profiling=self.config.enable_profiling,
            reward_config=self.config.reward_config,
            augmentation_config={
                "p": self.config.augmentation.p,
                "intensity": self.config.augmentation.intensity,
                "disable_validation": self.config.augmentation.disable_validation,
            },
            enable_visual_observations=self.config.enable_visual_observations,
            frame_skip=self.config.frame_skip,
            shared_level_cache=self.config.shared_level_cache,
            goal_curriculum_config=self.config.goal_curriculum_config,
        )

        # Initialize mixin systems using config
        self._init_graph_system(
            debug=self.config.graph.debug,
        )
        self._init_reachability_system(self.config.reachability.debug)
        self._init_debug_system(self.config.render.enable_debug_overlay)

        # Initialize reset retry counter for handling degenerate maps
        self._reset_retry_count = 0
        self._max_reset_retries = 3

        # Update configuration flags with new options
        self.config_flags.update(
            {"debug": self.config.graph.debug or self.config.reachability.debug}
        )

        # Store graph observation flag for conditional observation space building
        # When False, graph arrays are excluded from observation space (memory optimization)
        # Graph is still built internally for PBRS reward calculation
        self.enable_graph_observations = self.config.enable_graph_observations

        # Extend observation space with graph, reachability features
        self.observation_space = self._build_extended_observation_space()

        # Pre-allocate observation buffers
        self._game_state_buffer = np.zeros(GAME_STATE_CHANNELS, dtype=np.float32)

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
        self._cached_switch_states: Optional[np.ndarray] = None

        # Exit features cache (position-based invalidation)
        # Avoids recomputing path distances when ninja hasn't moved significantly
        self._last_exit_cache_ninja_pos: Optional[tuple] = None
        self._exit_cache_grid_size: int = 24

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

        base_adjacency = (
            graph_data.get("base_adjacency", adjacency) if graph_data else adjacency
        )
        distance = self._path_calculator.get_distance(
            start_pos,
            goal_pos,
            adjacency,
            base_adjacency,
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
        """Build the extended observation space with graph and reachability features.

        Graph observation spaces are conditionally included based on enable_graph_observations:
        - When True (default): Full graph arrays included (~21KB per observation)
        - When False: Graph arrays excluded (memory optimization for graph_free architecture)

        Note: Graph is still built internally for PBRS reward calculation regardless of this flag.
        This only controls whether graph arrays appear in the observation space.
        """
        obs_spaces = dict(self.observation_space.spaces)

        # Add reachability features (always available)
        obs_spaces["reachability_features"] = box.Box(
            low=0.0, high=1.0, shape=(REACHABILITY_FEATURES_DIM,), dtype=np.float32
        )

        # Add GCN-optimized minimal graph observation spaces
        # Only includes data actually used by GCN encoder (no edge features, no types)
        # This keeps SubprocVecEnv serialization overhead low
        # MEMORY OPTIMIZATION: Skip when enable_graph_observations=False (saves ~21KB per obs)
        if self.enable_graph_observations:
            obs_spaces["graph_node_feats"] = box.Box(
                low=-np.inf,
                high=np.inf,
                shape=(N_MAX_NODES, NODE_FEATURE_DIM),  # 7-dim features
                dtype=np.float32,
            )
            obs_spaces["graph_edge_index"] = box.Box(
                low=0,
                high=N_MAX_NODES - 1,
                shape=(2, E_MAX_EDGES),
                dtype=np.uint16,
            )
            obs_spaces["graph_node_mask"] = box.Box(
                low=0,
                high=1,
                shape=(N_MAX_NODES,),
                dtype=np.uint8,
            )
            obs_spaces["graph_edge_mask"] = box.Box(
                low=0,
                high=1,
                shape=(E_MAX_EDGES,),
                dtype=np.uint8,
            )

        # Switch states for locked doors (25 total: 5 doors * 5 features each)
        # Features: [switch_x, switch_y, door_x, door_y, collected] per door
        obs_spaces["switch_states"] = box.Box(
            low=0.0,
            high=1.0,
            shape=(SWITCH_STATES_DIM,),
            dtype=np.float32,
        )

        # Spatial context for graph-free observation (96 dims)
        # 64 dims: 8×8 local tile grid (simplified categories)
        # 32 dims: 8 nearest mines (relative_pos + state + radius)
        obs_spaces["spatial_context"] = box.Box(
            low=-1.0,
            high=1.0,
            shape=(SPATIAL_CONTEXT_DIM,),
            dtype=np.float32,
        )

        # Mine SDF features for actor safety awareness (3 dims)
        # Global danger signal from ALL mines (not just 8 nearest in spatial_context)
        # [SDF value, escape gradient X, escape gradient Y]
        obs_spaces["mine_sdf_features"] = box.Box(
            low=-1.0,
            high=1.0,
            shape=(MINE_SDF_FEATURES_DIM,),
            dtype=np.float32,
        )

        return SpacesDict(obs_spaces)

    def _post_action_hook(self):
        """Update graph after action execution if needed."""
        # Graph building happens if either flag is True
        should_update = self._should_update_graph()

        if should_update:
            self._update_graph_from_env_state()

    def _extend_info_hook(self, info: Dict[str, Any]):
        """Add NppEnvironment-specific info fields."""
        # Call parent to add diagnostic and observation metrics
        super()._extend_info_hook(info)

        # These fields are removed during observation processing but needed for reward calculation
        info["player_x"] = self.nplay_headless.ninja_position()[0]
        info["player_y"] = self.nplay_headless.ninja_position()[1]

        # Add level spawn position for corruption detection in Go-Explore
        # This helps detect checkpoints with spawn position but long action sequences
        if hasattr(self, "level_data") and self.level_data is not None:
            spawn_pos = self.level_data.start_position
            info["spawn_x"] = float(spawn_pos[0])
            info["spawn_y"] = float(spawn_pos[1])

        # CRITICAL: Use cached positions instead of direct simulator calls
        # Direct calls may return (0, 0) if entities not loaded during curriculum resets
        # The cached values are validated in base_environment._get_observation()
        switch_pos = self._cached_switch_pos
        if switch_pos is None:
            switch_pos = self.nplay_headless.exit_switch_position()
            # Log if we're falling back to (0, 0)
            if switch_pos == (0, 0):
                import logging

                _logger = logging.getLogger(__name__)
                _logger.warning(
                    f"_extend_info_hook: switch_pos is (0, 0)! "
                    f"_cached_switch_pos was None, entities may not be loaded. "
                    f"frame={self.nplay_headless.sim.frame}"
                )
        exit_pos = self._cached_exit_pos
        if exit_pos is None:
            exit_pos = self.nplay_headless.exit_door_position()
            if exit_pos == (0, 0):
                import logging

                _logger = logging.getLogger(__name__)
                _logger.warning(
                    "_extend_info_hook: exit_pos is (0, 0)! "
                    "_cached_exit_pos was None, entities may not be loaded."
                )

        info["switch_x"] = switch_pos[0]
        info["switch_y"] = switch_pos[1]
        info["switch_activated"] = self.nplay_headless.exit_switch_activated()
        info["exit_door_x"] = exit_pos[0]
        info["exit_door_y"] = exit_pos[1]
        info["player_dead"] = self.nplay_headless.ninja_has_died()
        info["player_won"] = self.nplay_headless.ninja_has_won()
        info["death_cause"] = self.nplay_headless.ninja_death_cause()

        # Additional fields for Go-Explore checkpoint system
        # These are needed for doomed checkpoint invalidation (falling into mines)
        ninja_vel = self.nplay_headless.ninja_velocity()
        info["player_xspeed"] = ninja_vel[0]
        info["player_yspeed"] = ninja_vel[1]
        info["airborne"] = self.nplay_headless.sim.ninja.airborn

        # Graph size info for level-size-aware optimizations
        # Used by Go-Explore to scale checkpoint thresholds for larger levels
        if hasattr(self, "current_graph") and self.current_graph is not None:
            info["graph_num_nodes"] = self.current_graph.num_nodes
        else:
            info["graph_num_nodes"] = 0

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

        # Check if map loading should be skipped and if this is a new level
        skip_map_load = False
        new_level = False  # True if a different level was loaded externally
        map_name = None
        if options is not None and isinstance(options, dict):
            skip_map_load = options.get("skip_map_load", False)
            # When skip_map_load=True, assume new level unless explicitly told otherwise
            new_level = options.get("new_level", skip_map_load)
            map_name = options.get("map_name", None)

        # Clear caches appropriately based on whether this is a new level
        from nclone.cache_management import (
            clear_all_caches_for_reset,
            clear_all_caches_for_new_level,
        )

        if new_level:
            # New level loaded externally - clear ALL caches including level-persistent ones
            clear_all_caches_for_new_level(self)
        else:
            # Same level being reset - clear only episode-specific caches
            clear_all_caches_for_reset(self)

        if not skip_map_load:
            # Load map - this calls sim.load() which calls sim.reset()
            self.map_loader.load_map()
        else:
            # Update map_loader.current_map_name if provided
            # This is critical for cache keying and momentum waypoint loading
            if map_name is not None:
                self.map_loader.current_map_name = map_name
                if self.enable_logging:
                    logger.debug(f"Updated map_loader.current_map_name to: {map_name}")

            # Only reset the sim if a map wasn't just loaded externally
            # (sim.load() already calls sim.reset(), so we shouldn't reset again)
            map_just_loaded = getattr(self.nplay_headless, "_map_just_loaded", False)
            if not map_just_loaded:
                # If map loading is skipped but no new map loaded, reset to spawn
                # This handles checkpoint replay and same-level resets
                self.nplay_headless.reset()
            else:
                # Clear the flag after checking it
                self.nplay_headless._map_just_loaded = False
                if self.enable_logging:
                    logger.debug(
                        "Skipped nplay_headless.reset() - map was just loaded externally"
                    )

        # === GOAL CURRICULUM: Move entities before graph building ===
        # Apply curriculum entity repositioning if enabled
        # TIMING: Must happen AFTER map load, BEFORE graph building
        # This ensures all observations reflect curriculum positions
        if self.goal_curriculum_manager is not None:
            # 1. Store original entity positions (from loaded map)
            switch_pos = self.nplay_headless.exit_switch_position()
            exit_pos = self.nplay_headless.exit_door_position()
            self.goal_curriculum_manager.store_original_positions(switch_pos, exit_pos)

            # 2. Build optimal paths using ORIGINAL positions
            # Need level_data for mine proximity calculations
            # Note: level_data is built from map tiles, doesn't depend on entity positions
            level_data_for_paths = self._extract_level_data()
            spawn_pos = level_data_for_paths.start_position

            # Build temporary graph for pathfinding (without curriculum positions)
            # We need this to extract the TRUE optimal path before moving entities
            temp_graph_data = self.graph_builder.build_graph(
                level_data_for_paths, ninja_pos=(int(spawn_pos[0]), int(spawn_pos[1]))
            )

            self.goal_curriculum_manager.build_paths_for_level(
                temp_graph_data, level_data_for_paths, spawn_pos
            )

            # 3. Move entities to curriculum positions
            self.goal_curriculum_manager.apply_to_simulator(self.nplay_headless.sim)

            # 4. Invalidate ALL entity-dependent caches (will rebuild with curriculum positions)
            # CRITICAL: Must clear ALL caches that contain entity positions/references
            self._cached_switch_pos = None
            self._cached_exit_pos = None
            self._cached_level_data = None  # Contains entity references
            self._cached_entities = None  # Contains entity list
            # Force switch cache invalidation to clear path distance caches
            self.invalidate_switch_cache()

            if self.enable_logging:
                logger.debug(
                    "Invalidated all entity-dependent caches after curriculum entity movement"
                )

        # CRITICAL: Reset and rebuild graph BEFORE getting observation
        self._reset_graph_state()
        self._reset_reachability_state()

        # Build graph from the newly loaded map (with curriculum entity positions)
        self._update_graph_from_env_state()

        # Extract path-based waypoints from optimal A* paths (if enabled)
        # This provides immediate dense guidance from first episode
        self._update_path_waypoints_for_current_level()

        # # Validate spawn position can reach graph nodes (detect degenerate maps early)
        # if hasattr(self, "current_graph_data") and self.current_graph_data:
        #     adjacency = self.current_graph_data.get("adjacency")
        #     if adjacency:
        #         # Warn if graph is very small
        #         if len(adjacency) < 10:
        #             logger.warning(
        #                 f"Small graph detected: {len(adjacency)} nodes. "
        #                 f"Map: {getattr(self.map_loader, 'current_map_name', 'unknown')}"
        #             )

        #         # Validate spawn can reach at least one graph node
        #         # First verify simulator and ninja entity are properly initialized
        #         if not (
        #             hasattr(self.nplay_headless, "sim")
        #             and hasattr(self.nplay_headless.sim, "ninja")
        #             and hasattr(self.nplay_headless.sim.ninja, "xpos")
        #             and hasattr(self.nplay_headless.sim.ninja, "ypos")
        #         ):
        #             # Simulator not fully initialized - skip validation
        #             # This can happen in multiprocessing environments during unpickling
        #             logger.warning(
        #                 "Simulator not fully initialized during reset - skipping spawn validation. "
        #                 "This is expected in multiprocessing environments."
        #             )
        #         else:
        #             from nclone.graph.reachability.pathfinding_utils import (
        #                 find_closest_node_to_position,
        #             )

        #             # Get ninja position (already in world space with 1-tile padding)
        #             ninja_pos = self.nplay_headless.ninja_position()
        #             start_pos = (
        #                 int(ninja_pos[0]),
        #                 int(ninja_pos[1]),
        #             )  # Use world space coordinates directly

        #             closest = find_closest_node_to_position(
        #                 start_pos,
        #                 adjacency,
        #                 threshold=50.0,
        #                 spatial_hash=self.current_graph_data.get("spatial_hash"),
        #                 subcell_lookup=None,
        #             )

        #             if closest is None:
        #                 # Critical: Spawn unreachable from graph
        #                 self._reset_retry_count += 1

        #                 if self._reset_retry_count >= self._max_reset_retries:
        #                     # All retries failed - raise with full diagnostics
        #                     sample_nodes = list(adjacency.keys())[:10]
        #                     raise RuntimeError(
        #                         f"Failed to generate valid map after {self._max_reset_retries} attempts. "
        #                         f"Spawn consistently unreachable from graph. "
        #                         f"Last attempt: spawn={start_pos} (world space), "
        #                         f"graph_nodes={len(adjacency)}, sample_nodes={sample_nodes}. "
        #                         f"This is a critical bug in map generation."
        #                     )

        #                 # Log and retry with new map
        #                 sample_nodes = list(adjacency.keys())[:5]
        #                 logger.error(
        #                     f"CRITICAL: Spawn unreachable from graph! "
        #                     f"Retry {self._reset_retry_count}/{self._max_reset_retries}. "
        #                     f"spawn={start_pos} (world space), graph_nodes={len(adjacency)}, "
        #                     f"sample_nodes={sample_nodes}. "
        #                     f"Regenerating map..."
        #                 )

        #                 # Regenerate map and retry reset
        #                 self.map_loader.load_map()
        #                 return self.reset(seed=seed, options=options)

        #             # Success - reset retry counter
        #             self._reset_retry_count = 0

        # Build precomputed door feature cache after graph is ready
        # PERFORMANCE: Precomputes ALL path distances for O(1) runtime lookup
        self._build_door_feature_cache()

        # Initialize locked door caches for this level
        # PERFORMANCE: Cache static door entities and initialize switch states
        self._initialize_locked_door_caches()

        # Set dynamic truncation limit based on level complexity
        # Uses PBRS surface area (mine count set to 0 since simplified reward system doesn't use mine proximity)
        pbrs_calculator = self.reward_calculator.pbrs_calculator
        if pbrs_calculator._cached_surface_area is not None:
            surface_area = pbrs_calculator._cached_surface_area
            # Simplified reward system: mine proximity removed, so mine count = 0 for truncation
            # Truncation now based purely on level size (surface area)
            reachable_mine_count = 0

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

        # Phase 4: Only keep ninja physics state (40 enhanced dims)
        # All other features (path objectives, mines, progress, etc.) are now in the graph
        base_game_state = obs["game_state"]
        self._game_state_buffer[:NINJA_STATE_DIM] = base_game_state[:NINJA_STATE_DIM]

        obs["game_state"] = np.array(self._game_state_buffer, copy=True)

        obs["reachability_features"] = self._get_reachability_features()

        obs.update(self._get_graph_observations())

        # Extract locked door switch states from environment
        switch_states_dict = self._get_switch_states_from_env()

        # Update level_data.switch_states for consistent cache key generation
        # CRITICAL: This must be done before adding level_data to observation
        # so that PBRS cache keys include correct switch states
        if hasattr(self, "level_data") and self.level_data is not None:
            self.level_data.switch_states = switch_states_dict

        # Store dict version for ICM and reachability systems
        obs["switch_states_dict"] = switch_states_dict

        switch_states_array = self._build_switch_states_array(obs)
        obs["switch_states"] = switch_states_array

        obs["level_data"] = self.level_data

        # Add adjacency graph and full graph data for PBRS path-aware reward shaping
        adjacency = self._get_adjacency_for_rewards()
        if adjacency is None:
            raise RuntimeError("Adjacency graph not available. Graph building failed.")
        obs["_adjacency_graph"] = adjacency
        # Include full graph_data for spatial indexing (contains spatial_hash)
        obs["_graph_data"] = self.current_graph_data

        # Add mine SDF features for actor safety awareness (3 dims)
        # These provide GLOBAL mine danger signal from ALL mines (not just 8 nearest)
        obs["mine_sdf_features"] = self._compute_mine_sdf_features(obs)

        # Add privileged features for asymmetric critic (if enabled)
        # These provide oracle information to the critic (not actor) for better value estimation
        obs["privileged_features"] = self._compute_privileged_features(obs)

        return obs

    def _compute_mine_sdf_features(self, obs: Dict[str, Any]) -> np.ndarray:
        """Compute mine SDF features for actor safety awareness.

        These features provide GLOBAL mine danger information from ALL mines,
        not just the 8 nearest mines in spatial_context. This helps the actor
        make better decisions about avoiding dangerous areas.

        MINE SDF FEATURES (3 dimensions):
        0. SDF value at ninja [-1,1]: distance to nearest mine (negative=danger zone)
        1. Escape gradient X [-1,1]: X component of escape direction
        2. Escape gradient Y [-1,1]: Y component of escape direction

        All features use O(1) lookups from pre-computed 88x50 SDF grid (12px resolution).

        Returns:
            np.ndarray: Mine SDF features [3]
        """
        mine_sdf_features = np.zeros(MINE_SDF_FEATURES_DIM, dtype=np.float32)

        try:
            ninja_x = obs.get("player_x", 0.0)
            ninja_y = obs.get("player_y", 0.0)

            pbrs_calculator = self.reward_calculator.pbrs_calculator
            if (
                pbrs_calculator is not None
                and pbrs_calculator.path_calculator is not None
            ):
                mine_sdf = pbrs_calculator.path_calculator.mine_sdf
                if mine_sdf is not None and mine_sdf.sdf_grid is not None:
                    # Feature 0: SDF value at ninja position
                    # Negative = inside danger zone, positive = safe
                    sdf_value = mine_sdf.get_sdf_at_position(ninja_x, ninja_y)
                    mine_sdf_features[0] = sdf_value

                    # Features 1-2: Escape gradient (unit vector away from nearest mine)
                    # Tells actor optimal direction to move for safety
                    grad_x, grad_y = mine_sdf.get_gradient_at_position(ninja_x, ninja_y)
                    mine_sdf_features[1] = grad_x
                    mine_sdf_features[2] = grad_y

        except Exception as e:
            logger.debug(f"Failed to compute mine SDF features: {e}")
            # Return safe zeros on failure

        return mine_sdf_features

    def _compute_privileged_features(self, obs: Dict[str, Any]) -> np.ndarray:
        """Compute privileged features for asymmetric critic.

        DESIGN RATIONALE:
        For graph_free architecture, actor sees:
        - game_state (41 dims): ninja physics state
        - reachability_features (22 dims): path distances, directions, mine context
        - spatial_context (112 dims): LOCAL 8x8 tiles + 8 nearest mines
        - mine_sdf_features (3 dims): GLOBAL SDF value + escape gradient

        Critic needs ADDITIONAL information actor cannot access. Features must be:
        1. O(1) lookups from pre-computed data (zero overhead)
        2. Genuinely unavailable to actor (not just reformatted)
        3. Predictive of future returns (helps value estimation)

        PRIVILEGED FEATURES (18 dimensions):

        Path Topology Features (6 dims) - future path structure:
        0. Combined remaining distance - total journey length
        1. Inflection point distance - next major direction change
        2. Path segments remaining - count of turns ahead
        3. Upcoming segment difficulty - look-ahead to next segment
        4. Alternative paths exist - multiple viable routes?
        5. Dead-end proximity - distance to nearest branch termination

        A* Pathfinding Internals (4 dims) - values actor cannot access:
        6. Current node g-cost - cost from start
        7. Current node f-cost - g + heuristic estimate
        8. Path optimality ratio - current/optimal cost
        9. Backtrack penalty - deviation from optimal

        Expert Demonstration Oracle (3 dims) - from replay data:
        10. Expert action - from nearest demo state
        11. Expert Q-estimate - interpolated from demos
        12. Demo distance - proximity to demonstrated states

        Global Level Context (3 dims) - level-wide metrics:
        13. Level difficulty score - precomputed complexity
        14. Progress fraction - traveled/total distance
        15. Estimated remaining steps - from path calculator

        Refined Physics Context (2 dims):
        16. Mine proximity cost - A* cost multipliers
        17. Graph connectivity - movement options

        All features use O(1) lookups from structures already computed for PBRS.

        Returns:
            np.ndarray: Privileged features [18]
        """
        # 18 dimensions: rich privileged information for critic
        privileged_features = np.zeros(18, dtype=np.float32)

        try:
            # Get ninja position and common data for lookups
            ninja_x = obs.get("player_x", 0.0)
            ninja_y = obs.get("player_y", 0.0)
            ninja_pos = (int(ninja_x), int(ninja_y))
            switch_activated = obs.get("switch_activated", 0.0)

            pbrs_calculator = self.reward_calculator.pbrs_calculator
            path_calculator = None
            if pbrs_calculator is not None:
                path_calculator = pbrs_calculator.path_calculator

            # === PATH TOPOLOGY FEATURES (0-5) ===
            if path_calculator is not None:
                level_data = obs.get("level_data")

                # Feature 0: Combined remaining distance
                # Total journey: distance to switch + distance to exit
                try:
                    if switch_activated < 0.5:
                        # Phase 1: Need to reach switch then exit
                        switch_dist = path_calculator.get_distance_to_switch(ninja_pos)
                        exit_dist_from_switch = (
                            path_calculator.get_distance_switch_to_exit()
                        )
                        combined_dist = switch_dist + exit_dist_from_switch
                    else:
                        # Phase 2: Just need to reach exit
                        combined_dist = path_calculator.get_distance_to_exit(ninja_pos)

                    # Normalize by level diagonal (~1200px)
                    from .constants import LEVEL_DIAGONAL

                    privileged_features[0] = min(
                        1.0, combined_dist / (LEVEL_DIAGONAL * 2.0)
                    )
                except Exception:
                    pass

                # Feature 1: Inflection point distance
                # Distance to next major direction change (>45°)
                try:
                    current_goal = "exit" if switch_activated > 0.5 else "switch"
                    path = path_calculator.get_full_path(ninja_pos, current_goal)
                    if path and len(path) > 2:
                        # Find first inflection point
                        for i in range(len(path) - 2):
                            dx1 = path[i + 1][0] - path[i][0]
                            dy1 = path[i + 1][1] - path[i][1]
                            dx2 = path[i + 2][0] - path[i + 1][0]
                            dy2 = path[i + 2][1] - path[i + 1][1]

                            # Compute angle change
                            mag1 = max(1.0, np.sqrt(dx1**2 + dy1**2))
                            mag2 = max(1.0, np.sqrt(dx2**2 + dy2**2))
                            dot = (dx1 * dx2 + dy1 * dy2) / (mag1 * mag2)
                            dot = np.clip(dot, -1.0, 1.0)

                            if np.arccos(dot) > np.pi / 4:  # >45° change
                                # Compute distance to this inflection
                                dist_to_inflection = sum(
                                    np.sqrt(
                                        (path[j + 1][0] - path[j][0]) ** 2
                                        + (path[j + 1][1] - path[j][1]) ** 2
                                    )
                                    for j in range(i + 1)
                                )
                                privileged_features[1] = min(
                                    1.0, dist_to_inflection / 500.0
                                )
                                break
                except Exception:
                    pass

                # Feature 2: Path segments remaining
                # Count of major direction changes ahead
                try:
                    current_goal = "exit" if switch_activated > 0.5 else "switch"
                    path = path_calculator.get_full_path(ninja_pos, current_goal)
                    if path and len(path) > 2:
                        segment_count = 0
                        for i in range(len(path) - 2):
                            dx1 = path[i + 1][0] - path[i][0]
                            dy1 = path[i + 1][1] - path[i][1]
                            dx2 = path[i + 2][0] - path[i + 1][0]
                            dy2 = path[i + 2][1] - path[i + 1][1]

                            mag1 = max(1.0, np.sqrt(dx1**2 + dy1**2))
                            mag2 = max(1.0, np.sqrt(dx2**2 + dy2**2))
                            dot = (dx1 * dx2 + dy1 * dy2) / (mag1 * mag2)
                            dot = np.clip(dot, -1.0, 1.0)

                            if np.arccos(dot) > np.pi / 4:
                                segment_count += 1

                        # Normalize by max expected (10 segments)
                        privileged_features[2] = min(1.0, segment_count / 10.0)
                except Exception:
                    pass

            # Features 3-5: Advanced path features (placeholder for future implementation)
            # These require more complex path analysis
            # Feature 3: Upcoming segment difficulty - would need segment-wise cost analysis
            # Feature 4: Alternative paths - would need multi-path A*
            # Feature 5: Dead-end proximity - would need branch detection

            # === A* PATHFINDING INTERNALS (6-9) ===
            # Features 6-9: Pathfinding cost metrics (placeholder for future implementation)
            # These would require storing g-cost, f-cost during pathfinding
            # Currently path_calculator provides distances but not A* internals

            # === EXPERT DEMONSTRATION ORACLE (10-12) ===
            # Features 10-12: Demo-based guidance (placeholder for future implementation)
            # Would require nearest-neighbor search in demonstration dataset
            # Integration with Go-Explore archive could provide this

            # === GLOBAL LEVEL CONTEXT (13-15) ===
            if path_calculator is not None:
                # Feature 13: Level difficulty score
                # Based on path length and mine count
                try:
                    level_data = obs.get("level_data")
                    if level_data and hasattr(level_data, "entities"):
                        mine_count = sum(
                            1
                            for e in level_data.entities
                            if hasattr(e, "entity_type")
                            and "TOGGLE_MINE" in str(e.entity_type).upper()
                        )

                        # Get total path length
                        spawn_pos = getattr(level_data, "start_position", (0, 0))
                        switch_dist = path_calculator.get_distance_to_switch(spawn_pos)
                        exit_dist = path_calculator.get_distance_switch_to_exit()
                        total_path = switch_dist + exit_dist

                        # Difficulty = (path_length/1000 + mine_count/10) / 2
                        path_difficulty = min(1.0, total_path / 2000.0)
                        mine_difficulty = min(1.0, mine_count / 20.0)
                        privileged_features[13] = (
                            path_difficulty + mine_difficulty
                        ) / 2.0
                except Exception:
                    pass

                # Feature 14: Progress fraction
                # How far along the optimal path?
                try:
                    level_data = obs.get("level_data")
                    if level_data and hasattr(level_data, "start_position"):
                        spawn_pos = getattr(level_data, "start_position", ninja_pos)

                        # Total optimal path length
                        switch_dist_from_spawn = path_calculator.get_distance_to_switch(
                            spawn_pos
                        )
                        exit_dist = path_calculator.get_distance_switch_to_exit()
                        total_optimal = switch_dist_from_spawn + exit_dist

                        # Remaining distance
                        if switch_activated < 0.5:
                            remaining = (
                                path_calculator.get_distance_to_switch(ninja_pos)
                                + exit_dist
                            )
                        else:
                            remaining = path_calculator.get_distance_to_exit(ninja_pos)

                        # Progress = (total - remaining) / total
                        if total_optimal > 0:
                            progress = max(
                                0.0, (total_optimal - remaining) / total_optimal
                            )
                            privileged_features[14] = min(1.0, progress)
                except Exception:
                    pass

                # Feature 15: Estimated remaining steps
                # Convert distance to frame estimates
                try:
                    if switch_activated < 0.5:
                        remaining_dist = path_calculator.get_distance_to_switch(
                            ninja_pos
                        )
                        remaining_dist += path_calculator.get_distance_switch_to_exit()
                    else:
                        remaining_dist = path_calculator.get_distance_to_exit(ninja_pos)

                    # Estimate: distance / (average_speed * frame_skip)
                    # Average speed ~3-4 px/frame, frame_skip=4
                    avg_speed_per_action = 12.0  # 3px/frame * 4 frames
                    estimated_actions = remaining_dist / avg_speed_per_action

                    # Normalize by max expected (1000 actions)
                    privileged_features[15] = min(1.0, estimated_actions / 1000.0)
                except Exception:
                    pass

            # === REFINED PHYSICS CONTEXT (16-17) ===
            # Feature 16: Mine proximity cost
            if pbrs_calculator is not None and path_calculator is not None:
                mine_cache = path_calculator.mine_proximity_cache
                if mine_cache is not None:
                    try:
                        cost_multiplier = mine_cache.get_cost_multiplier(ninja_pos)
                        from .reward_calculation.reward_constants import (
                            MINE_HAZARD_COST_MULTIPLIER,
                        )

                        if MINE_HAZARD_COST_MULTIPLIER > 1.0:
                            normalized_cost = (cost_multiplier - 1.0) / (
                                MINE_HAZARD_COST_MULTIPLIER - 1.0
                            )
                            privileged_features[16] = min(1.0, normalized_cost)
                    except Exception:
                        pass

            # Feature 17: Graph connectivity
            adjacency = obs.get("_adjacency_graph")
            if adjacency:
                try:
                    num_nodes = len(adjacency)
                    num_edges = sum(len(neighbors) for neighbors in adjacency.values())
                    avg_edges = num_edges / max(num_nodes, 1)
                    privileged_features[17] = min(1.0, avg_edges / 6.0)
                except Exception:
                    pass

        except Exception as e:
            logger.debug(f"Failed to compute privileged features: {e}")
            # Return safe zeros on failure - critic will learn with partial info

        return privileged_features

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

        # CRITICAL FIX: Also clear reward calculator's path calculator cache!
        # The reward calculator has its own separate path_calculator that needs clearing too
        if hasattr(self, "reward_calculator"):
            if hasattr(self.reward_calculator, "pbrs_calculator"):
                if hasattr(self.reward_calculator.pbrs_calculator, "path_calculator"):
                    reward_path_calc = (
                        self.reward_calculator.pbrs_calculator.path_calculator
                    )
                    if reward_path_calc is not None:
                        reward_path_calc.clear_cache()

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
        """Process the observation from the environment.

        Graph observations are conditionally included based on enable_graph_observations:
        - When True (default): Graph arrays added to observation (~21KB per obs)
        - When False: Graph arrays excluded (memory optimization for graph_free architecture)

        Note: Graph is still built internally for PBRS reward calculation regardless.
        """
        processed_obs = super()._process_observation(obs)

        # Add reachability features if not already added (always present in raw obs)
        if "reachability_features" not in processed_obs:
            processed_obs["reachability_features"] = obs.get(
                "reachability_features",
                np.zeros(REACHABILITY_FEATURES_DIM, dtype=np.float32),
            )

        # Add graph observations only if enabled (memory optimization)
        # Graph is still built internally for PBRS, but arrays excluded from obs when disabled
        if self.enable_graph_observations:
            graph_obs = self._get_graph_observations()
            for key, value in graph_obs.items():
                if key not in processed_obs:
                    processed_obs[key] = value

        if "switch_states" not in processed_obs:
            processed_obs["switch_states"] = obs.get(
                "switch_states", np.zeros(SWITCH_STATES_DIM, dtype=np.float32)
            )

        # Add spatial_context (graph-free local geometry features)
        if "spatial_context" not in processed_obs:
            processed_obs["spatial_context"] = self._compute_spatial_context()

        # Add mine_sdf_features (global mine danger signal for actor safety)
        if "mine_sdf_features" not in processed_obs:
            processed_obs["mine_sdf_features"] = obs.get(
                "mine_sdf_features",
                np.zeros(MINE_SDF_FEATURES_DIM, dtype=np.float32),
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

        return processed_obs

    def _compute_spatial_context(self) -> np.ndarray:
        """Compute spatial context features (graph-free local geometry).

        Returns 112-dimensional array:
        - 64 dims: 8×8 local tile grid (simplified tile categories)
        - 48 dims: 8 nearest mines (6 features each):
          - relative_pos (2), state (1), radius (1)
          - velocity_dot_direction (1), distance_rate (1)

        Markov Property: All features depend only on current state (position, velocity).

        Returns:
            np.ndarray: 112-dimensional spatial context features
        """
        # Get ninja position and velocity (current state only - Markovian)
        ninja_pos = self.nplay_headless.ninja_position()
        ninja_velocity = self.nplay_headless.ninja_velocity()

        # Get level data for tiles and entities
        level_data = self.level_data

        return compute_spatial_context(
            ninja_pos=ninja_pos,
            ninja_velocity=ninja_velocity,
            tiles=level_data.tiles,
            entities=level_data.entities,
            level_width=LEVEL_WIDTH_PX,
            level_height=LEVEL_HEIGHT_PX,
        )

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

    def _save_degenerate_map_debug_image(self, tile_dic, adjacency, num_nodes):
        """Save a debug visualization when a degenerate map (tiny graph) is detected.

        Args:
            tile_dic: Dictionary mapping (x, y) coordinates to tile type values
            adjacency: Graph adjacency structure
            num_nodes: Number of nodes in the graph
        """
        try:
            import matplotlib

            matplotlib.use("Agg")  # Non-interactive backend
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            from datetime import datetime
            import os

            # Get spawn position
            ninja_pos = self.nplay_headless.ninja_position()
            spawn_x, spawn_y = int(ninja_pos[0]), int(ninja_pos[1])

            # Create output directory
            output_dir = "debug_degenerate_maps"
            os.makedirs(output_dir, exist_ok=True)

            # Generate filename with timestamp and map name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            map_name = getattr(self.map_loader, "current_map_name", "unknown")
            filename = f"{output_dir}/{timestamp}_{map_name}_nodes{num_nodes}.png"

            # Create figure
            fig, ax = plt.subplots(figsize=(16, 10))

            # Render tiles - import from npp-rl if available, otherwise skip
            try:
                import sys

                # Try to import from npp-rl project
                npp_rl_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                    "npp-rl",
                )
                if os.path.exists(npp_rl_path):
                    sys.path.insert(0, npp_rl_path)

                from npp_rl.rendering.matplotlib_tile_renderer import (
                    render_tiles_to_axis,
                )

                render_tiles_to_axis(
                    ax,
                    tile_dic,
                    tile_size=24.0,
                    tile_color="#808080",
                    alpha=0.6,
                    show_tile_labels=False,
                )
            except ImportError:
                # Fallback: render simple rectangles for solid tiles
                for (x, y), tile_type in tile_dic.items():
                    if tile_type == 1:  # Solid tiles
                        rect = mpatches.Rectangle(
                            (x * 24, y * 24), 24, 24, facecolor="#808080", alpha=0.6
                        )
                        ax.add_patch(rect)

            # Draw graph nodes
            for node_pos in adjacency.keys():
                x, y = node_pos
                # Convert from tile data space (subtracts 24px) to display space
                screen_x = x + 24  # Add back the offset
                screen_y = y + 24
                circle = mpatches.Circle(
                    (screen_x, screen_y),
                    5,
                    facecolor="green",
                    edgecolor="black",
                    linewidth=1,
                    alpha=0.8,
                )
                ax.add_patch(circle)

            # Draw graph edges
            for node_pos, neighbors in adjacency.items():
                x1, y1 = node_pos
                screen_x1 = x1 + 24
                screen_y1 = y1 + 24
                for neighbor_pos, _ in neighbors:
                    x2, y2 = neighbor_pos
                    screen_x2 = x2 + 24
                    screen_y2 = y2 + 24
                    ax.plot(
                        [screen_x1, screen_x2],
                        [screen_y1, screen_y2],
                        "y-",
                        alpha=0.5,
                        linewidth=1,
                    )

            # Mark spawn position
            spawn_circle = mpatches.Circle(
                (spawn_x, spawn_y),
                8,
                facecolor="red",
                edgecolor="white",
                linewidth=2,
                alpha=1.0,
            )
            ax.add_patch(spawn_circle)

            # Add text annotation for spawn
            ax.text(
                spawn_x,
                spawn_y - 20,
                f"SPAWN\n({spawn_x}, {spawn_y})",
                ha="center",
                va="top",
                fontsize=10,
                color="red",
                weight="bold",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
            )

            # Configure plot
            ax.set_xlim(0, 44 * 24)  # Full map width with padding
            ax.set_ylim(25 * 24, 0)  # Full map height with padding (inverted Y)
            ax.set_aspect("equal")
            ax.set_title(
                f"Degenerate Map Debug: {map_name}\n"
                f"Nodes: {num_nodes}, Spawn: ({spawn_x}, {spawn_y})",
                fontsize=14,
                weight="bold",
            )
            ax.set_xlabel("X (pixels)", fontsize=10)
            ax.set_ylabel("Y (pixels)", fontsize=10)

            # Add grid
            ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.5)

            # Add legend
            legend_elements = [
                mpatches.Circle(
                    (0, 0),
                    1,
                    facecolor="red",
                    edgecolor="white",
                    linewidth=2,
                    label="Spawn Position",
                ),
                mpatches.Circle(
                    (0, 0), 1, facecolor="green", edgecolor="black", label="Graph Node"
                ),
                mpatches.Patch(facecolor="yellow", alpha=0.5, label="Graph Edge"),
                mpatches.Patch(facecolor="#808080", alpha=0.6, label="Solid Tile"),
            ]
            ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

            # Save figure
            plt.tight_layout()
            plt.savefig(filename, dpi=150, bbox_inches="tight")
            plt.close(fig)

            logger.info(f"Saved degenerate map debug image: {filename}")

        except Exception as e:
            logger.warning(f"Failed to save degenerate map debug image: {e}")

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
