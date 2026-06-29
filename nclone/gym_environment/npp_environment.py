"""
Refactored N++ environment class using mixins.

This environment provides a Gym interface for the N++ game, allowing reinforcement
learning agents to learn to play levels. The functionality is now organized using
mixins for better code organization and maintainability.
"""

import logging
import numpy as np
from gymnasium.spaces import box, Dict as SpacesDict
from typing import Dict, Any, Optional, Tuple


from ..constants import MAP_TILE_HEIGHT, MAP_TILE_WIDTH
from ..constants.physics_constants import (
    LEVEL_WIDTH_PX,
    LEVEL_HEIGHT_PX,
    MAX_HOR_SPEED,
)
from .constants import (
    GAME_STATE_CHANNELS,
    GAME_STATE_SWITCH_ACTIVATED_IDX,
    GAME_STATE_SWITCH_VEC_IDX,
    GAME_STATE_EXIT_VEC_IDX,
    GAME_STATE_RAYCAST_IDX,
    GAME_STATE_RAYCAST_DIM,
    GAME_STATE_FREE_SPACE_NAV_IDX,
    GAME_STATE_NEAREST_MINE_IDX,
    GAME_STATE_CLOSING_SPEED_IDX,
    GAME_STATE_JUMP_HELD_IDX,
    GAME_STATE_JUMP_USELESS_IDX,
    GAME_STATE_LAUNCHPAD_BUFFER_IDX,
    LEVEL_WIDTH,
    LEVEL_HEIGHT,
    LEVEL_DIAGONAL,
    HAZARD_MAP_CHANNELS,
    HAZARD_MAP_HEIGHT,
    HAZARD_MAP_WIDTH,
    LATTICE_NAV_DIM,
    MAX_LOCKED_DOORS,
    FEATURES_PER_DOOR,
    SWITCH_STATES_DIM,
    MAX_LOCKED_DOORS_ATTENTION,
)
from ..constants.entity_types import EntityType
from .base_environment import BaseNppEnvironment
from .entity_overlay_utils import classify_mines_by_state
from .layer3_macro_action_mask import Layer3MacroActionMasker
from .mixins import GraphMixin, ReachabilityMixin, DebugMixin
from .config import EnvironmentConfig
from .precomputed_door_features import PrecomputedDoorFeatureCache
from nclone_native import MineThreatMap

logger = logging.getLogger(__name__)


class NppEnvironment(BaseNppEnvironment, GraphMixin, ReachabilityMixin, DebugMixin):
    """
    Consolidated N++ environment class using mixins.

    This environment provides a Gym interface for the N++ game, allowing reinforcement
    learning agents to learn to play levels. We use a headless version of the game
    to speed up training.

    Features:
    - Multiple observation profiles (minimal/rich)
    - HWM A* distance shaping (single unified reward path)
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
            panel_width=self.config.render.panel_width,
            seed=self.config.seed,
            eval_mode=self.config.eval_mode,
            pbrs_gamma=self.config.pbrs.pbrs_gamma,
            custom_map_path=self.config.custom_map_path,
            test_dataset_path=self.config.test_dataset_path,
            defer_initial_map_load=self.config.defer_initial_map_load,
            enable_profiling=self.config.enable_profiling,
            profiling_console_interval=self.config.profiling_console_interval,
            reward_config=self.config.reward_config,
            frame_skip=self.config.frame_skip,
            shared_level_cache=self.config.shared_level_cache,
            sim_backend=self.config.sim_backend,
            enable_layer3_macro_action_mask=self.config.enable_layer3_macro_action_mask,
            layer3_duration_bins=self.config.layer3_duration_bins,
            layer3_all_masked_fallback=self.config.layer3_all_masked_fallback,
            enable_route_visualization=self.config.enable_route_visualization,
        )

        self.layer3_masker: Optional[Layer3MacroActionMasker]
        if self.config.enable_layer3_macro_action_mask:
            self.layer3_masker = Layer3MacroActionMasker(
                duration_bins=self.config.layer3_duration_bins,
                fallback_mode=self.config.layer3_all_masked_fallback,
            )
        else:
            self.layer3_masker = None

        # Initialize mixin systems using config
        self._init_graph_system(
            debug=self.config.graph.debug,
        )
        self._init_reachability_system(self.config.reachability.debug)
        # Single CachedPathDistanceCalculator: reward and reachability share SDF, mine
        # proximity, and level path caches (avoids duplicate rebuilds on graph/mine changes).
        self.reward_calculator.path_calculator = self._path_calculator
        self._init_debug_system(self.config.render.enable_debug_overlay)

        # Update configuration flags with new options
        self.config_flags.update(
            {"debug": self.config.graph.debug or self.config.reachability.debug}
        )

        # Rust mine SDF engines powering the 2-channel `hazard_map` Layer 1 key.
        # One engine per class (deadly / safe) so each keeps its own change-detect
        # state; both emit a row-major (H, W) buffer the env transposes to (W, H).
        self._mine_threat_map_deadly_rs = MineThreatMap()
        self._mine_threat_map_safe_rs = MineThreatMap()
        # Per-class transposed (W, H) channel caches and the stacked (2, W, H) map.
        self._hazard_deadly_chan = np.zeros(
            (HAZARD_MAP_WIDTH, HAZARD_MAP_HEIGHT), dtype=np.float32
        )
        self._hazard_safe_chan = np.zeros(
            (HAZARD_MAP_WIDTH, HAZARD_MAP_HEIGHT), dtype=np.float32
        )
        self._hazard_map_cache = np.zeros(
            (HAZARD_MAP_CHANNELS, HAZARD_MAP_WIDTH, HAZARD_MAP_HEIGHT),
            dtype=np.float32,
        )
        # Cheap mine-state signature for the hazard map. When it is unchanged
        # (the common case — a static mine field never toggles) we skip the whole
        # per-step prep (classify + build the (x,y) lists over ~1.7k mines + the
        # Rust change-detect) and return the cached SDF directly.
        self._hazard_sig = None

        # Deadly-mine position array (Nx2) + its mine signature, for the scalar
        # nearest-deadly-mine game_state feature. Recomputed only when the deadly
        # set changes (a toggle); otherwise the per-step cost is a vectorized min.
        self._deadly_mine_xy: Optional[np.ndarray] = None
        self._deadly_xy_sig = None

        # Cached tile grid for the current level (computed once per level load, reused each step)
        self._cached_tile_grid: Optional[np.ndarray] = None
        # Stable integer level ID derived from map name hash (for cache keying in the encoder)
        self._cached_level_id_int: Optional[np.ndarray] = None

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
        distance = self._path_calculator.get_path_cost(
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
        """Build the Layer 1 policy observation space.

        Exactly four structured tensors plus control masks:

        * ``tile_grid``       — inherited from base env
        * ``entity_overlays`` — inherited from base env
        * ``hazard_map``      — 1-channel deadly-mine SDF (NPP-46)
        * ``game_state``      — inherited from base env (overwritten with 28 dims)
        * ``action_mask`` / ``duration_mask`` — control masks

        Graph tensors, reachability vectors, spatial_context, mine_sdf_features,
        level_id and switch_states are kept internally by reward-side
        infrastructure but are NOT exposed to the policy.
        """
        obs_spaces = dict(self.observation_space.spaces)

        obs_spaces["hazard_map"] = box.Box(
            low=-1.0,
            high=1.0,
            shape=(HAZARD_MAP_CHANNELS, HAZARD_MAP_WIDTH, HAZARD_MAP_HEIGHT),
            dtype=np.float32,
        )

        # Optional physics-aware navigation hint from the lattice d* plan
        # (``[dx_hop, dy_hop, cost_to_go_norm, phase]``).  Off by default so the
        # strict Layer-1 contract / existing checkpoints are unaffected; enabled
        # via ReachabilityConfig.expose_lattice_features.  See lattice_distance.
        if getattr(getattr(self.config, "reachability", None),
                   "expose_lattice_features", False):
            obs_spaces["lattice_nav"] = box.Box(
                low=-1.0, high=1.0, shape=(LATTICE_NAV_DIM,), dtype=np.float32,
            )

        # Optional waypoint-subgoal nav hint (same 4-dim contract as lattice_nav).
        # Off by default; enabled via ReachabilityConfig.expose_waypoint_features.
        if getattr(getattr(self.config, "reachability", None),
                   "expose_waypoint_features", False):
            obs_spaces["waypoint_nav"] = box.Box(
                low=-1.0, high=1.0, shape=(LATTICE_NAV_DIM,), dtype=np.float32,
            )

        # Optional cost-to-go map (Phase-2 O2): the free-space cost-to-completion
        # field as a low-res spatial channel co-registered with hazard_map.
        if getattr(getattr(self.config, "reachability", None),
                   "expose_cost_to_go_map", False):
            obs_spaces["cost_to_go_map"] = box.Box(
                low=0.0, high=1.0,
                shape=(1, HAZARD_MAP_WIDTH, HAZARD_MAP_HEIGHT), dtype=np.float32,
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
        info["ninja_state"] = self.nplay_headless.sim.ninja.state
        # True whenever the ninja is touching a wall (wall_normal != 0), in ANY
        # state — the real wall-jump precondition (a REGULAR wall jump fires while
        # walled even when not in the wall-slide state). FiGAR uses this to
        # interrupt macros on wall contact so wall-jumps stay frame-actionable.
        info["ninja_walled"] = self.nplay_headless.sim.ninja.walled

        # Mine approach distance for FiGAR macro-action safety interrupts
        info["nearest_approaching_deadly_mine_distance"] = (
            self._compute_nearest_approaching_deadly_mine_distance()
        )

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

    def _compute_level_surface_area(self) -> Optional[float]:
        """Compute reachable surface area (node count) for the current level.

        Used to set a complexity-scaled truncation limit at level load.
        Returns ``None`` if graph data is not yet available.
        """
        graph_data = getattr(self, "current_graph_data", None)
        if not graph_data:
            return None
        adjacency = graph_data.get("adjacency")
        level_data = self.level_data
        if not adjacency or level_data is None:
            return None
        start_pos = getattr(level_data, "start_position", None)
        if start_pos is None:
            return None
        try:
            from ..graph.reachability.surface_area import get_cached_surface_area

            cache_key = level_data.get_cache_key_for_reachability(
                include_switch_states=True
            )
            return get_cached_surface_area(
                cache_key, start_pos, adjacency, graph_data
            )
        except Exception as e:
            logger.debug(f"Failed to compute level surface area: {e}")
            return None

    def _spawn_cost_to_completion(self) -> Optional[float]:
        """Best-effort free-space cost-to-completion from spawn (~pixels).

        Used to give long, winding levels a path-length-aware truncation limit.
        Builds the (disk-cached, ~ms) free-space source for the current map and
        queries the spawn cost. Returns ``None`` on any failure or non-finite cost
        so the caller falls back to the surface-area formula.
        """
        try:
            md = getattr(self.nplay_headless, "current_map_data", None)
            pos = self.nplay_headless.ninja_position()
            if not md or pos is None:
                return None
            from ..value_function.free_space_distance import FreeSpaceDistanceSource

            fs = FreeSpaceDistanceSource(md)
            if not getattr(fs, "available", False):
                return None
            c = fs.cost_to_completion(float(pos[0]), float(pos[1]), False)
            return float(c) if c is not None and c < float("inf") else None
        except Exception as e:  # noqa: BLE001
            logger.debug(f"spawn cost-to-completion unavailable: {e}")
            return None

    def reset(self, seed=None, options=None):
        """Reset the environment with planning components and visualization."""

        # Start profiling total reset time
        # reset_start_time = self._profile_start("reset_total")

        # Handle reinitialization after unpickling
        if hasattr(self, "_needs_reinit") and self._needs_reinit:
            self.observation_processor.reset()
            self.reward_calculator.reset()
            self._needs_reinit = False

        # === FAST RESET PATH DETECTION ===
        # Detect if this is a same-level reset (enables optimized reset path)
        same_level_reset = self._is_same_level_reset(options)

        # Check if map loading should be skipped and if this is a new level
        skip_map_load = False
        new_level = False  # True if a different level was loaded externally
        map_name = None
        option_map_data = None
        if options is not None and isinstance(options, dict):
            skip_map_load = options.get("skip_map_load", False)
            # When skip_map_load=True, assume new level unless explicitly told otherwise
            new_level = options.get("new_level", skip_map_load)
            map_name = options.get("map_name", None)
            option_map_data = options.get("map_data", None)
            if option_map_data is not None:
                # Explicit payload-driven load: force the standard (non-fast)
                # reset path so caches get cleared and the new map sticks.
                new_level = True

        # === FAST RESET PATH (2-3x faster for single-level training) ===
        # Use fast reset when:
        # 1. Same level detected (no map change)
        # 2. Not a new level load
        # 3. Entities already exist (not first reset)
        use_fast_reset = (
            same_level_reset
            and not new_level
            and hasattr(self.nplay_headless, "sim")
            and self.nplay_headless.sim.ninja is not None
        )

        if use_fast_reset:
            if self.enable_logging:
                logger.info(
                    "[FAST_RESET] Using optimized reset path for same-level training"
                )

            # Fast reset: entities in-place, no recreation
            # fast_sim_start = self._profile_start("reset_fast_sim")
            self.nplay_headless.fast_reset()
            # self._profile_end("reset_fast_sim", fast_sim_start)

            # Clear only episode-specific caches (not level-persistent ones)
            from nclone.cache_management import clear_episode_caches_only

            clear_episode_caches_only(self)

            # Reset observation processor (buffer clearing, no reallocation)
            self.observation_processor.reset()

            # ============================================================
            # Reset tracking variables from BaseNppEnvironment
            # ============================================================
            # Mirror the base reset path for per-episode state so fast resets
            # don't retain step-by-step history across the whole training run.
            self.last_action = None
            self._action_sequence = []
            self._checkpoint_replay_in_progress = False
            self._checkpoint_route = []
            self._checkpoint_source = None
            self._from_checkpoint = False
            self._checkpoint_base_reward = 0.0
            self._checkpoint_source_frame_skip = self.frame_skip
            self._checkpoint_replay_frame_count = 0
            self._switch_activation_frame = None
            self._switch_activation_step = None
            self.current_route = []
            self._warned_about_position = False
            # Force a hazard-map recompute on the first step of the new episode
            # (the level/mine field may have changed) — content-based after that.
            self._hazard_sig = None
            self._deterministic_mask_applied_count = 0
            self._total_steps_count = 0

            self._reset_reward_calculator()

            self.truncation_checker.reset()

            self.current_ep_reward = 0

            # Clear per-step path caches (ninja position resets to spawn).
            # Level-scoped caches remain valid because entity positions are unchanged.
            if hasattr(self, "reward_calculator") and hasattr(
                self.reward_calculator, "path_calculator"
            ):
                self.reward_calculator.path_calculator.clear_step_cache()

            # Reset graph state (fast - graph structure unchanged)
            # graph_start = self._profile_start("reset_fast_graph")
            self._reset_graph_state()
            # self._reset_reachability_state()

            # Update graph from current env state (fast - uses cached graph structure)
            self._update_graph_from_env_state()
            # self._profile_end("reset_fast_graph", graph_start)

            # Optional start-state curriculum: restore a mid-route ninja state.
            self._maybe_apply_start_state(options)

            # Get initial observation and return
            # obs_start = self._profile_start("reset_fast_observation")
            initial_obs = self._get_observation()
            processed_obs = self._process_observation(initial_obs)
            # self._profile_end("reset_fast_observation", obs_start)

            # Track current map name for fast reset detection
            self._last_reset_map_name = self.map_loader.current_map_name

            # End profiling and return
            # self._profile_end("reset_total", reset_start_time)

            return (processed_obs, {})

        # === STANDARD RESET PATH (full reset) ===
        # Only reached if NOT using fast reset
        else:
            # Reset observation processor
            self.observation_processor.reset()

            # ============================================================
            # CRITICAL: Reset tracking variables from BaseNppEnvironment
            # NppEnvironment doesn't call super().reset(), so we must do it here
            # ============================================================

            # Reset switch activation tracking (BaseNppEnvironment.reset line 1519)
            self._switch_activation_frame = None
            self._switch_activation_step = None

            # Reset position tracking for route visualization (BaseNppEnvironment.reset line 1515-1516)
            self.current_route = []
            self._warned_about_position = False
            # Force a hazard-map recompute on the first step of the new episode
            # (the level/mine field may have changed) — content-based after that.
            self._hazard_sig = None

            self._reset_reward_calculator()

            self.truncation_checker.reset()

            self.current_ep_reward = 0

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

        if option_map_data is not None:
            # Benchmark-driven training / eval: the level payload is supplied
            # directly via reset options. Bypass every dataset map-selection
            # branch and load the payload.
            effective_name = map_name if map_name is not None else "benchmark_level"
            self.map_loader.load_map_data(option_map_data, str(effective_name))
        elif not skip_map_load:
            self.map_loader.load_map()
        else:
            # Map loading skipped
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

        # CRITICAL: Reset all entity activation states after map/sim operations
        # Checkpoint replay and previous episodes can leave switches activated
        # New episodes must always start with switches deactivated
        try:
            sim = self.nplay_headless.sim

            # Reset exit DOOR (type 3) - switch_hit=False and remove from grid_entity
            exit_doors = sim.entity_dic.get(3, [])
            for door in exit_doors:
                if type(door).__name__ == "EntityExit":
                    # Reset switch_hit flag
                    if hasattr(door, "switch_hit"):
                        door.switch_hit = False
                    # CRITICAL: Remove exit door from grid_entity if present
                    # The door should only be in grid after switch collection
                    if (
                        door.cell in sim.grid_entity
                        and door in sim.grid_entity[door.cell]
                    ):
                        sim.grid_entity[door.cell].remove(door)
                        logger.debug(f"Removed exit door from grid_entity[{door.cell}]")

            # Reset exit SWITCH (type 4) - active=True (not collected)
            exit_switches = sim.entity_dic.get(4, [])
            for switch in exit_switches:
                if type(switch).__name__ == "EntityExitSwitch":
                    if hasattr(switch, "active"):
                        switch.active = True  # True = not collected
                    logger.debug("Reset exit switch: active=True")

            # Reset locked door switches to deactivated
            locked_door_entities = sim.entity_dic.get(6, [])
            for door in locked_door_entities:
                if hasattr(door, "active"):
                    door.active = True  # True = not collected
                if hasattr(door, "open"):
                    door.open = False  # False = door closed

            if self.enable_logging:
                logger.debug("Reset all entity activation states for new episode")
        except Exception as e:
            logger.warning(f"Failed to reset entity states: {e}")

        # CRITICAL: Reset and rebuild graph BEFORE getting observation
        self._reset_graph_state()
        self._reset_reachability_state()

        # Build graph from the newly loaded map (with curriculum entity positions)
        self._update_graph_from_env_state()

        # Trigger feature extractor cache precomputation (for hybrid_cached_graph)
        # This precomputes static graph embeddings once per level for O(1) runtime extraction
        if hasattr(self, "_feature_extractor_cache_callback"):
            level_id = self.map_loader.current_map_name
            self._feature_extractor_cache_callback(level_id, self.current_graph_data)

        # Build precomputed door feature cache after graph is ready
        # PERFORMANCE: Precomputes ALL path distances for O(1) runtime lookup
        self._build_door_feature_cache()

        # Initialize locked door caches for this level
        # PERFORMANCE: Cache static door entities and initialize switch states
        self._initialize_locked_door_caches()

        # Set dynamic truncation limit based on level complexity (reachable surface area).
        # Computed once per level via the shared module-level surface-area cache.
        surface_area = self._compute_level_surface_area()
        if surface_area is not None:
            truncation_limit = self.truncation_checker.set_level_truncation_limit(
                surface_area, 0, cost_to_completion=self._spawn_cost_to_completion()
            )
            if self.enable_logging:
                logger.info(
                    f"Level truncation limit: {truncation_limit} frames "
                    f"(surface_area={surface_area:.0f})"
                )

        # Optional start-state curriculum: restore a mid-route ninja state.
        self._maybe_apply_start_state(options)

        # NOW get initial observation (with valid graph data)
        initial_obs = self._get_observation()
        processed_obs = self._process_observation(initial_obs)

        # Track current map name for fast reset detection
        self._last_reset_map_name = self.map_loader.current_map_name

        # End profiling (standard reset path)
        # self._profile_end("reset_total", reset_start_time)

        return (processed_obs, {})

    # Frame-input table for start-action prefixes (matches the planner ACT map).
    _START_ACT = {0: (0, 0), 1: (1, 0), 2: (-1, 0), 3: (0, 1), 4: (1, 1), 5: (-1, 1)}

    def _maybe_apply_start_state(self, options):
        """Optional start-state curriculum hook (general, sim-agnostic).

        ``options['start_state']`` is a curriculum start state — a dict with
        ``start_actions`` (a frame-input prefix, list of action ids 0-5) — or
        ``options['start_state_curriculum']`` is an object with ``.sample(difficulty)``
        returning one. After the normal reset we **replay that prefix on the env's
        own sim**, so the episode begins mid-route and the state is faithful to the
        env's physics by construction (works whether the env is Python- or
        native-backed; no save/restore or cross-engine parity concern). If the
        prefix dies partway (e.g. a native-found prefix drifting on the Python sim),
        we stop at the last surviving frame — always a valid start state.

        This is the basis for a reverse curriculum (start near the goal, schedule
        the start back toward spawn) for fine-tuning hard levels, and for
        restart-state diversity in generalized training."""
        ss = None
        if options and isinstance(options, dict):
            ss = options.get("start_state")
            if ss is None:
                cur = options.get("start_state_curriculum")
                if cur is not None and hasattr(cur, "sample"):
                    ss = cur.sample(options.get("start_state_difficulty", 1.0))
        # Lazy POST-SWITCH reverse curriculum (FAR_OBJECTIVE loop levels): on the
        # first reset of a gated-bucket level, build (disk-cached) a per-level pool
        # of switch-already-activated start states and seed from it. Fixes the
        # return-to-exit leg never being practised (exit_given_switch == 0).
        if ss is None and getattr(self.config, "post_switch_curriculum", False):
            ss = self._maybe_sample_post_switch(options)
        # Fall back to a STORED curriculum (set via set_start_state_curriculum) so
        # the reverse curriculum also fires on VecEnv AUTO-resets, where SB3 calls
        # reset() with no options -- this is what drives it during RL rollouts.
        if ss is None:
            cur = getattr(self, "_ssc", None)
            if cur is None:
                ssc_map = getattr(self, "_ssc_map", None)
                if ssc_map:
                    lvl = getattr(
                        getattr(self, "map_loader", None), "current_map_name", None
                    )
                    cur = ssc_map.get(lvl)
            if cur is not None and hasattr(cur, "sample"):
                ss = cur.sample(float(getattr(self, "_ssc_difficulty", 1.0)))
        if not ss:
            return
        actions = ss.get("start_actions") if isinstance(ss, dict) else ss
        if not actions:
            return
        nh = self.nplay_headless
        try:
            for a in actions:
                h, j = self._START_ACT.get(int(a), (0, 0))
                nh.tick(h, j)
                if nh.ninja_has_died() or nh.ninja_has_won():
                    break
        except Exception:
            return
        # Re-decide per-episode potential gates at the new start position.
        for attr in ("_lattice_episode_decided", "_freespace_episode_decided"):
            if hasattr(self, attr):
                setattr(self, attr, False)

    def _maybe_sample_post_switch(self, options):
        """Lazily build+sample a per-level POST-SWITCH start state.

        Gated to the buckets in ``config.post_switch_curriculum_buckets`` (the
        FAR_OBJECTIVE loop levels). The per-level pool is generated on first
        encounter and disk-cached (keyed by map_data hash), so multi-level
        training pays the RRT-build cost once per level, shared across workers.
        Returns a start-state dict (``{'start_actions': [...], ...}``) or ``None``
        when the level isn't gated, has no map_data this reset, or yields no
        switch-consuming prefix."""
        if not options or not isinstance(options, dict):
            return None
        bucket = options.get("bucket")
        gated = getattr(self.config, "post_switch_curriculum_buckets", ())
        if bucket is None or str(bucket) not in {str(b) for b in gated}:
            return None
        map_data = options.get("map_data")
        if not map_data:
            return None
        try:
            from .post_switch_curriculum_cache import get_or_build

            cur = get_or_build(
                list(map_data),
                n_states=int(getattr(self.config, "post_switch_curriculum_n_states", 300)),
                max_iter=int(getattr(self.config, "post_switch_curriculum_max_iter", 120000)),
            )
        except Exception:
            return None
        if not cur or len(cur) == 0:
            return None
        # The reverse-curriculum difficulty is annealed by the trainer callback
        # via set_start_state_difficulty (default 1.0 == full return leg).
        return cur.sample(float(getattr(self, "_ssc_difficulty", 1.0)))

    def set_start_state_curriculum(self, cur) -> None:
        """Store a single StartStateCurriculum (single-level reverse curriculum).
        Used on VecEnv auto-resets via the stored-curriculum fallback above."""
        self._ssc = cur

    def set_start_state_curricula(self, mapping: dict) -> None:
        """Store a {level_id: StartStateCurriculum} map (multi-level/curriculum
        reverse curriculum). The env applies the curriculum matching the level it
        just loaded, so only the hard levels we precomputed routes for get it."""
        self._ssc_map = dict(mapping or {})

    def set_start_state_difficulty(self, d: float) -> None:
        """Reverse-curriculum difficulty in [0,1] (0=near goal, 1=spawn). The
        callback anneals this UP as the agent's success on the hard bucket rises."""
        self._ssc_difficulty = float(d)

    def _get_observation(self) -> Dict[str, Any]:
        """Get the current observation from the game state."""
        t_sup = self._profile_start("obs_super_layers")
        obs = super()._get_observation()
        self._profile_end("obs_super_layers", t_sup)

        from .constants import NINJA_STATE_DIM, PHYSICS_DERIVED_DIM

        # Phase 4: Only keep ninja physics state (40 enhanced dims)
        # All other features (path objectives, mines, progress, etc.) are now in the graph
        base_game_state = obs["game_state"]
        self._game_state_buffer[:NINJA_STATE_DIM] = base_game_state[:NINJA_STATE_DIM]

        # Append physics-derived features: horizontal_range, max_jump_height, time_to_ground
        t_phys = self._profile_start("obs_npp_physics_derived")
        self._game_state_buffer[
            NINJA_STATE_DIM : NINJA_STATE_DIM + PHYSICS_DERIVED_DIM
        ] = self._compute_physics_derived_features()
        self._profile_end("obs_npp_physics_derived", t_phys)

        # L1.4: phase indicator for policy (same source as obs["switch_activated"])
        switch_activated = bool(self.nplay_headless.exit_switch_activated())
        self._game_state_buffer[GAME_STATE_SWITCH_ACTIVATED_IDX] = float(
            switch_activated
        )

        # Goal geometry: ninja-relative vectors to the exit switch and exit door,
        # normalized by level extent and clipped to [-1, 1]. Static entities sit on
        # a 6px grid, so these are exact (the 24px overlay tile would otherwise
        # quantize the goal to 1 of 16 sub-positions). Exposing both keeps the
        # policy's goal geometry continuous across the switch->exit phase change.
        nx, ny = self.nplay_headless.ninja_position()
        sx = float(obs.get("switch_x", nx))
        sy = float(obs.get("switch_y", ny))
        ex = float(obs.get("exit_door_x", nx))
        ey = float(obs.get("exit_door_y", ny))
        self._game_state_buffer[GAME_STATE_SWITCH_VEC_IDX] = np.clip(
            (sx - nx) / LEVEL_WIDTH, -1.0, 1.0
        )
        self._game_state_buffer[GAME_STATE_SWITCH_VEC_IDX + 1] = np.clip(
            (sy - ny) / LEVEL_HEIGHT, -1.0, 1.0
        )
        self._game_state_buffer[GAME_STATE_EXIT_VEC_IDX] = np.clip(
            (ex - nx) / LEVEL_WIDTH, -1.0, 1.0
        )
        self._game_state_buffer[GAME_STATE_EXIT_VEC_IDX + 1] = np.clip(
            (ey - ny) / LEVEL_HEIGHT, -1.0, 1.0
        )

        # Surface ray-casts: distance to the nearest grounded platform in 8 compass
        # directions (blind-jump / gap verification beyond the 24px tile view).
        t_ray = self._profile_start("obs_npp_raycasts")
        self._game_state_buffer[
            GAME_STATE_RAYCAST_IDX : GAME_STATE_RAYCAST_IDX + GAME_STATE_RAYCAST_DIM
        ] = self._compute_surface_raycasts()
        self._profile_end("obs_npp_raycasts", t_ray)

        # Free-space nav hint: physics-free steepest-descent direction toward the
        # current objective that rounds obstacles (vs the straight-line goal vectors
        # above). Observation only — never reused as a reward potential.
        t_fsn = self._profile_start("obs_npp_free_space_nav")
        self._refresh_free_space_nav(nx, ny, switch_activated)
        self._game_state_buffer[
            GAME_STATE_FREE_SPACE_NAV_IDX : GAME_STATE_FREE_SPACE_NAV_IDX + 4
        ] = getattr(self, "_cached_free_space_nav", np.zeros(4, dtype=np.float32))
        self._profile_end("obs_npp_free_space_nav", t_fsn)

        # Scalar nearest-deadly-mine distance (danger gauge for the MLP stream).
        self._game_state_buffer[GAME_STATE_NEAREST_MINE_IDX] = (
            self._compute_nearest_deadly_mine_distance_norm()
        )

        # Closing speed toward the current objective (switch pre-activation, else
        # exit): the rate at which the ninja is approaching the live goal.
        gx, gy = (ex, ey) if switch_activated else (sx, sy)
        gdx, gdy = gx - nx, gy - ny
        gmag = (gdx * gdx + gdy * gdy) ** 0.5
        if gmag > 1e-6:
            vx, vy = self.nplay_headless.ninja_velocity()
            closing = (vx * gdx + vy * gdy) / gmag  # px/frame toward goal
            self._game_state_buffer[GAME_STATE_CLOSING_SPEED_IDX] = float(
                np.clip(closing / MAX_HOR_SPEED, -1.0, 1.0)
            )
        else:
            self._game_state_buffer[GAME_STATE_CLOSING_SPEED_IDX] = 0.0

        # Jump-execution features (precise jump / wall-jump timing). The feed-forward
        # policy can't track its own action history, so expose: whether jump was held
        # last frame (jumps fire only on the rising edge), whether a jump now would be
        # useless, and the launch-pad buffer window (the 4th buffer). Read straight off
        # the ninja (both python + native expose these fields).
        _ninja = self.nplay_headless.sim.ninja
        self._game_state_buffer[GAME_STATE_JUMP_HELD_IDX] = (
            1.0 if _ninja.jump_input_old != 0 else -1.0
        )
        _has_buf = (
            -1 < _ninja.jump_buffer < 5
            or -1 < _ninja.floor_buffer < 5
            or -1 < _ninja.wall_buffer < 5
            or -1 < _ninja.launch_pad_buffer < 4
        )
        _jump_useless = (
            bool(_ninja.airborn)
            and _ninja.state != 3
            and _ninja.jump_input_old != 0
            and not _has_buf
        )
        self._game_state_buffer[GAME_STATE_JUMP_USELESS_IDX] = (
            1.0 if _jump_useless else -1.0
        )
        self._game_state_buffer[GAME_STATE_LAUNCHPAD_BUFFER_IDX] = (
            (max(_ninja.launch_pad_buffer, 0) / 4.0) * 2 - 1
        )

        obs["game_state"] = np.array(self._game_state_buffer, copy=True)

        # Populate the internal A*-path-distance cache used by PBRS. The feature
        # vector is discarded; only `_cached_switch_distance` / `_cached_exit_distance`
        # (set as a side-effect of `_compute_reachability`) matter for the reward
        # path. These are not exposed to the policy.
        t_path = self._profile_start("obs_npp_path_dist")
        self._update_path_distance_cache()
        self._profile_end("obs_npp_path_dist", t_path)
        if hasattr(self, "_cached_switch_distance"):
            obs["_cached_switch_distance"] = self._cached_switch_distance
        if hasattr(self, "_cached_exit_distance"):
            obs["_cached_exit_distance"] = self._cached_exit_distance
        # Number of necessary locked doors opened (from the lattice door plan);
        # consumed by the reward calculator for the locked-door-switch bonus +
        # HWM sub-goal phase. 0 when no plan / no necessary doors.
        obs["necessary_doors_open"] = int(getattr(self, "_necessary_doors_open", 0))

        # Extract locked door switch states from environment — kept internally
        # for the reward path even though `switch_states` is no longer in the
        # policy obs dict.
        t_sa = self._profile_start("obs_npp_switch_adj")
        switch_states_dict = self._get_switch_states_from_env()
        if hasattr(self, "level_data") and self.level_data is not None:
            self.level_data.switch_states = switch_states_dict
        obs["switch_states_dict"] = switch_states_dict

        obs["level_data"] = self.level_data

        # Add adjacency graph and full graph data for PBRS path-aware reward shaping.
        adjacency = self._get_adjacency_for_rewards()
        if adjacency is None:
            raise RuntimeError("Adjacency graph not available. Graph building failed.")
        obs["_adjacency_graph"] = adjacency
        obs["_graph_data"] = self.current_graph_data
        self._profile_end("obs_npp_switch_adj", t_sa)

        t_haz = self._profile_start("obs_npp_hazard")
        obs["hazard_map"] = self._compute_hazard_map()
        self._profile_end("obs_npp_hazard", t_haz)

        # Optional physics-aware nav hint (lattice d*).  ``_cached_lattice_nav``
        # is refreshed by `_compute_reachability` (called via
        # `_update_path_distance_cache` above) on cell change; zeros when no plan.
        if getattr(getattr(self.config, "reachability", None),
                   "expose_lattice_features", False):
            obs["lattice_nav"] = np.array(
                getattr(self, "_cached_lattice_nav", np.zeros(LATTICE_NAV_DIM)),
                copy=True, dtype=np.float32,
            )

        # Waypoint subgoal feed (long-horizon distance fix).  Reward-only distance +
        # index drive the per-leg waypoint PBRS (stripped from the policy obs in
        # _process_observation); the 4-dim nav hint is exposed to the policy only when
        # expose_waypoint_features is set.  See ReachabilityConfig.enable_waypoint_tracker.
        if getattr(self, "_waypoint_leg_distance", None) is not None:
            obs["waypoint_leg_distance"] = self._waypoint_leg_distance
            obs["waypoint_index"] = self._waypoint_index
        if getattr(getattr(self.config, "reachability", None),
                   "expose_waypoint_features", False):
            obs["waypoint_nav"] = np.array(
                getattr(self, "_cached_waypoint_nav", np.zeros(LATTICE_NAV_DIM)),
                copy=True, dtype=np.float32,
            )

        # Cost-to-go map (Phase-2 O2): global route field for the policy.
        if getattr(getattr(self.config, "reachability", None),
                   "expose_cost_to_go_map", False):
            obs["cost_to_go_map"] = self._compute_cost_to_go_map(switch_activated)

        # Return full observation - conversion to Layer 1 dict happens in
        # _process_observation(). This ensures the reward calculator has
        # access to all metadata.
        return obs

    def _update_path_distance_cache(self) -> None:
        """Run the A* path distance computation to populate reward caches.

        This used to be part of `_get_reachability_features`, which also
        produced a feature vector for the policy. The feature vector is no
        longer part of the observation (Layer 1), but the cached path
        distances (`_cached_switch_distance`, `_cached_exit_distance`) are
        still consumed by ``RewardCalculator._get_distances`` to avoid a
        duplicate A* call. Any exception here is non-fatal since the reward
        calculator falls back to direct path cost queries.
        """
        try:
            # Compute the reward distances EVERY frame at sub-node resolution
            # (not the 24px-cell-cached _compute_reachability, whose feature vector
            # is unused in Layer 1). This keeps the PBRS/HWM gradient alive during
            # slow <=1px movement while skipping the wasted feature-vector work.
            self._refresh_reward_distances(self.nplay_headless.ninja_position())
        except Exception:
            pass

    def _compute_physics_derived_features(self) -> np.ndarray:
        """Compute 3 physics-derived features for value function alignment with Φ.

        Features: horizontal_range_norm, max_jump_height_norm, time_to_ground_norm.
        Time-to-ground uses geometry-accurate downward raycast via adjacency graph.
        """
        import math
        from .constants import PHYSICS_DERIVED_DIM
        from ..constants.physics_constants import (
            DRAG_REGULAR,
            GRAVITY_FALL,
            GRAVITY_JUMP,
            MAX_HOR_SPEED,
            MAX_JUMP_DURATION,
        )
        from ..graph.reachability.directional_connectivity import find_platform_in_direction
        from ..graph.reachability.path_distance_cache import NODE_WORLD_COORD_OFFSET

        features = np.zeros(PHYSICS_DERIVED_DIM, dtype=np.float32)
        ninja = self.nplay_headless.sim.ninja
        vx, vy = ninja.xspeed, ninja.yspeed
        g = ninja.applied_gravity

        # time_to_ground: geometry-accurate via downward raycast
        if ninja.floor_count > 0:
            time_to_ground = 0.0
        else:
            graph_data = getattr(self, "current_graph_data", None) or {}
            spatial_hash = graph_data.get("spatial_hash")
            physics_cache = graph_data.get("node_physics")
            base_adjacency = graph_data.get("base_adjacency")

            if spatial_hash is not None and physics_cache is not None and base_adjacency:
                node_pos = (
                    int(ninja.xpos) - NODE_WORLD_COORD_OFFSET,
                    int(ninja.ypos) - NODE_WORLD_COORD_OFFSET,
                )
                # The downward raycast result is a pure function of node_pos and
                # the (static, per-level) graph, but it costs ~40 spatial-hash
                # queries. Memoize it per level keyed by node_pos. ``graph_data``
                # is rebuilt (fresh dict) on level change, so the memo auto-
                # invalidates. Bit-identical: only the raycast distance is cached;
                # the min() with the continuous height bound is applied after.
                memo = graph_data.get("_downcast_memo")
                if memo is None:
                    memo = {}
                    graph_data["_downcast_memo"] = memo
                dy = memo.get(node_pos)
                if dy is None:
                    dy = find_platform_in_direction(
                        node_pos,
                        (0, 1),  # South = downward in screen coords
                        base_adjacency,
                        physics_cache,
                        spatial_hash,
                        max_distance=500.0,
                        step_size=12,
                    )
                    memo[node_pos] = dy
                dy = min(dy, max(0.0, LEVEL_HEIGHT_PX - ninja.ypos))
            else:
                dy = max(0.0, LEVEL_HEIGHT_PX - ninja.ypos)

            discriminant = vy * vy + 2.0 * g * dy
            if g > 1e-9 and discriminant >= 0:
                time_to_ground = max(0.0, (-vy + math.sqrt(discriminant)) / g)
            else:
                time_to_ground = 90.0

        time_to_ground_norm = min(time_to_ground / 90.0, 1.0)
        features[2] = float(time_to_ground_norm)

        # horizontal_range_norm: 0 when grounded, else abs(vx)*time*drag^(t/2)
        if ninja.floor_count > 0:
            horizontal_range_norm = 0.0
        else:
            drag_factor = DRAG_REGULAR ** (time_to_ground / 2.0) if time_to_ground > 0 else 1.0
            horizontal_range = abs(vx) * time_to_ground * drag_factor
            max_range = MAX_HOR_SPEED * 90.0
            horizontal_range_norm = min(horizontal_range / max_range, 1.0)
        features[0] = float(horizontal_range_norm)

        # max_jump_height_norm: remaining upward displacement if in jump state
        if ninja.state == 3 and ninja.jump_duration < MAX_JUMP_DURATION and vy < 0:
            frames_left = MAX_JUMP_DURATION - ninja.jump_duration
            # Discrete: vy (negative) decreases toward 0 by GRAVITY_JUMP each frame.
            # Approx displacement = avg_|vy| * frames_left
            remaining_height = abs(vy) * frames_left * 0.5
            max_jump_height_norm = min(remaining_height / 60.0, 1.0)
        else:
            max_jump_height_norm = 0.0
        features[1] = float(max_jump_height_norm)

        return features

    # 8 compass directions in (tile_dx, tile_dy) screen space (+x east, +y south),
    # order [E, NE, N, NW, W, SW, S, SE] to match COMPASS_DIRECTIONS.
    _RAYCAST_DIRS = (
        (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1),
    )
    _RAYCAST_NORM_PX = 500.0  # distance normalizer; 1.0 = nearest surface >= 500px

    def _compute_surface_raycasts(self) -> np.ndarray:
        """8-dir normalized distance to the nearest solid surface (wall / floor).

        Order [E, NE, N, NW, W, SW, S, SE]; 0.0 = adjacent surface, 1.0 = none
        within 500px. Ray-marches the cached tile grid (cheap: <=42 int8 lookups
        per direction) from the ninja's tile to the first non-empty tile, treating
        the enclosing border as a wall. Lets the policy sense walls / verify
        landing surfaces and gaps beyond the local view. Memoized per ninja tile
        (24px) in ``current_graph_data`` (auto-invalidated on level change)."""
        from ..constants.physics_constants import (
            MAP_TILE_HEIGHT,
            MAP_TILE_WIDTH,
            TILE_PIXEL_SIZE,
        )

        grid = getattr(self, "_cached_tile_grid", None)
        if grid is None:
            return np.ones(GAME_STATE_RAYCAST_DIM, dtype=np.float32)

        ninja = self.nplay_headless.sim.ninja
        ntx = int(ninja.xpos / TILE_PIXEL_SIZE) - 1
        nty = int(ninja.ypos / TILE_PIXEL_SIZE) - 1

        graph_data = getattr(self, "current_graph_data", None)
        memo = None
        if graph_data is not None:
            memo = graph_data.get("_raycast_memo")
            if memo is None:
                memo = {}
                graph_data["_raycast_memo"] = memo
            cached = memo.get((ntx, nty))
            if cached is not None:
                return cached

        out = np.ones(GAME_STATE_RAYCAST_DIM, dtype=np.float32)
        ts = float(TILE_PIXEL_SIZE)
        for i, (dtx, dty) in enumerate(self._RAYCAST_DIRS):
            tx, ty, steps = ntx, nty, 0
            while True:
                tx += dtx
                ty += dty
                steps += 1
                if not (0 <= tx < MAP_TILE_WIDTH and 0 <= ty < MAP_TILE_HEIGHT):
                    break  # enclosing border acts as a wall
                if grid[tx, ty] != 0:
                    break  # first non-empty tile = surface
            diag = 1.41421356 if (dtx and dty) else 1.0
            out[i] = min(steps * ts * diag / self._RAYCAST_NORM_PX, 1.0)

        if memo is not None:
            memo[(ntx, nty)] = out
        return out

    def _compute_cost_to_go_map(self, switch_activated: bool) -> np.ndarray:
        """Phase-2 (O2): the free-space cost-to-go FIELD as a (1, W, H) spatial map,
        co-registered with hazard_map (6px grid). Phase-aware (cost-to-switch pre-
        activation, cost-to-exit after), normalized by the PBRS K and clamped to
        [0,1]; solid/unreachable cells read 1.0 (max cost). Reuses the per-level
        FreeSpaceDistanceSource already built for the nav hint (line ~943), so this
        is a cheap slice — no extra Dijkstra. All-1.0 if the source is unavailable."""
        W, H = HAZARD_MAP_WIDTH, HAZARD_MAP_HEIGHT
        K = float(getattr(self, "_cost_to_go_map_k", 1000.0))
        src = getattr(self, "_free_space_nav_source", None)
        if src is None or not getattr(src, "available", False):
            return np.ones((1, W, H), dtype=np.float32)
        field = getattr(src, "_d_exit" if switch_activated else "_d_switch", None)
        if field is None:
            return np.ones((1, W, H), dtype=np.float32)
        m = np.asarray(field, dtype=np.float32)[:W, :H]   # (W,H), co-registered 6px
        out = np.clip(m / K, 0.0, 1.0)
        out[~np.isfinite(m)] = 1.0
        return out[None, :, :]

    def _compute_hazard_map(self) -> np.ndarray:
        """NPP-46: 2-channel mine SDF at 6px (hazard_map key).

        Each channel is `sdf = clip((d - 14) / 24, -1, 1)` where `d` is the
        Euclidean distance from each 6px cell center to the nearest mine of that
        class. Positive outside the 14px kill zone, zero at the boundary, negative
        inside. Channels: ch0 = deadly (toggled) mines, ch1 = safe (untoggled)
        mines. Shape: (2, W, H) — same axis order as tile_grid / entity_overlays.

        Recomputes a channel only when its mine set changes vs the last call.
        Classification uses the shared ``classify_mines_by_state`` helper so the
        overlay and the hazard map never disagree after a mid-episode toggle.
        """
        toggle_mines, toggled_mines = self.nplay_headless.get_mine_entities()
        # Fast path: a cheap (#deadly, #safe) signature. type-21 (toggled) mines
        # are always deadly; type-1 (toggle) mines are deadly when state==0. On a
        # field that never toggles this is constant after step 1, so we skip the
        # full rebuild entirely. (A simultaneous opposite-toggle that keeps both
        # counts stable would leave a 1-step-stale hazard OBS — harmless, self-heals
        # next step.) MINE_STATE_DEADLY == 0.
        n_deadly = len(toggled_mines) + sum(
            1 for m in toggle_mines if getattr(m, "state", 1) == 0
        )
        n_safe = len(toggle_mines) + len(toggled_mines) - n_deadly
        sig = (len(toggle_mines), len(toggled_mines), n_deadly, n_safe)
        if sig == self._hazard_sig and self._hazard_map_cache is not None:
            return self._hazard_map_cache
        self._hazard_sig = sig

        deadly_mines, safe_mines = classify_mines_by_state(
            (*toggle_mines, *toggled_mines)
        )
        # Blind the hazard SDF to mines sealed away from reachable space.
        deadly_mines = self._filter_relevant_mines(deadly_mines)
        safe_mines = self._filter_relevant_mines(safe_mines)
        deadly = [(float(m.xpos), float(m.ypos)) for m in deadly_mines]
        safe = [(float(m.xpos), float(m.ypos)) for m in safe_mines]

        # Each Rust engine emits a row-major (H, W) flat buffer (or None when its
        # mine set is unchanged); transpose to (W, H) so both channels co-register
        # with tile_grid / entity_overlays, then stack.
        deadly_flat = self._mine_threat_map_deadly_rs.compute_hazard_map_if_changed(
            deadly
        )
        safe_flat = self._mine_threat_map_safe_rs.compute_hazard_map_if_changed(safe)
        if deadly_flat is not None:
            self._hazard_deadly_chan = (
                np.asarray(deadly_flat, dtype=np.float32)
                .reshape(HAZARD_MAP_HEIGHT, HAZARD_MAP_WIDTH)
                .T.copy()
            )
        if safe_flat is not None:
            self._hazard_safe_chan = (
                np.asarray(safe_flat, dtype=np.float32)
                .reshape(HAZARD_MAP_HEIGHT, HAZARD_MAP_WIDTH)
                .T.copy()
            )
        self._hazard_map_cache = np.stack(
            (self._hazard_deadly_chan, self._hazard_safe_chan)
        )
        return self._hazard_map_cache

    # Normalizer for the scalar nearest-deadly-mine feature: 7 tiles. Near-mine
    # resolution is fine; threats beyond this saturate to 1.0 (effectively "safe").
    _NEAREST_MINE_NORM_PX = 168.0

    def _get_deadly_mine_xy(self) -> np.ndarray:
        """(N, 2) array of deadly-mine pixel centers, cached by mine signature.

        Recomputed only when the deadly set changes (a toggle / level change);
        otherwise the per-step nearest-mine read is just a vectorized min."""
        toggle_mines, toggled_mines = self.nplay_headless.get_mine_entities()
        n_deadly = len(toggled_mines) + sum(
            1 for m in toggle_mines if getattr(m, "state", 1) == 0
        )
        sig = (len(toggle_mines), len(toggled_mines), n_deadly)
        if sig == self._deadly_xy_sig and self._deadly_mine_xy is not None:
            return self._deadly_mine_xy
        self._deadly_xy_sig = sig
        pts = [
            (float(m.xpos), float(m.ypos))
            for m in self._filter_relevant_mines([*toggle_mines, *toggled_mines])
            if getattr(m, "state", 1) == 0  # STATE_TOGGLED = 0, deadly
        ]
        self._deadly_mine_xy = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        return self._deadly_mine_xy

    def _compute_nearest_deadly_mine_distance_norm(self) -> float:
        """Normalized [0, 1] distance to the nearest deadly mine (1.0 = none /
        beyond the local normalizer, 0.0 = at the ninja center)."""
        arr = self._get_deadly_mine_xy()
        if arr.shape[0] == 0:
            return 1.0
        ninja = self.nplay_headless.sim.ninja
        d2 = ((arr[:, 0] - ninja.xpos) ** 2 + (arr[:, 1] - ninja.ypos) ** 2).min()
        return float(min((d2 ** 0.5) / self._NEAREST_MINE_NORM_PX, 1.0))

    def _compute_nearest_approaching_deadly_mine_distance(self) -> float:
        """Distance in pixels to the nearest deadly mine the ninja is approaching.

        Uses dot(velocity, mine_direction) > 0 as the approach filter — identical
        to the velocity_dot_direction feature in spatial_context.py. Only mines in
        STATE_TOGGLED (state == 0) are considered deadly.

        Returns:
            Euclidean distance to the nearest approaching deadly mine, or
            float('inf') if no deadly mine is currently being approached.
        """
        ninja = self.nplay_headless.sim.ninja
        ninja_x, ninja_y = ninja.xpos, ninja.ypos
        vx, vy = self.nplay_headless.ninja_velocity()
        toggle_mines, toggled_mines = self.nplay_headless.get_mine_entities()

        min_dist = float("inf")
        for mine in self._filter_relevant_mines([*toggle_mines, *toggled_mines]):
            if getattr(mine, "state", 1) != 0:  # STATE_TOGGLED = 0, deadly
                continue
            dx = mine.xpos - ninja_x
            dy = mine.ypos - ninja_y
            dist_sq = dx * dx + dy * dy
            if dist_sq < 1e-12:
                continue
            # Positive dot product means velocity has a component toward the mine
            if (vx * dx + vy * dy) > 0:
                dist = dist_sq**0.5
                if dist < min_dist:
                    min_dist = dist
        return min_dist

    def _get_action_and_duration_mask(
        self,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Compose the legacy action mask with the Layer 3 mine veto."""
        action_mask, duration_mask = super()._get_action_and_duration_mask()
        if not self.enable_layer3_macro_action_mask or self.layer3_masker is None:
            return action_mask, duration_mask

        layer3_result = self.layer3_masker.compute(
            headless=self.nplay_headless,
            base_action_mask=action_mask,
            actions_to_execute=self._actions_to_execute,
        )

        self._layer3_decision_points += 1
        self._layer3_masked_actions_total += layer3_result.masked_actions
        self._layer3_masked_candidates_total += layer3_result.masked_candidates
        self._layer3_mask_override_count += layer3_result.mask_overrides
        if layer3_result.all_masked_event:
            self._layer3_all_masked_events += 1

        return layer3_result.action_mask, layer3_result.duration_mask

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

        # Also clear the reward calculator's (separate) path_calculator cache.
        if hasattr(self, "reward_calculator") and hasattr(
            self.reward_calculator, "path_calculator"
        ):
            self.reward_calculator.path_calculator.clear_cache()

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
        """Reduce the raw observation dict to the Layer 1 policy contract.

        Raw obs carries reward-side metadata (`_adjacency_graph`, `_graph_data`,
        `level_data`, `switch_states_dict`, cached path distances). That
        metadata is consumed by the reward calculator BEFORE this call and is
        stripped here so the rollout buffer only holds the four policy
        tensors plus control masks.
        """
        processed_obs = super()._process_observation(obs)

        if "hazard_map" not in processed_obs:
            processed_obs["hazard_map"] = obs.get(
                "hazard_map",
                np.zeros(
                    (HAZARD_MAP_CHANNELS, HAZARD_MAP_WIDTH, HAZARD_MAP_HEIGHT),
                    dtype=np.float32,
                ),
            )

        # Strip reward-only / legacy metadata so rollout buffer holds just
        # tile_grid, entity_overlays, hazard_map, game_state (+ masks).
        internal_only_keys = [
            "_adjacency_graph",
            "_graph_data",
            "_cached_switch_distance",
            "_cached_exit_distance",
            "waypoint_leg_distance",
            "waypoint_index",
            "necessary_doors_open",
            "switch_states_dict",
            "level_data",
            "entity_positions",
        ]
        for key in internal_only_keys:
            processed_obs.pop(key, None)

        # Final safety net: the policy/value network must NEVER see a non-finite
        # input. A graph-OOB query makes path_distance_calculator return inf,
        # which can reach a normalized obs feature (e.g. the reachability path-
        # distance dims) and produce NaN logits / a NaN critic output -> NaN SIL
        # priorities -> crash. Distances are sanitized at the reward boundary, but
        # the OBS path is independent, so clamp every float feature array here.
        for key, val in processed_obs.items():
            if isinstance(val, np.ndarray) and val.dtype.kind == "f":
                if not np.isfinite(val).all():
                    # Capture the level BEFORE sanitizing (bad procgen or a
                    # graph/grid logic error); best-effort, deduped, gated.
                    try:
                        from .invalid_value_capture import capture_invalid_value

                        capture_invalid_value(
                            f"nonfinite_obs_{key}",
                            getattr(self.nplay_headless, "current_map_data", None),
                            {
                                "obs_key": key,
                                "shape": list(val.shape),
                                "n_nonfinite": int((~np.isfinite(val)).sum()),
                                "step": int(getattr(self, "_episode_step", -1)),
                            },
                        )
                    except Exception:
                        pass
                    np.nan_to_num(
                        val, copy=False, nan=0.0, posinf=1.0, neginf=0.0
                    )

        return processed_obs

    def _get_cached_tile_grid(self) -> np.ndarray:
        """Return the tile grid for the current level, computing it once per level load.

        The tile grid is static for the lifetime of a level episode, so we cache it
        on first access and invalidate whenever the map name changes.

        Returns:
            np.ndarray: (42, 23) int8 tile type IDs, values in [0, 37]
        """
        current_map_name = getattr(self.map_loader, "current_map_name", None)
        # Invalidate cache when level changes (compare against stored name)
        if self._cached_tile_grid is None or current_map_name != getattr(
            self, "_tile_grid_map_name", None
        ):
            tile_data = self.nplay_headless.sim.get_tile_grid()
            self._cached_tile_grid = (
                np.array(tile_data, dtype=np.int8).reshape(23, MAP_TILE_WIDTH).T
            )  # -> (42, 23)
            self._tile_grid_map_name = current_map_name
            # Recompute stable integer level ID when tile grid changes
            self._cached_level_id_int = None

        return self._cached_tile_grid

    def set_feature_extractor_callback(self, callback):
        """Set callback for feature extractor cache precomputation.

        Called by EnvironmentFactory to wire hybrid_cached_graph extractor.
        The callback is triggered at reset() to precompute static graph embeddings.

        Args:
            callback: Function(level_id, graph_data) to call at level load
        """
        self._feature_extractor_cache_callback = callback

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

        if getattr(self, "_path_calculator", None) is not None and hasattr(
            self, "reward_calculator"
        ):
            self.reward_calculator.path_calculator = self._path_calculator

        # Mark that we need to reinitialize on next reset
        self._needs_reinit = True
