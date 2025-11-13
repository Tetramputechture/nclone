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

logger = logging.getLogger(__name__)

from ..graph.common import N_MAX_NODES, E_MAX_EDGES, NODE_FEATURE_DIM, EDGE_FEATURE_DIM
from ..constants.physics_constants import LEVEL_WIDTH_PX, LEVEL_HEIGHT_PX
from .constants import (
    GAME_STATE_CHANNELS,
    PATH_AWARE_OBJECTIVES_DIM,
    MINE_FEATURES_DIM,
    PROGRESS_FEATURES_DIM,
    LEVEL_DIAGONAL,
    ENTITY_POSITIONS_DIM,
    SWITCH_STATES_DIM,
    REACHABILITY_FEATURES_DIM,
    SUBTASK_FEATURES_DIM,
    MAX_LOCKED_DOORS,
    FEATURES_PER_DOOR,
    MAX_LOCKED_DOORS_ATTENTION,
    LOCKED_DOOR_FEATURES_DIM,
)
from ..constants.entity_types import EntityType
from .base_environment import BaseNppEnvironment
from .mixins import GraphMixin, ReachabilityMixin, DebugMixin, HierarchicalMixin
from .config import EnvironmentConfig
from .precomputed_door_features import PrecomputedDoorFeatureCache


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

        # Initialize mixin systems using config
        self._init_graph_system(
            debug=self.config.graph.debug,
        )
        self._init_reachability_system(self.config.reachability.debug)
        self._init_debug_system(self.config.render.enable_debug_overlay)
        self._init_hierarchical_system(self.config.hierarchical)

        # Update configuration flags with new options
        self.config_flags.update(
            {
                "enable_hierarchical": self.config.hierarchical.enable_hierarchical,
                "debug": self.config.graph.debug
                or self.config.reachability.debug
                or self.config.hierarchical.debug,
            }
        )

        # Extend observation space with graph, reachability, and hierarchical features
        self.observation_space = self._build_extended_observation_space(
            self.config.hierarchical.enable_hierarchical,
        )

        # Pre-allocate observation buffers

        self._game_state_buffer = np.zeros(
            GAME_STATE_CHANNELS, dtype=np.float32
        )  # Now 58
        self._path_aware_objectives_buffer = np.zeros(
            PATH_AWARE_OBJECTIVES_DIM, dtype=np.float32
        )
        self._mine_features_buffer = np.zeros(
            MINE_FEATURES_DIM, dtype=np.float32
        )  # Now 8
        self._progress_features_buffer = np.zeros(
            PROGRESS_FEATURES_DIM, dtype=np.float32
        )

        # Cache for locked door features (keyed by switch state hash + ninja grid cell)
        # PERFORMANCE: Path distances are expensive (~9ms per call). Cache by both:
        #   1. Switch states (rarely change)
        #   2. Ninja grid cell (24px quantization to balance cache hits vs accuracy)
        self._locked_door_cache: Dict[str, np.ndarray] = {}
        self._last_switch_state_hash: Optional[str] = None
        self._last_ninja_grid_cell: Optional[tuple] = None

        # Cache invalidation flag (set by entity interactions)
        # PERFORMANCE: Avoids computing hash on every frame, only when switches change
        self._switch_states_changed: bool = True  # Start True to compute initial state

        # Precomputed door feature cache (aggressive optimization)
        # PERFORMANCE: Precomputes ALL path distances at level load for O(1) lookup
        self.door_feature_cache = PrecomputedDoorFeatureCache()

        # Level-specific locked door caches (static for level duration)
        # PERFORMANCE: Eliminates redundant loops and observation calls every step
        self._has_locked_doors: bool = False  # Flag: does level have locked doors?
        self._cached_locked_doors: list = []  # Cached locked door entities (static)
        self._cached_switch_states: Optional[np.ndarray] = (
            None  # Cached switch states (updated on invalidation)
        )

        # DO NOT build graph in __init__!
        # Graph building requires a loaded map, which only happens during reset().
        # Building graph here creates invalid initial state with (0,0) positions.

        # Set environment reference in simulator for cache invalidation
        # PERFORMANCE: Enables entities to invalidate caches on state changes
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

    def _build_extended_observation_space(
        self,
        enable_hierarchical: bool,
    ) -> SpacesDict:
        """Build the extended observation space with graph and reachability features."""
        obs_spaces = dict(self.observation_space.spaces)

        # Add entity positions (always available)
        obs_spaces["entity_positions"] = box.Box(
            low=0.0, high=1.0, shape=(ENTITY_POSITIONS_DIM,), dtype=np.float32
        )

        # Add switch states (always available for hierarchical and ICM systems)
        obs_spaces["switch_states"] = box.Box(
            low=0.0, high=1.0, shape=(SWITCH_STATES_DIM,), dtype=np.float32
        )

        # Add reachability features (always available)
        obs_spaces["reachability_features"] = box.Box(
            low=0.0, high=1.0, shape=(REACHABILITY_FEATURES_DIM,), dtype=np.float32
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

        # Add hierarchical features
        if enable_hierarchical:
            obs_spaces["subtask_features"] = box.Box(
                low=0.0, high=1.0, shape=(SUBTASK_FEATURES_DIM,), dtype=np.float32
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
            start_time = time.time()
            self._update_graph_from_env_state()
            update_time = (time.time() - start_time) * 1000  # Convert to ms

            # Update performance stats
            self._update_graph_performance_stats(update_time)

            if self.config.graph.debug:
                print(f"Graph updated in {update_time:.2f}ms")

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
        # Call parent to add diagnostic and observation metrics
        super()._extend_info_hook(info)

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

        if self.reachability_times:
            avg_time = np.mean(self.reachability_times[-10:])  # Last 10 samples
            info["reachability_time_ms"] = avg_time * 1000

        # Add hierarchical info if enabled
        if self.enable_hierarchical:
            info["hierarchical"] = self._get_hierarchical_info()

    def _build_mine_death_lookup_table(self):
        """
        Build hybrid mine death predictor.

        Uses graph system's reachability data to filter mines and build
        a simple danger zone grid. Much faster than old lookup table approach.
        Called once per episode after graph building is complete.
        """
        from ..mine_death_predictor import MineDeathPredictor

        # Get reachable positions from graph system
        if (
            self.current_graph_data is None
            or "reachable" not in self.current_graph_data
        ):
            raise ValueError("Graph data not available for mine death predictor.")

        reachable_positions = self.current_graph_data.get("reachable", set())

        # Skip if no reachable positions (shouldn't happen but be safe)
        if not reachable_positions:
            raise ValueError("No reachable positions found for mine death predictor.")

        # Create and build predictor
        predictor = MineDeathPredictor(self.nplay_headless.sim)

        # Build danger zone grid with verbose output if debug enabled
        verbose = getattr(self, "debug", False)
        predictor.build_lookup_table(reachable_positions, verbose=verbose)

        # Attach predictor to ninja
        self.nplay_headless.sim.ninja.mine_death_predictor = predictor

        if verbose:
            stats = predictor.get_stats()
            logger.info(
                f"Hybrid mine predictor built: "
                f"{stats.build_time_ms:.1f}ms, "
                f"{stats.reachable_mines} mines, "
                f"{stats.danger_zone_cells} danger cells"
            )

    def _build_terminal_velocity_lookup_table(self):
        """
        Build terminal velocity death predictor with graph optimization.

        Uses graph system's reachability and adjacency data to:
        1. Build lookup table ONLY for reachable positions (50-75% smaller)
        2. Enable fast reachability checks during queries (5x fewer simulations)
        3. Maintain accurate physics simulation for predictions

        Hybrid approach: Graph optimizes search space, physics ensures accuracy.
        Called once per episode after graph building is complete.

        Implements per-level caching: Same tiles = same terminal velocity physics.
        Cache hit avoids expensive rebuild (88% time reduction).
        """
        from ..terminal_velocity_predictor import TerminalVelocityPredictor

        # Extract level_id for caching
        # Use level_id from level_data if available, otherwise generate from tiles
        level_id = None
        if hasattr(self, "level_data") and self.level_data is not None:
            level_id = getattr(self.level_data, "level_id", None)

        # If no level_id, generate from tiles (fallback)
        if level_id is None and hasattr(self, "nplay_headless"):
            try:
                tiles = self.nplay_headless.get_tile_data()
                if tiles is not None:
                    level_id = f"level_{hash(tiles.tobytes())}"
            except Exception:
                # If tile extraction fails, proceed without caching
                pass

        # Get reachable positions for lookup table
        reachable_positions = (
            self.current_graph_data.get("reachable", set())
            if self.current_graph_data
            else set()
        )

        # Skip if no reachable positions
        if not reachable_positions:
            logger.warning(
                "No reachable positions found for terminal velocity predictor. Skipping."
            )
            self.nplay_headless.sim.ninja.terminal_velocity_predictor = None
            return

        # Calculate reachability ratio for strategy selection
        from ..constants.physics_constants import LEVEL_WIDTH_PX, LEVEL_HEIGHT_PX

        total_cells = (LEVEL_WIDTH_PX // 24) * (LEVEL_HEIGHT_PX // 24)
        reachability_ratio = (
            len(reachable_positions) / total_cells if total_cells > 0 else 0.0
        )

        verbose = getattr(self, "debug", False)

        # Smart strategy selection based on level characteristics
        if reachability_ratio > 0.8:
            # Open level: use lazy building (instant initialization)
            # Terminal impacts are rare in open spaces, lazy caching is more efficient
            if verbose:
                logger.info(
                    f"Terminal velocity strategy: LAZY building (reachability={reachability_ratio:.2f}, open level)"
                )

            if self.current_graph_data is None:
                predictor = TerminalVelocityPredictor(
                    self.nplay_headless.sim, graph_data=None, lazy_build=True
                )
            else:
                predictor = TerminalVelocityPredictor(
                    self.nplay_headless.sim,
                    graph_data=self.current_graph_data,
                    lazy_build=True,
                )

            # No build_lookup_table() call - start with empty table

        elif reachability_ratio < 0.5:
            # Dense level: use eager building (precompute hot paths)
            # Terminal impacts more likely in tight spaces, worth upfront cost
            if verbose:
                logger.info(
                    f"Terminal velocity strategy: EAGER building (reachability={reachability_ratio:.2f}, dense level)"
                )

            if self.current_graph_data is None:
                predictor = TerminalVelocityPredictor(
                    self.nplay_headless.sim, graph_data=None, lazy_build=False
                )
            else:
                predictor = TerminalVelocityPredictor(
                    self.nplay_headless.sim,
                    graph_data=self.current_graph_data,
                    lazy_build=False,
                )

            # Build full lookup table for all reachable positions
            predictor.build_lookup_table(
                reachable_positions, level_id=level_id, verbose=verbose
            )

        else:
            # Medium level: hybrid strategy (surface-adjacent only)
            # Balance between build time and runtime performance
            if verbose:
                logger.info(
                    f"Terminal velocity strategy: HYBRID building (reachability={reachability_ratio:.2f}, medium level)"
                )

            if self.current_graph_data is None:
                predictor = TerminalVelocityPredictor(
                    self.nplay_headless.sim, graph_data=None, lazy_build=True
                )
            else:
                predictor = TerminalVelocityPredictor(
                    self.nplay_headless.sim,
                    graph_data=self.current_graph_data,
                    lazy_build=True,
                )

            # Build lookup table for surface-adjacent positions only
            # Terminal impacts primarily occur near surfaces
            surface_positions = predictor._filter_surface_adjacent_positions(
                reachable_positions, max_distance=48
            )

            if len(surface_positions) > 0:
                # Temporarily disable lazy_build for this partial build
                predictor.lazy_build = False
                predictor.build_lookup_table(
                    surface_positions, level_id=level_id, verbose=verbose
                )
                # Re-enable lazy_build for runtime caching
                predictor.lazy_build = True

                if verbose:
                    logger.info(
                        f"Hybrid build: {len(surface_positions)}/{len(reachable_positions)} "
                        f"surface-adjacent positions precomputed ({100 * len(surface_positions) / len(reachable_positions):.1f}%)"
                    )
            else:
                # No surface positions found, fall back to pure lazy
                if verbose:
                    logger.info(
                        "Hybrid build: No surface positions found, using pure lazy mode"
                    )

        # Attach predictor to ninja
        self.nplay_headless.sim.ninja.terminal_velocity_predictor = predictor

        if verbose:
            stats = predictor.get_stats()
            logger.info(
                f"Terminal velocity predictor built: "
                f"{stats.build_time_ms:.1f}ms, "
                f"{stats.lookup_table_size} entries (reachability-optimized)"
            )

    def _build_door_feature_cache(self):
        """
        Build precomputed door feature cache.

        Precomputes path distances from all reachable grid cells to all doors/switches.
        This is a one-time cost at level load that eliminates expensive runtime path
        distance calculations.

        PERFORMANCE: Build time ~50-200ms, saves ~25s per 858 steps (26x speedup)
        """
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
        if self.enable_hierarchical:
            self._reset_hierarchical_state()

        # Build graph from the newly loaded map
        self._update_graph_from_env_state()

        # Build mine death predictor lookup table after graph is ready
        self._build_mine_death_lookup_table()

        # Build terminal velocity predictor lookup table after graph is ready
        self._build_terminal_velocity_lookup_table()

        # Build precomputed door feature cache after graph is ready
        # PERFORMANCE: Precomputes ALL path distances for O(1) runtime lookup
        self._build_door_feature_cache()

        # Initialize locked door caches for this level
        # PERFORMANCE: Cache static door entities and initialize switch states
        self._initialize_locked_door_caches()

        # NOW get initial observation (with valid graph data)
        initial_obs = self._get_observation()
        processed_obs = self._process_observation(initial_obs)

        return (processed_obs, {})

    def _get_observation(self) -> Dict[str, Any]:
        """Get the current observation from the game state."""
        # Get base observation
        obs = super()._get_observation()

        from .constants import (
            NINJA_STATE_DIM,
            PATH_AWARE_OBJECTIVES_DIM,
            MINE_FEATURES_DIM,
            PROGRESS_FEATURES_DIM,
            SEQUENTIAL_GOAL_DIM,
            ACTION_DEATH_PROBABILITIES_DIM,
            TERMINAL_VELOCITY_DEATH_PROBABILITIES_DIM,
        )

        base_game_state = obs["game_state"]
        self._game_state_buffer[:NINJA_STATE_DIM] = base_game_state[:NINJA_STATE_DIM]

        if hasattr(self, "_path_calculator") and self._path_calculator is not None:
            self._compute_path_aware_objectives(obs, self._path_aware_objectives_buffer)
            self._extract_mine_features(obs, self._mine_features_buffer)
            self._compute_progress_features(obs, self._progress_features_buffer)

            idx = NINJA_STATE_DIM
            self._game_state_buffer[idx : idx + PATH_AWARE_OBJECTIVES_DIM] = (
                self._path_aware_objectives_buffer
            )
            idx += PATH_AWARE_OBJECTIVES_DIM
            self._game_state_buffer[idx : idx + MINE_FEATURES_DIM] = (
                self._mine_features_buffer
            )
            idx += MINE_FEATURES_DIM
            self._game_state_buffer[idx : idx + PROGRESS_FEATURES_DIM] = (
                self._progress_features_buffer
            )
            idx += PROGRESS_FEATURES_DIM

            # NEW: Sequential goal features
            if hasattr(self.nplay_headless, "get_sequential_goal_features"):
                seq_features = self.nplay_headless.get_sequential_goal_features()
                self._game_state_buffer[idx : idx + SEQUENTIAL_GOAL_DIM] = seq_features
            else:
                self._game_state_buffer[idx : idx + SEQUENTIAL_GOAL_DIM] = 0.0
            idx += SEQUENTIAL_GOAL_DIM

            # NEW: Mine death probabilities (6 features)
            if hasattr(self.nplay_headless, "sim") and hasattr(
                self.nplay_headless.sim, "ninja"
            ):
                ninja = self.nplay_headless.sim.ninja
                # Calculate mine death probabilities for all 6 actions
                death_prob_result = (
                    ninja.mine_death_predictor.calculate_death_probability(
                        frames_to_simulate=10
                    )
                )
                self._game_state_buffer[idx : idx + ACTION_DEATH_PROBABILITIES_DIM] = (
                    death_prob_result.action_death_probs
                )
            else:
                self._game_state_buffer[idx : idx + ACTION_DEATH_PROBABILITIES_DIM] = (
                    0.0
                )
            idx += ACTION_DEATH_PROBABILITIES_DIM

            # NEW: Terminal velocity death probabilities (6 features)
            if (
                hasattr(self.nplay_headless, "sim")
                and hasattr(self.nplay_headless.sim, "ninja")
                and hasattr(
                    self.nplay_headless.sim.ninja, "terminal_velocity_predictor"
                )
                and self.nplay_headless.sim.ninja.terminal_velocity_predictor
                is not None
            ):
                ninja = self.nplay_headless.sim.ninja
                predictor = ninja.terminal_velocity_predictor

                # Conditional computation: only compute when ninja is in risky state
                # This matches the Tier 1 filter logic in is_action_deadly()
                from ..constants.physics_constants import TERMINAL_IMPACT_SAFE_VELOCITY
                from ..terminal_velocity_predictor import DeathProbabilityResult

                # Check if chaining wall jumps (TWO wall jumps within 6 frames of each other)
                # Single wall jump doesn't build enough velocity to be lethal
                frames_between_jumps = (
                    ninja.last_wall_jump_frame - ninja.second_last_wall_jump_frame
                )
                is_chaining_wall_jumps = (
                    frames_between_jumps <= 6 and frames_between_jumps > 0
                )

                is_risky_state = (
                    ninja.airborn
                    and (
                        ninja.yspeed
                        > TERMINAL_IMPACT_SAFE_VELOCITY  # Dangerous downward velocity
                        or (
                            ninja.yspeed
                            < -TERMINAL_IMPACT_SAFE_VELOCITY  # Dangerous upward velocity
                            and is_chaining_wall_jumps  # Only dangerous if chaining wall jumps
                        )
                    )
                )

                if is_risky_state:
                    # Calculate terminal velocity death probabilities for all 6 actions
                    tv_death_prob_result = predictor.calculate_death_probability(
                        frames_to_simulate=10
                    )
                else:
                    # Safe state: return cached zero result (no computation needed)
                    tv_death_prob_result = DeathProbabilityResult(
                        action_death_probs=[0.0] * 6,
                        masked_actions=[],
                        frames_simulated=0,
                        nearest_surface_distance=float("inf"),
                    )

                self._game_state_buffer[
                    idx : idx + TERMINAL_VELOCITY_DEATH_PROBABILITIES_DIM
                ] = tv_death_prob_result.action_death_probs
            else:
                self._game_state_buffer[
                    idx : idx + TERMINAL_VELOCITY_DEATH_PROBABILITIES_DIM
                ] = 0.0
        else:
            self._game_state_buffer[NINJA_STATE_DIM:] = 0.0

        # Validate game_state for inf/NaN with detailed diagnostics
        inf_mask = np.isinf(self._game_state_buffer)
        nan_mask = np.isnan(self._game_state_buffer)

        if np.any(inf_mask) or np.any(nan_mask):
            # Feature name mapping for readable diagnostics
            def get_feature_name(idx):
                if idx < NINJA_STATE_DIM:
                    return f"ninja_physics[{idx}]"
                elif idx < NINJA_STATE_DIM + PATH_AWARE_OBJECTIVES_DIM:
                    local_idx = idx - NINJA_STATE_DIM
                    return f"path_objectives[{local_idx}]"
                elif (
                    idx
                    < NINJA_STATE_DIM + PATH_AWARE_OBJECTIVES_DIM + MINE_FEATURES_DIM
                ):
                    local_idx = idx - NINJA_STATE_DIM - PATH_AWARE_OBJECTIVES_DIM
                    return f"mine_features[{local_idx}]"
                else:
                    local_idx = (
                        idx
                        - NINJA_STATE_DIM
                        - PATH_AWARE_OBJECTIVES_DIM
                        - MINE_FEATURES_DIM
                    )
                    return f"progress_features[{local_idx}]"

            # Log inf values with detail
            inf_indices = np.where(inf_mask)[0]
            for idx in inf_indices:
                print(
                    f"[OBSERVATION] Inf at game_state[{idx}] ({get_feature_name(idx)}), "
                    f"value={self._game_state_buffer[idx]}, "
                    f"ninja_pos={self.nplay_headless.ninja_position()}"
                )

            # Log NaN values with detail
            nan_indices = np.where(nan_mask)[0]
            for idx in nan_indices:
                print(
                    f"[OBSERVATION] NaN at game_state[{idx}] ({get_feature_name(idx)}), "
                    f"ninja_pos={self.nplay_headless.ninja_position()}"
                )

            # Raise error instead of silently fixing NaN/inf values
            # This helps identify the root cause of invalid values
            error_parts = []
            if np.any(inf_mask):
                inf_indices = np.where(inf_mask)[0]
                error_parts.append(
                    f"Inf at indices: {inf_indices[:10]} (total {len(inf_indices)})"
                )
            if np.any(nan_mask):
                nan_indices = np.where(nan_mask)[0]
                error_parts.append(
                    f"NaN at indices: {nan_indices[:10]} (total {len(nan_indices)})"
                )

            raise ValueError(
                f"Invalid values detected in game_state buffer: {', '.join(error_parts)}. "
                f"This indicates a problem upstream in observation processing. "
                f"Check mine positions, ninja state, or feature calculations."
            )

        obs["game_state"] = np.array(self._game_state_buffer, copy=True)

        obs["reachability_features"] = self._get_reachability_features()

        obs.update(self._get_graph_observations())

        # Add hierarchical features if enabled
        if self.enable_hierarchical:
            obs["subtask_features"] = self._get_subtask_features()

        # Add switch states for hierarchical PPO and ICM
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

        # Add level data for reachability analysis and hierarchical planning
        # This is needed by ICM and reachability-aware exploration
        obs["level_data"] = self._extract_level_data()

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

    def _compute_path_aware_objectives(
        self, obs: Dict[str, Any], buffer: np.ndarray
    ) -> np.ndarray:
        """
        Compute path-aware objective features using graph-based pathfinding.

        Returns PATH_AWARE_OBJECTIVES_DIM (15) features:
        - Exit switch (4): collected, rel_x, rel_y, path_distance
        - Exit door (3): rel_x, rel_y, path_distance
        - Nearest locked door (8): present, switch_collected, switch_rel_x, switch_rel_y,
          switch_path_distance, door_rel_x, door_rel_y, door_path_distance
        """

        features = buffer
        features.fill(0.0)

        # Extract positions directly from raw observation (not dependent on processing order)
        ninja_x = obs.get("player_x", 0.0)
        ninja_y = obs.get("player_y", 0.0)
        switch_x = obs.get("switch_x", 0.0)
        switch_y = obs.get("switch_y", 0.0)
        exit_door_x = obs.get("exit_door_x", 0.0)
        exit_door_y = obs.get("exit_door_y", 0.0)

        adjacency = self.current_graph_data.get("adjacency")
        if adjacency is None or len(adjacency) == 0:
            raise RuntimeError("Adjacency graph not available. Graph building failed.")

        ninja_pos = (
            int(ninja_x),
            int(ninja_y),
        )

        # Exit switch [0-3]
        from ..constants.physics_constants import EXIT_SWITCH_RADIUS

        exit_switch_collected = 1.0 if obs.get("switch_activated", False) else 0.0
        exit_switch_pos = (
            int(switch_x),
            int(switch_y),
        )
        rel_switch_x = (exit_switch_pos[0] - ninja_pos[0]) / (LEVEL_WIDTH_PX / 2)
        rel_switch_y = (exit_switch_pos[1] - ninja_pos[1]) / (LEVEL_HEIGHT_PX / 2)
        switch_path_dist = self._safe_path_distance(
            ninja_pos,
            exit_switch_pos,
            adjacency,
            "exit_switch",
            level_data=self.level_data,
            graph_data=self.current_graph_data,
            entity_radius=EXIT_SWITCH_RADIUS,
        )
        features[0] = exit_switch_collected
        features[1] = np.clip(rel_switch_x, -1.0, 1.0)
        features[2] = np.clip(rel_switch_y, -1.0, 1.0)
        area_scale = self._get_reachable_area_scale()
        features[3] = np.clip(switch_path_dist / area_scale, 0.0, 1.0)

        # Exit door [4-6]
        from ..constants.physics_constants import EXIT_DOOR_RADIUS

        exit_door_pos = (
            int(exit_door_x),
            int(exit_door_y),
        )
        rel_door_x = (exit_door_pos[0] - ninja_pos[0]) / (LEVEL_WIDTH_PX / 2)
        rel_door_y = (exit_door_pos[1] - ninja_pos[1]) / (LEVEL_HEIGHT_PX / 2)
        door_path_dist = self._safe_path_distance(
            ninja_pos,
            exit_door_pos,
            adjacency,
            "exit_door",
            level_data=self.level_data,
            graph_data=self.current_graph_data,
            entity_radius=EXIT_DOOR_RADIUS,
        )
        features[4] = np.clip(rel_door_x, -1.0, 1.0)
        features[5] = np.clip(rel_door_y, -1.0, 1.0)
        features[6] = np.clip(door_path_dist / area_scale, 0.0, 1.0)

        # Nearest locked door [7-14]
        locked_doors = obs.get("locked_doors", [])
        if locked_doors:
            # Find nearest active (uncollected) locked door by Euclidean distance
            nearest_door = None
            nearest_door_dist = float("inf")

            for door in locked_doors:
                if getattr(door, "active", True):  # Only consider uncollected doors
                    # Extract door position using shared helper
                    _, _, door_x, door_y, _ = self._extract_locked_door_positions(door)

                    euclidean_dist = np.sqrt(
                        (door_x - ninja_pos[0]) ** 2 + (door_y - ninja_pos[1]) ** 2
                    )
                    if euclidean_dist < nearest_door_dist:
                        nearest_door_dist = euclidean_dist
                        nearest_door = door

            if nearest_door is not None:
                # Extract all positions using shared helper
                switch_x, switch_y, door_x, door_y, switch_collected = (
                    self._extract_locked_door_positions(nearest_door)
                )

                features[7] = 1.0  # Locked door present
                features[8] = switch_collected

                # Switch features
                from ..constants.physics_constants import LOCKED_DOOR_SWITCH_RADIUS

                switch_pos = (int(switch_x), int(switch_y))
                rel_switch_x = (switch_pos[0] - ninja_pos[0]) / (LEVEL_WIDTH_PX / 2)
                rel_switch_y = (switch_pos[1] - ninja_pos[1]) / (LEVEL_HEIGHT_PX / 2)
                switch_path_dist = self._safe_path_distance(
                    ninja_pos,
                    switch_pos,
                    adjacency,
                    "locked_door_switch",
                    level_data=self.level_data,
                    graph_data=self.current_graph_data,
                    entity_radius=LOCKED_DOOR_SWITCH_RADIUS,
                )
                features[9] = np.clip(rel_switch_x, -1.0, 1.0)
                features[10] = np.clip(rel_switch_y, -1.0, 1.0)
                features[11] = np.clip(switch_path_dist / area_scale, 0.0, 1.0)

                # Door features
                door_pos = (int(door_x), int(door_y))
                rel_door_x = (door_pos[0] - ninja_pos[0]) / (LEVEL_WIDTH_PX / 2)
                rel_door_y = (door_pos[1] - ninja_pos[1]) / (LEVEL_HEIGHT_PX / 2)
                door_path_dist = self._safe_path_distance(
                    ninja_pos,
                    door_pos,
                    adjacency,
                    "locked_door",
                    level_data=self.level_data,
                    graph_data=self.current_graph_data,
                    entity_radius=0.0,  # Door is a line segment, use center point
                )
                features[12] = np.clip(rel_door_x, -1.0, 1.0)
                features[13] = np.clip(rel_door_y, -1.0, 1.0)
                features[14] = np.clip(door_path_dist / area_scale, 0.0, 1.0)

        # Check for NaN before returning
        if np.isnan(features).any():
            logger.error(
                f"[PATH_OBJECTIVES_NAN] NaN detected in path_aware_objectives features. "
                f"NaN count: {np.isnan(features).sum()}, "
                f"features: {features}, ninja_pos=({ninja_x}, {ninja_y})"
            )

        return features

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

    def invalidate_switch_cache(self):
        """
        Invalidate switch-dependent caches.

        Called by entity collision methods when switch states change.
        This is more efficient than computing hash keys every frame.

        Clears:
        - Pathfinding visualization cache (debug overlay renderer)
        - Path distance calculator cache (forces rebuild with new goals)
        - Cached level_data (entities may be stale after switch activation)
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

        # Validate ninja position before computing features
        if np.isnan(ninja_x) or np.isinf(ninja_x):
            raise ValueError(f"Invalid ninja_x in mine features: {ninja_x}")
        if np.isnan(ninja_y) or np.isinf(ninja_y):
            raise ValueError(f"Invalid ninja_y in mine features: {ninja_y}")

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

                # Validate mine distance
                if np.isnan(dist) or (np.isinf(dist) and dist != float("inf")):
                    raise ValueError(
                        f"Invalid mine distance: {dist}, "
                        f"mine_pos=({mx}, {my}), ninja_pos=({ninja_x}, {ninja_y})"
                    )

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

        # Check for NaN before returning
        if np.isnan(features).any():
            print(
                f"[MINE_FEATURES_NAN] NaN detected in mine_features. "
                f"NaN count: {np.isnan(features).sum()}, "
                f"features: {features}, ninja_pos=({ninja_x}, {ninja_y})"
            )

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

        # Check for NaN before returning
        if np.isnan(features).any():
            print(
                f"[PROGRESS_FEATURES_NAN] NaN detected in progress_features. "
                f"NaN count: {np.isnan(features).sum()}, "
                f"features: {features}, ninja_pos=({ninja_x}, {ninja_y}), "
                f"total_path_dist={total_path_dist}, completed={completed}, total={total}"
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

        # Add subtask features if enabled and not already added
        if self.enable_hierarchical and "subtask_features" not in processed_obs:
            processed_obs["subtask_features"] = obs.get(
                "subtask_features", np.zeros(SUBTASK_FEATURES_DIM, dtype=np.float32)
            )

        # Add switch states dict if not already added (dict version for ICM - not in obs space)
        if "switch_states_dict" not in processed_obs:
            processed_obs["switch_states_dict"] = obs.get("switch_states_dict", {})

        # Add level data if not already added (for reachability/hierarchical - not in obs space)
        if "level_data" not in processed_obs:
            processed_obs["level_data"] = obs.get("level_data", None)

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
