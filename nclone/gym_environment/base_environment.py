"""
Base N++ environment class with core functionality.

This module contains the core environment functionality without the specialized
mixins for graph, reachability, and debug features.
"""

import logging
import gymnasium
from gymnasium.spaces import box, discrete, Dict as SpacesDict
import random
import numpy as np
from typing import Tuple, Optional, Dict, Any, List

# Core nclone imports
from ..constants import MAP_TILE_WIDTH, MAP_TILE_HEIGHT
from ..nplay_headless import NPlayHeadless

# Graph and level data imports
from ..graph.level_data import LevelData, extract_start_position_from_map_data

from .constants import (
    GAME_STATE_CHANNELS,
    NUM_TILE_TYPES,
    LEVEL_WIDTH,
    LEVEL_HEIGHT,
)
from .reward_calculation.main_reward_calculator import RewardCalculator
from .reward_calculation.reward_config import RewardConfig
from .entity_overlay_utils import compute_entity_overlays, ENTITY_OVERLAY_CHANNELS
from .observation_processor import ObservationProcessor
from .profiling_mixin import ProfilingMixin
from .truncation_checker import TruncationChecker
from .entity_extractor import EntityExtractor
from .env_map_loader import EnvMapLoader
from .reward_calculation.reward_constants import PBRS_GAMMA
from ..constants.entity_types import EntityType
from ..entity_classes.entity_door_locked import EntityDoorLocked

logger = logging.getLogger(__name__)


class BaseNppEnvironment(gymnasium.Env, ProfilingMixin):
    """
    Base N++ environment class with core functionality.

    This class provides the core Gym interface for the N++ game, allowing reinforcement
    learning agents to learn to play levels. Specialized functionality is provided
    through mixins.

    Core Features:
    - Gym environment interface
    - Action execution and observation processing
    - Reward calculation and episode termination
    - Map loading and basic game state management
    """

    metadata = {"render.modes": ["human", "grayscale_array"]}
    RANDOM_MAP_CHANCE = 0.5

    def __init__(
        self,
        render_mode: str = "grayscale_array",
        enable_animation: bool = False,
        enable_logging: bool = False,
        enable_debug_overlay: bool = False,
        seed: Optional[int] = None,
        eval_mode: bool = False,
        custom_map_path: Optional[str] = None,
        test_dataset_path: Optional[str] = None,
        pbrs_gamma: float = PBRS_GAMMA,
        enable_profiling: bool = False,
        reward_config: Optional[
            RewardConfig
        ] = None,  # RewardConfig for curriculum-aware reward system
        frame_skip: int = 4,  # Frame skip (action repeat) for temporal abstraction
        shared_level_cache: Optional[
            Any
        ] = None,  # Shared level cache for multi-worker training
        shared_level_caches_by_stage: Optional[
            Dict[int, Any]
        ] = None,  # Multi-stage shared caches for goal curriculum
        goal_curriculum_config: Optional[
            Any
        ] = None,  # GoalCurriculumConfig for intermediate goal curriculum
        enable_layer3_macro_action_mask: bool = False,
        layer3_duration_bins: Optional[tuple[int, ...]] = None,
        layer3_all_masked_fallback: str = "noop",
        value_function_dir: Optional[
            str
        ] = None,  # Directory with precomputed value functions
        use_precomputed_pbrs: bool = True,  # Use precomputed value function for PBRS
        simulation_phi_dir: Optional[
            str
        ] = None,  # Directory with precomputed simulation-based phi grids
        use_reachability_pbrs: bool = False,  # Use analytical reachability for PBRS
    ):
        """
        Initialize the base N++ environment.

        Args:
            render_mode: Rendering mode ("human" or "grayscale_array")
            enable_animation: Enable animation in rendering
            enable_logging: Enable debug logging
            enable_debug_overlay: Enable debug overlay visualization
            seed: Random seed for reproducibility
            eval_mode: Use evaluation maps instead of training maps
            custom_map_path: Path to custom map file
            test_dataset_path: Path to test dataset directory for evaluation
            pbrs_gamma: PBRS discount factor
            enable_profiling: Enable detailed performance profiling (~5% overhead)
            value_function_dir: Directory containing precomputed value functions (.npz files)
            use_precomputed_pbrs: If True, use precomputed value function instead of path-based PBRS
        """
        _logger = logging.getLogger(__name__)

        super().__init__()

        # Initialize profiling before other components
        self._init_profiling(enable_profiling)

        self.render_mode = render_mode
        self.enable_animation = enable_animation
        self.enable_logging = enable_logging
        self.debug_mode = enable_logging
        self.custom_map_path = custom_map_path
        self.test_dataset_path = test_dataset_path
        self.eval_mode = eval_mode

        # Cache for previous observation to avoid redundant computation
        self._prev_obs_cache = None

        # Initialize core game interface. Rendering (pygame + `render()`) is only
        # required for human playback / video capture / debug overlays, never
        # for the policy observation — which is structured tensors (tile_grid,
        # entity_overlays, hazard_map, game_state) produced elsewhere.
        self.nplay_headless = NPlayHeadless(
            render_mode=render_mode,
            enable_animation=enable_animation,
            enable_logging=enable_logging,
            enable_debug_overlay=enable_debug_overlay,
            seed=seed,
            enable_rendering=(render_mode == "human"),
        )

        # Set environment reference in simulator for cache invalidation
        # This allows entity collision methods to invalidate caches when switch states change
        self.nplay_headless.sim.gym_env = self

        # Initialize action space (6 actions: NOOP, Left, Right, Jump, Jump+Left, Jump+Right)
        self.action_space = discrete.Discrete(6)

        # Initialize RNG
        self.rng = random.Random(seed)

        # Track reward for the current episode
        self.current_ep_reward = 0
        self._last_episode_reward = 0  # For reset validation

        # When True, skip the inner HWM/PBRS reward calculation entirely.
        # Retained for offline dataset generation where only the raw obs is
        # needed and reward is recomputed downstream. Avoids expensive
        # A*/path-distance calls when they would be discarded anyway.
        self.skip_reward_calculation = False

        # Track last action for death context learning
        self.last_action = None

        # Track episode count for periodic cache clearing in worker processes
        # Used by SharedMemorySubprocVecEnv to trigger cache clearing every N episodes
        self._episode_count = 0

        # Position tracking for route visualization (integrated directly)
        self.current_route = []
        self._warned_about_position = False

        # Track deterministic masking application rate (for TensorBoard monitoring)
        self._deterministic_mask_applied_count = 0
        self._total_steps_count = 0

        # Track switch activation frame and action-step index for route visualization
        self._switch_activation_frame = None
        self._switch_activation_step: Optional[int] = None

        # Track curriculum stage that was active at EPISODE START (not current stage)
        # This is captured at reset() and used in _build_episode_info() for correct visualization
        self._episode_start_curriculum_stage: Optional[int] = None

        # Per-episode PBRS diagnostics tracking (also reset in reset())
        self._phi_at_spawn: Optional[float] = None
        self._phi_at_switch_collection: Optional[float] = None
        self._phi_at_terminal: Optional[float] = None
        self._contact_floor_frames: int = 0
        self._contact_wall_frames: int = 0
        self._contact_airborne_frames: int = 0
        self._first_frame_of_episode: bool = True

        # Frame skip for temporal abstraction (always enabled)
        self.frame_skip = max(1, frame_skip)  # Ensure at least 1
        self.enable_layer3_macro_action_mask = enable_layer3_macro_action_mask
        self.layer3_duration_bins = tuple(layer3_duration_bins or ())
        self.layer3_all_masked_fallback = layer3_all_masked_fallback
        self._layer3_num_durations = len(self.layer3_duration_bins)

        # Precomputed value function support
        self.value_function_dir = value_function_dir
        self.use_precomputed_pbrs = use_precomputed_pbrs

        # Simulation-based phi grid support
        self.simulation_phi_dir = simulation_phi_dir
        self._current_precomputed_pbrs = None  # Loaded per level

        # Analytical reachability PBRS support
        self.use_reachability_pbrs = use_reachability_pbrs
        self._current_reachability_pbrs = None  # Built per level

        # Layer 3 episode-level diagnostics (updated by mask builders at decision points)
        self._layer3_decision_points = 0
        self._layer3_masked_actions_total = 0
        self._layer3_masked_candidates_total = 0
        self._layer3_mask_override_count = 0
        self._layer3_all_masked_events = 0

        if use_reachability_pbrs:
            _logger.info("Analytical reachability PBRS enabled")

        if use_precomputed_pbrs and value_function_dir:
            _logger.info(
                f"Precomputed value function PBRS enabled (dir: {value_function_dir})"
            )
        elif use_precomputed_pbrs and not value_function_dir:
            _logger.warning(
                "use_precomputed_pbrs=True but value_function_dir not specified, falling back to path-based PBRS"
            )
            self.use_precomputed_pbrs = False

        # Action sequence tracking for Go-Explore checkpoint replay
        # Stores all actions since last reset for deterministic replay
        self._action_sequence: List[int] = []
        self._checkpoint_replay_in_progress: bool = False

        # Checkpoint route for visualization (stores path from checkpoint replay)
        # Persists through episode so it can be included in episode end info
        self._checkpoint_route: List[Tuple[float, float]] = []

        # Checkpoint episode tracking for logging/analysis
        # Allows distinguishing checkpoint vs spawn episodes in TensorBoard
        self._from_checkpoint: bool = False
        self._checkpoint_base_reward: float = (
            0.0  # Cumulative reward to reach checkpoint
        )
        self._checkpoint_source_frame_skip: int = (
            frame_skip  # Frame skip used to create checkpoint (1=demo, 4=agent)
        )
        self._checkpoint_replay_frame_count: int = 0  # Frames executed during checkpoint replay (to subtract from episode length)

        # Initialize entity extractor
        self.entity_extractor = EntityExtractor(self.nplay_headless)

        # Initialize map loader
        self.map_loader = EnvMapLoader(
            self.nplay_headless,
            self.rng,
            eval_mode,
            custom_map_path,
            test_dataset_path=test_dataset_path,
        )

        # Store shared level cache references for mixins (ReachabilityMixin needs it)
        self.shared_level_cache = shared_level_cache
        self.shared_level_caches_by_stage = shared_level_caches_by_stage

        # Initialize goal curriculum manager BEFORE reward calculator
        # This manages entity repositioning for intermediate goal curriculum
        self.goal_curriculum_manager = None
        if goal_curriculum_config is not None and goal_curriculum_config.enabled:
            from .reward_calculation.intermediate_goal_manager import (
                IntermediateGoalManager,
            )

            self.goal_curriculum_manager = IntermediateGoalManager(
                goal_curriculum_config
            )
            logger.info(
                "Goal curriculum enabled: entities will be repositioned along optimal path"
            )

            # INFO: Multi-stage SharedLevelCache support for goal curriculum
            # If shared_level_caches_by_stage is provided, the system will automatically
            # select the appropriate cache for each curriculum stage, providing correct
            # distances at all difficulty levels with shared memory efficiency.
            if shared_level_caches_by_stage is not None:
                num_stages = len(shared_level_caches_by_stage)
                total_memory = sum(
                    c.memory_usage_kb for c in shared_level_caches_by_stage.values()
                )
                logger.info(
                    f"Multi-stage SharedLevelCache active with goal curriculum: "
                    f"{num_stages} stages, {total_memory:.0f}KB total. "
                    f"Cache will automatically switch as curriculum advances."
                )
            elif shared_level_cache is not None:
                # Single cache with curriculum - should not happen with proper factory setup
                logger.warning(
                    "Single SharedLevelCache with goal curriculum - positions may mismatch! "
                    "Expected multi-stage caches but got single cache. "
                    "This may cause validation failures and BFS fallbacks."
                )

        # Layer 1 observation processor — structured tensors only, no visual processing.
        self.observation_processor = ObservationProcessor()

        self.reward_calculator = RewardCalculator(
            reward_config=reward_config,
            pbrs_gamma=pbrs_gamma,
            shared_level_cache=shared_level_cache,
            shared_level_caches_by_stage=shared_level_caches_by_stage,
            goal_curriculum_manager=self.goal_curriculum_manager,
            simulation_phi_dir=simulation_phi_dir,
        )

        # Initialize primitive-level reward calculator for FiGAR-PPO per-frame PBRS
        from .reward_calculation.primitive_reward_calculator import (
            PrimitiveRewardCalculator,
        )
        from .reward_calculation.reward_constants import (
            R_SWITCH,
            R_EXIT,
            R_DEATH,
            R_DEATH_MINE,
            TIME_PENALTY_PER_STEP,
        )

        self.primitive_reward_calculator = PrimitiveRewardCalculator(
            gamma=pbrs_gamma,
            alpha=5.0,  # Stronger PBRS vs TIME_PENALTY_PER_STEP for FiGAR primitive rewards
            r_switch=R_SWITCH,  # 1.0
            r_exit=R_EXIT,      # 5.0
            r_death=R_DEATH,    # -0.5
            r_death_mine=R_DEATH_MINE,  # -0.5 extra when death_cause=='mine'
            time_penalty_per_frame=TIME_PENALTY_PER_STEP,
        )

        # Initialize truncation checker
        self.truncation_checker = TruncationChecker(self)

        # Store all configuration flags for logging and debugging
        self.config_flags = {
            "render_mode": render_mode,
            "enable_animation": enable_animation,
            "enable_logging": enable_logging,
            "enable_debug_overlay": enable_debug_overlay,
            "eval_mode": eval_mode,
            "pbrs_gamma": pbrs_gamma,
        }

        # Track last reset map name for fast reset detection
        self._last_reset_map_name = None

        # Build base observation space (will be extended by mixins)
        self.observation_space = self._build_base_observation_space()

        self.mirror_map = False

        # Cache for level data and entities (invalidated only on reset)
        # PERFORMANCE: Eliminates 8217+ redundant calls to _extract_level_data()
        self._cached_level_data: Optional[LevelData] = None
        self._cached_entities: Optional[list] = None

        # Cache for observation (invalidated after each action execution)
        # PERFORMANCE: Eliminates redundant _get_observation() calls within single step
        self._cached_observation: Optional[Dict[str, Any]] = None

        # Cache for STATIC positions/entities (invalidated only on reset/new level)
        # MEMORY OPTIMIZATION: These values never change during an episode
        self._cached_switch_pos: Optional[Tuple[float, float]] = None
        self._cached_exit_pos: Optional[Tuple[float, float]] = None
        self._cached_locked_doors: Optional[list] = None
        self._cached_locked_door_switches: Optional[list] = None
        self._cached_toggle_mines: Optional[list] = None
        self._cached_tile_grid: Optional[np.ndarray] = None

        # Load the initial map
        self.map_loader.load_map()

    def _build_base_observation_space(self) -> SpacesDict:
        """Build the base observation space (Layer 1 contract).

        Always exactly the four structured tensors plus masks. Visual
        observations (`player_frame`, `global_view`, `screen`) are not part of
        the policy observation.
        """
        obs_spaces = {}

        # Game state features (ninja state + entity states) - always available
        obs_spaces["game_state"] = box.Box(
            low=-1,
            high=1,
            shape=(
                GAME_STATE_CHANNELS,
            ),  # NppEnvironment overwrites with 24 ninja + 3 physics-derived + switch (28); base supplies 25 here
            dtype=np.float32,
        )
        # Action mask for invalid action filtering - always available
        obs_spaces["action_mask"] = box.Box(
            low=0,
            high=1,
            shape=(6,),  # 6 actions in N++
            dtype=np.int8,
        )
        if self.enable_layer3_macro_action_mask:
            obs_spaces["duration_mask"] = box.Box(
                low=0,
                high=1,
                shape=(6, self._layer3_num_durations),
                dtype=np.int8,
            )

        # Layer 1 structured observations (NPP-9): tile indices + entity overlays
        obs_spaces["tile_grid"] = box.Box(
            low=0,
            high=NUM_TILE_TYPES - 1,
            shape=(MAP_TILE_WIDTH, MAP_TILE_HEIGHT),
            dtype=np.int8,
        )
        obs_spaces["entity_overlays"] = box.Box(
            low=0.0,
            high=1.0,
            shape=(ENTITY_OVERLAY_CHANNELS, MAP_TILE_WIDTH, MAP_TILE_HEIGHT),
            dtype=np.float32,
        )

        # Note: NppEnvironment._get_observation replaces game_state with 28-dim vector (L1.4).

        return SpacesDict(obs_spaces)

    def _actions_to_execute(self, action: int) -> Tuple[int, int]:
        """
        Execute the specified action using the game controller.

        Args:
            action: Action to execute (0-5)

        Returns:
            Tuple of (horizontal_input, jump_input)

        The action mapping is:
        0: NOOP - No action
        1: Left - Press 'A' key
        2: Right - Press 'D' key
        3: Jump - Press Space key
        4: Jump + Left - Press Space + 'A' keys
        5: Jump + Right - Press Space + 'D' keys
        """
        hoz_input = 0
        jump_input = 0

        if action == 0:  # NOOP
            pass
        elif action == 1:  # Left
            hoz_input = -1
        elif action == 2:  # Right
            hoz_input = 1
        elif action == 3:  # Jump
            jump_input = 1
        elif action == 4:  # Jump + Left
            jump_input = 1
            hoz_input = -1
        elif action == 5:  # Jump + Right
            jump_input = 1
            hoz_input = 1

        return hoz_input, jump_input

    def set_curriculum_stage(self, new_stage: int) -> None:
        """Set curriculum stage externally (called by global curriculum callback).

        This method allows the global curriculum callback to synchronize all parallel
        environments to the same curriculum stage. The callback aggregates success rates
        across all environments and coordinates stage advancement globally.

        CRITICAL: Stage change is DEFERRED until next episode reset to prevent mid-episode
        corruption. When callback calls this on all 256+ workers, some are mid-episode with
        collected switches. Changing unified_stage immediately would cause cache/entity mismatches.

        Args:
            new_stage: New curriculum stage index
        """
        if self.goal_curriculum_manager is None:
            logger.warning(
                "set_curriculum_stage called but no curriculum manager exists"
            )
            return

        old_stage = self.goal_curriculum_manager.state.unified_stage

        if old_stage == new_stage:
            return  # Already at this stage

        # CRITICAL: Defer BOTH stage change and entity repositioning to next reset
        # Store pending stage instead of applying immediately
        # This prevents mid-episode corruption when callback updates all workers simultaneously
        if not hasattr(self, "_pending_curriculum_stage"):
            self._pending_curriculum_stage = None

        # PROTECTION: Detect stage skipping (rapid advancement)
        # If pending stage already exists and is different from new_stage, we're skipping a stage
        if (
            self._pending_curriculum_stage is not None
            and self._pending_curriculum_stage != new_stage
        ):
            logger.warning(
                f"[CURRICULUM] Rapid stage advancement detected! "
                f"Overwriting pending stage {self._pending_curriculum_stage} with {new_stage}. "
                f"Current: {old_stage}, Pending was: {self._pending_curriculum_stage}, New: {new_stage}. "
                f"This worker will skip stage {self._pending_curriculum_stage}!"
            )

        self._pending_curriculum_stage = new_stage

        # ANNEALED MASKING (2026-01-05): Update RewardConfig with curriculum stage
        # This allows action_masking_mode property to track curriculum progress
        # for annealed masking transitions (hard -> soft -> none)
        if hasattr(self, "reward_calculator") and self.reward_calculator.config:
            num_stages = self.goal_curriculum_manager._num_stages
            # Update immediately (not deferred) since masking mode only affects action selection
            # No mid-episode corruption risk - just changes which actions are masked/biased
            self.reward_calculator.config.set_curriculum_stage(new_stage, num_stages)
            logger.debug(
                f"[CURRICULUM] Updated RewardConfig curriculum stage: {new_stage}/{num_stages}, "
                f"masking_mode={self.reward_calculator.config.action_masking_mode}"
            )

        frame_count = (
            self.nplay_headless.sim.frame if hasattr(self, "nplay_headless") else 0
        )

        print(
            f"[set_curriculum_stage] Stage change queued: {old_stage} → {new_stage} "
            f"(will apply at next reset with sub-goal rebuild, current frame={frame_count})"
        )
        logger.info(
            f"[CURRICULUM] Stage change queued: {old_stage} → {new_stage} "
            f"(will apply at next reset with sub-goal rebuild, current frame={frame_count})"
        )

        # DEFERRED: Both stage change AND cache invalidation happen at next reset
        # This prevents mid-episode cache/entity mismatches where:
        # - unified_stage changes → SharedLevelCache switches to new stage
        # - But actual entities haven't moved yet → cache positions ≠ entity positions
        # - PBRS pathfinds to cache positions, not actual entity positions
        # All changes are deferred to next reset when entities are freshly loaded

    def step(self, action: int):
        """Execute one environment step with frame skip and enhanced episode info.

        With frame_skip enabled, repeats the action for N frames and calculates
        reward once using the initial→final transition.

        Template method that defines the step execution flow with hooks
        for subclasses to extend behavior at specific points.
        """
        logger = logging.getLogger(__name__)

        # Granular step timing (when profile_enabled)
        t_step = self._profile_start("step_total")
        try:
            # Get initial observation for frame skip reward calculation
            initial_obs = self._prev_obs_cache
            if initial_obs is None:
                initial_obs = self._get_observation()

            # Record action for debug visualization
            if hasattr(self, "_record_action_for_debug"):
                self._record_action_for_debug(action)

            # Track last action for death context
            self.last_action = action

            # Track action in action sequence for Go-Explore checkpoint replay
            self._action_sequence.append(action)

            # Track total steps for deterministic masking rate calculation
            self._total_steps_count += 1

            # Execute action for frame_skip frames
            # OPTIMIZATION: Compute action inputs once before loop
            action_hoz, action_jump = self._actions_to_execute(action)
            terminated = False
            truncated = False
            frames_executed = 0

            # OPTIMIZATION: Cache headless reference to avoid attribute lookup in loop
            headless = self.nplay_headless

            # OPTIMIZATION: Check profiling state once before loop
            _profile = getattr(self, "profile_enabled", False)

            # Per-frame primitive rewards for FiGAR-PPO primitive-level GAE
            primitive_rewards = []

            # Track switch activation state across frames for per-frame PBRS delta detection
            prev_switch_activated = getattr(self, "_prev_switch_activated", False)

            for i in range(self.frame_skip):
                headless.tick(action_hoz, action_jump)

                frames_executed = i + 1

                # Track switch activation frame and action-step index precisely (~1μs)
                current_switch_activated = headless.exit_switch_activated()
                if self._switch_activation_frame is None and current_switch_activated:
                    self._switch_activation_frame = headless.sim.frame
                    # current_route has not been appended yet for this step, so
                    # len(current_route) is the 0-based index of the current action.
                    self._switch_activation_step = len(self.current_route)

                # Check death/win per frame
                player_won = headless.ninja_has_won()
                player_dead = headless.ninja_has_died()

                # Per-frame PBRS reward for FiGAR-PPO primitive-level GAE.
                # Only computed when a precomputed surface-graph PBRS is loaded.
                if self._current_precomputed_pbrs is not None:
                    pos = headless.ninja_position()
                    vel = headless.ninja_velocity()
                    # Map Rust ninja state codes to PBRS contact categories:
                    # 0-2 (Immobile/Running/GroundSlide) → floor, 5 (WallSlide) → wall, else → airborne
                    ninja_sc = headless.sim.ninja.state
                    contact = (
                        0 if ninja_sc in (0, 1, 2) else (1 if ninja_sc == 5 else 2)
                    )
                    death_cause = headless.ninja_death_cause() if player_dead else None
                    frame_r = self.primitive_reward_calculator.calculate_frame_reward(
                        player_x=float(pos[0]),
                        player_y=float(pos[1]),
                        player_dead=player_dead,
                        player_won=player_won,
                        switch_activated=current_switch_activated,
                        prev_switch_activated=prev_switch_activated,
                        phi_grid_cache=None,
                        precomputed_pbrs=self._current_precomputed_pbrs,
                        player_xspeed=float(vel[0]),
                        player_yspeed=float(vel[1]),
                        contact_state=contact,
                        death_cause=death_cause,
                    )
                    primitive_rewards.append(frame_r)

                    # Diagnostic tracking: contact distribution and phi at key moments
                    comps = self.primitive_reward_calculator.last_components
                    if contact == 0:
                        self._contact_floor_frames += 1
                    elif contact == 1:
                        self._contact_wall_frames += 1
                    else:
                        self._contact_airborne_frames += 1
                    if self._first_frame_of_episode:
                        self._phi_at_spawn = comps.get("prev_potential")
                        self._first_frame_of_episode = False
                    if current_switch_activated and not prev_switch_activated:
                        self._phi_at_switch_collection = comps.get("phi_state")
                    self._phi_at_terminal = comps.get("phi_state")

                    prev_switch_activated = current_switch_activated

                if player_won or player_dead:
                    terminated = True
                    break

            # Persist switch activation state for the next step's first-frame delta
            self._prev_switch_activated = current_switch_activated

            # Check truncation only once after all frames (expensive curriculum-aware check)
            if not terminated:
                truncated = self._check_curriculum_aware_truncation()

            # Primitive / FiGAR path: append truncation penalty before episode reward + info
            # so current_ep_reward, info["r"], and info["primitive_rewards"] stay consistent.
            if self._current_precomputed_pbrs is not None and truncated and primitive_rewards:
                from .reward_calculation.reward_constants import TRUNCATION_PENALTY

                primitive_rewards = list(primitive_rewards)
                primitive_rewards.append(TRUNCATION_PENALTY)

            self._track_position_after_step()

            # Invalidate observation cache since game state changed
            self._cached_observation = None

            # Hook: After action execution, before observation
            self._post_action_hook()

            # Get final observation - OBSERVATION GENERATION
            final_obs = self._get_observation()

            # Cache observation for reward calculation in next step
            self._prev_obs_cache = final_obs

            # Get player_won for reward calculation
            player_won = final_obs.get("player_won", False)

            use_primitive_reward_path = self._current_precomputed_pbrs is not None

            if not self.skip_reward_calculation:
                # Hook: After observation, before reward calculation
                self._pre_reward_hook(final_obs, player_won)

                if use_primitive_reward_path:
                    # FiGAR-PPO: only PrimitiveRewardCalculator runs (precomputed PBRS path).
                    reward = float(sum(primitive_rewards)) if primitive_rewards else 0.0
                    self.current_ep_reward += reward
                else:
                    # Standard PPO: main HWM / grid reward calculator
                    reward = self.reward_calculator.calculate_reward(
                        final_obs,
                        initial_obs,
                        action,
                        frames_executed=frames_executed,
                        curriculum_manager=self.goal_curriculum_manager,
                        truncated=truncated,
                    )

                    # Update dynamic truncation limit if PBRS surface area is now available
                    self._update_dynamic_truncation_if_needed()

                    # Hook: Modify reward if needed
                    reward = self._modify_reward_hook(
                        reward, final_obs, player_won, terminated
                    )

                    # === TRUNCATION HANDLING ===
                    # No additional penalty for truncation. HWM pipeline:
                    # - HWM shaping rewards progress toward objectives (zero when no progress)
                    # - Sparse terminal rewards (R_SWITCH, R_EXIT, R_DEATH) define task hierarchy
                    # - Truncation yields 0 reward; agent learns from HWM accumulated during episode

                    self.current_ep_reward += reward
            else:
                reward = 0.0

            # Process observation for training
            t_proc = self._profile_start("observation_process")
            processed_obs = self._process_observation(final_obs)
            self._profile_end("observation_process", t_proc)

            # Build episode info
            t_info = self._profile_start("info_build")
            info = self._build_episode_info(player_won, terminated, truncated)
            self._profile_end("info_build", t_info)

            # DIAGNOSTIC: Store reward breakdown in info for route visualization
            # This allows seeing reward components without needing logging enabled
            if not self.skip_reward_calculation and truncated and not terminated:
                from .reward_calculation.reward_constants import GLOBAL_REWARD_SCALE

                combined_path_distance = final_obs.get(
                    "_pbrs_combined_path_distance", 1000.0
                )
                final_distance = final_obs.get("_pbrs_last_distance_to_goal", None)

                if use_primitive_reward_path:
                    prc = self.primitive_reward_calculator
                    sc = prc.global_reward_scale
                    total_pbrs_u = (
                        sum(prc.episode_pbrs_rewards)
                        if prc.episode_pbrs_rewards
                        else 0.0
                    )
                    n_fr = len(prc.episode_pbrs_rewards)
                    pbrs_total_scaled = prc.alpha * total_pbrs_u * sc
                    time_total_scaled = n_fr * prc.time_penalty_per_frame * sc
                    # No main PBRS path distance cache in primitive-only mode
                    if final_distance is None or final_distance == float("inf"):
                        progress = 0.0
                    elif combined_path_distance > 0:
                        progress = 1.0 - (final_distance / combined_path_distance)
                        progress = max(0.0, min(1.0, progress))
                    else:
                        progress = 0.0
                    info["_timeout_reward_breakdown"] = {
                        "pbrs_total": pbrs_total_scaled,
                        "time_total": time_total_scaled,
                        "final_progress": progress,
                        "final_distance": final_distance,
                        "combined_path": combined_path_distance,
                        "forward_steps": prc.episode_forward_steps,
                        "backtrack_steps": prc.episode_backtrack_steps,
                    }
                else:
                    pbrs_total_unscaled = (
                        sum(self.reward_calculator.episode_pbrs_rewards)
                        if hasattr(self.reward_calculator, "episode_pbrs_rewards")
                        else 0.0
                    )
                    time_total_unscaled = (
                        sum(self.reward_calculator.episode_time_penalties)
                        if hasattr(self.reward_calculator, "episode_time_penalties")
                        else 0.0
                    )
                    pbrs_total_scaled = pbrs_total_unscaled * GLOBAL_REWARD_SCALE
                    time_total_scaled = time_total_unscaled * GLOBAL_REWARD_SCALE

                    if final_distance is None or final_distance == float("inf"):
                        final_distance = (
                            self.reward_calculator.closest_distance_this_episode
                        )

                    if combined_path_distance > 0 and final_distance != float("inf"):
                        progress = 1.0 - (final_distance / combined_path_distance)
                        progress = max(0.0, min(1.0, progress))
                    else:
                        progress = 0.0

                    info["_timeout_reward_breakdown"] = {
                        "pbrs_total": pbrs_total_scaled,
                        "time_total": time_total_scaled,
                        "final_progress": progress,
                        "final_distance": final_distance,
                        "combined_path": combined_path_distance,
                        "forward_steps": self.reward_calculator.episode_forward_steps,
                        "backtrack_steps": self.reward_calculator.episode_backtrack_steps,
                    }

            # Route viz: full primitive reward breakdown (death, timeout, success)
            if (
                not self.skip_reward_calculation
                and use_primitive_reward_path
                and (terminated or truncated)
            ):
                info["_primitive_reward_breakdown"] = self._build_primitive_reward_breakdown(
                    final_obs, player_won, terminated, truncated
                )

            # Add frame skip stats to info
            info["frame_skip_stats"] = {
                "skip_value": self.frame_skip,
                "frames_executed": frames_executed,
            }

            # Per-frame primitive rewards (truncation already applied above for precomputed path)
            info["primitive_rewards"] = primitive_rewards

            # Hook: Add additional info fields
            self._extend_info_hook(info)

            # OPTIMIZATION: Clear step-level cache at end (not start) for obs↔reward sharing.
            # Observation populated cache for final state; reward reused it. Next step's
            # initial = this step's final, so we clear now to avoid stale cache across steps.
            if hasattr(self, "_path_calculator") and self._path_calculator is not None:
                self._path_calculator.clear_step_cache()

            # End step profiling
            self._profile_end("step_total", t_step)

            return processed_obs, reward, terminated, truncated, info

        except Exception as e:
            logger.error(
                f"[STEP_EXCEPTION] Exception in step() method: {type(e).__name__}: {e}. "
                f"Action: {action}, "
                f"Current ep reward: {self.current_ep_reward}"
            )
            # Re-raise to maintain normal error handling
            raise

    def _try_load_value_function_pbrs(self) -> None:
        """Try to load value function PBRS for the current level.

        Priority order:
            1. Neural IQL V(s) checkpoint (if iql_checkpoint_path is set)
            2. Surface graph .npz (existing TwoPhaseValuePBRS)
            3. Fall through to phi grid / path-based fallback

        Sets _current_precomputed_pbrs when a value function is loaded.
        Clears it when no files are found.
        """
        if not getattr(self, "use_precomputed_pbrs", False):
            self._current_precomputed_pbrs = None
            return

        level_id = getattr(self.map_loader, "current_map_name", None)
        if not level_id:
            self._current_precomputed_pbrs = None
            return

        # Priority 1: Neural IQL V(s) checkpoint
        iql_path = getattr(self, "iql_checkpoint_path", None)
        if iql_path:
            try:
                from nclone.value_function.two_phase_pbrs import create_neural_pbrs

                iql_device = getattr(self, "iql_device", "cpu")
                adapter = create_neural_pbrs(
                    checkpoint_path=iql_path,
                    device=iql_device,
                )

                # Set level data from current headless simulator
                headless = getattr(self, "headless", None)
                if headless is not None:
                    self._setup_neural_pbrs_level(adapter, headless)

                self._current_precomputed_pbrs = adapter
                logger.info(
                    "Loaded neural IQL PBRS for level %s from %s",
                    level_id,
                    iql_path,
                )
                return
            except (FileNotFoundError, ImportError) as e:
                logger.warning("Failed to load neural IQL PBRS: %s, falling back", e)

        # Priority 2: Surface graph .npz — prefer RoutePBRS (route-tracking) when
        # the .npz contains predecessor arrays; fall back to legacy SurfaceGraphPBRS.
        vf_dir = getattr(self, "value_function_dir", None)
        if not vf_dir:
            self._current_precomputed_pbrs = None
            return

        # Try RoutePBRS first (requires predecessor arrays in the .npz)
        try:
            from nclone.value_function.route_pbrs import create_route_pbrs
            route_pbrs = create_route_pbrs(level_id, vf_dir)
            if route_pbrs is not None:
                self._current_precomputed_pbrs = route_pbrs
                logger.info(f"Loaded RoutePBRS (route-tracking) for level {level_id}")
                return
        except Exception as e:
            logger.warning(f"RoutePBRS load failed for {level_id}: {e}, falling back")

        # Fall back to legacy SurfaceGraphPBRS
        from nclone.value_function import create_two_phase_pbrs

        self._current_precomputed_pbrs = create_two_phase_pbrs(
            level_id,
            value_function_dir=vf_dir,
            interpolation_method="linear",
            blend_mode="continuous",
        )
        logger.info(f"Loaded legacy SurfaceGraphPBRS for level {level_id}")

    def _setup_neural_pbrs_level(self, adapter: Any, headless: Any) -> None:
        """Configure neural PBRS adapter with current level data from headless.

        Extracts tile grid, entity positions, and mine data from the headless
        simulator to set up the NeuralValuePBRS adapter for the current level.

        Args:
            adapter: NeuralValuePBRS instance.
            headless: NativePlayHeadless or NPlayHeadless instance.
        """
        import numpy as np

        try:
            sim = headless.sim if hasattr(headless, "sim") else headless
            sim_state = sim.get_state()

            # Extract tile grid (42x23)
            tile_data = sim.get_tile_grid()
            tile_grid = np.array(tile_data, dtype=np.int8).reshape(23, 42).T

            # Find switch and exit positions from entities
            switch_pos = (0.0, 0.0)
            exit_pos = (0.0, 0.0)
            mine_positions = []

            for ent in sim_state.entities:
                if ent.entity_type == 4:  # exit switch
                    switch_pos = (ent.x, ent.y)
                elif ent.entity_type == 3:  # exit door
                    exit_pos = (ent.x, ent.y)
                elif ent.entity_type in (1, 21):  # toggle mines
                    mine_positions.append((ent.x, ent.y, ent.entity_type))

            adapter.set_level(
                tile_grid=tile_grid,
                switch_pos=switch_pos,
                exit_pos=exit_pos,
                mine_positions=mine_positions,
                headless=headless,
            )
        except Exception as e:
            logger.warning("Failed to set up neural PBRS level data: %s", e)

    def _try_build_reachability_pbrs(self) -> None:
        """Build analytical reachability PBRS from current graph data.

        Requires self.current_graph_data to be populated (by graph mixin).
        Sets self._current_precomputed_pbrs so the reward calculator picks
        it up via the existing precomputed_pbrs code path (same interface).
        """
        if not getattr(self, "use_reachability_pbrs", False):
            return

        graph_data = getattr(self, "current_graph_data", None)
        if not graph_data:
            return

        try:
            from nclone.analytical_physics.reachability_builder import (
                build_reachability_pbrs,
                should_rebuild,
            )

            # Get level data
            try:
                level_data = self.level_data
            except Exception:
                return

            # Check if rebuild is needed (mine state may have changed)
            existing = self._current_reachability_pbrs
            if existing is not None and not should_rebuild(existing, level_data):
                # No mine state change, reuse existing PBRS
                self._current_precomputed_pbrs = existing
                return

            switch_pos = self.nplay_headless.exit_switch_position()
            exit_pos = self.nplay_headless.exit_door_position()

            if switch_pos is None or exit_pos is None:
                return

            pbrs = build_reachability_pbrs(
                graph_data=graph_data,
                level_data=level_data,
                switch_pos=switch_pos,
                exit_pos=exit_pos,
            )

            self._current_reachability_pbrs = pbrs
            # Use the same slot as precomputed PBRS (same interface)
            self._current_precomputed_pbrs = pbrs

        except Exception as e:
            logger.warning(f"Failed to build reachability PBRS: {e}")
            self._current_reachability_pbrs = None

    def get_observation(self) -> Dict[str, Any]:
        """Public accessor for current observation (for frame skip wrapper).

        Note: Returns observation directly without copying. The caller should not
        mutate the returned dictionary. If mutation is needed, the caller should
        make their own copy.
        """
        return self._get_observation()

    def step_without_reward(
        self, action: int, generate_observation: bool = True
    ) -> Tuple[Dict[str, Any], bool, bool, Dict[str, Any]]:
        """Execute one physics frame without reward calculation (for frame skip).

        Args:
            action: Action to execute
            generate_observation: If False, returns None for observation (for intermediate frames)

        Returns:
            Tuple of (observation, terminated, truncated, info)
            observation will be None if generate_observation=False
        """
        logger = logging.getLogger(__name__)

        try:
            # Record action for debug visualization
            if hasattr(self, "_record_action_for_debug"):
                self._record_action_for_debug(action)

            # Track last action for death context
            self.last_action = action

            # Execute action
            action_hoz, action_jump = self._actions_to_execute(action)
            self.nplay_headless.tick(action_hoz, action_jump)

            # Track position for route visualization
            self._track_position_after_step()

            # Invalidate observation cache since game state changed
            self._cached_observation = None

            # Hook: After action execution, before observation
            self._post_action_hook()

            # Get current observation only if needed
            if generate_observation:
                curr_obs = self._get_observation()
                # Cache observation reference for next step (no copy needed)
                self._prev_obs_cache = curr_obs
            else:
                curr_obs = None

            # Check termination
            terminated, truncated, _ = self._check_termination()

            # Minimal info dict (extended info added later by wrapper)
            info = {}

            # Return observation directly (no copy needed - wrapper will handle if needed)
            return curr_obs, terminated, truncated, info

        except Exception as e:
            logger.error(
                f"[STEP_EXCEPTION] Exception in step_without_reward(): {type(e).__name__}: {e}. "
                f"Action: {action}"
            )
            raise

    def step_n_frames(
        self,
        action: int,
        n_frames: int,
        gamma: float = 0.99,
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Execute action for n_frames, accumulating discounted rewards.

        This method is designed for FiGAR-PPO temporal abstraction where the agent
        selects (action, duration) pairs. The action is held for the specified
        duration, and rewards are accumulated with proper discounting.

        The accumulated reward is: R = Σ_{t=0}^{k-1} γ^t * r_t
        The effective discount to the next state is: γ^k

        Args:
            action: Action to execute (0-5 for N++)
            n_frames: Number of environment steps to execute (each step is frame_skip frames)
            gamma: Discount factor for reward accumulation (should match PPO gamma)

        Returns:
            Tuple of:
                - observation: Final observation after n_frames
                - accumulated_reward: Discounted sum of rewards over n_frames
                - terminated: Whether episode ended (death or win)
                - truncated: Whether episode was truncated (time limit)
                - info: Info dict with additional fields:
                    - 'frames_executed': Actual frames executed (may be < n_frames if done)
                    - 'gamma_power': γ^k for SMDP discounting
                    - 'frame_rewards': List of per-frame rewards (for debugging)
        """
        logger = logging.getLogger(__name__)

        accumulated_reward = 0.0
        gamma_power = 1.0
        frames_executed = 0
        frame_rewards = []
        terminated = False
        truncated = False
        info = {}

        for frame in range(n_frames):
            # Execute one step (with frame_skip built in)
            obs, reward, frame_terminated, frame_truncated, step_info = self.step(
                action
            )

            # Accumulate discounted reward
            accumulated_reward += gamma_power * reward
            gamma_power *= gamma
            frames_executed += 1
            frame_rewards.append(reward)

            # Check termination
            if frame_terminated:
                terminated = True
                info.update(step_info)  # Preserve terminal info
                break
            if frame_truncated:
                truncated = True
                info.update(step_info)  # Preserve truncation info
                break

        # Build info dict
        info["frames_executed"] = frames_executed
        info["gamma_power"] = gamma_power
        info["frame_rewards"] = frame_rewards
        info["planned_duration"] = n_frames
        info["actual_duration"] = frames_executed

        # Log if episode ended early
        if frames_executed < n_frames:
            logger.debug(
                f"step_n_frames: Early termination at frame {frames_executed}/{n_frames}. "
                f"terminated={terminated}, truncated={truncated}"
            )

        return obs, accumulated_reward, terminated, truncated, info

    def calculate_reward_transition(
        self,
        initial_obs: Dict[str, Any],
        final_obs: Dict[str, Any],
        action: int,
        frames_executed: int = 1,
    ) -> float:
        """Calculate reward for a multi-frame transition (for frame skip).

        Args:
            initial_obs: Observation at start of frame skip
            final_obs: Observation at end of frame skip
            action: Action executed
            frames_executed: Number of frames executed (for time penalty scaling)

        Returns:
            Total reward for the transition
        """
        player_won = final_obs.get("player_won", False)

        if self.skip_reward_calculation:
            return 0.0

        # Hook: After observation, before reward calculation
        self._pre_reward_hook(final_obs, player_won)

        # Calculate reward with frame count for time penalty scaling
        reward = self.reward_calculator.calculate_reward(
            final_obs,
            initial_obs,
            action,
            frames_executed=frames_executed,
            curriculum_manager=self.goal_curriculum_manager,
        )

        # Update dynamic truncation limit if PBRS surface area is now available
        self._update_dynamic_truncation_if_needed()

        # Truncation handling: NO penalty applied (TIMEOUT_PENALTY = 0.0)
        # See lines 442-455 for full explanation of why truncation is not penalized
        terminated = final_obs.get("player_dead", False)

        # Hook: Modify reward if needed
        reward = self._modify_reward_hook(reward, final_obs, player_won, terminated)

        self.current_ep_reward += reward

        return reward

    def _post_action_hook(self):
        """Hook called after action execution, before getting observation.

        Subclasses can override this to inject behavior at this point.
        """
        pass

    def _pre_reward_hook(self, curr_obs: Dict[str, Any], player_won: bool):
        """Hook called after observation, before reward calculation.

        Args:
            curr_obs: Current observation dictionary
            player_won: Whether the player won

        Subclasses can override this to inject behavior at this point.
        """
        pass

    def _modify_reward_hook(
        self,
        reward: float,
        curr_obs: Dict[str, Any],
        player_won: bool,
        terminated: bool,
    ) -> float:
        """Hook to modify reward after base calculation.

        Args:
            reward: Base reward value
            curr_obs: Current observation dictionary
            player_won: Whether the player won
            terminated: Whether episode terminated

        Returns:
            Modified reward value

        Subclasses can override this to add reward shaping.
        """
        return reward

    def _get_observation_metrics(self, obs: Dict[str, Any]) -> Dict[str, float]:
        """Extract key observation values for TensorBoard logging.

        Extracts individual feature values from observations to help debug
        incorrect computations and track sensible trends.

        Args:
            obs: Raw observation dictionary from _get_observation()

        Returns:
            Dictionary with observation metrics, keys prefixed with "obs/"
        """
        metrics = {}

        # === GAME STATE FEATURES ===
        # Log first 15 key game_state features (ninja physics state)
        if "game_state" in obs:
            game_state = obs["game_state"]
            if isinstance(game_state, np.ndarray):
                # Log first 15 features (covers ninja state: position, velocity, physics)
                for i in range(min(15, len(game_state))):
                    metrics[f"obs/game_state/feature_{i}"] = float(game_state[i])

        # === ENTITY POSITIONS ===
        # Log raw positions (will be normalized in processed observation)
        metrics["obs/entity_positions/ninja_x"] = float(obs["player_x"])
        metrics["obs/entity_positions/ninja_y"] = float(obs["player_y"])
        # Also log normalized versions (as they appear in processed observation)
        metrics["obs/entity_positions/ninja_x_norm"] = float(
            obs["player_x"] / LEVEL_WIDTH
        )
        metrics["obs/entity_positions/ninja_y_norm"] = float(
            obs["player_y"] / LEVEL_HEIGHT
        )

        metrics["obs/entity_positions/switch_x"] = float(obs["switch_x"])
        metrics["obs/entity_positions/switch_y"] = float(obs["switch_y"])
        metrics["obs/entity_positions/switch_x_norm"] = float(
            obs["switch_x"] / LEVEL_WIDTH
        )
        metrics["obs/entity_positions/switch_y_norm"] = float(
            obs["switch_y"] / LEVEL_HEIGHT
        )

        metrics["obs/entity_positions/exit_x"] = float(obs["exit_door_x"])
        metrics["obs/entity_positions/exit_y"] = float(obs["exit_door_y"])
        metrics["obs/entity_positions/exit_x_norm"] = float(
            obs["exit_door_x"] / LEVEL_WIDTH
        )
        metrics["obs/entity_positions/exit_y_norm"] = float(
            obs["exit_door_y"] / LEVEL_HEIGHT
        )

        # === KEY PHYSICS STATE ===
        metrics["obs/physics/xspeed"] = float(obs["player_xspeed"])
        metrics["obs/physics/yspeed"] = float(obs["player_yspeed"])
        metrics["obs/switch_activated"] = float(obs["switch_activated"])

        return metrics

    def _check_curriculum_aware_truncation(self) -> bool:
        """Check truncation with curriculum-aware policy that respects reward config.

        Philosophy: Truncation serves computational efficiency and training stability,
        but should align with reward curriculum phases:

        - Early phase (no time penalty): 2.0x more generous (more exploration time)
        - Mid phase (conditional penalty): 1.5x more generous (balanced approach)
        - Late phase (full time penalty): 1.0x standard (efficiency focus)

        This balances computational limits with curriculum consistency.

        Returns:
            True if truncation should occur, False otherwise
        """
        reward_config = self.reward_calculator.config

        # Determine curriculum-aware truncation multiplier
        if reward_config is not None:
            phase = reward_config.training_phase

            if phase == "early":
                # No time penalty in rewards -> use standard truncation for faster learning
                multiplier = (
                    1.0  # Changed from 2.0 to fix episode-length vs rollout mismatch
                )
            elif phase == "mid":
                # Conditional time penalty -> standard truncation
                multiplier = 1.0  # Changed from 1.5
            elif phase == "late":
                # Full time penalty -> standard truncation for efficiency
                multiplier = 1.0
            else:
                # Unknown phase -> conservative standard truncation
                multiplier = 1.0
        else:
            # No reward config -> use standard truncation
            multiplier = 1.0

        # Check against curriculum-adjusted limit
        return self.truncation_checker.update_and_check_for_truncation(multiplier)

    def _get_curriculum_aware_truncation_limit(self) -> int:
        """Get the curriculum-aware truncation limit without side effects.

        Used for time_remaining calculation in observations to ensure consistency
        between what the agent observes and when truncation actually occurs.

        Returns:
            Curriculum-adjusted truncation limit in frames
        """
        reward_config = self.reward_calculator.config
        base_limit = self.truncation_checker.current_truncation_limit

        # Determine curriculum-aware truncation multiplier
        if reward_config is not None:
            phase = reward_config.training_phase

            if phase == "early":
                # No time penalty in rewards -> use standard truncation for faster learning
                multiplier = (
                    1.0  # Changed from 4.0 to fix episode-length vs rollout mismatch
                )
            elif phase == "mid":
                # Conditional time penalty -> standard truncation
                multiplier = 1.0  # Changed from 2.0
            elif phase == "late":
                # Full time penalty -> standard truncation for efficiency
                multiplier = 1.0
            else:
                # Unknown phase -> conservative standard truncation
                multiplier = 1.0
        else:
            # No reward config -> use standard truncation
            multiplier = 1.0

        return int(base_limit * multiplier)

    def _update_dynamic_truncation_if_needed(self):
        """Update dynamic truncation limit if PBRS surface area becomes available.

        This ensures that dynamic truncation is applied as soon as the level complexity
        can be determined from PBRS calculations, even if it wasn't available during
        initial level loading.
        """
        from .constants import MAX_TIME_IN_FRAMES

        pbrs_calculator = self.reward_calculator.pbrs_calculator
        if (
            pbrs_calculator._cached_surface_area is not None
            and self.truncation_checker.current_truncation_limit == MAX_TIME_IN_FRAMES
        ):
            # Surface area is now available but we're still using fallback limit
            surface_area = pbrs_calculator._cached_surface_area
            reachable_mine_count = 0  # Simplified reward system

            truncation_limit = self.truncation_checker.set_level_truncation_limit(
                surface_area, reachable_mine_count
            )

            if self.enable_logging:
                logger.info(
                    f"Updated to dynamic truncation limit: {truncation_limit} frames "
                    f"(surface_area={surface_area:.0f})"
                )

    def _extend_info_hook(self, info: Dict[str, Any]):
        """Hook to add additional fields to info dictionary.

        Args:
            info: Info dictionary to extend (modified in place)

        Subclasses can override this to add custom info fields.
        """
        # CRITICAL FIX: Add goal_curriculum info to EVERY step's info dict with ACTUAL positions
        # Go-Explore creates checkpoints during episodes (in _on_step), not just at episode end
        # Without this, checkpoints would be created with None for curriculum positions
        #
        # BUG FIX: When curriculum stage changes mid-episode via set_curriculum_stage(),
        # unified_stage updates immediately but entity positions don't change until next reset.
        # Checkpoints MUST store ACTUAL simulator positions, not calculated positions.
        if self.goal_curriculum_manager is not None:
            curriculum_info = self.goal_curriculum_manager.get_curriculum_info()

            # Override calculated positions with ACTUAL simulator positions
            # This ensures checkpoints capture the true entity state at checkpoint creation time
            actual_switch_pos = self.nplay_headless.exit_switch_position()
            actual_exit_pos = self.nplay_headless.exit_door_position()

            curriculum_info["curriculum_switch_pos"] = actual_switch_pos
            curriculum_info["curriculum_exit_pos"] = actual_exit_pos

            info["goal_curriculum"] = curriculum_info

    def _track_position_after_step(self):
        """Track ninja position after physics step for route visualization.

        Called after each physics tick to record the ninja's trajectory.
        """
        try:
            position = self.nplay_headless.ninja_position()
            if position is not None:
                self.current_route.append((float(position[0]), float(position[1])))
                # # DEBUG: Log first 30 position appends to diagnose 2x issue
                # if len(self.current_route) <= 30:
                #     logger.warning(
                #         f"[POSITION_APPEND] count={len(self.current_route)}, "
                #         f"pos=({position[0]:.1f}, {position[1]:.1f})"
                #     )
            else:
                if not self._warned_about_position:
                    logger.warning(
                        "ninja_position() returned None during route tracking"
                    )
                    self._warned_about_position = True
        except Exception as e:
            if not self._warned_about_position:
                logger.warning(
                    f"Exception in _track_position_after_step: {e}",
                    exc_info=True,
                )
                self._warned_about_position = True

    def _build_primitive_reward_breakdown(
        self,
        final_obs: Dict[str, Any],
        player_won: bool,
        terminated: bool,
        truncated: bool,
    ) -> Dict[str, Any]:
        """Scaled reward components for route visualization (FiGAR / precomputed PBRS).

        ``episode_return`` is ``current_ep_reward`` (sum of ``primitive_rewards`` per
        step, including truncation penalty when applicable). ``components_sum`` is
        sparse + α·PBRS + time + truncation in scaled units; ``residual`` catches
        float drift or inconsistent truncation scaling.
        """
        from .reward_calculation.reward_constants import TRUNCATION_PENALTY

        prc = self.primitive_reward_calculator
        sc = prc.global_reward_scale
        total_pbrs_u = sum(prc.episode_pbrs_rewards) if prc.episode_pbrs_rewards else 0.0
        total_sparse_u = (
            sum(prc.episode_sparse_rewards) if prc.episode_sparse_rewards else 0.0
        )
        n = len(prc.episode_pbrs_rewards)
        pbrs_scaled = prc.alpha * total_pbrs_u * sc
        sparse_scaled = total_sparse_u * sc
        time_scaled = n * prc.time_penalty_per_frame * sc
        trunc_scaled = (TRUNCATION_PENALTY * sc) if truncated else 0.0

        components_sum = pbrs_scaled + sparse_scaled + time_scaled + trunc_scaled
        ep_ret = float(self.current_ep_reward)
        residual = ep_ret - components_sum

        if truncated and not terminated:
            combined_path_distance = final_obs.get("_pbrs_combined_path_distance", 1000.0)
            final_distance = final_obs.get("_pbrs_last_distance_to_goal", None)
            if final_distance is None or final_distance == float("inf"):
                progress = 0.0
            elif combined_path_distance > 0:
                progress = 1.0 - (final_distance / combined_path_distance)
                progress = max(0.0, min(1.0, progress))
            else:
                progress = 0.0
        else:
            combined_path_distance = None
            final_distance = None
            progress = None

        death_cause = None
        if terminated and not player_won:
            death_cause = self.nplay_headless.ninja_death_cause()

        if player_won:
            terminal_label = "win"
        elif truncated:
            terminal_label = "timeout"
        else:
            terminal_label = "death"

        return {
            "pbrs_scaled": pbrs_scaled,
            "sparse_scaled": sparse_scaled,
            "time_scaled": time_scaled,
            "truncation_scaled": trunc_scaled,
            "components_sum": components_sum,
            "episode_return": ep_ret,
            "residual": residual,
            "forward_steps": prc.episode_forward_steps,
            "backtrack_steps": prc.episode_backtrack_steps,
            "n_primitive_frames": n,
            "death_cause": death_cause,
            "terminal": terminal_label,
            "alpha": prc.alpha,
            "gamma": prc.gamma,
            "global_reward_scale": sc,
            "final_progress": progress,
            "final_distance": final_distance,
            "combined_path_distance": combined_path_distance,
        }

    def _build_episode_info(
        self, player_won: bool, terminated: bool, truncated: bool
    ) -> Dict[str, Any]:
        """Build the base episode info dictionary.

        This method can be extended by subclasses to add additional info fields.

        Args:
            player_won: Whether the player successfully completed the level

        Returns:
            Dictionary containing episode information
        """
        # Calculate episode frame count (excluding checkpoint replay frames)
        # For checkpoint episodes, only count frames executed after replay
        total_frames = self.nplay_headless.sim.frame
        replay_frames = self._checkpoint_replay_frame_count

        # DEFENSIVE: Sanity check for stale _checkpoint_replay_frame_count
        # If replay_frames > total_frames, it means the value wasn't reset between episodes
        # This causes negative episode_frames and corrupts route visualization
        if replay_frames > total_frames:
            # logger.error(
            #     f"[STALE_REPLAY_FRAMES] _checkpoint_replay_frame_count ({replay_frames}) > "
            #     f"total_frames ({total_frames}). This indicates stale value from previous episode. "
            #     f"from_checkpoint={self._from_checkpoint}, "
            #     f"checkpoint_base_reward={self._checkpoint_base_reward:.3f}. "
            #     f"Resetting to 0."
            # )
            replay_frames = 0
            self._checkpoint_replay_frame_count = 0

        episode_frames = total_frames - replay_frames

        # Determine curriculum vs full level success
        curriculum_success = False
        full_level_success = False

        if player_won:
            # Agent reached exit door (which is at curriculum-adjusted position)
            curriculum_success = True

            # Full level success only if curriculum is at final stage
            if self.goal_curriculum_manager is not None:
                full_level_success = self.goal_curriculum_manager.is_at_final_stage()
            else:
                # No curriculum active - any completion is full level
                full_level_success = True
        else:
            curriculum_success = False
            full_level_success = False

        # CRITICAL: Capture entity positions FRESH from simulator at episode end
        # Do NOT use _cached_switch_pos/_cached_exit_pos - these can be stale!
        # With goal curriculum, entities are repositioned at reset, and the cached values
        # may not reflect the actual curriculum positions used during the episode.
        # Fresh reads ensure visualization shows correct positions.
        exit_switch_pos = self.nplay_headless.exit_switch_position()
        exit_door_pos = self.nplay_headless.exit_door_position()

        info = {
            "is_success": player_won,
            "terminated": terminated,
            "truncated": truncated,
            "r": self.current_ep_reward,  # Cumulative reward for entire episode
            "l": episode_frames,  # Episode frames only (excluding checkpoint replay)
            "l_total": total_frames,  # Total frames including replay (for debugging)
            "l_checkpoint_replay": self._checkpoint_replay_frame_count,  # Replay frames
            "level_id": self.map_loader.current_map_name,
            "config_flags": self.config_flags.copy(),
            "terminal_impact": self.nplay_headless.get_ninja_terminal_impact(),
            "exit_switch_pos": exit_switch_pos,
            "exit_door_pos": exit_door_pos,
            # CRITICAL: Include switch_activated for PBRS visualization and waypoint phase
            # This must be in base info dict (not just _extend_info_hook) because it's
            # accessed within _build_episode_info itself for waypoint phase selection
            "switch_activated": self.nplay_headless.exit_switch_activated(),
            # Switch activation frame and action-step index for route visualization
            # (None if switch was never activated this episode)
            "switch_activation_frame": self._switch_activation_frame,
            "switch_activation_step": self._switch_activation_step,
        }

        # # DIAGNOSTIC: Log switch activation frame in episode info
        # if terminated or truncated:
        #     print(
        #         f"[EPISODE_INFO] switch_activation_frame={self._switch_activation_frame}, "
        #         f"switch_activated={self.nplay_headless.exit_switch_activated()}, "
        #         f"episode_length={episode_frames}, "
        #         f"curriculum_stage={self._episode_start_curriculum_stage}"
        #     )

        info.update(
            {
                # Curriculum success tracking: separate metrics for curriculum vs full level
                "curriculum_success": curriculum_success,
                "full_level_success": full_level_success,
            }
        )

        # Death and timeout diagnostics for TensorBoard
        if terminated or truncated:
            ninja_pos = self.nplay_headless.ninja_position()
            if ninja_pos is not None and exit_switch_pos is not None:
                px, py = float(ninja_pos[0]), float(ninja_pos[1])
                sx, sy = exit_switch_pos[0], exit_switch_pos[1]
                dist_to_switch = ((px - sx) ** 2 + (py - sy) ** 2) ** 0.5
                if terminated and not player_won:
                    info["death_x"] = px
                    info["death_y"] = py
                    info["distance_to_switch_at_death"] = dist_to_switch
                if truncated:
                    info["distance_to_switch_at_timeout"] = dist_to_switch

        # Add episode route for visualization if episode is ending
        if terminated or truncated:
            route_length = len(self.current_route)
            if route_length > 0:
                info["episode_route"] = list(self.current_route)
                info["route_length"] = route_length

                # Precompute the level-cache shortest path from the episode's starting
                # position so the route visualization callback can overlay it without
                # needing an env_method call from an async thread.
                episode_switch_activated = self.nplay_headless.exit_switch_activated()
                info["episode_next_hop_path"] = (
                    self._get_next_hop_path_for_visualization(
                        self.current_route[0],
                        switch_activated=episode_switch_activated,
                    )
                )

                # CRITICAL: Clear route immediately after copying to prevent accumulation
                # This fixes a bug where routes would leak across episodes if reset()
                # was not called reliably (e.g., in SubprocVecEnv worker scenarios)
                self.current_route = []
            else:
                logger.warning(
                    f"Episode ended with EMPTY route! "
                    f"frames={self.nplay_headless.sim.frame}"
                )

            # Add deterministic masking statistics for TensorBoard monitoring
            # Tracks how often trivial straight paths trigger deterministic masking
            if self._total_steps_count > 0:
                mask_rate = (
                    self._deterministic_mask_applied_count / self._total_steps_count
                )
                info["deterministic_mask_applied_count"] = (
                    self._deterministic_mask_applied_count
                )
                info["deterministic_mask_rate"] = mask_rate
            else:
                info["deterministic_mask_applied_count"] = 0
                info["deterministic_mask_rate"] = 0.0

            if self.enable_layer3_macro_action_mask:
                decisions = max(self._layer3_decision_points, 1)
                info["layer3_mask_enabled"] = True
                info["layer3_decision_points"] = self._layer3_decision_points
                info["layer3_actions_masked_per_decision"] = (
                    self._layer3_masked_actions_total / decisions
                )
                info["layer3_candidates_masked_per_decision"] = (
                    self._layer3_masked_candidates_total / decisions
                )
                info["layer3_mask_overrides_per_episode"] = (
                    self._layer3_mask_override_count
                )
                info["layer3_all_actions_masked_events"] = (
                    self._layer3_all_masked_events
                )

            # CRITICAL: Capture level data for route visualization BEFORE auto-reset
            # With VecEnv, the environment auto-resets after done=True, so by the time
            # the callback runs, a NEW level may already be loaded (curriculum).
            # We must capture the CURRENT level's data here in the info dict.
            info["episode_tiles"] = self.get_route_visualization_tile_data()
            info["episode_mines"] = self.get_route_visualization_mine_data()
            info["episode_locked_doors"] = (
                self.get_route_visualization_locked_door_data()
            )

            # Capture curriculum stage for visualization (shows which stage this episode was)
            # CRITICAL: Use _episode_start_curriculum_stage (captured at reset), NOT current stage!
            # The current stage might have advanced mid-episode due to success rates.
            if self.goal_curriculum_manager is not None:
                # Use the stage that was ACTIVE when this episode started
                episode_stage = (
                    self._episode_start_curriculum_stage
                    if self._episode_start_curriculum_stage is not None
                    else self.goal_curriculum_manager.state.unified_stage
                )
                info["episode_curriculum_stage"] = episode_stage
                # Also capture original positions for comparison visualization
                info["goal_curriculum"] = {
                    "enabled": True,
                    "unified_stage": episode_stage,  # Use episode start stage
                    "curriculum_switch_pos": exit_switch_pos,  # Fresh read from above
                    "curriculum_exit_pos": exit_door_pos,  # Fresh read from above
                    "original_switch_pos": (
                        self.goal_curriculum_manager._original_switch_pos
                    ),
                    "original_exit_pos": (
                        self.goal_curriculum_manager._original_exit_pos
                    ),
                }

        # DIAGNOSTIC: Validate episode stats are reasonable
        if (terminated or truncated) and self.enable_logging:
            if self.current_ep_reward == 0.0:
                logger.warning(
                    f"Episode ended with zero cumulative reward. "
                    f"Won: {player_won}, Dead: {self.nplay_headless.ninja_has_died()}, "
                    f"Frames: {self.nplay_headless.sim.frame}"
                )

        # Store last episode reward for reset validation
        self._last_episode_reward = self.current_ep_reward

        # Store last episode reward for reset validation
        self._last_episode_reward = self.current_ep_reward

        # Add goal curriculum info for callbacks (Go-Explore, route visualization)
        if self.goal_curriculum_manager is not None:
            info["goal_curriculum"] = self.goal_curriculum_manager.get_curriculum_info()

            # Update manager with episode outcomes for stage advancement
            if terminated or truncated:
                # Get switch activation from observation (most reliable)
                # Use final observation from step() or current state
                switch_activated = self.nplay_headless.exit_switch_activated()
                # Use curriculum success (not full level) for stage advancement
                self.goal_curriculum_manager.update_from_episode(
                    switch_activated=switch_activated, completed=curriculum_success
                )

                # Check if curriculum stage advanced - invalidate PBRS cache and switch to new stage cache
                if self.goal_curriculum_manager.needs_cache_rebuild():
                    new_stage = self.goal_curriculum_manager.state.unified_stage
                    logger.info(
                        f"[CURRICULUM] Stage advanced to {new_stage}, switching shared cache and invalidating PBRS caches. "
                        "Next PBRS calculation will use new stage's SharedLevelCache with curriculum goal positions."
                    )
                    # Clear PBRS normalization cache (will rebuild on next step with curriculum positions)
                    # CRITICAL: All position-dependent PBRS caches must be cleared on curriculum advancement
                    if hasattr(self.reward_calculator, "pbrs_calculator"):
                        pbrs_calc = self.reward_calculator.pbrs_calculator

                        # Clear path distance caches (used for PBRS normalization)
                        pbrs_calc._cached_combined_path_distance = None
                        pbrs_calc._cached_spawn_to_switch_distance = None
                        pbrs_calc._cached_switch_to_exit_distance = None
                        pbrs_calc._cached_combined_physics_cost = None

                        # Clear surface area cache (geometry-based, not position-based, but clear for safety)
                        pbrs_calc._cached_surface_area = None

                        # Clear level/state tracking (forces recomputation on next step)
                        pbrs_calc._cached_level_id = None
                        pbrs_calc._cached_switch_states = None

                        # Clear per-step caches (distance to current goal)
                        pbrs_calc._cached_distance = None
                        pbrs_calc._cached_start_node = None
                        pbrs_calc._cached_next_hop = None

                    # Clear level_data cache (contains old stage's entity positions)
                    self._cached_level_data = None
                    self._cached_entities = None
                    self._cached_switch_pos = None
                    self._cached_exit_pos = None

                    # Stage-aware level_ids automatically trigger cache rebuilds
                    # Just clear step cache for immediate effect
                    if hasattr(
                        self.reward_calculator.pbrs_calculator, "path_calculator"
                    ):
                        path_calc = (
                            self.reward_calculator.pbrs_calculator.path_calculator
                        )
                        # Clear step cache only (level cache will rebuild due to new level_id)
                        path_calc.clear_step_cache()
                        logger.debug(
                            f"Cleared step cache for curriculum stage {new_stage}. "
                            f"Level cache will rebuild automatically due to stage-aware level_id."
                        )

        # Add curriculum info if using custom map (for route visualization)
        if self.custom_map_path:
            info["curriculum_stage"] = "custom_map"
            info["curriculum_generator"] = "custom"

        # Episode-end reward metrics (only populated when the episode is actually done)
        if terminated or truncated:
            # FiGAR / precomputed PBRS: TensorBoard + PBRSLoggingCallback use primitive stats only
            if self._current_precomputed_pbrs is not None:
                info["episode_pbrs_metrics"] = (
                    self.primitive_reward_calculator.get_episode_reward_metrics()
                )
            elif hasattr(self.reward_calculator, "get_episode_reward_metrics"):
                info["episode_pbrs_metrics"] = (
                    self.reward_calculator.get_episode_reward_metrics()
                )

            # Surface-graph precomputed PBRS diagnostics
            if self._current_precomputed_pbrs is not None:
                sg = self.primitive_reward_calculator.get_episode_stats()
                sg["phi_at_spawn"] = self._phi_at_spawn
                sg["phi_at_switch_collection"] = self._phi_at_switch_collection
                sg["phi_at_terminal"] = self._phi_at_terminal
                sg["contact_floor_frames"] = self._contact_floor_frames
                sg["contact_wall_frames"] = self._contact_wall_frames
                sg["contact_airborne_frames"] = self._contact_airborne_frames
                if hasattr(self._current_precomputed_pbrs, "get_statistics"):
                    sg["vf_stats"] = self._current_precomputed_pbrs.get_statistics()
                else:
                    sg["vf_stats"] = {}
                # RoutePBRS route-tracking metrics (absent for SurfaceGraphPBRS)
                if hasattr(self._current_precomputed_pbrs, "get_route_metrics"):
                    sg["route_metrics"] = self._current_precomputed_pbrs.get_route_metrics()
                # Full episode return (includes truncation penalty on last step when timed out)
                sg["scaled_episode_return"] = float(self.current_ep_reward)
                info["episode_precomputed_pbrs_metrics"] = sg
                info["precomputed_pbrs_active"] = True
            else:
                info["precomputed_pbrs_active"] = False

        return info

    def _is_same_level_reset(self, options: Optional[Dict] = None) -> bool:
        """Detect if this reset is for the same level (enables fast reset path).

        Fast reset can be used when:
        1. The map has not changed since the last reset
        2. No new map loading is required
        3. Entities have already been created

        Returns:
            True if this is a same-level reset, False otherwise
        """
        # If skip_map_load is explicitly True, this is likely a same-level reset
        # (used by curriculum wrapper, test environment, etc.)
        if options is not None and options.get("skip_map_load", False):
            return True

        # Check if we're using a custom map (single-level training)
        if self.custom_map_path is not None:
            # Single map training - check if map name hasn't changed
            current_map_name = self.map_loader.current_map_name
            if (
                current_map_name is not None
                and current_map_name == self._last_reset_map_name
            ):
                return True

        # Otherwise, this is a new level (different map or first reset)
        return False

    def reset(self, seed=None, options=None):
        """Reset the environment.

        Args:
            seed: Random seed (optional)
            options: Dictionary of reset options. Supported keys:
                - skip_map_load (bool): If True, skip loading a new map.
                  Use this when map has already been loaded externally
                  (e.g., by curriculum wrapper, test_environment.py).
                - new_level (bool): If True, treat this as a new level requiring
                  full cache clearing. Defaults to True when skip_map_load=True.
                - map_name (str): Name of the externally loaded map (for cache keying).
                  When skip_map_load=True, this updates map_loader.current_map_name.
                - checkpoint: StateCheckpoint or CheckpointEntry to restore via action replay.
                  If provided, the environment will reset and replay the checkpoint's
                  action sequence to restore the exact game state.
        """
        # Clear observation cache on reset
        self._prev_obs_cache = None

        # Handle reinitialization after unpickling
        if hasattr(self, "_needs_reinit") and self._needs_reinit:
            # Reinitialize components that may have been affected by pickling
            self.observation_processor.reset()
            self.reward_calculator.reset()
            self._needs_reinit = False

        # Reset observation processor
        self.observation_processor.reset()

        # Reset reward calculator
        self.reward_calculator.reset()

        # Reset primitive reward calculator
        self.primitive_reward_calculator.reset()

        # Reset per-episode tracking for precomputed PBRS diagnostics
        self._phi_at_spawn: Optional[float] = None
        self._phi_at_switch_collection: Optional[float] = None
        self._phi_at_terminal: Optional[float] = None
        self._contact_floor_frames: int = 0
        self._contact_wall_frames: int = 0
        self._contact_airborne_frames: int = 0
        self._first_frame_of_episode: bool = True

        # Reset per-episode fallback counters so vf_stats reports per-episode
        # rates rather than lifetime cumulative fractions.
        if self._current_precomputed_pbrs is not None and hasattr(
            self._current_precomputed_pbrs, "reset_fallback_stats"
        ):
            self._current_precomputed_pbrs.reset_fallback_stats()

        # RoutePBRS: extract initial route from spawn position.
        # Must be called after the level is loaded (headless is ready) so that
        # ninja_position() returns the actual spawn coordinates.
        if self._current_precomputed_pbrs is not None and hasattr(
            self._current_precomputed_pbrs, "reset"
        ):
            try:
                pos = self.nplay_headless.ninja_position()
                vel = self.nplay_headless.ninja_velocity()
                sc = getattr(self.nplay_headless.sim.ninja, "state", None)
                if pos is not None and vel is not None:
                    self._current_precomputed_pbrs.reset(
                        float(pos[0]), float(pos[1]),
                        float(vel[0]), float(vel[1]),
                        contact_state=sc,
                    )
            except Exception as e:
                logger.debug(f"RoutePBRS.reset failed (non-fatal): {e}")

        # Reset truncation checker
        self.truncation_checker.reset()

        # Reset episode reward (CRITICAL: must be 0 at episode start)
        self.current_ep_reward = 0

        # SAFETY: Verify reward was properly accumulated in previous episode
        # This catches bugs where reward might not be reset or accumulated correctly
        if hasattr(self, "_last_episode_reward"):
            if self._last_episode_reward != 0 and self.enable_logging:
                logger.debug(
                    f"Previous episode reward: {self._last_episode_reward:.2f}, "
                    f"resetting to 0 for new episode"
                )

        # Increment episode counter for periodic cache clearing
        # Worker processes use this to trigger cache clearing every 100 episodes
        self._episode_count += 1

        # Reset last action tracking for death context
        self.last_action = None

        # Reset action sequence tracking for Go-Explore
        self._action_sequence = []
        self._checkpoint_replay_in_progress = False

        # Reset checkpoint route for visualization (will be populated if checkpoint replay)
        self._checkpoint_route = []

        # Reset checkpoint episode tracking (will be set if checkpoint used)
        self._from_checkpoint = False
        self._checkpoint_base_reward = 0.0
        self._checkpoint_source_frame_skip = self.frame_skip
        self._checkpoint_replay_frame_count = 0

        # Reset position tracking for route visualization
        self.current_route = []
        self._warned_about_position = False

        # Reset deterministic masking counters for episode-level tracking
        self._deterministic_mask_applied_count = 0
        self._total_steps_count = 0
        self._layer3_decision_points = 0
        self._layer3_masked_actions_total = 0
        self._layer3_masked_candidates_total = 0
        self._layer3_mask_override_count = 0
        self._layer3_all_masked_events = 0

        # Reset switch activation tracking
        self._switch_activation_frame = None
        self._switch_activation_step = None

        # Reset per-frame switch state tracker for surface-graph PBRS delta computation
        self._prev_switch_activated = False

        # Capture curriculum stage at EPISODE START for correct visualization
        # This is the stage that determines entity positions for THIS episode
        if self.goal_curriculum_manager is not None:
            self._episode_start_curriculum_stage = (
                self.goal_curriculum_manager.state.unified_stage
            )
        else:
            self._episode_start_curriculum_stage = None

        # Invalidate observation cache
        self._cached_observation = None

        # Invalidate level data cache on reset (new level loaded)
        # PERFORMANCE: Cache will be rebuilt on first access
        self._cached_level_data = None
        self._cached_entities = None

        # Invalidate static position caches (new level = new positions)
        self._cached_switch_pos = None
        self._cached_exit_pos = None
        self._cached_locked_doors = None
        self._cached_locked_door_switches = None
        self._cached_toggle_mines = None
        self._cached_tile_grid = None

        # Check if map loading should be skipped (e.g., curriculum already loaded one)
        skip_map_load = False
        checkpoint = None
        map_name = None
        if options is not None and isinstance(options, dict):
            skip_map_load = options.get("skip_map_load", False)
            checkpoint = options.get("checkpoint", None)
            map_name = options.get("map_name", None)

        if not skip_map_load:
            # Load map - this calls sim.load() which calls sim.reset()
            self.map_loader.load_map()
        else:
            # Update map_loader.current_map_name if provided
            # This is critical for cache keying and momentum waypoint loading
            if map_name is not None and hasattr(self, "map_loader"):
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

        # NOTE: Curriculum stage change logic removed from BaseNppEnvironment.reset()
        # NppEnvironment completely overrides reset() without calling super(), making
        # stage logic here unreachable dead code. Curriculum stage application now
        # happens in NppEnvironment.reset() where it can properly coordinate with
        # entity repositioning before graph building.

        # If checkpoint provided, replay action sequence to restore state
        if checkpoint is not None:
            return self._reset_to_checkpoint(checkpoint)

        # SIMPLIFIED: Removed path waypoint extraction - no longer using waypoint system

        # Track initial position for route visualization
        try:
            position = self.nplay_headless.ninja_position()
            if position is not None:
                self.current_route.append((float(position[0]), float(position[1])))
        except Exception as e:
            logger.warning(f"Could not get initial position for route tracking: {e}")

        # Get initial observation and process it
        initial_obs = self._get_observation()
        processed_obs = self._process_observation(initial_obs)

        return (processed_obs, {})

    def _reset_to_checkpoint(self, checkpoint) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset environment to a checkpoint state via action replay.

        Since N++ physics is fully deterministic, replaying the same action
        sequence from spawn will always produce identical results.

        Args:
            checkpoint: StateCheckpoint or CheckpointEntry with action_sequence

        Returns:
            Tuple of (processed_observation, info_dict)
        """
        self._checkpoint_replay_in_progress = True

        # Extract action sequence from checkpoint
        action_sequence = getattr(checkpoint, "action_sequence", None)

        # Check if action_sequence is empty (avoid numpy array boolean ambiguity)
        # Use explicit length check instead of "if not action_sequence"
        if action_sequence is None:
            has_actions = False
        elif hasattr(action_sequence, "__len__"):
            has_actions = len(action_sequence) > 0
        else:
            has_actions = False

        if not has_actions:
            logger.warning("Checkpoint has empty action_sequence, starting from spawn")
            self._checkpoint_replay_in_progress = False
            # Get initial observation and return
            initial_obs = self._get_observation()
            processed_obs = self._process_observation(initial_obs)
            return (processed_obs, {"checkpoint_replay": False, "replay_frames": 0})

        # Convert numpy array to list if needed (do this early to avoid further ambiguity)
        if hasattr(action_sequence, "tolist"):
            action_sequence = action_sequence.tolist()
        elif not isinstance(action_sequence, list):
            action_sequence = list(action_sequence)

        # Mark this episode as starting from checkpoint for logging/analysis
        self._from_checkpoint = True
        self._checkpoint_base_reward = getattr(checkpoint, "cumulative_reward", 0.0)
        self._checkpoint_source_frame_skip = getattr(
            checkpoint, "source_frame_skip", self.frame_skip
        )

        # Track replay statistics for debugging
        actions_replayed = 0
        frames_replayed = 0
        ninja_died_during_replay = False

        # Track positions during checkpoint replay for visualization
        # This captures the path taken during action replay
        checkpoint_route = []

        # Get spawn position for debugging and add to checkpoint route
        spawn_pos = self.nplay_headless.ninja_position()
        if spawn_pos is not None:
            checkpoint_route.append((float(spawn_pos[0]), float(spawn_pos[1])))

        # DEBUG: Log checkpoint replay details
        # checkpoint_cell = getattr(checkpoint, "cell", None)
        checkpoint_expected_pos = getattr(checkpoint, "ninja_position", None)
        # checkpoint_frame_count = getattr(checkpoint, "frame_count", 0)

        # CRITICAL: Use checkpoint's source_frame_skip for accurate replay
        # Training checkpoints use training frame_skip (e.g., 4)
        # Demo checkpoints use frame_skip=1 (recorded at 60fps)
        checkpoint_frame_skip = getattr(
            checkpoint, "source_frame_skip", self.frame_skip
        )

        # DEBUG: Log checkpoint type and attributes to diagnose pickling issues
        checkpoint_source = getattr(checkpoint, "source", "unknown")
        if checkpoint_source == "demo" and checkpoint_frame_skip != 1:
            logger.warning(
                f"[CHECKPOINT_FRAME_SKIP_BUG] Demo checkpoint has wrong frame_skip! "
                f"source={checkpoint_source}, source_frame_skip={checkpoint_frame_skip} (expected 1), "
                f"checkpoint_type={type(checkpoint).__name__}, "
                f"actions={len(action_sequence)}, "
                f"has_source_frame_skip={'source_frame_skip' in dir(checkpoint)}"
            )
            # Force correct frame_skip for demo checkpoints
            checkpoint_frame_skip = 1
            logger.info(
                "[CHECKPOINT_FRAME_SKIP_BUG] Forcing frame_skip=1 for demo checkpoint"
            )

        # print(
        #     f"[CHECKPOINT REPLAY] Starting replay: "
        #     f"cell={checkpoint_cell}, actions={len(action_sequence)}, "
        #     f"expected_pos={checkpoint_expected_pos}, "
        #     f"frame_count={checkpoint_frame_count}, "
        #     f"spawn={spawn_pos}, training_frame_skip={self.frame_skip}, "
        #     f"checkpoint_frame_skip={checkpoint_frame_skip}"
        # )

        # CRITICAL: Restore entity positions AND switch state to what they were when checkpoint was created
        # This is essential for curriculum learning where entity positions change between stages
        # and for scenarios where switch activation state differs between episodes
        checkpoint_switch_pos = getattr(checkpoint, "curriculum_switch_position", None)
        checkpoint_exit_pos = getattr(checkpoint, "curriculum_exit_position", None)
        checkpoint_switch_activated = getattr(checkpoint, "switch_activated", False)
        checkpoint_locked_door_states = getattr(checkpoint, "locked_door_states", {})

        # DIAGNOSTIC: Log checkpoint curriculum positions to help debug position mismatches
        logger.info(
            f"[CHECKPOINT REPLAY] Checkpoint positions - "
            f"switch_pos={checkpoint_switch_pos}, exit_pos={checkpoint_exit_pos}, "
            f"switch_activated={checkpoint_switch_activated}"
        )
        if checkpoint_switch_pos is None or checkpoint_exit_pos is None:
            # CRITICAL FIX: If curriculum is active but checkpoint has no positions, abort replay
            # Replaying without proper entity positions will cause position mismatches
            if (
                hasattr(self, "goal_curriculum_manager")
                and self.goal_curriculum_manager is not None
            ):
                logger.error(
                    "[CHECKPOINT REPLAY] Cannot replay checkpoint without curriculum positions "
                    "when curriculum is active. Checkpoint missing positions: "
                    f"switch={checkpoint_switch_pos}, exit={checkpoint_exit_pos}. "
                    "Falling back to normal spawn reset."
                )
                self._checkpoint_replay_in_progress = False
                # Fall back to normal reset from spawn
                return self.reset(seed=None, options=None)

            # Curriculum disabled - log warning and proceed
            logger.warning(
                "[CHECKPOINT REPLAY] Checkpoint missing curriculum positions, "
                f"but curriculum is disabled. Checkpoint positions: switch={checkpoint_switch_pos}, exit={checkpoint_exit_pos}. "
                "Proceeding with replay using current entity positions."
            )

        # Save current entity positions so we can restore them after replay
        # NOTE: We do NOT save/restore activation states - those persist from replay
        saved_switch_pos = None
        saved_exit_pos = None
        entity_state_modified = False

        # Always restore entity state (positions and/or switch activation)
        if True:
            try:
                from nclone.physics import clamp_cell
                import math

                sim = self.nplay_headless.sim
                exit_switches = sim.entity_dic.get(3, [])

                if exit_switches:
                    switch_entity = exit_switches[-1]
                    # Save current entity positions only
                    saved_switch_pos = (switch_entity.xpos, switch_entity.ypos)

                    # Restore switch activation state to checkpoint creation state
                    # This is critical: if checkpoint was created pre-activation but we're
                    # currently post-activation, the replay will fail
                    # NOTE: This state will PERSIST after replay - we do NOT restore it!
                    if hasattr(switch_entity, "activated"):
                        switch_entity.activated = checkpoint_switch_activated
                        entity_state_modified = True
                        logger.debug(
                            f"Set switch activation for replay (will persist): "
                            f"activated → {checkpoint_switch_activated}"
                        )

                    # Restore locked door states to checkpoint creation state
                    # These states will PERSIST after replay - we do NOT restore them!
                    door_entities = sim.entity_dic.get(4, [])
                    for door in door_entities:
                        door_id = getattr(door, "id", None)
                        if door_id and hasattr(door, "open"):
                            door_key = f"locked_door_{door_id}"

                            # Restore to checkpoint state if available
                            if door_key in checkpoint_locked_door_states:
                                checkpoint_door_state = checkpoint_locked_door_states[
                                    door_key
                                ]
                                door.open = checkpoint_door_state
                                entity_state_modified = True
                                logger.debug(
                                    f"Set door state for replay (will persist): "
                                    f"{door_key} open={checkpoint_door_state}"
                                )

                    # Restore to checkpoint creation position if available
                    if checkpoint_switch_pos is not None:
                        old_cell = switch_entity.cell
                        switch_entity.xpos = checkpoint_switch_pos[0]
                        switch_entity.ypos = checkpoint_switch_pos[1]

                        # Update grid placement
                        new_cell = clamp_cell(
                            math.floor(checkpoint_switch_pos[0] / 24),
                            math.floor(checkpoint_switch_pos[1] / 24),
                        )
                        if new_cell != old_cell and old_cell in sim.grid_entity:
                            if switch_entity in sim.grid_entity[old_cell]:
                                sim.grid_entity[old_cell].remove(switch_entity)
                            switch_entity.cell = new_cell
                            if new_cell not in sim.grid_entity:
                                sim.grid_entity[new_cell] = []
                            sim.grid_entity[new_cell].append(switch_entity)

                        entity_state_modified = True
                        logger.info(
                            f"[CHECKPOINT REPLAY] Set switch to checkpoint position for replay: "
                            f"{saved_switch_pos} → {checkpoint_switch_pos}"
                        )
                    else:
                        logger.warning(
                            f"[CHECKPOINT REPLAY] No checkpoint_switch_pos! Switch stays at: "
                            f"{saved_switch_pos}. Checkpoint may not have curriculum positions."
                        )

                    # Restore exit door position if available
                    if hasattr(switch_entity, "parent") and switch_entity.parent:
                        door_entity = switch_entity.parent
                        saved_exit_pos = (door_entity.xpos, door_entity.ypos)

                        if checkpoint_exit_pos is not None:
                            old_cell = door_entity.cell
                            door_entity.xpos = checkpoint_exit_pos[0]
                            door_entity.ypos = checkpoint_exit_pos[1]

                            # Update cell attribute but do NOT add to grid_entity!
                            # Exit door should only be in grid after switch activation
                            new_cell = clamp_cell(
                                math.floor(checkpoint_exit_pos[0] / 24),
                                math.floor(checkpoint_exit_pos[1] / 24),
                            )
                            if new_cell != old_cell and old_cell in sim.grid_entity:
                                if door_entity in sim.grid_entity[old_cell]:
                                    sim.grid_entity[old_cell].remove(door_entity)
                            # Also remove from new cell if present
                            if new_cell in sim.grid_entity:
                                if door_entity in sim.grid_entity[new_cell]:
                                    sim.grid_entity[new_cell].remove(door_entity)
                            door_entity.cell = new_cell
                            # NOTE: Do NOT append to grid_entity - awaiting switch

                            entity_state_modified = (
                                True  # Mark that we modified exit position
                            )
                            logger.info(
                                f"[CHECKPOINT REPLAY] Set exit to checkpoint position for replay: "
                                f"{saved_exit_pos} → {checkpoint_exit_pos} (not in grid until switch)"
                            )
                        else:
                            logger.warning(
                                f"[CHECKPOINT REPLAY] No checkpoint_exit_pos! Exit stays at: "
                                f"{saved_exit_pos}. Checkpoint may not have curriculum positions."
                            )
            except Exception as e:
                logger.warning(f"Failed to restore entity positions for replay: {e}")

        # DIAGNOSTIC: Log actual entity positions BEFORE replay starts
        # This verifies that entity positions were correctly set to checkpoint positions
        try:
            sim = self.nplay_headless.sim
            exit_switches = sim.entity_dic.get(3, [])
            if exit_switches:
                switch_entity = exit_switches[-1]
                actual_switch_pos = (switch_entity.xpos, switch_entity.ypos)
                actual_exit_pos = None
                if hasattr(switch_entity, "parent") and switch_entity.parent:
                    door_entity = switch_entity.parent
                    actual_exit_pos = (door_entity.xpos, door_entity.ypos)
                logger.info(
                    f"[CHECKPOINT REPLAY] BEFORE REPLAY - Actual entity positions: "
                    f"switch={actual_switch_pos}, exit={actual_exit_pos}"
                )
        except Exception as e:
            logger.warning(f"Failed to log entity positions before replay: {e}")

        # Replay action sequence with frame skip matching original checkpoint creation
        # Training checkpoints: use training frame_skip (e.g., 4 frames/action)
        # Demo checkpoints: use frame_skip=1 (60fps, 1 frame/action)
        hor_input, jump_input = 0, 0
        for action in action_sequence:
            hor_input, jump_input = self._actions_to_execute(action)
            # Replay for checkpoint_frame_skip ticks (not self.frame_skip!)
            for _ in range(checkpoint_frame_skip):
                self.nplay_headless.tick(hor_input, jump_input)
                frames_replayed += 1

                # Track position for checkpoint route visualization
                pos = self.nplay_headless.ninja_position()
                if pos is not None:
                    checkpoint_route.append((float(pos[0]), float(pos[1])))

                # Check if ninja died during replay - THIS SHOULD NEVER HAPPEN
                if self.nplay_headless.ninja_has_died():
                    ninja_died_during_replay = True
                    checkpoint_cell = getattr(checkpoint, "cell", None)
                    checkpoint_source = getattr(checkpoint, "source", "unknown")
                    checkpoint_level_id = getattr(checkpoint, "level_id", None)

                    error_msg = (
                        f"[CRITICAL BUG] Ninja died during checkpoint replay! "
                        f"This should NEVER happen with deterministic physics. "
                        f"frame={frames_replayed}, action={actions_replayed + 1}/{len(action_sequence)}, "
                        f"checkpoint_cell={checkpoint_cell}, source={checkpoint_source}, "
                        f"level_id={checkpoint_level_id}, spawn={spawn_pos}, "
                        f"checkpoint_frame_skip={checkpoint_frame_skip}, training_frame_skip={self.frame_skip}, "
                        f"first_10_actions={action_sequence[:10]}, last_10_actions={action_sequence[-10:]}"
                    )

                    logger.error(error_msg)
                    print(f"\n{'=' * 80}")
                    print(error_msg)
                    print(f"{'=' * 80}\n")

                    # FAIL FAST: Raise exception to halt training and force bug fix
                    raise RuntimeError(
                        f"Checkpoint replay caused ninja death (checkpoint corruption bug). "
                        f"Cell={checkpoint_cell}, actions={len(action_sequence)}, "
                        f"died_at_action={actions_replayed + 1}. "
                        f"This indicates corrupted action buffer. See logs for details."
                    )

            # Track action in current sequence (checkpoint actions become our history)
            self._action_sequence.append(action)
            actions_replayed += 1

            if ninja_died_during_replay:
                break

        self._checkpoint_replay_in_progress = False

        # CRITICAL FIX: DO NOT overwrite entity positions after checkpoint replay!
        # The checkpoint was created with entities at specific curriculum stage positions,
        # and those positions MUST persist for the entire episode. Overwriting them with
        # current curriculum positions breaks the deterministic replay guarantee.
        #
        # Entity positions were already correctly set BEFORE replay (lines 1802-1868).
        # Switch activation states correctly persist from replay (not restored).
        #
        # Only invalidate position caches to ensure observations reflect checkpoint positions
        if (
            hasattr(self, "goal_curriculum_manager")
            and self.goal_curriculum_manager is not None
        ):
            # CRITICAL: Invalidate position caches so observations use checkpoint entity positions
            # Without this, cached positions from pre-replay would be stale
            self._cached_switch_pos = None
            self._cached_exit_pos = None

            logger.info(
                f"[CHECKPOINT REPLAY] Keeping checkpoint entity positions: "
                f"switch={checkpoint_switch_pos}, exit={checkpoint_exit_pos} "
                f"(NOT overwriting with current curriculum stage positions)"
            )
        elif entity_state_modified:
            # Fallback: restore to saved positions if no curriculum manager
            try:
                from nclone.physics import clamp_cell
                import math

                sim = self.nplay_headless.sim
                exit_switches = sim.entity_dic.get(3, [])

                if exit_switches and saved_switch_pos is not None:
                    switch_entity = exit_switches[-1]
                    old_cell = switch_entity.cell

                    switch_entity.xpos = saved_switch_pos[0]
                    switch_entity.ypos = saved_switch_pos[1]

                    new_cell = clamp_cell(
                        math.floor(saved_switch_pos[0] / 24),
                        math.floor(saved_switch_pos[1] / 24),
                    )
                    if new_cell != old_cell and old_cell in sim.grid_entity:
                        if switch_entity in sim.grid_entity[old_cell]:
                            sim.grid_entity[old_cell].remove(switch_entity)
                        switch_entity.cell = new_cell
                        if new_cell not in sim.grid_entity:
                            sim.grid_entity[new_cell] = []
                        sim.grid_entity[new_cell].append(switch_entity)

                    # Restore exit door
                    if (
                        saved_exit_pos is not None
                        and hasattr(switch_entity, "parent")
                        and switch_entity.parent
                    ):
                        door_entity = switch_entity.parent
                        old_cell = door_entity.cell

                        door_entity.xpos = saved_exit_pos[0]
                        door_entity.ypos = saved_exit_pos[1]

                        # Update cell but do NOT add to grid_entity - awaiting switch
                        new_cell = clamp_cell(
                            math.floor(saved_exit_pos[0] / 24),
                            math.floor(saved_exit_pos[1] / 24),
                        )
                        if new_cell != old_cell and old_cell in sim.grid_entity:
                            if door_entity in sim.grid_entity[old_cell]:
                                sim.grid_entity[old_cell].remove(door_entity)
                        # Also remove from new cell if present
                        if new_cell in sim.grid_entity:
                            if door_entity in sim.grid_entity[new_cell]:
                                sim.grid_entity[new_cell].remove(door_entity)
                        door_entity.cell = new_cell
                        # NOTE: Do NOT append to grid_entity - awaiting switch

                    # CRITICAL: Invalidate position caches since entity positions changed!
                    self._cached_switch_pos = None
                    self._cached_exit_pos = None

                    logger.warning(
                        f"[CHECKPOINT REPLAY] No curriculum manager, restored to saved positions: "
                        f"switch={saved_switch_pos}, exit={saved_exit_pos}"
                    )
            except Exception as e:
                logger.warning(f"Failed to restore entity state after replay: {e}")

        # Track frames executed during checkpoint replay
        # This allows us to report only episode frames (excluding replay) in info dict
        self._checkpoint_replay_frame_count = frames_replayed

        # Store checkpoint route for inclusion in episode end info
        # This persists through the episode so route visualization can access it
        self._checkpoint_route = checkpoint_route

        # Track position after replay for current_route (exploration phase starts here)
        actual_position = None
        try:
            actual_position = self.nplay_headless.ninja_position()
            if actual_position is not None:
                # Start current_route fresh for exploration phase
                self.current_route.append(
                    (float(actual_position[0]), float(actual_position[1]))
                )
        except Exception as e:
            logger.warning(f"Could not get position after checkpoint replay: {e}")

        # # DEBUG: Log replay completion
        # checkpoint_expected_pos = getattr(checkpoint, "ninja_position", None)
        # checkpoint_frame_count = getattr(checkpoint, "frame_count", 0)
        # print(
        #     f"[CHECKPOINT REPLAY] Complete: "
        #     f"actions={actions_replayed}/{len(action_sequence)}, "
        #     f"frames={frames_replayed}, died={ninja_died_during_replay}, "
        #     f"expected_pos={checkpoint_expected_pos}, actual_pos={actual_position}, "
        #     f"expected_frames={checkpoint_frame_count}"
        # )

        # Validate position if checkpoint has expected position
        expected_position = checkpoint_expected_pos
        position_valid = True
        if expected_position is not None and actual_position is not None:
            dx = actual_position[0] - expected_position[0]
            dy = actual_position[1] - expected_position[1]
            error = (dx * dx + dy * dy) ** 0.5

            # TOLERANCE: 5px accounts for spawn position drift between creation and replay
            # Spawn positions can vary by ~0.5px, and with many actions this compounds
            # Real corruption (wrong action sequences) causes 100+ px errors and death
            position_valid = error < 5.0
            if not position_valid:
                # Log spawn position comparison to diagnose drift
                spawn_dx = spawn_pos[0] - 312.0 if spawn_pos else 0
                spawn_dy = spawn_pos[1] - 444.0 if spawn_pos else 0
                spawn_drift = (spawn_dx * spawn_dx + spawn_dy * spawn_dy) ** 0.5

                logger.warning(
                    f"Checkpoint replay position mismatch: "
                    f"expected={expected_position}, actual={actual_position}, "
                    f"error={error:.6f}px, spawn={spawn_pos}, spawn_drift={spawn_drift:.2f}px | "
                    f"actions={actions_replayed}/{len(action_sequence)}, "
                    f"frames={frames_replayed}, died={ninja_died_during_replay}"
                )

        # Get observation after replay
        initial_obs = self._get_observation()
        processed_obs = self._process_observation(initial_obs)

        # Get current position and velocity for info dict (CRITICAL FIX)
        # This ensures the info dict has the correct position after checkpoint replay,
        # preventing stale terminal positions from previous episode from persisting
        current_pos = self.nplay_headless.ninja_position()
        current_vel = self.nplay_headless.ninja_velocity()

        # Determine if replay failed and why
        replay_failed = ninja_died_during_replay or not position_valid
        replay_failure_reason = None
        if ninja_died_during_replay:
            replay_failure_reason = "ninja_died"
        elif not position_valid:
            replay_failure_reason = "position_mismatch"

        info = {
            "checkpoint_replay": True,
            "replay_frames": len(action_sequence),
            "position_valid": position_valid,
            "checkpoint_cell": getattr(checkpoint, "cell", None),
            "checkpoint_distance": getattr(checkpoint, "distance_to_goal", None),
            "checkpoint_route": checkpoint_route,  # Path taken during checkpoint replay
            "checkpoint_source": getattr(
                checkpoint, "source", "unknown"
            ),  # Demo vs agent checkpoint
            "replay_failed": replay_failed,
            "replay_failure_reason": replay_failure_reason,
            # CRITICAL FIX: Include position so info dict has correct position after replay
            # Without this, stale terminal position from previous episode persists, causing
            # corrupted checkpoints to be created with wrong position but checkpoint's action buffer
            "player_x": current_pos[0] if current_pos else 0.0,
            "player_y": current_pos[1] if current_pos else 0.0,
            "player_xspeed": current_vel[0] if current_vel else 0.0,
            "player_yspeed": current_vel[1] if current_vel else 0.0,
        }

        if self.enable_logging:
            logger.debug(
                f"Checkpoint replay complete: {len(action_sequence)} actions, "
                f"position_valid={position_valid}, "
                f"checkpoint_route has {len(checkpoint_route)} positions"
            )

        return (processed_obs, info)

    def get_current_action_sequence(self) -> List[int]:
        """Get the action sequence since last reset.

        Used by Go-Explore callback to save checkpoints with action sequences
        for deterministic replay.

        Returns:
            List of action indices executed since last reset
        """
        return list(self._action_sequence)

    def get_action_sequence_length(self) -> int:
        """Get the number of actions in current episode.

        Returns:
            Number of actions executed since last reset
        """
        return len(self._action_sequence)

    def is_checkpoint_replay_active(self) -> bool:
        """Check if checkpoint replay is currently in progress.

        Returns:
            True if currently replaying a checkpoint's action sequence
        """
        return self._checkpoint_replay_in_progress

    def _reset_to_checkpoint_from_wrapper(
        self, checkpoint
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset to checkpoint from VecEnv wrapper (for Go-Explore integration).

        This method is called via env_method from GoExploreVecEnv wrapper
        to apply checkpoint resets after environment auto-resets.

        Since the environment has already auto-reset (VecEnv behavior after done=True),
        we replay the checkpoint's action sequence from the current spawn state.

        Args:
            checkpoint: CheckpointEntry or StateCheckpoint with action_sequence

        Returns:
            Tuple of (processed_observation, info_dict)
        """
        # The environment is already at spawn (auto-reset happened)
        # We just need to replay the action sequence
        action_sequence = getattr(checkpoint, "action_sequence", None)

        # Check if action_sequence is empty (avoid numpy array boolean ambiguity)
        # Use explicit length check instead of "if not action_sequence"
        if action_sequence is None:
            has_actions = False
        elif hasattr(action_sequence, "__len__"):
            has_actions = len(action_sequence) > 0
        else:
            has_actions = False

        if not has_actions:
            # No actions to replay - just return current state
            obs = self._get_observation()
            processed_obs = self._process_observation(obs)
            return (processed_obs, {"checkpoint_replay": False, "replay_frames": 0})

        # Convert numpy array to list if needed (do this early to avoid further ambiguity)
        if hasattr(action_sequence, "tolist"):
            action_sequence = action_sequence.tolist()
        elif not isinstance(action_sequence, list):
            action_sequence = list(action_sequence)

        self._checkpoint_replay_in_progress = True

        # Mark this episode as starting from checkpoint for logging/analysis
        self._from_checkpoint = True
        self._checkpoint_base_reward = getattr(checkpoint, "cumulative_reward", 0.0)
        self._checkpoint_source_frame_skip = getattr(
            checkpoint, "source_frame_skip", self.frame_skip
        )

        # CRITICAL FIX: Explicitly reset to level spawn before replaying actions
        # The assumption that "VecEnv auto-reset already happened" is WRONG when
        # checkpoints are retrieved at episode START (after 1 action was taken).
        # We must reset to spawn to ensure deterministic replay.
        self.nplay_headless.reset()

        # CRITICAL: Restore checkpoint curriculum entity positions AFTER reset
        # The reset() call above loads entities at ORIGINAL positions, but we need
        # them at the CHECKPOINT's curriculum positions for accurate replay
        checkpoint_switch_pos = getattr(checkpoint, "curriculum_switch_position", None)
        checkpoint_exit_pos = getattr(checkpoint, "curriculum_exit_position", None)
        checkpoint_switch_activated = getattr(checkpoint, "switch_activated", False)

        if checkpoint_switch_pos is not None and checkpoint_exit_pos is not None:
            try:
                from nclone.physics import clamp_cell
                import math

                sim = self.nplay_headless.sim
                exit_entities = sim.entity_dic.get(3, [])

                # Find switch entity (type-safe)
                switch_entity = None
                for entity in reversed(exit_entities):
                    if type(entity).__name__ == "EntityExitSwitch":
                        switch_entity = entity
                        break

                if switch_entity:
                    # Restore switch position
                    old_cell = switch_entity.cell
                    switch_entity.xpos = checkpoint_switch_pos[0]
                    switch_entity.ypos = checkpoint_switch_pos[1]

                    # Update grid placement
                    new_cell = clamp_cell(
                        math.floor(checkpoint_switch_pos[0] / 24),
                        math.floor(checkpoint_switch_pos[1] / 24),
                    )
                    if new_cell != old_cell:
                        if (
                            old_cell in sim.grid_entity
                            and switch_entity in sim.grid_entity[old_cell]
                        ):
                            sim.grid_entity[old_cell].remove(switch_entity)
                        switch_entity.cell = new_cell
                        if new_cell not in sim.grid_entity:
                            sim.grid_entity[new_cell] = []
                        sim.grid_entity[new_cell].append(switch_entity)

                    # Restore switch activation state
                    if hasattr(switch_entity, "activated"):
                        switch_entity.activated = checkpoint_switch_activated

                    # Restore exit door position
                    if hasattr(switch_entity, "parent") and switch_entity.parent:
                        door_entity = switch_entity.parent
                        old_cell = door_entity.cell
                        door_entity.xpos = checkpoint_exit_pos[0]
                        door_entity.ypos = checkpoint_exit_pos[1]

                        # Update cell attribute (do NOT add to grid_entity until switch activates)
                        new_cell = clamp_cell(
                            math.floor(checkpoint_exit_pos[0] / 24),
                            math.floor(checkpoint_exit_pos[1] / 24),
                        )
                        if new_cell != old_cell:
                            if (
                                old_cell in sim.grid_entity
                                and door_entity in sim.grid_entity[old_cell]
                            ):
                                sim.grid_entity[old_cell].remove(door_entity)
                            if (
                                new_cell in sim.grid_entity
                                and door_entity in sim.grid_entity[new_cell]
                            ):
                                sim.grid_entity[new_cell].remove(door_entity)
                        door_entity.cell = new_cell

                    logger.info(
                        f"[CHECKPOINT WRAPPER] Restored curriculum positions: "
                        f"switch={checkpoint_switch_pos}, exit={checkpoint_exit_pos}"
                    )
            except Exception as e:
                logger.error(f"Failed to restore checkpoint curriculum positions: {e}")

            # Invalidate position caches so observations use checkpoint positions
            self._cached_switch_pos = None
            self._cached_exit_pos = None
        else:
            logger.warning(
                f"[CHECKPOINT WRAPPER] Checkpoint missing curriculum positions! "
                f"switch={checkpoint_switch_pos}, exit={checkpoint_exit_pos}. "
                f"Entities will be at original level positions during replay."
            )

        # CRITICAL: Clear current_route to remove spawn position added by auto-reset
        # Checkpoint episodes should track positions starting from checkpoint, not spawn
        self.current_route = []

        # Track replay statistics for debugging
        actions_replayed = 0
        frames_replayed = 0
        ninja_died_during_replay = False

        # Track positions during checkpoint replay for visualization
        # This captures the path taken during action replay
        checkpoint_route = []

        # Get spawn position for debugging and add to checkpoint route
        # Now this will be the actual level spawn (312, 444) after the reset
        spawn_pos = self.nplay_headless.ninja_position()
        if spawn_pos is not None:
            checkpoint_route.append((float(spawn_pos[0]), float(spawn_pos[1])))

        # DEBUG: Log checkpoint replay start (wrapper method)
        # checkpoint_cell = getattr(checkpoint, "cell", None)
        # checkpoint_expected_pos = getattr(checkpoint, "ninja_position", None)
        # checkpoint_frame_count = getattr(checkpoint, "frame_count", 0)

        # CRITICAL: Use checkpoint's source_frame_skip for accurate replay
        # Training checkpoints use training frame_skip (e.g., 4)
        # Demo checkpoints use frame_skip=1 (recorded at 60fps)
        checkpoint_frame_skip = getattr(
            checkpoint, "source_frame_skip", self.frame_skip
        )

        # DEBUG: Log checkpoint type and attributes to diagnose pickling issues
        checkpoint_source = getattr(checkpoint, "source", "unknown")
        if checkpoint_source == "demo" and checkpoint_frame_skip != 1:
            logger.warning(
                f"[CHECKPOINT_FRAME_SKIP_BUG] Demo checkpoint has wrong frame_skip! "
                f"source={checkpoint_source}, source_frame_skip={checkpoint_frame_skip} (expected 1), "
                f"checkpoint_type={type(checkpoint).__name__}, "
                f"actions={len(action_sequence)}, "
                f"has_source_frame_skip={'source_frame_skip' in dir(checkpoint)}"
            )
            # Force correct frame_skip for demo checkpoints
            checkpoint_frame_skip = 1
            logger.info(
                "[CHECKPOINT_FRAME_SKIP_BUG] Forcing frame_skip=1 for demo checkpoint"
            )

        # DEBUG: Log detailed replay info to diagnose position mismatch
        checkpoint_cell = getattr(checkpoint, "cell", None)
        checkpoint_expected_pos = getattr(checkpoint, "ninja_position", None)
        checkpoint_level_id = getattr(checkpoint, "level_id", None)

        # Enable debug logging via environment variable or enable_logging flag
        import os

        _verbose_debug = self.enable_logging or os.environ.get(
            "CHECKPOINT_REPLAY_DEBUG", ""
        ).lower() in ("1", "true", "yes")
        if _verbose_debug:
            logger.error(
                f"[REPLAY_DEBUG] Starting checkpoint replay: "
                f"cell={checkpoint_cell}, source={checkpoint_source}, "
                f"spawn_pos={spawn_pos}, expected_pos={checkpoint_expected_pos}, "
                f"actions={len(action_sequence)}, frame_skip={checkpoint_frame_skip}, "
                f"level_id={checkpoint_level_id}, "
                f"first_10_actions={list(action_sequence[:10])}"
            )

        # Track positions during replay for debugging (first 10 only)
        _debug_positions = []

        # Replay action sequence with frame skip matching original checkpoint creation
        # Training checkpoints: use training frame_skip (e.g., 4 frames/action)
        # Demo checkpoints: use frame_skip=1 (60fps, 1 frame/action)
        for action in action_sequence:
            hor_input, jump_input = self._actions_to_execute(action)
            # Replay for checkpoint_frame_skip ticks (not self.frame_skip!)
            for _ in range(checkpoint_frame_skip):
                self.nplay_headless.tick(hor_input, jump_input)
                frames_replayed += 1

                # Track position for checkpoint route visualization
                pos = self.nplay_headless.ninja_position()
                if pos is not None:
                    checkpoint_route.append((float(pos[0]), float(pos[1])))

                # Check if ninja died during replay - THIS SHOULD NEVER HAPPEN
                if self.nplay_headless.ninja_has_died():
                    ninja_died_during_replay = True
                    checkpoint_cell = getattr(checkpoint, "cell", None)
                    checkpoint_source = getattr(checkpoint, "source", "unknown")
                    checkpoint_level_id = getattr(checkpoint, "level_id", None)

                    error_msg = (
                        f"[CRITICAL BUG] Ninja died during checkpoint replay! "
                        f"This should NEVER happen with deterministic physics. "
                        f"frame={frames_replayed}, action={actions_replayed + 1}/{len(action_sequence)}, "
                        f"checkpoint_cell={checkpoint_cell}, source={checkpoint_source}, "
                        f"level_id={checkpoint_level_id}, spawn={spawn_pos}, "
                        f"checkpoint_frame_skip={checkpoint_frame_skip}, training_frame_skip={self.frame_skip}, "
                        f"first_10_actions={action_sequence[:10]}, last_10_actions={action_sequence[-10:]}"
                    )

                    logger.error(error_msg)
                    print(f"\n{'=' * 80}")
                    print(error_msg)
                    print(f"{'=' * 80}\n")

                    # FAIL FAST: Raise exception to halt training and force bug fix
                    raise RuntimeError(
                        f"Checkpoint replay caused ninja death (checkpoint corruption bug). "
                        f"Cell={checkpoint_cell}, actions={len(action_sequence)}, "
                        f"died_at_action={actions_replayed + 1}. "
                        f"This indicates corrupted action buffer. See logs for details."
                    )

            self._action_sequence.append(action)
            actions_replayed += 1

            # Track first 10 positions for debugging position mismatch
            if actions_replayed <= 10:
                pos = self.nplay_headless.ninja_position()
                if pos is not None:
                    _debug_positions.append(
                        (actions_replayed, action, (pos[0], pos[1]))
                    )

            if ninja_died_during_replay:
                break

        self._checkpoint_replay_in_progress = False

        # Track frames executed during checkpoint replay
        # This allows us to report only episode frames (excluding replay) in info dict
        self._checkpoint_replay_frame_count = frames_replayed

        # DEBUG: Log position trajectory for first 10 actions
        if _verbose_debug and _debug_positions:
            pos_log = ", ".join(
                [f"a{i}:{a}->({p[0]:.1f},{p[1]:.1f})" for i, a, p in _debug_positions]
            )
            logger.error(f"[REPLAY_DEBUG] First 10 positions: {pos_log}")

        # DEBUG: Log replay completion
        actual_position = self.nplay_headless.ninja_position()
        # print(
        #     f"[CHECKPOINT REPLAY FROM WRAPPER] Complete: "
        #     f"actions={actions_replayed}/{len(action_sequence)}, "
        #     f"frames={frames_replayed}, died={ninja_died_during_replay}, "
        #     f"actual_pos={actual_position}, expected_frames={checkpoint_frame_count}"
        # )

        # Track frames executed during checkpoint replay
        # This allows us to report only episode frames (excluding replay) in info dict
        self._checkpoint_replay_frame_count = frames_replayed

        # Store checkpoint route for inclusion in episode end info
        # This persists through the episode so route visualization can access it
        self._checkpoint_route = checkpoint_route

        # Track position after replay for current_route (exploration phase starts here)
        actual_position = None
        try:
            actual_position = self.nplay_headless.ninja_position()
            if actual_position is not None:
                # Start current_route fresh for exploration phase
                self.current_route.append(
                    (float(actual_position[0]), float(actual_position[1]))
                )
        except Exception as e:
            logger.warning(f"Could not get position after checkpoint replay: {e}")

        # Validate position if checkpoint has expected position
        expected_position = getattr(checkpoint, "ninja_position", None)
        position_valid = True
        if expected_position is not None and actual_position is not None:
            dx = actual_position[0] - expected_position[0]
            dy = actual_position[1] - expected_position[1]
            error = (dx * dx + dy * dy) ** 0.5

            # TOLERANCE: 5px accounts for spawn position drift between creation and replay
            # Spawn positions can vary by ~0.5px, and with many actions this compounds
            # Real corruption (wrong action sequences) causes 100+ px errors and death
            position_valid = error < 5.0

            # DEBUG: Always log position comparison when logging enabled
            if _verbose_debug:
                logger.error(
                    f"[REPLAY_DEBUG] Position validation: "
                    f"expected={expected_position}, actual={actual_position}, "
                    f"error={error:.2f}px, valid={position_valid}"
                )

            if not position_valid:
                # Log spawn position comparison to diagnose drift
                spawn_dx = spawn_pos[0] - 312.0 if spawn_pos else 0
                spawn_dy = spawn_pos[1] - 444.0 if spawn_pos else 0
                spawn_drift = (spawn_dx * spawn_dx + spawn_dy * spawn_dy) ** 0.5
                # Sample actions for debugging
                first_10 = (
                    action_sequence[:10]
                    if len(action_sequence) >= 10
                    else action_sequence
                )
                last_10 = action_sequence[-10:] if len(action_sequence) >= 10 else []
                # Count action distribution to detect corruption
                action_dist = {}
                for a in action_sequence:
                    action_dist[a] = action_dist.get(a, 0) + 1
                logger.error(
                    f"[REPLAY_DEBUG] Position mismatch FAILURE: "
                    f"expected={expected_position}, actual={actual_position}, "
                    f"error={error:.6f}px, spawn={spawn_pos}, spawn_drift={spawn_drift:.2f}px | "
                    f"actions={actions_replayed}/{len(action_sequence)}, "
                    f"frames={frames_replayed}, died={ninja_died_during_replay} | "
                    f"first_actions={list(first_10)}, last_actions={list(last_10)}, "
                    f"action_dist={action_dist}"
                )

        # Get observation after replay
        obs = self._get_observation()
        processed_obs = self._process_observation(obs)

        # Get current position and velocity for info dict (CRITICAL FIX)
        # This ensures infos[env_idx] has the correct position after checkpoint replay,
        # preventing stale terminal positions from previous episode from persisting
        current_pos = self.nplay_headless.ninja_position()
        current_vel = self.nplay_headless.ninja_velocity()

        # Determine if replay failed and why
        replay_failed = ninja_died_during_replay or not position_valid
        replay_failure_reason = None
        if ninja_died_during_replay:
            replay_failure_reason = "ninja_died"
        elif not position_valid:
            replay_failure_reason = "position_mismatch"

        info = {
            "checkpoint_replay": True,
            "replay_frames": len(action_sequence),
            "position_valid": position_valid,
            "checkpoint_cell": getattr(checkpoint, "cell", None),
            "checkpoint_distance": getattr(checkpoint, "distance_to_goal", None),
            "checkpoint_route": checkpoint_route,  # Path taken during checkpoint replay
            "checkpoint_source": getattr(
                checkpoint, "source", "unknown"
            ),  # Demo vs agent checkpoint
            "replay_failed": replay_failed,
            "replay_failure_reason": replay_failure_reason,
            # CRITICAL FIX: Include position so infos[env_idx] has correct position after replay
            # Without this, stale terminal position from previous episode persists, causing
            # corrupted checkpoints to be created with wrong position but checkpoint's action buffer
            "player_x": current_pos[0] if current_pos else 0.0,
            "player_y": current_pos[1] if current_pos else 0.0,
            "player_xspeed": current_vel[0] if current_vel else 0.0,
            "player_yspeed": current_vel[1] if current_vel else 0.0,
        }

        if self.enable_logging:
            logger.debug(
                f"Checkpoint replay from wrapper complete: {len(action_sequence)} actions, "
                f"checkpoint_route has {len(checkpoint_route)} positions"
            )

        # CRITICAL FIX: Check if checkpoint replay exceeded truncation limit
        # Large checkpoints (3000+ actions) can replay 12,000+ frames, exceeding
        # the 6000 frame truncation limit. Without this check, episodes continue
        # indefinitely and overflow the Go-Explore action buffer.
        current_frame = self.nplay_headless.sim.frame
        truncated_after_replay = self._check_curriculum_aware_truncation()

        if truncated_after_replay:
            logger.warning(
                f"[CHECKPOINT_TRUNCATION] Checkpoint replay exceeded truncation limit! "
                f"Replayed {len(action_sequence)} actions ({frames_replayed} frames), "
                f"current_frame={current_frame}, limit={self.truncation_checker.current_truncation_limit}. "
                f"Marking episode as truncated."
            )
            # Mark as truncated in info so Go-Explore knows to reset the buffer
            info["truncated_after_checkpoint_replay"] = True
            info["checkpoint_replay_frames"] = frames_replayed

        return (processed_obs, info)

    def render(self):
        """Render the environment."""
        # Get debug info from mixin if available, otherwise None
        debug_info = self._debug_info() if hasattr(self, "_debug_info") else None
        return self.nplay_headless.render(debug_info)

    def _get_observation(self) -> Dict[str, Any]:
        """Get the current observation from the game state."""
        # Return cached observation if valid
        # CRITICAL: DO NOT recompute action_mask! It must remain exactly as it was computed.
        # If we recompute it, ninja state might have changed slightly between action selection
        # and validation, causing the mask to change and trigger false positives in
        # masked action detection. The action_mask must stay consistent throughout the step.
        if self._cached_observation is not None:
            return self._cached_observation

        # Calculate time remaining feature using curriculum-aware truncation limit
        # This ensures agents see accurate time pressure that aligns with actual truncation behavior
        current_frame = self.nplay_headless.sim.frame
        curriculum_limit = self._get_curriculum_aware_truncation_limit()

        # Robust time_remaining calculation (avoid division by zero, clamp to [0, 1])
        if curriculum_limit <= 0:
            time_remaining = 1.0  # Fallback: full time remaining
        else:
            time_remaining = max(
                0.0, (curriculum_limit - current_frame) / curriculum_limit
            )

        ninja_state = self.nplay_headless.get_ninja_state()

        # Base game_state: get_ninja_state() (24) + time_remaining (1) = 25 floats.
        # NppEnvironment expands to 28 (adds physics-derived + switch_activated).
        import numpy as np

        game_state = np.append(ninja_state, time_remaining)

        # Get entity states for PBRS hazard detection
        # MEMORY OPTIMIZATION: Use cached static entity/position data
        # These values never change during an episode, only on reset

        # Cache static positions on first access
        # CRITICAL: Validate positions are not (0, 0) which indicates entities not loaded
        # CRITICAL: For curriculum, ensure we're not caching positions before repositioning
        # CRITICAL: During first few frames of episode, always read fresh positions if curriculum active
        # This prevents caching stale positions from before curriculum repositioning
        current_frame = self.nplay_headless.sim.frame
        force_fresh_read = (
            self.goal_curriculum_manager is not None
            and current_frame <= 10
            and self._cached_switch_pos is not None
        )

        if self._cached_switch_pos is None or force_fresh_read:
            switch_pos = self.nplay_headless.exit_switch_position()
            # Only cache if valid (not the default (0, 0) returned when entities missing)
            if switch_pos != (0, 0):
                # DIAGNOSTIC: If curriculum is active, verify position matches curriculum expectations
                # This catches cases where we're caching positions before curriculum repositioning
                if self.goal_curriculum_manager is not None:
                    expected_pos = (
                        self.goal_curriculum_manager.get_curriculum_switch_position()
                    )
                    dx = abs(switch_pos[0] - expected_pos[0])
                    dy = abs(switch_pos[1] - expected_pos[1])

                    # If position doesn't match curriculum, log warning
                    # During first 10 frames, don't cache mismatched positions (force re-read)
                    should_cache = True
                    if dx > 1.0 or dy > 1.0:
                        stage = self.goal_curriculum_manager.state.unified_stage
                        if current_frame <= 10:
                            logger.error(
                                f"[OBS_CACHE] Switch position mismatch in first 10 frames! "
                                f"frame={current_frame}, stage={stage}, "
                                f"read_pos={switch_pos}, expected={expected_pos}, "
                                f"error=({dx:.1f}, {dy:.1f})px. "
                                f"NOT caching - will force fresh read next time."
                            )
                            # Don't cache - force fresh read next time
                            should_cache = False
                        else:
                            logger.warning(
                                f"[OBS_CACHE] Caching switch position that doesn't match curriculum! "
                                f"frame={current_frame}, stage={stage}, "
                                f"caching_pos={switch_pos}, expected={expected_pos}, "
                                f"error=({dx:.1f}, {dy:.1f})px."
                            )

                    # Only cache if position matches curriculum (or curriculum not active)
                    if should_cache:
                        self._cached_switch_pos = switch_pos
                else:
                    # No curriculum check needed, cache normally
                    self._cached_switch_pos = switch_pos
            else:
                # Don't cache (0, 0) - force re-fetch next time
                # Log warning to help diagnose curriculum loading issues
                logger.warning(
                    f"exit_switch_position() returned (0, 0) - entities may not be loaded yet. "
                    f"entity_dic keys: {list(self.nplay_headless.sim.entity_dic.keys())}"
                )
        # Same safeguard for exit position as switch position
        force_fresh_exit_read = (
            self.goal_curriculum_manager is not None
            and current_frame <= 10
            and self._cached_exit_pos is not None
        )

        if self._cached_exit_pos is None or force_fresh_exit_read:
            exit_pos = self.nplay_headless.exit_door_position()
            if exit_pos != (0, 0):
                # DIAGNOSTIC: If curriculum is active, verify position matches curriculum expectations
                if self.goal_curriculum_manager is not None:
                    expected_pos = (
                        self.goal_curriculum_manager.get_curriculum_exit_position()
                    )
                    dx = abs(exit_pos[0] - expected_pos[0])
                    dy = abs(exit_pos[1] - expected_pos[1])

                    should_cache = True
                    if dx > 1.0 or dy > 1.0:
                        stage = self.goal_curriculum_manager.state.unified_stage
                        if current_frame <= 10:
                            logger.error(
                                f"[OBS_CACHE] Exit position mismatch in first 10 frames! "
                                f"frame={current_frame}, stage={stage}, "
                                f"read_pos={exit_pos}, expected={expected_pos}, "
                                f"error=({dx:.1f}, {dy:.1f})px. "
                                f"NOT caching - will force fresh read next time."
                            )
                            should_cache = False
                        else:
                            logger.warning(
                                f"[OBS_CACHE] Caching exit position that doesn't match curriculum! "
                                f"frame={current_frame}, stage={stage}, "
                                f"caching_pos={exit_pos}, expected={expected_pos}, "
                                f"error=({dx:.1f}, {dy:.1f})px."
                            )

                    if should_cache:
                        self._cached_exit_pos = exit_pos
                else:
                    # No curriculum check needed, cache normally
                    self._cached_exit_pos = exit_pos
            else:
                logger.warning(
                    "exit_door_position() returned (0, 0) - entities may not be loaded yet."
                )
        if self._cached_locked_doors is None:
            self._cached_locked_doors = self.nplay_headless.locked_doors()
        if self._cached_locked_door_switches is None:
            self._cached_locked_door_switches = (
                self.nplay_headless.locked_door_switches()
            )
        if self._cached_toggle_mines is None:
            self._cached_toggle_mines = self.nplay_headless.sim.entity_dic.get(
                EntityType.TOGGLE_MINE, []
            )

        if self._cached_tile_grid is None:
            tile_data = self.nplay_headless.sim.get_tile_grid()
            self._cached_tile_grid = np.array(tile_data, dtype=np.int8).reshape(23, 42).T

        # NPP-38: launch pads, bounce blocks, and thwumps are out of scope for
        # the Layer 1 toggle-mines chunk; their entity caches were removed.

        # Use cached static values (with fallback to fresh fetch if cache failed)
        switch_pos = self._cached_switch_pos
        if switch_pos is None:
            # Cache miss (likely due to entities not loaded) - try fresh fetch
            switch_pos = self.nplay_headless.exit_switch_position()
            # CRITICAL: If still (0, 0), entities are definitely not loaded
            # Log detailed diagnostics to help identify root cause
            if switch_pos == (0, 0):
                entity_keys = list(self.nplay_headless.sim.entity_dic.keys())
                entity_counts = {
                    k: len(v) for k, v in self.nplay_headless.sim.entity_dic.items()
                }
                logger.error(
                    f"CRITICAL: exit_switch_position() returned (0, 0) after fresh fetch! "
                    f"Entities not loaded. entity_dic keys: {entity_keys}, counts: {entity_counts}, "
                    f"frame: {self.nplay_headless.sim.frame}"
                )
        exit_pos = self._cached_exit_pos
        if exit_pos is None:
            exit_pos = self.nplay_headless.exit_door_position()
            if exit_pos == (0, 0):
                logger.error(
                    "CRITICAL: exit_door_position() returned (0, 0) after fresh fetch!"
                )
        locked_doors = self._cached_locked_doors
        locked_door_switches = self._cached_locked_door_switches

        # Build entity list from cached static entities
        entities = []
        entities.extend(self._cached_toggle_mines)

        # Toggled mines need fresh lookup (state can change during episode)
        toggled_mines = self.nplay_headless.sim.entity_dic.get(
            EntityType.TOGGLE_MINE_TOGGLED, []
        )
        entities.extend(toggled_mines)
        entities.extend(locked_doors)
        entities.extend(locked_door_switches)

        # Get ninja properties for PBRS impact risk calculation
        ninja_vel_old = self.nplay_headless.ninja_velocity_old()

        # Get current ninja velocity for momentum rewards
        ninja_vel = self.nplay_headless.ninja_velocity()

        # Get current ninja position (this is dynamic, not cached)
        ninja_pos = self.nplay_headless.ninja_position()

        # Get map name for waypoint lookups (use actual filename instead of auto-generated level_id)
        map_name = getattr(self.map_loader, "current_map_name", None)

        # CRITICAL: Get fresh switch_activated state (NEVER cache this value)
        # This must always reflect current entity state, not stale cached value
        switch_activated = self.nplay_headless.exit_switch_activated()

        # DIAGNOSTIC: Verify switch position matches curriculum expectations at episode start
        if (
            current_frame <= 10
            and self.goal_curriculum_manager is not None
            and switch_pos != (0, 0)
        ):
            expected_pos = self.goal_curriculum_manager.get_curriculum_switch_position()
            dx = abs(switch_pos[0] - expected_pos[0])
            dy = abs(switch_pos[1] - expected_pos[1])

            if dx > 1.0 or dy > 1.0:
                stage = self.goal_curriculum_manager.state.unified_stage
                logger.error(
                    f"[OBS_CORRUPTION] Observation has WRONG switch position at episode start! "
                    f"frame={current_frame}, stage={stage}, "
                    f"observed_switch_pos={switch_pos}, expected={expected_pos}, "
                    f"error=({dx:.1f}, {dy:.1f})px. "
                    f"Entity repositioning failed or was undone! Switch will be collected immediately!"
                )

        entity_overlays = compute_entity_overlays(
            ninja_pos=(float(ninja_pos[0]), float(ninja_pos[1])),
            switch_pos=switch_pos,
            switch_activated=switch_activated,
            exit_pos=exit_pos,
            deadly_mines=self._cached_toggle_mines or [],
            safe_mines=toggled_mines,
        )

        # Extract jump_duration_remaining from ninja_state (index 11, 0-indexed)
        # This is normalized [0, 1]: 1.0 = full jump available, 0.0 = no jump frames left
        jump_duration_remaining = (
            float(ninja_state[11]) if len(ninja_state) > 11 else 0.0
        )

        action_mask, duration_mask = self._get_action_and_duration_mask()

        obs = {
            "game_state": game_state,
            "player_dead": self.nplay_headless.ninja_has_died(),
            "player_won": self.nplay_headless.ninja_has_won(),
            "death_cause": self.nplay_headless.ninja_death_cause(),
            "player_x": ninja_pos[0],
            "player_y": ninja_pos[1],
            "player_xspeed": ninja_vel[0],
            "player_yspeed": ninja_vel[1],
            "player_xspeed_old": ninja_vel_old[0],
            "player_yspeed_old": ninja_vel_old[1],
            "player_airborn_old": self.nplay_headless.ninja_airborn_old(),
            "jump_duration_remaining": jump_duration_remaining,  # For physics-aware PBRS
            "switch_activated": switch_activated,  # Use fresh value, not cached
            "switch_x": switch_pos[0],
            "switch_y": switch_pos[1],
            "exit_door_x": exit_pos[0],
            "exit_door_y": exit_pos[1],
            "sim_frame": self.nplay_headless.sim.frame,
            "entities": entities,  # Entity objects (mines, locked doors, switches)
            "locked_doors": locked_doors,
            "locked_door_switches": locked_door_switches,
            "action_mask": action_mask,
            "map_name": map_name,  # For waypoint lookups by filename
            "tile_grid": self._cached_tile_grid
            if self._cached_tile_grid is not None
            else np.zeros((MAP_TILE_WIDTH, MAP_TILE_HEIGHT), dtype=np.int8),
            "entity_overlays": entity_overlays,
        }
        if duration_mask is not None:
            obs["duration_mask"] = duration_mask

        action_mask = obs["action_mask"]
        if not action_mask.any():
            logger.error(
                f"[MASK_VALIDATION] All actions masked! Mask: {action_mask}. "
                f"This should never happen - at least one action must be valid. "
                f"Forcing NOOP to be valid as emergency fallback."
            )
            # Force NOOP to be valid as emergency fallback
            action_mask = action_mask.copy()
            action_mask[0] = 1
            obs["action_mask"] = action_mask
            if "duration_mask" in obs:
                duration_mask = obs["duration_mask"].copy()
                duration_mask[0, 0] = 1
                obs["duration_mask"] = duration_mask

        # Cache the computed observation before returning
        self._cached_observation = obs
        return obs

    def _get_action_mask_with_path_update(self) -> np.ndarray:
        """Get action mask with path guidance predictor updated.

        Updates the path guidance predictor's cached path before retrieving
        the action mask. This ensures path-based masking uses current ninja
        position and goal state.

        Also detects trivial straight horizontal paths and passes path direction
        to the Ninja for deterministic action masking (ensuring 100% success
        on simple corridors).

        ANNEALED MASKING (2026-01-05): Path-direction masking now only applies
        in 'hard' mode, transitioning to 'soft' (logit bias) and then 'none'
        as training progresses.

        Args:
            ninja_pos: Current ninja position (x, y)
            switch_pos: Exit switch position (x, y)
            exit_pos: Exit door position (x, y)

        Returns:
            Action mask as numpy array of int8 (defensive copy to prevent memory sharing)
        """
        # Get annealed masking mode from reward config
        masking_mode = self.reward_calculator.config.action_masking_mode
        self.nplay_headless.sim._action_masking_mode = masking_mode

        return np.array(self.nplay_headless.get_action_mask(), dtype=np.int8)

    def _get_action_and_duration_mask(
        self,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Return the action mask and optional duration mask for the current state."""
        action_mask = self._get_action_mask_with_path_update()
        if not self.enable_layer3_macro_action_mask:
            return action_mask, None
        return action_mask, self._build_default_duration_mask(action_mask)

    def _build_default_duration_mask(self, action_mask: np.ndarray) -> np.ndarray:
        """Broadcast the action mask across all duration bins when Layer 3 is disabled."""
        mask = np.array(action_mask, dtype=np.int8, copy=True).reshape(6, 1)
        return np.repeat(mask, self._layer3_num_durations, axis=1)

    def _check_termination(self) -> Tuple[bool, bool, bool]:
        """
        Check if the episode should be terminated.

        Uses curriculum-aware truncation that respects reward config phases:
        - Early phase (no time penalty): More generous truncation for exploration
        - Mid phase (optional time penalty): Moderate truncation
        - Late phase (full time penalty): Standard truncation for efficiency

        Returns:
            Tuple containing:
            - terminated: True if episode should be terminated, False otherwise
            - truncated: True if episode should be truncated, False otherwise
            - player_won: True if player won, False otherwise
        """
        player_won = self.nplay_headless.ninja_has_won()
        player_dead = self.nplay_headless.ninja_has_died()
        terminated = player_won or player_dead

        # Check truncation using curriculum-aware policy
        should_truncate = self._check_curriculum_aware_truncation()

        # Return proper truncated flag (don't merge with terminated)
        # Truncation is now treated separately so we can apply death penalty
        return terminated, should_truncate, player_won

    def _calculate_reward(self, curr_obs, prev_obs, action=None):
        """Calculate the reward for the environment."""
        return self.reward_calculator.calculate_reward(
            curr_obs, prev_obs, action, curriculum_manager=self.goal_curriculum_manager
        )

    def _process_observation(self, obs):
        """Process the observation from the environment."""
        return self.observation_processor.process_observation(obs)

    def _reset_reward_calculator(self):
        """Reset the reward calculator."""
        self.reward_calculator.reset()
        self.primitive_reward_calculator.reset()

    def _get_switch_states_from_env(self) -> Dict[str, bool]:
        """
        Extract locked door switch states from environment.

        Returns:
            Dictionary mapping switch IDs to activation states (True = activated/collected)
        """
        switch_states = {}

        # Get locked door entities (type 6)
        # Each locked door has a switch that the ninja must collect to open the door
        locked_doors = self.nplay_headless.locked_doors()

        for i, locked_door in enumerate(locked_doors):
            # The 'active' attribute indicates if the switch is still collectible
            # active=True means switch not yet collected (door still locked)
            # active=False means switch has been collected (door is open)
            is_activated = not getattr(locked_door, "active", True)
            switch_states[f"locked_door_{i}"] = is_activated

        # Get locked door switch entities (type 7) if they exist separately
        locked_door_switches = self.nplay_headless.locked_door_switches()

        for i, switch in enumerate(locked_door_switches):
            # Similar logic for separate switch entities
            is_activated = not getattr(switch, "active", True)
            switch_states[f"locked_door_switch_{i}"] = is_activated

        return switch_states

    def invalidate_switch_cache(self):
        """Invalidate switch-dependent caches when switch states change.

        This is a no-op stub in BaseNppEnvironment. Subclasses that use
        switch-dependent caching (like NppEnvironment with mixins) should
        override this method to clear their caches.

        Called by entity collision methods (EntityExitSwitch, EntityDoorLocked)
        when switches are activated, ensuring path distances and other cached
        values are recomputed with updated goal positions.
        """
        pass

    def _extract_level_data(self) -> LevelData:
        """
        Extract level structure data for graph construction.

        Returns:
            LevelData object containing tiles and entities
        """
        # Build level tiles as a compact 2D array of inner playable area [23 x 42]
        tile_dic = self.nplay_headless.get_tile_data()
        tiles = np.zeros((MAP_TILE_HEIGHT, MAP_TILE_WIDTH), dtype=np.int32)
        # Simulator tiles include a 1-tile border; map inner (1..42, 1..23) -> (0..41, 0..22)
        for (x, y), tile_id in tile_dic.items():
            inner_x = x - 1
            inner_y = y - 1
            if 0 <= inner_x < MAP_TILE_WIDTH and 0 <= inner_y < MAP_TILE_HEIGHT:
                tiles[inner_y, inner_x] = int(tile_id)

        # Extract entities
        entities = self.entity_extractor.extract_graph_entities()

        # Extract ninja spawn position from map_data
        start_position = extract_start_position_from_map_data(
            self.nplay_headless.sim.map_data
        )

        # CRITICAL FIX: Populate switch_states for cache key generation
        # This ensures PBRS path distance caches are built with correct switch states
        # Without this, caches may not invalidate properly when switches activate
        switch_states = {}
        if hasattr(self, "_get_switch_states_from_env"):
            switch_states = self._get_switch_states_from_env()

        # Include curriculum stage in level_id for proper cache invalidation
        # Each stage gets a unique level_id to force cache rebuilds when stage advances
        curriculum_stage = None
        if (
            hasattr(self, "goal_curriculum_manager")
            and self.goal_curriculum_manager is not None
        ):
            curriculum_stage = self.goal_curriculum_manager.state.unified_stage

        return LevelData(
            start_position=start_position,
            tiles=tiles,
            entities=entities,
            switch_states=switch_states,
            curriculum_stage=curriculum_stage,
        )

    @property
    def level_data(self) -> LevelData:
        """
        Get current level data for external access.

        Uses cached value to avoid rebuilding level data thousands of times per episode.
        Level geometry never changes during an episode, so caching is safe.

        Returns:
            LevelData object containing tiles and entities
        """
        if self._cached_level_data is None:
            self._cached_level_data = self._extract_level_data()

        return self._cached_level_data

    @property
    def entities(self) -> list:
        """
        Get current entities for external access.

        Uses cached value to avoid rebuilding entity list thousands of times per episode.
        Entity positions never change during an episode, so caching is safe.

        Returns:
            List of entity dictionaries
        """
        if self._cached_entities is None:
            self._cached_entities = self.entity_extractor.extract_graph_entities()
        return self._cached_entities

    @property
    def current_map_name(self) -> str:
        """Get the current map name."""
        return self.map_loader.current_map_name

    def get_route_visualization_tile_data(self) -> Dict[Tuple[int, int], int]:
        """Get tile data for route visualization.

        This method is callable via env_method from SubprocVecEnv to access
        tile data from remote environments.

        Returns:
            Dictionary mapping (x, y) grid coordinates to tile type values
        """
        tile_dic = self.nplay_headless.get_tile_data()
        return dict(tile_dic)  # Make a copy

    def get_route_visualization_mine_data(self) -> List[Dict[str, Any]]:
        """Get mine data for route visualization.

        This method is callable via env_method from SubprocVecEnv to access
        mine data from remote environments.

        Returns:
            List of mine dictionaries with keys: x, y, state, radius
        """
        return self.nplay_headless.get_all_mine_data_for_visualization()

    def get_route_visualization_locked_door_data(self) -> List[Dict[str, Any]]:
        """Get locked door data for route visualization.

        This method is callable via env_method from SubprocVecEnv to access
        locked door data from remote environments.

        Returns:
            List of locked door dictionaries with keys:
            - switch_x, switch_y: Switch position
            - door_x, door_y: Door segment center position
            - segment_x1, segment_y1, segment_x2, segment_y2: Door segment endpoints
            - closed: Whether door is closed (True) or open (False)
            - active: Whether switch is still collectible (True = not collected, door closed)
            - switch_radius: Switch radius
        """
        locked_doors = []
        locked_door_entities = self.nplay_headless.locked_doors()

        for door_entity in locked_door_entities:
            # Get switch position (entity position)
            switch_x = float(getattr(door_entity, "xpos", 0.0))
            switch_y = float(getattr(door_entity, "ypos", 0.0))

            # Get door segment
            segment = getattr(door_entity, "segment", None)
            if segment:
                segment_x1 = float(getattr(segment, "x1", 0.0))
                segment_y1 = float(getattr(segment, "y1", 0.0))
                segment_x2 = float(getattr(segment, "x2", 0.0))
                segment_y2 = float(getattr(segment, "y2", 0.0))
                door_x = (segment_x1 + segment_x2) * 0.5
                door_y = (segment_y1 + segment_y2) * 0.5
            else:
                # Fallback if segment not available
                segment_x1 = segment_y1 = segment_x2 = segment_y2 = 0.0
                door_x = door_y = 0.0

            # Get door state
            closed = bool(getattr(door_entity, "closed", True))
            active = bool(getattr(door_entity, "active", True))

            locked_doors.append(
                {
                    "switch_x": switch_x,
                    "switch_y": switch_y,
                    "door_x": door_x,
                    "door_y": door_y,
                    "segment_x1": segment_x1,
                    "segment_y1": segment_y1,
                    "segment_x2": segment_x2,
                    "segment_y2": segment_y2,
                    "closed": closed,
                    "active": active,
                    "switch_radius": float(EntityDoorLocked.RADIUS),
                }
            )

        return locked_doors

    def get_route_visualization_terminal_impact(self) -> bool:
        """Get terminal impact value for route visualization.

        This method is callable via env_method from SubprocVecEnv to access
        terminal impact from remote environments.

        Returns:
            Boolean indicating if the ninja died from terminal impact
        """
        return self.nplay_headless.get_ninja_terminal_impact()

    def get_route_visualization_data(self) -> Dict[str, Any]:
        """Get all visualization data in a single batched call.

        This method provides 10-15% speedup by batching multiple env_method calls
        into a single remote call for SubprocVecEnv environments.

        Returns:
            Dictionary containing:
                - tiles: Dict mapping (x, y) grid coordinates to tile type
                - mines: List of mine dictionaries
                - locked_doors: List of locked door dictionaries
                - graph_data: Graph data for pathfinding visualization (or None)
                - level_data: Level data for mine proximity (or None)
        """
        return {
            "tiles": self.get_route_visualization_tile_data(),
            "mines": self.get_route_visualization_mine_data(),
            "locked_doors": self.get_route_visualization_locked_door_data(),
            "graph_data": self.get_graph_data_for_visualization(),
            "level_data": self.get_level_data_for_visualization(),
        }

    def get_path_distances_for_visualization(
        self,
        agent_pos: Tuple[float, float],
        switch_pos: Tuple[float, float],
        exit_pos: Tuple[float, float],
        switch_activated: bool,
    ) -> Tuple[Optional[float], Optional[float]]:
        """Calculate shortest path distances from agent position to goals.

        This method is callable via env_method from SubprocVecEnv to provide
        path distance debugging information for route visualization.

        Uses the PBRS path calculator to get geometric distances along
        physics-optimal paths, matching the reward calculation logic.

        Args:
            agent_pos: Agent's position (x, y)
            switch_pos: Exit switch position (x, y)
            exit_pos: Exit door position (x, y)
            switch_activated: Whether switch has been activated

        Returns:
            Tuple of (switch_distance, exit_distance) in pixels
            Returns (None, None) if calculation not possible
        """
        try:
            # Get path calculator from reward calculator's PBRS calculator
            if not hasattr(self.reward_calculator, "pbrs_calculator"):
                return (None, None)

            pbrs_calc = self.reward_calculator.pbrs_calculator
            if not hasattr(pbrs_calc, "path_calculator"):
                return (None, None)

            path_calculator = pbrs_calc.path_calculator

            # Get current level data and graph
            level_data = self.level_data

            # Get graph data - need to check if GraphMixin is available
            adjacency = None
            graph_data = None
            if hasattr(self, "current_graph_data") and self.current_graph_data:
                adjacency = self.current_graph_data.get("adjacency")
                graph_data = self.current_graph_data

            if adjacency is None:
                return (None, None)

            base_adjacency = (
                graph_data.get("base_adjacency", adjacency) if graph_data else adjacency
            )

            from ..constants.physics_constants import (
                EXIT_SWITCH_RADIUS,
                EXIT_DOOR_RADIUS,
                NINJA_RADIUS,
            )

            # Calculate distance to switch (if not activated)
            switch_distance = None
            if not switch_activated:
                try:
                    switch_distance = path_calculator.get_path_cost(
                        (int(agent_pos[0]), int(agent_pos[1])),
                        (int(switch_pos[0]), int(switch_pos[1])),
                        adjacency,
                        base_adjacency,
                        level_data=level_data,
                        graph_data=graph_data,
                        entity_radius=EXIT_SWITCH_RADIUS,
                        ninja_radius=NINJA_RADIUS,
                        goal_id="switch",
                    )
                    if switch_distance == float("inf"):
                        switch_distance = None
                except Exception as e:
                    logger.debug(f"Failed to calculate switch distance: {e}")
                    switch_distance = None

            # Calculate distance to exit
            exit_distance = None
            try:
                exit_distance = path_calculator.get_path_cost(
                    (int(agent_pos[0]), int(agent_pos[1])),
                    (int(exit_pos[0]), int(exit_pos[1])),
                    adjacency,
                    base_adjacency,
                    level_data=level_data,
                    graph_data=graph_data,
                    entity_radius=EXIT_DOOR_RADIUS,
                    ninja_radius=NINJA_RADIUS,
                    goal_id="exit",
                )
                if exit_distance == float("inf"):
                    exit_distance = None
            except Exception as e:
                logger.debug(f"Failed to calculate exit distance: {e}")
                exit_distance = None

            return (switch_distance, exit_distance)

        except Exception as e:
            logger.debug(f"get_path_distances_for_visualization failed: {e}")
            return (None, None)

    def get_graph_data_for_visualization(self) -> Optional[Dict[str, Any]]:
        """Get graph data for route visualization.

        This method is callable via env_method from SubprocVecEnv to provide
        graph data for PBRS path visualization in route callbacks.

        Returns:
            Dictionary with graph data suitable for pathfinding visualization,
            or None if graph data is not available
        """
        try:
            if hasattr(self, "current_graph_data") and self.current_graph_data:
                return {
                    "adjacency": self.current_graph_data.get("adjacency"),
                    "base_adjacency": self.current_graph_data.get("base_adjacency"),
                    "node_physics": self.current_graph_data.get("node_physics"),
                    "spatial_hash": self.current_graph_data.get("spatial_hash"),
                    "subcell_lookup": self.current_graph_data.get("subcell_lookup"),
                }
            return None
        except Exception as e:
            logger.debug(f"get_graph_data_for_visualization failed: {e}")
            return None

    def get_level_data_for_visualization(self) -> Optional[Any]:
        """Get level data for route visualization.

        This method is callable via env_method from SubprocVecEnv to provide
        level data for PBRS path visualization with mine proximity calculations.

        IMPORTANT: Returns a PICKLABLE copy of level data.
        Entity 'entity_ref' fields are stripped because they reference simulation
        objects that cannot be pickled (they may indirectly reference mp.Array
        ctypes objects from SharedLevelCache).

        Returns:
            LevelData object (picklable copy) or None if not available
        """
        try:
            if not hasattr(self, "level_data"):
                return None

            level_data = self.level_data

            # Create picklable copy of entities by stripping entity_ref fields
            # entity_ref contains actual Entity objects that may have references
            # to unpicklable objects (simulation state, ctypes arrays, etc.)
            picklable_entities = []
            for entity in level_data.entities:
                # Create a copy without entity_ref
                clean_entity = {k: v for k, v in entity.items() if k != "entity_ref"}
                picklable_entities.append(clean_entity)

            # Create a new LevelData with picklable entities
            from ..graph.level_data import LevelData as LevelDataClass

            return LevelDataClass(
                start_position=level_data.start_position,
                tiles=level_data.tiles.copy(),  # Copy numpy array to ensure it's independent
                entities=picklable_entities,
                player=level_data.player,
                level_id=level_data.level_id,
                metadata=level_data.metadata,
                switch_states=dict(level_data.switch_states),
                entity_start_positions=list(level_data.entity_start_positions),
                curriculum_stage=level_data.curriculum_stage,
            )
        except Exception as e:
            logger.debug(f"get_level_data_for_visualization failed: {e}")
            return None

    def _get_next_hop_path_for_visualization(
        self,
        start_pos: Tuple[float, float],
        switch_activated: bool,
    ) -> Optional[List[Tuple[int, int]]]:
        """Follow the level-cache next_hop chain from start_pos to the active goal.

        Intended to be called from _build_episode_info() so the path is computed
        inside the environment process and delivered via the info dict (avoiding any
        env_method call overhead in the callback).

        Args:
            start_pos: World-space (x, y) position to start the path from.
            switch_activated: True if the exit switch has already been collected
                (path should lead to the exit door rather than the switch).

        Returns:
            List of (x, y) graph-node positions along the cached geometric shortest
            path, or None if the cache is not available or the start is unreachable.
        """
        try:
            path_calculator = getattr(self, "_path_calculator", None)
            if path_calculator is None:
                return None
            level_cache = getattr(path_calculator, "level_cache", None)
            if level_cache is None:
                return None

            if not hasattr(self, "current_graph_data") or not self.current_graph_data:
                return None
            adjacency = self.current_graph_data.get("adjacency")
            if not adjacency:
                return None

            from nclone.graph.reachability.pathfinding_utils import find_ninja_node
            from nclone.constants.physics_constants import NINJA_RADIUS

            spatial_hash = self.current_graph_data.get("spatial_hash")
            subcell_lookup = self.current_graph_data.get("subcell_lookup")

            start_node = find_ninja_node(
                start_pos,
                adjacency,
                spatial_hash=spatial_hash,
                subcell_lookup=subcell_lookup,
                ninja_radius=NINJA_RADIUS,
            )
            if start_node is None:
                return None

            goal_id = "exit" if switch_activated else "switch"

            path: List[Tuple[int, int]] = []
            current = start_node
            visited: set = set()
            max_hops = 1000

            while current is not None and len(path) < max_hops:
                if current in visited:
                    break
                path.append(current)
                visited.add(current)
                next_hop = level_cache.get_next_hop(current, goal_id)
                if next_hop is None or next_hop == current:
                    break
                current = next_hop

            return path if len(path) >= 2 else None

        except Exception:
            return None

    def get_pbrs_path_for_visualization(
        self,
        start_pos: Tuple[float, float],
        goal_pos: Tuple[float, float],
        switch_activated: bool,
    ) -> Optional[List[Tuple[int, int]]]:
        """Calculate PBRS path using next_hop chain from level cache.

        IMPORTANT: Level cache stores only GEOMETRIC distances (direction-independent).
        The next_hop chain gives topologically-correct path, but NOT physics-optimal!

        For PHYSICS-OPTIMAL paths (correct costs with momentum, gravity, mines):
        Use find_shortest_path FROM current position TO goal instead.

        This method is callable via env_method from SubprocVecEnv. It extracts
        the topological path by following the next_hop chain from the level cache.

        STRICT VALIDATION: Requires physics_cache to be available in graph data.

        Args:
            start_pos: Agent's starting position (x, y)
            goal_pos: Goal position (x, y) - switch or exit
            switch_activated: Whether switch has been activated

        Returns:
            List of (x, y) node positions forming the topological path, or None if unreachable
            NOTE: This path uses geometric next_hop, not physics-optimal costs!

        Raises:
            RuntimeError: If physics_cache not available (strict validation)
        """
        try:
            # Import pathfinding utilities
            from nclone.graph.reachability.pathfinding_utils import find_ninja_node
            from nclone.constants.physics_constants import NINJA_RADIUS

            # Get path calculator from reward calculator
            if not hasattr(self.reward_calculator, "pbrs_calculator"):
                raise RuntimeError("PBRS calculator not found in reward calculator")
            pbrs_calc = self.reward_calculator.pbrs_calculator
            if not hasattr(pbrs_calc, "path_calculator"):
                raise RuntimeError("Path calculator not found in PBRS calculator")
            path_calculator = pbrs_calc.path_calculator

            # Get level cache
            if not hasattr(path_calculator, "level_cache"):
                raise RuntimeError("Path calculator has no level_cache attribute")
            level_cache = path_calculator.level_cache
            if level_cache is None:
                raise RuntimeError("Level cache is not built yet")

            # Get graph data
            if not hasattr(self, "current_graph_data") or not self.current_graph_data:
                raise RuntimeError("Current graph data not available")

            adjacency = self.current_graph_data.get("adjacency")
            if not adjacency:
                raise RuntimeError("Adjacency graph is empty")

            base_adjacency = self.current_graph_data.get("base_adjacency")
            if not base_adjacency:
                raise RuntimeError(
                    "Base adjacency not found in graph_data. "
                    "Base adjacency is required for physics edge classification."
                )

            # DIAGNOSTIC: Log adjacency sizes to detect masking issues
            logger.warning(
                f"[PBRS_PATH_EXTRACT] Adjacency state: "
                f"masked={len(adjacency)} nodes, base={len(base_adjacency)} nodes "
                f"({'SAME' if len(adjacency) == len(base_adjacency) else 'DIFFERENT - entity masking active'})"
            )

            # STRICT VALIDATION: Mine proximity cache must be available
            if not hasattr(path_calculator, "mine_proximity_cache"):
                raise RuntimeError(
                    "Path calculator missing mine_proximity_cache attribute. "
                    "Mine avoidance costs will not be applied to paths."
                )
            mine_cache = path_calculator.mine_proximity_cache
            if mine_cache is None:
                raise RuntimeError(
                    "Mine proximity cache is None. "
                    "Paths will be computed without mine avoidance costs."
                )

            # Log mine cache state for diagnostics
            mine_cache_size = (
                len(mine_cache.cache) if hasattr(mine_cache, "cache") else 0
            )

            # CRITICAL DIAGNOSTIC: Check if level cache was built WITH mine costs
            # Compare cached level_data to see if mines were present when cache was built
            level_cache_valid = False
            if (
                hasattr(level_cache, "_cached_level_data")
                and level_cache._cached_level_data
            ):
                cached_level_id = getattr(
                    level_cache._cached_level_data, "level_id", "unknown"
                )
                level_cache_size = len(level_cache.cache)
                level_cache_valid = True
                logger.warning(
                    f"[PBRS_PATH_EXTRACT] Level cache state: "
                    f"level_id={cached_level_id}, "
                    f"cache_size={level_cache_size}, "
                    f"current_mine_cache_size={mine_cache_size}"
                )

            logger.warning(
                f"[PBRS_PATH_EXTRACT] Mine proximity cache: {mine_cache_size} nodes with costs "
                f"(avoidance {'ACTIVE' if mine_cache_size > 0 else 'INACTIVE'}), "
                f"level_cache_built={level_cache_valid}"
            )

            # STRICT: If mine cache is empty but level has mines, cache is stale
            if mine_cache_size == 0 and self.level_data:
                mines = self.level_data.get_all_toggle_mines()
                if len(mines) > 0:
                    raise RuntimeError(
                        f"Level has {len(mines)} mines but mine proximity cache is EMPTY! "
                        f"This means paths will NOT avoid mines. "
                        f"This indicates the level cache was built before mine cache was populated."
                    )

            spatial_hash = self.current_graph_data.get("spatial_hash")
            subcell_lookup = self.current_graph_data.get("subcell_lookup")

            # Find start node from agent position
            start_node = find_ninja_node(
                start_pos,
                adjacency,
                spatial_hash=spatial_hash,
                subcell_lookup=subcell_lookup,
                ninja_radius=NINJA_RADIUS,
            )

            if start_node is None:
                raise RuntimeError(f"Start node not found for position {start_pos}")

            # Determine goal_id (same as reward calculator uses)
            goal_id = "exit" if switch_activated else "switch"

            # DEBUG: Check if level cache has data for this goal_id
            # Test by trying to get next_hop from start_node
            test_next_hop = level_cache.get_next_hop(start_node, goal_id)
            if test_next_hop is None:
                # This is the first node and next_hop is already None
                # This means we're AT the goal, or the cache wasn't built for this goal_id
                logger.debug(
                    f"PBRS path: next_hop is None for start_node={start_node}, goal_id={goal_id}. "
                    f"Either at goal already or cache not built for this goal_id."
                )
                # Check if we're actually at the goal
                goal_node = find_ninja_node(
                    goal_pos,
                    adjacency,
                    spatial_hash=spatial_hash,
                    subcell_lookup=subcell_lookup,
                    ninja_radius=NINJA_RADIUS,
                )
                if goal_node == start_node:
                    logger.debug("Agent is at goal node - returning single-node path")
                    return [start_node]
                else:
                    # Cache not built properly for this goal_id
                    # Check if it's a SharedPathCacheView issue
                    cache_type = type(level_cache).__name__
                    has_shared_cache = hasattr(level_cache, "shared_cache")

                    # Try to diagnose the issue
                    if has_shared_cache:
                        shared = level_cache.shared_cache
                        goal_id_in_mapping = goal_id in shared.goal_id_to_idx
                        start_node_in_mapping = start_node in shared.node_pos_to_idx

                        logger.warning(
                            f"PBRS path extraction failed: level_cache.get_next_hop({start_node}, '{goal_id}') "
                            f"returned None, but agent not at goal. "
                            f"Cache type: {cache_type}, "
                            f"goal_id_in_shared_mapping: {goal_id_in_mapping}, "
                            f"start_node_in_shared_mapping: {start_node_in_mapping}, "
                            f"available_goal_ids: {list(shared.goal_id_to_idx.keys())[:10]}, "
                            f"start_pos={start_pos}, goal_pos={goal_pos}, start_node={start_node}, goal_node={goal_node}"
                        )
                    else:
                        logger.warning(
                            f"PBRS path extraction failed: level_cache.get_next_hop({start_node}, '{goal_id}') "
                            f"returned None, but agent not at goal. Cache may not be built for this goal_id. "
                            f"Cache type: {cache_type}, "
                            f"start_pos={start_pos}, goal_pos={goal_pos}, start_node={start_node}, goal_node={goal_node}"
                        )
                    return None

            # Extract path by following next_hop chain
            path = []
            current = start_node
            visited = set()  # Prevent infinite loops
            max_hops = 1000  # Safety limit

            while current is not None and len(path) < max_hops:
                if current in visited:
                    logger.warning(
                        f"PBRS path extraction: infinite loop detected at node {current}. "
                        f"Path so far: {len(path)} nodes"
                    )
                    break  # Infinite loop detected

                path.append(current)
                visited.add(current)

                # Get next hop toward goal
                next_hop = level_cache.get_next_hop(current, goal_id)

                if next_hop is None or next_hop == current:
                    # Reached goal or dead end
                    break

                current = next_hop

            if len(path) < 2:
                logger.debug(
                    f"PBRS path too short: {len(path)} nodes. "
                    f"start_node={start_node}, goal_id={goal_id}"
                )
                return None

            # DIAGNOSTIC: Log path quality for validation
            # Sample first few hops for diagnosis
            sample_hops = path[: min(5, len(path))]
            logger.warning(
                f"[PBRS_PATH_EXTRACT] Extracted {len(path)} nodes from level cache. "
                f"start_node={start_node}, goal_id={goal_id}, "
                f"first_hops={sample_hops}"
            )

            return path

        except Exception as e:
            logger.debug(f"get_pbrs_path_for_visualization failed: {e}")
            return None

    def get_reward_config(self) -> "RewardConfig":
        """Get reward configuration for curriculum updates.

        This method is callable via env_method from SubprocVecEnv to access
        the reward configuration for updating curriculum-aware reward components.

        Returns:
            RewardConfig instance
        """
        return self.reward_calculator.config

    def __getstate__(self):
        """Custom pickle method to handle non-picklable pygame objects and support vectorization."""
        state = self.__dict__.copy()

        # Remove the entire nplay_headless object as it contains pygame objects
        # It will be recreated when needed after unpickling
        if "nplay_headless" in state:
            # Store initialization parameters instead
            state["_nplay_headless_params"] = {
                "render_mode": getattr(
                    self.nplay_headless, "render_mode", "grayscale_array"
                ),
                "enable_animation": getattr(
                    self.nplay_headless, "enable_animation", False
                ),
                "enable_logging": getattr(self.nplay_headless, "enable_logging", False),
                "enable_debug_overlay": getattr(
                    self.nplay_headless, "enable_debug_overlay", False
                ),
                "seed": getattr(self.nplay_headless, "seed", None),
            }
            del state["nplay_headless"]

        # Remove non-picklable objects that will be recreated
        non_picklable_attrs = ["entity_extractor", "map_loader"]

        for attr in non_picklable_attrs:
            if attr in state:
                del state[attr]

        # Clear observation cache (shouldn't be pickled)
        if "_cached_observation" in state:
            del state["_cached_observation"]

        return state

    def __setstate__(self, state):
        """Custom unpickle method to restore the environment and support vectorization."""
        self.__dict__.update(state)

        # Recreate nplay_headless if it was removed during pickling
        if not hasattr(self, "nplay_headless") and hasattr(
            self, "_nplay_headless_params"
        ):
            from ..nplay_headless import NPlayHeadless

            # Recreate nplay_headless with stored parameters
            params = self._nplay_headless_params
            self.nplay_headless = NPlayHeadless(
                render_mode=params["render_mode"],
                enable_animation=params["enable_animation"],
                enable_logging=params["enable_logging"],
                enable_debug_overlay=params["enable_debug_overlay"],
                seed=params["seed"],
            )
            # Clean up temporary params
            delattr(self, "_nplay_headless_params")

        # Recreate entity extractor
        if not hasattr(self, "entity_extractor"):
            self.entity_extractor = EntityExtractor(self.nplay_headless)

        # Recreate map loader
        if not hasattr(self, "map_loader"):
            self.map_loader = EnvMapLoader(
                self.nplay_headless,
                getattr(self, "rng", None),
                getattr(self, "eval_mode", False),
                getattr(self, "custom_map_path", None),
                test_dataset_path=getattr(self, "test_dataset_path", None),
            )

        # Initialize observation cache
        if not hasattr(self, "_cached_observation"):
            self._cached_observation = None

        # Mark that we need to reinitialize on next reset
        self._needs_reinit = True
