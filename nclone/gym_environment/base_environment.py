"""
Base N++ environment class with core functionality.

This module contains the core environment functionality without the specialized
mixins for graph, reachability, and debug features.
"""

import logging
import time
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
    PLAYER_FRAME_WIDTH,
    PLAYER_FRAME_HEIGHT,
    RENDERED_VIEW_WIDTH,
    RENDERED_VIEW_HEIGHT,
    LEVEL_WIDTH,
    LEVEL_HEIGHT,
    FEATURES_PER_DOOR,
)
from .reward_calculation.main_reward_calculator import RewardCalculator
from .reward_calculation.reward_config import RewardConfig
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
        enable_augmentation: bool = True,
        augmentation_config: Optional[Dict[str, Any]] = None,
        pbrs_gamma: float = PBRS_GAMMA,
        enable_visual_observations: bool = False,
        enable_profiling: bool = False,
        reward_config: Optional[
            RewardConfig
        ] = None,  # RewardConfig for curriculum-aware reward system
        frame_skip: int = 4,  # Frame skip (action repeat) for temporal abstraction
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
            enable_augmentation: Enable frame augmentation
            augmentation_config: Augmentation configuration dictionary
            pbrs_gamma: PBRS discount factor
            enable_visual_observations: If False, skip rendering and visual observation processing
                (graph + state + reachability contain all necessary information)
            enable_profiling: Enable detailed performance profiling (~5% overhead)
        """
        _init_start = time.perf_counter()
        _logger = logging.getLogger(__name__)
        print("[PROFILE] BaseNppEnvironment.__init__ starting...")

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
        self.enable_visual_observations = (
            enable_visual_observations or render_mode == "human"
        )

        # Cache for previous observation to avoid redundant computation
        self._prev_obs_cache = None

        # Initialize core game interface
        # Note: Grayscale rendering is automatic in headless mode (grayscale_array)
        # Skip rendering entirely if visual observations disabled for maximum performance
        self.nplay_headless = NPlayHeadless(
            render_mode=render_mode,
            enable_animation=enable_animation,
            enable_logging=enable_logging,
            enable_debug_overlay=enable_debug_overlay,
            seed=seed,
            enable_rendering=enable_visual_observations,
        )

        # Initialize action space (6 actions: NOOP, Left, Right, Jump, Jump+Left, Jump+Right)
        self.action_space = discrete.Discrete(6)

        # Initialize RNG
        self.rng = random.Random(seed)

        # Track reward for the current episode
        self.current_ep_reward = 0
        self._last_episode_reward = 0  # For reset validation

        # Track last action for death context learning
        self.last_action = None

        # Track episode count for periodic cache clearing in worker processes
        # Used by SharedMemorySubprocVecEnv to trigger cache clearing every N episodes
        self._episode_count = 0

        # Position tracking for route visualization (integrated directly)
        self.current_route = []
        self._warned_about_position = False

        # Frame skip for temporal abstraction (always enabled)
        self.frame_skip = max(1, frame_skip)  # Ensure at least 1

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

        # Initialize observation processor with performance optimizations
        # training_mode=True disables validation for ~12% performance boost
        self.observation_processor = ObservationProcessor(
            enable_augmentation=enable_augmentation,
            augmentation_config=augmentation_config,
            training_mode=not eval_mode,  # Disable validation in training mode
            enable_visual_processing=enable_visual_observations,
        )

        self.reward_calculator = RewardCalculator(
            reward_config=reward_config, pbrs_gamma=pbrs_gamma
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

        # Momentum waypoints cache (per level)
        # Loaded from demonstration analysis, used for momentum-aware PBRS
        self._cached_momentum_waypoints: Optional[List[Any]] = None
        self._cached_waypoints_level_id: Optional[str] = None

        # Kinodynamic database cache (per level)
        # Exhaustive precomputed (position, velocity) reachability for perfect accuracy
        self._cached_kinodynamic_db: Optional[Any] = None
        self._cached_kinodynamic_level_id: Optional[str] = None

        # Load the initial map
        self.map_loader.load_map()

    def _build_base_observation_space(self) -> SpacesDict:
        """Build the base observation space."""
        obs_spaces = {}

        # Only include visual observation spaces if visual observations are enabled
        if self.enable_visual_observations:
            # Player-centered frame
            obs_spaces["player_frame"] = box.Box(
                low=0,
                high=255,
                shape=(
                    PLAYER_FRAME_HEIGHT,
                    PLAYER_FRAME_WIDTH,
                    1,
                ),
                dtype=np.uint8,
            )
            # Global view frame
            obs_spaces["global_view"] = box.Box(
                low=0,
                high=255,
                shape=(RENDERED_VIEW_HEIGHT, RENDERED_VIEW_WIDTH, 1),
                dtype=np.uint8,
            )

        # Game state features (ninja state + entity states) - always available
        obs_spaces["game_state"] = box.Box(
            low=-1,
            high=1,
            shape=(
                GAME_STATE_CHANNELS,
            ),  # Now 41: enhanced physics state (40) + time_remaining (1)
            dtype=np.float32,
        )
        # Action mask for invalid action filtering - always available
        obs_spaces["action_mask"] = box.Box(
            low=0,
            high=1,
            shape=(6,),  # 6 actions in N++
            dtype=np.int8,
        )

        # Note: time_remaining is now included as the 41st feature in game_state

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

    def step(self, action: int):
        """Execute one environment step with frame skip and enhanced episode info.

        With frame_skip enabled, repeats the action for N frames and calculates
        reward once using the initial→final transition.

        Template method that defines the step execution flow with hooks
        for subclasses to extend behavior at specific points.
        """
        logger = logging.getLogger(__name__)

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

            for i in range(self.frame_skip):
                # Execute physics tick (inline profiling check to avoid function call overhead)
                if _profile:
                    t_physics = self._profile_start("physics_tick")
                    headless.tick(action_hoz, action_jump)
                    self._profile_end("physics_tick", t_physics)
                else:
                    headless.tick(action_hoz, action_jump)

                frames_executed = i + 1

                # OPTIMIZATION: Only check death/win per frame (cheap attribute lookup)
                # Skip truncation check in intermediate frames (it's time-based, won't change in 4 frames)
                player_won = headless.ninja_has_won()
                player_dead = headless.ninja_has_died()
                if player_won or player_dead:
                    terminated = True
                    break

            # Check truncation only once after all frames (expensive curriculum-aware check)
            if not terminated:
                truncated = self._check_curriculum_aware_truncation()

            # OPTIMIZATION: Track position only at end of frame_skip (not every physics frame)
            # Route visualization doesn't need sub-frame granularity
            self._track_position_after_step()

            # Invalidate observation cache since game state changed
            self._cached_observation = None

            # Hook: After action execution, before observation
            self._post_action_hook()

            # Get final observation - OBSERVATION GENERATION
            t_obs = self._profile_start("observation_get")
            final_obs = self._get_observation()
            self._profile_end("observation_get", t_obs)

            # Cache observation for reward calculation in next step
            self._prev_obs_cache = final_obs

            # Get player_won for reward calculation
            player_won = final_obs.get("player_won", False)

            # Hook: After observation, before reward calculation
            self._pre_reward_hook(final_obs, player_won)

            # Calculate reward using initial→final transition with frame count scaling
            t_reward = self._profile_start("reward_calc")
            reward = self.reward_calculator.calculate_reward(
                final_obs, initial_obs, action, frames_executed=frames_executed
            )
            self._profile_end("reward_calc", t_reward)

            # Update dynamic truncation limit if PBRS surface area is now available
            self._update_dynamic_truncation_if_needed()

            # Apply timeout penalty for truncation (distinct from death)
            # Timeout indicates inefficient navigation or getting stuck - should be strongly discouraged
            if truncated and not terminated:
                from .reward_calculation.reward_constants import TIMEOUT_PENALTY

                reward += TIMEOUT_PENALTY

            # Hook: Modify reward if needed
            reward = self._modify_reward_hook(reward, final_obs, player_won, terminated)

            self.current_ep_reward += reward

            # Process observation for training
            processed_obs = self._process_observation(final_obs)

            # Build episode info
            info = self._build_episode_info(player_won, terminated, truncated)

            # Add frame skip stats to info
            info["frame_skip_stats"] = {
                "skip_value": self.frame_skip,
                "frames_executed": frames_executed,
            }

            # Hook: Add additional info fields
            self._extend_info_hook(info)

            return processed_obs, reward, terminated, truncated, info

        except Exception as e:
            logger.error(
                f"[STEP_EXCEPTION] Exception in step() method: {type(e).__name__}: {e}. "
                f"Action: {action}, "
                f"Current ep reward: {self.current_ep_reward}"
            )
            # Re-raise to maintain normal error handling
            raise

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
        # Hook: After observation, before reward calculation
        player_won = final_obs.get("player_won", False)
        self._pre_reward_hook(final_obs, player_won)

        # Calculate reward with frame count for time penalty scaling
        reward = self.reward_calculator.calculate_reward(
            final_obs, initial_obs, action, frames_executed=frames_executed
        )

        # Update dynamic truncation limit if PBRS surface area is now available
        self._update_dynamic_truncation_if_needed()

        # Apply timeout penalty for truncation (treat truncation as failure)
        terminated = final_obs.get("player_dead", False)
        truncated = self._check_curriculum_aware_truncation()
        if truncated and not terminated:
            from .reward_calculation.reward_constants import TIMEOUT_PENALTY

            reward += TIMEOUT_PENALTY

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

        # === REACHABILITY FEATURES ===
        if "reachability_features" in obs:
            reachability = obs["reachability_features"]
            if isinstance(reachability, np.ndarray):
                for i in range(len(reachability)):
                    metrics[f"obs/reachability_features/feature_{i}"] = float(
                        reachability[i]
                    )

        # === SWITCH STATES ===
        # Log first door's features as example
        if "switch_states" in obs:
            switch_states = obs["switch_states"]
            if isinstance(switch_states, np.ndarray) and len(switch_states) > 0:
                # Log first door's features (5 features: switch_x, switch_y, door_x, door_y, collected)
                for i in range(min(FEATURES_PER_DOOR, len(switch_states))):
                    feature_names = [
                        "switch_x_norm",
                        "switch_y_norm",
                        "door_x_norm",
                        "door_y_norm",
                        "collected",
                    ]
                    if i < len(feature_names):
                        metrics[f"obs/switch_states/door_0_{feature_names[i]}"] = float(
                            switch_states[i]
                        )

        # === KEY PHYSICS STATE ===
        metrics["obs/physics/xspeed"] = float(obs["player_xspeed"])
        metrics["obs/physics/yspeed"] = float(obs["player_yspeed"])
        metrics["obs/switch_activated"] = float(obs["switch_activated"])
        metrics["obs/time_remaining"] = float(obs["time_remaining"])

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
                # No time penalty in rewards -> more generous truncation for exploration
                multiplier = 2.0
            elif phase == "mid":
                # Conditional time penalty -> moderate truncation
                multiplier = 1.5
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
                # No time penalty in rewards -> more generous truncation for exploration
                multiplier = 4.0
            elif phase == "mid":
                # Conditional time penalty -> moderate truncation
                multiplier = 2.0
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
        pass

    def _track_position_after_step(self):
        """Track ninja position after physics step for route visualization.

        Called after each physics tick to record the ninja's trajectory.
        """
        try:
            position = self.nplay_headless.ninja_position()
            if position is not None:
                self.current_route.append((float(position[0]), float(position[1])))
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
        info = {
            "is_success": player_won,
            "terminated": terminated,
            "truncated": truncated,
            "r": self.current_ep_reward,  # Cumulative reward for entire episode
            "l": self.nplay_headless.sim.frame,  # Total frames executed (not actions)
            "level_id": self.map_loader.current_map_name,
            "config_flags": self.config_flags.copy(),
            "terminal_impact": self.nplay_headless.get_ninja_terminal_impact(),
            "exit_switch_pos": self._cached_switch_pos,
            "exit_door_pos": self._cached_exit_pos,
        }

        # Add episode route for visualization if episode is ending
        if terminated or truncated:
            route_length = len(self.current_route)
            if route_length > 0:
                info["episode_route"] = list(self.current_route)
                info["route_length"] = route_length

                # CRITICAL: Clear route immediately after copying to prevent accumulation
                # This fixes a bug where routes would leak across episodes if reset()
                # was not called reliably (e.g., in SubprocVecEnv worker scenarios)
                self.current_route = []
            else:
                logger.warning(
                    f"Episode ended with EMPTY route! "
                    f"frames={self.nplay_headless.sim.frame}"
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

            # Add checkpoint route if this episode started from a Go-Explore checkpoint
            # This path was recorded during checkpoint replay at episode start
            if self._checkpoint_route and len(self._checkpoint_route) > 0:
                info["checkpoint_route"] = list(self._checkpoint_route)
                # Clear after copying to prevent leaking to next episode
                self._checkpoint_route = []

            # Add checkpoint tracking for logging/analysis
            # Allows distinguishing checkpoint vs spawn episodes in TensorBoard
            info["from_checkpoint"] = self._from_checkpoint
            if self._from_checkpoint:
                info["checkpoint_base_reward"] = self._checkpoint_base_reward
                # Total cumulative reward = base (to reach checkpoint) + episode (from checkpoint)
                info["total_cumulative_reward"] = (
                    self._checkpoint_base_reward + self.current_ep_reward
                )

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

        # Add curriculum info if using custom map (for route visualization)
        if self.custom_map_path:
            info["curriculum_stage"] = "custom_map"
            info["curriculum_generator"] = "custom"

        # Add PBRS component rewards (always available for TensorBoard logging)
        # This provides detailed breakdown: pbrs_reward, time_penalty, total_reward, etc.
        if hasattr(self.reward_calculator, "last_pbrs_components"):
            info["pbrs_components"] = self.reward_calculator.last_pbrs_components.copy()
        else:
            # Fallback: provide empty dict if reward calculator doesn't track components
            info["pbrs_components"] = {}

        # Add episode-level PBRS metrics for PBRSLoggingCallback (SubprocVecEnv compatible)
        # Only include when episode ends (terminated or truncated)
        if terminated or truncated:
            if hasattr(self.reward_calculator, "get_episode_reward_metrics"):
                info["episode_pbrs_metrics"] = (
                    self.reward_calculator.get_episode_reward_metrics()
                )

            # Add reward config state for curriculum tracking
            if hasattr(self.reward_calculator, "get_config_state"):
                info["reward_config_state"] = self.reward_calculator.get_config_state()

        return info

    def reset(self, seed=None, options=None):
        """Reset the environment.

        Args:
            seed: Random seed (optional)
            options: Dictionary of reset options. Supported keys:
                - skip_map_load (bool): If True, skip loading a new map.
                  Use this when map has already been loaded externally
                  (e.g., by curriculum wrapper).
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

        # Reset position tracking for route visualization
        self.current_route = []
        self._warned_about_position = False

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

        # Invalidate momentum waypoints cache (new level = new waypoints)
        self._cached_momentum_waypoints = None
        self._cached_waypoints_level_id = None

        # Invalidate kinodynamic database cache (new level = new database)
        self._cached_kinodynamic_db = None
        self._cached_kinodynamic_level_id = None

        # Check if map loading should be skipped (e.g., curriculum already loaded one)
        skip_map_load = False
        checkpoint = None
        if options is not None and isinstance(options, dict):
            skip_map_load = options.get("skip_map_load", False)
            checkpoint = options.get("checkpoint", None)

        if not skip_map_load:
            # Load map - this calls sim.load() which calls sim.reset()
            self.map_loader.load_map()
        else:
            # If map loading is skipped, we still need to reset the sim
            # This happens when curriculum wrapper loads the map externally
            self.nplay_headless.reset()

        # If checkpoint provided, replay action sequence to restore state
        if checkpoint is not None:
            return self._reset_to_checkpoint(checkpoint)

        # Load momentum waypoints for current level (if available)
        # This enables momentum-aware PBRS for levels requiring backtracking
        self._update_momentum_waypoints_for_current_level()

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

        # Replay action sequence with frame skip matching original step() behavior
        # Each action in the sequence was executed for frame_skip frames originally
        hor_input, jump_input = 0, 0
        for action in action_sequence:
            hor_input, jump_input = self._actions_to_execute(action)
            # Replay for frame_skip ticks to match original step() execution
            for _ in range(self.frame_skip):
                self.nplay_headless.tick(hor_input, jump_input)
                frames_replayed += 1

                # Track position for checkpoint route visualization
                pos = self.nplay_headless.ninja_position()
                if pos is not None:
                    checkpoint_route.append((float(pos[0]), float(pos[1])))

                # Check if ninja died during replay - stop if so
                if self.nplay_headless.ninja_has_died():
                    ninja_died_during_replay = True
                    logger.warning(
                        f"Ninja died during checkpoint replay at frame {frames_replayed} "
                        f"(action {actions_replayed + 1}/{len(action_sequence)})"
                    )
                    break

            # Track action in current sequence (checkpoint actions become our history)
            self._action_sequence.append(action)
            actions_replayed += 1

            if ninja_died_during_replay:
                break

        self._checkpoint_replay_in_progress = False

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
            position_valid = error < 0.1  # Allow small floating point error
            if not position_valid:
                logger.warning(
                    f"Checkpoint replay position mismatch: "
                    f"expected={expected_position}, actual={actual_position}, "
                    f"error={error:.6f}px | "
                    f"spawn={spawn_pos}, actions={actions_replayed}/{len(action_sequence)}, "
                    f"frames={frames_replayed}, died={ninja_died_during_replay}"
                )

        # Get observation after replay
        initial_obs = self._get_observation()
        processed_obs = self._process_observation(initial_obs)

        info = {
            "checkpoint_replay": True,
            "replay_frames": len(action_sequence),
            "position_valid": position_valid,
            "checkpoint_cell": getattr(checkpoint, "cell", None),
            "checkpoint_distance": getattr(checkpoint, "distance_to_goal", None),
            "checkpoint_route": checkpoint_route,  # Path taken during checkpoint replay
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

        # Replay action sequence with frame skip matching original step() behavior
        # Each action in the sequence was executed for frame_skip frames originally
        for action in action_sequence:
            hor_input, jump_input = self._actions_to_execute(action)
            # Replay for frame_skip ticks to match original step() execution
            for _ in range(self.frame_skip):
                self.nplay_headless.tick(hor_input, jump_input)
                frames_replayed += 1

                # Track position for checkpoint route visualization
                pos = self.nplay_headless.ninja_position()
                if pos is not None:
                    checkpoint_route.append((float(pos[0]), float(pos[1])))

                # Check if ninja died during replay - stop if so
                if self.nplay_headless.ninja_has_died():
                    ninja_died_during_replay = True
                    logger.warning(
                        f"Ninja died during checkpoint replay at frame {frames_replayed} "
                        f"(action {actions_replayed + 1}/{len(action_sequence)})"
                    )
                    break

            self._action_sequence.append(action)
            actions_replayed += 1

            if ninja_died_during_replay:
                break

        self._checkpoint_replay_in_progress = False

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
            position_valid = error < 0.1
            if not position_valid:
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
                logger.warning(
                    f"Checkpoint replay position mismatch: "
                    f"expected={expected_position}, actual={actual_position}, "
                    f"error={error:.6f}px | "
                    f"spawn={spawn_pos}, actions={actions_replayed}/{len(action_sequence)}, "
                    f"frames={frames_replayed}, died={ninja_died_during_replay} | "
                    f"first_actions={first_10}, last_actions={last_10}, "
                    f"action_dist={action_dist}"
                )

        # Get observation after replay
        obs = self._get_observation()
        processed_obs = self._process_observation(obs)

        info = {
            "checkpoint_replay": True,
            "replay_frames": len(action_sequence),
            "position_valid": position_valid,
            "checkpoint_cell": getattr(checkpoint, "cell", None),
            "checkpoint_distance": getattr(checkpoint, "distance_to_goal", None),
            "checkpoint_route": checkpoint_route,  # Path taken during checkpoint replay
        }

        if self.enable_logging:
            logger.debug(
                f"Checkpoint replay from wrapper complete: {len(action_sequence)} actions, "
                f"checkpoint_route has {len(checkpoint_route)} positions"
            )

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

        # game_state contains ninja_state (40 enhanced physics features) + time_remaining (1) = 41 total
        # This is the complete physics state - path-aware objectives, mines, etc. are now in the graph
        # Always append time_remaining to create consistent 41D state
        import numpy as np

        game_state = np.append(ninja_state, time_remaining)

        # Get entity states for PBRS hazard detection
        # MEMORY OPTIMIZATION: Use cached static entity/position data
        # These values never change during an episode, only on reset

        # Cache static positions on first access
        # CRITICAL: Validate positions are not (0, 0) which indicates entities not loaded
        if self._cached_switch_pos is None:
            switch_pos = self.nplay_headless.exit_switch_position()
            # Only cache if valid (not the default (0, 0) returned when entities missing)
            if switch_pos != (0, 0):
                self._cached_switch_pos = switch_pos
            else:
                # Don't cache (0, 0) - force re-fetch next time
                # Log warning to help diagnose curriculum loading issues
                logger.warning(
                    f"exit_switch_position() returned (0, 0) - entities may not be loaded yet. "
                    f"entity_dic keys: {list(self.nplay_headless.sim.entity_dic.keys())}"
                )
        if self._cached_exit_pos is None:
            exit_pos = self.nplay_headless.exit_door_position()
            if exit_pos != (0, 0):
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
            "switch_activated": self.nplay_headless.exit_switch_activated(),
            "switch_x": switch_pos[0],
            "switch_y": switch_pos[1],
            "exit_door_x": exit_pos[0],
            "exit_door_y": exit_pos[1],
            "sim_frame": self.nplay_headless.sim.frame,
            "entities": entities,  # Entity objects (mines, locked doors, switches)
            "locked_doors": locked_doors,
            "locked_door_switches": locked_door_switches,
            "action_mask": self._get_action_mask_with_path_update(),
        }

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

        # Only add screen if visual observations enabled
        # When disabled, graph + state + reachability contain all necessary information
        if self.enable_visual_observations:
            obs["screen"] = self.render()

        # Cache the computed observation before returning
        self._cached_observation = obs
        return obs

    def _get_action_mask_with_path_update(self) -> np.ndarray:
        """Get action mask with path guidance predictor updated.

        Updates the path guidance predictor's cached path before retrieving
        the action mask. This ensures path-based masking uses current ninja
        position and goal state.

        Args:
            ninja_pos: Current ninja position (x, y)
            switch_pos: Exit switch position (x, y)
            exit_pos: Exit door position (x, y)

        Returns:
            Action mask as numpy array of int8 (defensive copy to prevent memory sharing)
        """
        return np.array(self.nplay_headless.get_action_mask(), dtype=np.int8)

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
        return self.reward_calculator.calculate_reward(curr_obs, prev_obs, action)

    def _process_observation(self, obs):
        """Process the observation from the environment."""
        return self.observation_processor.process_observation(obs)

    def _reset_reward_calculator(self):
        """Reset the reward calculator."""
        self.reward_calculator.reset()

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

        return LevelData(
            start_position=start_position,
            tiles=tiles,
            entities=entities,
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

    def _load_momentum_waypoints_if_available(self) -> Optional[List[Any]]:
        """Load momentum waypoints for current level from cache if available.

        Momentum waypoints are extracted from expert demonstrations and cached
        per level. They inform PBRS potential calculation for momentum-aware
        reward shaping.

        Returns:
            List of MomentumWaypoint objects if cache exists, None otherwise
        """
        level_id = self.map_loader.current_map_name

        # Check if already cached for this level
        if (
            self._cached_waypoints_level_id == level_id
            and self._cached_momentum_waypoints is not None
        ):
            return self._cached_momentum_waypoints

        # Try to load from cache
        try:
            from ..analysis.momentum_waypoint_extractor import MomentumWaypointExtractor

            waypoints = MomentumWaypointExtractor.load_waypoints_from_cache(level_id)

            # Cache the result (even if None)
            self._cached_momentum_waypoints = waypoints
            self._cached_waypoints_level_id = level_id

            if waypoints:
                logger.debug(
                    f"Loaded {len(waypoints)} momentum waypoints for level {level_id}"
                )
            else:
                logger.debug(f"No momentum waypoints cache found for level {level_id}")

            return waypoints

        except Exception as e:
            logger.debug(f"Failed to load momentum waypoints: {e}")
            return None

    def _load_kinodynamic_database_if_available(self) -> Optional[Any]:
        """Load kinodynamic database for current level if available.

        Kinodynamic databases are precomputed using build_kinodynamic_database.py
        and provide perfect velocity-aware reachability with O(1) queries.

        Returns:
            KinodynamicDatabase instance if available, None otherwise
        """
        level_id = self.map_loader.current_map_name

        # Check if already cached
        if (
            self._cached_kinodynamic_level_id == level_id
            and self._cached_kinodynamic_db is not None
        ):
            return self._cached_kinodynamic_db

        # Try to load from cache
        try:
            from ..graph.reachability.kinodynamic_database import KinodynamicDatabase

            db = KinodynamicDatabase.load(f"kinodynamic_db/{level_id}.npz")

            # Cache the result (even if None)
            self._cached_kinodynamic_db = db
            self._cached_kinodynamic_level_id = level_id

            if db:
                stats = db.get_statistics()
                logger.info(
                    f"Loaded kinodynamic database for {level_id}: "
                    f"{stats['num_nodes']} nodes, {stats['reachable_pairs']:,} reachable pairs"
                )
            else:
                logger.debug(f"No kinodynamic database found for {level_id}")

            return db

        except Exception as e:
            logger.debug(f"Failed to load kinodynamic database: {e}")
            return None

    def _update_momentum_waypoints_for_current_level(self) -> None:
        """Update PBRS calculator with momentum waypoints and kinodynamic database.

        Called during reset after map is loaded. Loads:
        1. Kinodynamic database (if available) - most accurate
        2. Momentum waypoints (if available) - fallback for complex cases
        """
        # Load kinodynamic database (highest priority)
        kinodynamic_db = self._load_kinodynamic_database_if_available()

        # Load momentum waypoints (fallback)
        waypoints = self._load_momentum_waypoints_if_available()

        # Update PBRS calculator
        if hasattr(self.reward_calculator, "pbrs_calculator"):
            pbrs_calc = self.reward_calculator.pbrs_calculator

            # Set kinodynamic database (if available)
            if hasattr(pbrs_calc, "set_kinodynamic_database"):
                pbrs_calc.set_kinodynamic_database(kinodynamic_db)

            # Set momentum waypoints (if available)
            if hasattr(pbrs_calc, "set_momentum_waypoints"):
                pbrs_calc.set_momentum_waypoints(waypoints)
                if waypoints and not kinodynamic_db:
                    logger.info(
                        f"Enabled momentum-aware PBRS with {len(waypoints)} waypoints "
                        f"for level {self.map_loader.current_map_name}"
                    )
            else:
                logger.debug("PBRS calculator does not support momentum waypoints")

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
