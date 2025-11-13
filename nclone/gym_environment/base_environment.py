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
    PLAYER_FRAME_WIDTH,
    PLAYER_FRAME_HEIGHT,
    MAX_TIME_IN_FRAMES,
    RENDERED_VIEW_WIDTH,
    RENDERED_VIEW_HEIGHT,
    LEVEL_WIDTH,
    LEVEL_HEIGHT,
    FEATURES_PER_DOOR,
)
from .reward_calculation.main_reward_calculator import RewardCalculator
from .observation_processor import ObservationProcessor
from .truncation_checker import TruncationChecker
from .entity_extractor import EntityExtractor
from .env_map_loader import EnvMapLoader
from .reward_calculation.reward_constants import PBRS_GAMMA
from ..constants.entity_types import EntityType
from ..entity_classes.entity_door_locked import EntityDoorLocked

logger = logging.getLogger(__name__)


class BaseNppEnvironment(gymnasium.Env):
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
        enable_short_episode_truncation: bool = False,
        seed: Optional[int] = None,
        eval_mode: bool = False,
        custom_map_path: Optional[str] = None,
        test_dataset_path: Optional[str] = None,
        enable_augmentation: bool = True,
        augmentation_config: Optional[Dict[str, Any]] = None,
        pbrs_gamma: float = PBRS_GAMMA,
    ):
        """
        Initialize the base N++ environment.

        Args:
            render_mode: Rendering mode ("human" or "grayscale_array")
            enable_animation: Enable animation in rendering
            enable_logging: Enable debug logging
            enable_debug_overlay: Enable debug overlay visualization
            enable_short_episode_truncation: Enable episode truncation on lack of progress
            seed: Random seed for reproducibility
            eval_mode: Use evaluation maps instead of training maps
            custom_map_path: Path to custom map file
            test_dataset_path: Path to test dataset directory for evaluation
            enable_augmentation: Enable frame augmentation
            augmentation_config: Augmentation configuration dictionary
            pbrs_gamma: PBRS discount factor
        """
        super().__init__()

        # Store configuration
        self.render_mode = render_mode
        self.enable_animation = enable_animation
        self.enable_logging = enable_logging
        self.custom_map_path = custom_map_path
        self.test_dataset_path = test_dataset_path
        self.eval_mode = eval_mode

        # Initialize core game interface
        # Note: Grayscale rendering is automatic in headless mode (grayscale_array)
        self.nplay_headless = NPlayHeadless(
            render_mode=render_mode,
            enable_animation=enable_animation,
            enable_logging=enable_logging,
            enable_debug_overlay=enable_debug_overlay,
            seed=seed,
        )

        # Initialize action space (6 actions: NOOP, Left, Right, Jump, Jump+Left, Jump+Right)
        self.action_space = discrete.Discrete(6)

        # Initialize RNG
        self.rng = random.Random(seed)

        # Track reward for the current episode
        self.current_ep_reward = 0

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
        )

        self.reward_calculator = RewardCalculator(pbrs_gamma=pbrs_gamma)

        # Initialize truncation checker
        self.truncation_checker = TruncationChecker(
            self, enable_short_episode_truncation=enable_short_episode_truncation
        )

        # Store all configuration flags for logging and debugging
        self.config_flags = {
            "render_mode": render_mode,
            "enable_animation": enable_animation,
            "enable_logging": enable_logging,
            "enable_debug_overlay": enable_debug_overlay,
            "enable_short_episode_truncation": enable_short_episode_truncation,
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

        # Load the initial map
        self.map_loader.load_map()

    def _build_base_observation_space(self) -> SpacesDict:
        """Build the base observation space."""
        obs_spaces = {
            # Player-centered frame
            "player_frame": box.Box(
                low=0,
                high=255,
                shape=(
                    PLAYER_FRAME_HEIGHT,
                    PLAYER_FRAME_WIDTH,
                    1,
                ),
                dtype=np.uint8,
            ),
            # Global view frame
            "global_view": box.Box(
                low=0,
                high=255,
                shape=(RENDERED_VIEW_HEIGHT, RENDERED_VIEW_WIDTH, 1),
                dtype=np.uint8,
            ),
            # Game state features (ninja state + entity states)
            "game_state": box.Box(
                low=-1,
                high=1,
                shape=(
                    GAME_STATE_CHANNELS,
                ),  # Now 52: 29 physics + 15 objectives + 5 mines + 3 progress
                dtype=np.float32,
            ),
            # Action mask for invalid action filtering
            "action_mask": box.Box(
                low=0,
                high=1,
                shape=(6,),  # 6 actions in N++
                dtype=np.int8,
            ),
        }
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

    def _check_observation_for_nan(
        self, obs: Dict[str, Any], obs_name: str = "observation"
    ):
        """Check observation dictionary for NaN values and log details.

        Args:
            obs: Observation dictionary to check
            obs_name: Name identifier for logging (e.g., "prev_obs", "curr_obs")
        """
        nan_found = False
        nan_components = []

        for key, value in obs.items():
            if isinstance(value, np.ndarray):
                if np.isnan(value).any():
                    nan_found = True
                    nan_count = np.isnan(value).sum()
                    nan_components.append(f"{key} (array, {nan_count} NaN values)")
            elif isinstance(value, (float, np.floating)):
                if np.isnan(value):
                    nan_found = True
                    nan_components.append(f"{key} (scalar, value={value})")
            elif isinstance(value, (list, tuple)):
                # Check list/tuple elements
                for i, item in enumerate(value):
                    if isinstance(item, (float, np.floating)) and np.isnan(item):
                        nan_found = True
                        nan_components.append(f"{key}[{i}] (scalar, value={item})")
                    elif isinstance(item, np.ndarray) and np.isnan(item).any():
                        nan_found = True
                        nan_count = np.isnan(item).sum()
                        nan_components.append(
                            f"{key}[{i}] (array, {nan_count} NaN values)"
                        )

        if nan_found:
            raise ValueError(
                f"[OBS_NAN] NaN detected in {obs_name}! Components with NaN: {nan_components}. "
                f"Raw values: player_x={obs.get('player_x')}, player_y={obs.get('player_y')}, "
                f"player_xspeed={obs.get('player_xspeed')}, player_yspeed={obs.get('player_yspeed')}"
            )

        return nan_found

    def step(self, action: int):
        """Execute one environment step with enhanced episode info.

        Template method that defines the step execution flow with hooks
        for subclasses to extend behavior at specific points.
        """
        try:
            # Get previous observation
            prev_obs = self._get_observation()

            # Check for NaN in previous observation
            if self._check_observation_for_nan(prev_obs, "prev_obs"):
                print(
                    f"[STEP_NAN] NaN detected in prev_obs before action execution. "
                    f"Action: {action}"
                )

            # Record action for debug visualization
            if hasattr(self, "_record_action_for_debug"):
                self._record_action_for_debug(action)

            # Execute action
            action_hoz, action_jump = self._actions_to_execute(action)
            self.nplay_headless.tick(action_hoz, action_jump)

            # Invalidate observation cache since game state changed
            self._cached_observation = None

            # Hook: After action execution, before observation
            self._post_action_hook()

            # Get current observation
            curr_obs = self._get_observation()

            # Check for NaN in current observation BEFORE reward calculation
            if self._check_observation_for_nan(curr_obs, "curr_obs"):
                print(
                    f"[STEP_NAN] NaN detected in curr_obs after action execution. "
                    f"Action: {action}, prev_obs had NaN: {self._check_observation_for_nan(prev_obs, 'prev_obs')}"
                )

            terminated, truncated, player_won = self._check_termination()

            # Hook: After observation, before reward calculation
            self._pre_reward_hook(curr_obs, player_won)

            # Calculate reward (pass action for NOOP penalty)
            reward = self._calculate_reward(curr_obs, prev_obs, action)

            # Check reward for NaN
            if np.isnan(reward):
                print(
                    f"[STEP_NAN] NaN detected in reward after calculation. "
                    f"Reward value: {reward}, action: {action}, "
                    f"player_pos=({curr_obs.get('player_x')}, {curr_obs.get('player_y')})"
                )

            # Hook: Modify reward if needed
            reward = self._modify_reward_hook(reward, curr_obs, player_won, terminated)

            # Check reward again after modification
            if np.isnan(reward):
                print(
                    f"[STEP_NAN] NaN detected in reward after modification hook. "
                    f"Reward value: {reward}"
                )

            self.current_ep_reward += reward

            # Process observation for training
            processed_obs = self._process_observation(curr_obs)

            # Check processed observation for NaN
            if isinstance(processed_obs, dict):
                if self._check_observation_for_nan(processed_obs, "processed_obs"):
                    print(
                        f"[STEP_NAN] NaN detected in processed_obs after processing. "
                        f"Action: {action}"
                    )

            # Build episode info
            info = self._build_episode_info(player_won, terminated, truncated)

            # Hook: Add additional info fields
            self._extend_info_hook(info)

            # Validate observation before returning
            if self._check_observation_for_nan(processed_obs, "step_obs"):
                raise ValueError(
                    f"NaN detected after step (action={action}). "
                    f"Frame: {self.nplay_headless.sim.frame}"
                )

            return processed_obs, reward, terminated, truncated, info

        except Exception as e:
            logger.error(
                f"[STEP_EXCEPTION] Exception in step() method: {type(e).__name__}: {e}. "
                f"Action: {action}, "
                f"Current ep reward: {self.current_ep_reward}"
            )
            # Re-raise to maintain normal error handling
            raise

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

        # === SUBTASK FEATURES (if hierarchical enabled) ===
        if "subtask_features" in obs:
            subtask = obs["subtask_features"]
            if isinstance(subtask, np.ndarray):
                feature_names = [
                    "subtask_type",
                    "progress",
                    "priority",
                    "completion_bonus",
                ]
                for i in range(min(len(subtask), len(feature_names))):
                    metrics[f"obs/subtask_features/{feature_names[i]}"] = float(
                        subtask[i]
                    )

        # === KEY PHYSICS STATE ===
        metrics["obs/physics/xspeed"] = float(obs["player_xspeed"])
        metrics["obs/physics/yspeed"] = float(obs["player_yspeed"])
        metrics["obs/switch_activated"] = float(obs["switch_activated"])
        metrics["obs/time_remaining"] = float(obs["time_remaining"])

        return metrics

    def _extend_info_hook(self, info: Dict[str, Any]):
        """Hook to add additional fields to info dictionary.

        Args:
            info: Info dictionary to extend (modified in place)

        Subclasses can override this to add custom info fields.
        """
        # Generate comprehensive diagnostic metrics for TensorBoard
        curr_obs = self._get_observation()
        diagnostic_metrics = self.reward_calculator.get_diagnostic_metrics(curr_obs)
        # Add to info dict for callback to log
        info["diagnostic_metrics"] = diagnostic_metrics

        # Generate observation metrics for TensorBoard logging
        observation_metrics = self._get_observation_metrics(curr_obs)
        info["observation_metrics"] = observation_metrics

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
            "r": self.current_ep_reward,
            "l": self.nplay_headless.sim.frame,
            "level_id": self.map_loader.current_map_name,
            "config_flags": self.config_flags.copy(),
            "terminal_impact": self.nplay_headless.get_ninja_terminal_impact(),
        }

        # Add PBRS component rewards if available
        if hasattr(self.reward_calculator, "last_pbrs_components"):
            info["pbrs_components"] = self.reward_calculator.last_pbrs_components.copy()

        return info

    def reset(self, seed=None, options=None):
        """Reset the environment.

        Args:
            seed: Random seed (optional)
            options: Dictionary of reset options. Supported keys:
                - skip_map_load (bool): If True, skip loading a new map.
                  Use this when map has already been loaded externally
                  (e.g., by curriculum wrapper).
        """
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

        # Reset episode reward
        self.current_ep_reward = 0

        # Invalidate observation cache
        self._cached_observation = None

        # Invalidate level data cache on reset (new level loaded)
        # PERFORMANCE: Cache will be rebuilt on first access
        self._cached_level_data = None
        self._cached_entities = None

        # Check if map loading should be skipped (e.g., curriculum already loaded one)
        skip_map_load = False
        if options is not None and isinstance(options, dict):
            skip_map_load = options.get("skip_map_load", False)

        if not skip_map_load:
            # Load map - this calls sim.load() which calls sim.reset()
            self.map_loader.load_map()
        else:
            # If map loading is skipped, we still need to reset the sim
            # This happens when curriculum wrapper loads the map externally
            self.nplay_headless.reset()

        # Get initial observation and process it
        initial_obs = self._get_observation()
        processed_obs = self._process_observation(initial_obs)

        # Validate observation before returning
        if self._check_observation_for_nan(processed_obs, "reset_obs"):
            raise ValueError(
                "NaN detected immediately after reset. Initialization problem."
            )

        return (processed_obs, {})

    def render(self):
        """Render the environment."""
        # Get debug info from mixin if available, otherwise None
        debug_info = self._debug_info() if hasattr(self, "_debug_info") else None
        return self.nplay_headless.render(debug_info)

    def _get_observation(self) -> Dict[str, Any]:
        """Get the current observation from the game state."""
        # Return cached observation if valid
        if self._cached_observation is not None:
            return self._cached_observation

        # Calculate time remaining feature
        time_remaining = (
            MAX_TIME_IN_FRAMES - self.nplay_headless.sim.frame
        ) / MAX_TIME_IN_FRAMES

        ninja_state = self.nplay_headless.get_ninja_state()

        # game_state contains ninja_state (29 features) + path-aware objectives (15) +
        # mine features (8) + progress features (3) + sequential goals (3) = 58 total features
        # Start with ninja_state, will be extended by NppEnvironment if path-aware features enabled
        game_state = ninja_state

        # Get entity states for PBRS hazard detection
        # Try to use reachable area scale if available (from GraphMixin)
        area_scale = None
        area_scale = self._get_reachable_area_scale()
        entity_states_raw = self.nplay_headless.get_entity_states(area_scale=area_scale)

        entities = []
        # Extract entities relevant for Deep RL agent:
        # - Toggle mines (hazards)
        toggle_mines = self.nplay_headless.sim.entity_dic.get(
            EntityType.TOGGLE_MINE, []
        )
        entities.extend(toggle_mines)

        # - Toggled mines (active hazards)
        toggled_mines = self.nplay_headless.sim.entity_dic.get(
            EntityType.TOGGLE_MINE_TOGGLED, []
        )
        entities.extend(toggled_mines)

        # - Locked doors (obstacles requiring switch activation)
        locked_doors = self.nplay_headless.locked_doors()
        entities.extend(locked_doors)

        # - Locked door switches (objectives for opening locked doors)
        locked_door_switches = self.nplay_headless.locked_door_switches()
        entities.extend(locked_door_switches)

        # Get ninja properties for PBRS impact risk calculation
        ninja_vel_old = self.nplay_headless.ninja_velocity_old()
        ninja_floor_norm = self.nplay_headless.ninja_floor_normal()
        ninja_ceiling_norm = self.nplay_headless.ninja_ceiling_normal()

        # Get current ninja velocity for momentum rewards
        ninja_vel = self.nplay_headless.ninja_velocity()

        # DIAGNOSTIC: Log what positions we're extracting from simulator
        ninja_pos = self.nplay_headless.ninja_position()
        switch_pos = self.nplay_headless.exit_switch_position()
        exit_pos = self.nplay_headless.exit_door_position()

        obs = {
            "screen": self.render(),
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
            "buffered_jump_executed": self.nplay_headless.ninja_last_jump_was_buffered(),
            "floor_normal_x": ninja_floor_norm[0],
            "floor_normal_y": ninja_floor_norm[1],
            "ceiling_normal_x": ninja_ceiling_norm[0],
            "ceiling_normal_y": ninja_ceiling_norm[1],
            "switch_activated": self.nplay_headless.exit_switch_activated(),
            "switch_x": switch_pos[0],
            "switch_y": switch_pos[1],
            "exit_door_x": exit_pos[0],
            "exit_door_y": exit_pos[1],
            "time_remaining": time_remaining,
            "sim_frame": self.nplay_headless.sim.frame,
            "entity_states": entity_states_raw,  # For PBRS hazard detection
            "entities": entities,  # Entity objects (mines, locked doors, switches)
            "locked_doors": locked_doors,  # For hierarchical navigation
            "locked_door_switches": locked_door_switches,  # For hierarchical navigation
            # Update path guidance predictor before getting action mask
            # This ensures path-based masking has current data
            "action_mask": self._get_action_mask_with_path_update(
                ninja_pos, switch_pos, exit_pos
            ),
        }

        # Cache the computed observation before returning
        self._cached_observation = obs
        return obs

    def _get_action_mask_with_path_update(
        self,
        ninja_pos: Tuple[float, float],
        switch_pos: Tuple[float, float],
        exit_pos: Tuple[float, float],
    ) -> np.ndarray:
        """Get action mask with path guidance predictor updated.

        Updates the path guidance predictor's cached path before retrieving
        the action mask. This ensures path-based masking uses current ninja
        position and goal state.

        Args:
            ninja_pos: Current ninja position (x, y)
            switch_pos: Exit switch position (x, y)
            exit_pos: Exit door position (x, y)

        Returns:
            Action mask as numpy array of int8
        """
        # Get action mask from ninja
        return np.array(self.nplay_headless.get_action_mask(), dtype=np.int8)

    def _check_termination(self) -> Tuple[bool, bool, bool]:
        """
        Check if the episode should be terminated.

        Returns:
            Tuple containing:
            - terminated: True if episode should be terminated, False otherwise
            - truncated: True if episode should be truncated, False otherwise
            - player_won: True if player won, False otherwise
        """
        player_won = self.nplay_headless.ninja_has_won()
        player_dead = self.nplay_headless.ninja_has_died()
        terminated = player_won or player_dead

        # Check truncation using our truncation checker
        ninja_x, ninja_y = self.nplay_headless.ninja_position()
        should_truncate, reason = self.truncation_checker.update(ninja_x, ninja_y)
        if should_truncate and self.enable_logging:
            print(f"Episode terminated due to time: {reason}")

        # We also terminate if the truncation state is reached, that way we can
        # learn from the episode, since our time remaining is in our observation
        return terminated or should_truncate, False, player_won

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
