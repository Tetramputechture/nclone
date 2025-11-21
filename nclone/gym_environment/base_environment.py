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
    RENDERED_VIEW_WIDTH,
    RENDERED_VIEW_HEIGHT,
    LEVEL_WIDTH,
    LEVEL_HEIGHT,
    FEATURES_PER_DOOR,
)
from .reward_calculation.main_reward_calculator import RewardCalculator
from .reward_calculation.reward_config import RewardConfig
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
        seed: Optional[int] = None,
        eval_mode: bool = False,
        custom_map_path: Optional[str] = None,
        test_dataset_path: Optional[str] = None,
        enable_augmentation: bool = True,
        augmentation_config: Optional[Dict[str, Any]] = None,
        pbrs_gamma: float = PBRS_GAMMA,
        enable_visual_observations: bool = False,
        reward_config: Optional[
            RewardConfig
        ] = None,  # RewardConfig for curriculum-aware reward system
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
        """
        super().__init__()

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

        # Track last action for death context learning
        self.last_action = None

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
        """Execute one environment step with enhanced episode info.

        Template method that defines the step execution flow with hooks
        for subclasses to extend behavior at specific points.
        """
        logger = logging.getLogger(__name__)

        try:
            # Get previous observation for reward calculation
            # Note: We don't validate actions against prev_obs mask because:
            # 1. The policy already validated with the mask from _last_obs
            # 2. The environment state may have changed since policy selection
            # 3. Validation belongs at policy level, not environment level
            prev_obs = self._prev_obs_cache
            if prev_obs is None:
                prev_obs = self._get_observation()

            # Record action for debug visualization
            if hasattr(self, "_record_action_for_debug"):
                self._record_action_for_debug(action)

            # Track last action for death context
            self.last_action = action

            # Execute action
            action_hoz, action_jump = self._actions_to_execute(action)
            self.nplay_headless.tick(action_hoz, action_jump)

            # Invalidate observation cache since game state changed
            self._cached_observation = None

            # Hook: After action execution, before observation
            self._post_action_hook()

            # Get current observation
            curr_obs = self._get_observation()

            # Cache observation for reward calculation in next step
            # Shallow copy dict + explicit action_mask copy prevents wrapper mutations
            self._prev_obs_cache = curr_obs.copy()
            self._prev_obs_cache["action_mask"] = curr_obs["action_mask"].copy()

            # Return a copy so wrappers can't mutate our cached observation
            obs_to_return = curr_obs.copy()
            obs_to_return["action_mask"] = curr_obs["action_mask"].copy()

            terminated, truncated, player_won = self._check_termination()

            # Hook: After observation, before reward calculation
            self._pre_reward_hook(curr_obs, player_won)

            # Calculate reward (pass action for NOOP penalty)
            reward = self._calculate_reward(curr_obs, prev_obs, action)

            # Update dynamic truncation limit if PBRS surface area is now available
            self._update_dynamic_truncation_if_needed()

            # Apply death penalty for truncation (treat truncation as failure)
            # Note: Truncation policy is curriculum-aware and aligns with reward config:
            # - Early phase: No time penalty + generous truncation (exploration focus)
            # - Late phase: Full time penalty + standard truncation (efficiency focus)
            if truncated and not terminated:
                from .reward_calculation.reward_constants import DEATH_PENALTY

                reward += DEATH_PENALTY

            # Hook: Modify reward if needed
            reward = self._modify_reward_hook(reward, curr_obs, player_won, terminated)

            self.current_ep_reward += reward

            # Process observation for training (use returned copy, not original)
            processed_obs = self._process_observation(obs_to_return)

            # Build episode info
            info = self._build_episode_info(player_won, terminated, truncated)

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

    def _check_curriculum_aware_truncation(
        self, ninja_x: float, ninja_y: float
    ) -> Tuple[bool, str]:
        """Check truncation with curriculum-aware policy that respects reward config.

        Philosophy: Truncation serves computational efficiency and training stability,
        but should align with reward curriculum phases:

        - Early phase (no time penalty): 2.0x more generous (more exploration time)
        - Mid phase (conditional penalty): 1.5x more generous (balanced approach)
        - Late phase (full time penalty): 1.0x standard (efficiency focus)

        This balances computational limits with curriculum consistency.

        Args:
            ninja_x: Current ninja x position
            ninja_y: Current ninja y position

        Returns:
            Tuple of (should_truncate: bool, reason: str)
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

        # Apply curriculum-aware truncation limit
        base_limit = self.truncation_checker.current_truncation_limit
        curriculum_limit = int(base_limit * multiplier)

        # Add current position to history (always do this)
        self.truncation_checker.position_history.append((ninja_x, ninja_y))

        # Check against curriculum-adjusted limit
        frames_elapsed = len(self.truncation_checker.position_history)
        should_truncate = frames_elapsed >= curriculum_limit

        if should_truncate:
            if reward_config is not None:
                reason = f"Max frames reached ({curriculum_limit}) [curriculum: {reward_config.training_phase} phase, {multiplier:.1f}x base limit of {base_limit}]"
            else:
                reason = f"Max frames reached ({curriculum_limit})"
        else:
            reason = ""

        return should_truncate, reason

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

        # Reset episode reward
        self.current_ep_reward = 0

        # Reset last action tracking for death context
        self.last_action = None

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

        return (processed_obs, {})

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
        # Try to use reachable area scale if available (from GraphMixin)
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

        # Get deterministic death probabilities for auxiliary learning
        death_probabilities = self.nplay_headless.get_death_probabilities()

        # DIAGNOSTIC: Log what positions we're extracting from simulator
        ninja_pos = self.nplay_headless.ninja_position()
        switch_pos = self.nplay_headless.exit_switch_position()
        exit_pos = self.nplay_headless.exit_door_position()

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
            "buffered_jump_executed": self.nplay_headless.ninja_last_jump_was_buffered(),
            "floor_normal_x": ninja_floor_norm[0],
            "floor_normal_y": ninja_floor_norm[1],
            "ceiling_normal_x": ninja_ceiling_norm[0],
            "ceiling_normal_y": ninja_ceiling_norm[1],
            # Deterministic death probabilities for auxiliary learning
            "mine_death_probability": death_probabilities["mine_death_probability"],
            "terminal_impact_probability": death_probabilities[
                "terminal_impact_probability"
            ],
            "switch_activated": self.nplay_headless.exit_switch_activated(),
            "switch_x": switch_pos[0],
            "switch_y": switch_pos[1],
            "exit_door_x": exit_pos[0],
            "exit_door_y": exit_pos[1],
            "sim_frame": self.nplay_headless.sim.frame,
            "entities": entities,  # Entity objects (mines, locked doors, switches)
            "locked_doors": locked_doors,
            "locked_door_switches": locked_door_switches,
            # Update path guidance predictor before getting action mask
            # This ensures path-based masking has current data
            "action_mask": self._get_action_mask_with_path_update(
                ninja_pos, switch_pos, exit_pos
            ),
            "death_context": self._get_death_context(),
        }

        # DEFENSIVE FIX: Validate mask consistency before returning observation
        # This catches corrupted masks early before they cause issues in the policy
        action_mask = obs["action_mask"]
        if action_mask.shape != (6,):
            logger.error(
                f"[MASK_VALIDATION] Invalid action_mask shape: {action_mask.shape}, expected (6,). "
                f"This indicates a critical bug in mask generation."
            )
            # Force correct shape with all actions valid as fallback
            obs["action_mask"] = np.ones(6, dtype=np.int8)
        elif not action_mask.any():
            logger.error(
                f"[MASK_VALIDATION] All actions masked! Mask: {action_mask}. "
                f"This should never happen - at least one action must be valid. "
                f"Forcing NOOP to be valid as emergency fallback."
            )
            # Force NOOP to be valid as emergency fallback
            action_mask = action_mask.copy()
            action_mask[0] = 1
            obs["action_mask"] = action_mask

        # Add ninja debug state for masked action bug detection
        # This is stored in the observation so reward calculator can access it
        # when a masked action is detected
        ninja = self.nplay_headless.sim.ninja
        if hasattr(ninja, "_mask_debug_state"):
            obs["_ninja_debug_state"] = ninja._mask_debug_state.copy()

        # Add debug tracking info when debug mode is enabled
        if self.nplay_headless.sim.sim_config.debug:
            obs["_env_id"] = id(self)
            obs["_step_num"] = self.nplay_headless.sim.frame
            if hasattr(ninja, "_mask_debug_state"):
                obs["_mask_fingerprint"] = ninja._mask_debug_state.get(
                    "mask_fingerprint"
                )
                obs["_mask_timestamp"] = ninja._mask_debug_state.get("mask_timestamp")

        # Only add screen if visual observations enabled
        # When disabled, graph + state + reachability contain all necessary information
        if self.enable_visual_observations:
            obs["screen"] = self.render()

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
            Action mask as numpy array of int8 (defensive copy to prevent memory sharing)
        """
        # Get action mask from ninja and force deep copy
        # DEFENSIVE FIX: Use np.array(..., copy=True) to ensure unique memory allocation
        # This prevents memory sharing issues in SubprocVecEnv where arrays might be
        # shared across process boundaries via IPC, leading to race conditions where
        # the mask used by the policy differs from the mask computed in the environment
        mask = np.array(self.nplay_headless.get_action_mask(), dtype=np.int8, copy=True)

        # Ensure C-contiguous layout to prevent memory aliasing
        if not mask.flags["C_CONTIGUOUS"]:
            mask = np.ascontiguousarray(mask, dtype=np.int8)

        # DIAGNOSTIC: Track mask creation for debugging (only in debug mode)
        if self.nplay_headless.sim.sim_config.debug:
            import os

            logger.debug(
                f"[MASK_CREATE] pid={os.getpid()} env_frame={self.nplay_headless.sim.frame} "
                f"id={id(mask)} owns_data={mask.flags['OWNDATA']} "
                f"c_contiguous={mask.flags['C_CONTIGUOUS']} hash={hash(mask.tobytes())}"
            )

        return mask

    def _get_death_context(self) -> np.ndarray:
        """Extract rich death context for agent learning.

        Provides 9 features describing what led to death:
        [0] death_occurred: 1.0 if died this step, 0.0 otherwise
        [1] death_velocity_magnitude: Speed at death (normalized 0-1)
        [2] death_velocity_x: X velocity at death (normalized -1 to 1)
        [3] death_velocity_y: Y velocity at death (normalized -1 to 1)
        [4] death_impact_normal_x: Surface normal X at impact (-1 to 1)
        [5] death_impact_normal_y: Surface normal Y at impact (-1 to 1)
        [6] frames_airborne_before_death: Frames airborne before death (normalized 0-1)
        [7] last_action_before_death: Action that led to death (normalized -1 to 1)
        [8] death_type: Categorical encoding (0=none, 0.25=mine, 0.5=terminal_impact, 0.75=hazard, 1.0=other)

        Returns:
            Array of shape (9,) with death context features
        """
        context = np.zeros(9, dtype=np.float32)

        if not self.nplay_headless.ninja_has_died():
            return context

        ninja = self.nplay_headless.sim.ninja
        context[0] = 1.0  # death_occurred flag

        # Velocity at death (from ninja.death_xspeed/yspeed)
        from ..constants.physics_constants import MAX_HOR_SPEED

        death_vel_mag = np.sqrt(ninja.death_xspeed**2 + ninja.death_yspeed**2)
        context[1] = np.clip(death_vel_mag / (MAX_HOR_SPEED * 2), 0, 1)
        context[2] = np.clip(ninja.death_xspeed / MAX_HOR_SPEED, -1, 1)
        context[3] = np.clip(ninja.death_yspeed / MAX_HOR_SPEED, -1, 1)

        # Surface normal (for terminal impact)
        if getattr(ninja, "terminal_impact", False):
            # Use floor or ceiling normal depending on impact type
            if ninja.floor_count > 0:
                context[4] = ninja.floor_normalized_x
                context[5] = ninja.floor_normalized_y
            elif ninja.ceiling_count > 0:
                context[4] = ninja.ceiling_normalized_x
                context[5] = ninja.ceiling_normalized_y

        # Frames airborne (tracked in ninja.py)
        if hasattr(ninja, "frames_airborne"):
            context[6] = np.clip(ninja.frames_airborne / 100.0, 0, 1)

        # Last action (tracked in environment)
        if hasattr(self, "last_action") and self.last_action is not None:
            context[7] = (self.last_action / 5.0) * 2 - 1  # Normalize 0-5 to [-1, 1]

        # Death type encoding
        death_cause = ninja.death_cause or "other"
        type_map = {
            None: 0.0,
            "mine": 0.25,
            "terminal_impact": 0.5,
            "hazard": 0.75,
            "other": 1.0,
        }
        context[8] = type_map.get(death_cause, 0.0)

        return context

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
        ninja_x, ninja_y = self.nplay_headless.ninja_position()
        should_truncate, reason = self._check_curriculum_aware_truncation(
            ninja_x, ninja_y
        )

        if should_truncate and self.enable_logging:
            reward_config = self.reward_calculator.config
            phase = reward_config.training_phase if reward_config else "N/A"
            time_penalty = (
                reward_config.time_penalty_per_step if reward_config else "N/A"
            )
            print(
                f"Episode truncated in {phase} phase (time_penalty={time_penalty}): {reason}"
            )

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
