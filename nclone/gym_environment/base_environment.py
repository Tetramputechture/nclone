"""
Base N++ environment class with core functionality.

This module contains the core environment functionality without the specialized
mixins for graph, reachability, and debug features.
"""

import gymnasium
from gymnasium.spaces import box, discrete, Dict as SpacesDict
import random
import numpy as np
from typing import Tuple, Optional, Dict, Any

# Core nclone imports
from ..constants import MAP_TILE_WIDTH, MAP_TILE_HEIGHT
from ..nplay_headless import NPlayHeadless

# Graph and level data imports
from ..graph.level_data import LevelData

from .constants import (
    GAME_STATE_CHANNELS,
    PLAYER_FRAME_WIDTH,
    PLAYER_FRAME_HEIGHT,
    MAX_TIME_IN_FRAMES,
    RENDERED_VIEW_WIDTH,
    RENDERED_VIEW_HEIGHT,
)
from .reward_calculation.main_reward_calculator import RewardCalculator
from .observation_processor import ObservationProcessor
from .truncation_checker import TruncationChecker
from .entity_extractor import EntityExtractor
from .env_map_loader import EnvMapLoader


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

    metadata = {"render.modes": ["human", "rgb_array"]}
    RANDOM_MAP_CHANCE = 0.5

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
        enable_augmentation: bool = True,
        augmentation_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the base N++ environment.

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
            enable_augmentation: Enable frame augmentation
            augmentation_config: Augmentation configuration dictionary
        """
        super().__init__()

        # Store configuration
        self.render_mode = render_mode
        self.enable_animation = enable_animation
        self.enable_logging = enable_logging
        self.custom_map_path = custom_map_path
        self.eval_mode = eval_mode

        # Initialize core game interface
        # Note: Grayscale rendering is automatic in headless mode (rgb_array)
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
            self.nplay_headless, self.rng, eval_mode, custom_map_path
        )

        # Initialize observation processor with performance optimizations
        # training_mode=True disables validation for ~12% performance boost
        self.observation_processor = ObservationProcessor(
            enable_augmentation=enable_augmentation,
            augmentation_config=augmentation_config,
            training_mode=not eval_mode,  # Disable validation in training mode
        )

        # Initialize reward calculator with PBRS configuration
        self.reward_calculator = RewardCalculator(
            enable_pbrs=enable_pbrs, pbrs_weights=pbrs_weights, pbrs_gamma=pbrs_gamma
        )

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
            "enable_pbrs": enable_pbrs,
            "pbrs_weights": pbrs_weights,
            "pbrs_gamma": pbrs_gamma,
        }

        # Build base observation space (will be extended by mixins)
        self.observation_space = self._build_base_observation_space()

        self.mirror_map = False

        # Load the initial map
        self.map_loader.load_initial_map()

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
                shape=(GAME_STATE_CHANNELS,),
                dtype=np.float32,
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

    def step(self, action: int):
        """Execute one environment step with enhanced episode info."""
        # Get previous observation
        prev_obs = self._get_observation()

        # Execute action
        action_hoz, action_jump = self._actions_to_execute(action)
        self.nplay_headless.tick(action_hoz, action_jump)

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

        return processed_obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """Reset the environment."""
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

        # Reset level and load map
        self.nplay_headless.reset()
        self.map_loader.load_map()

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
        # Calculate time remaining feature
        time_remaining = (
            MAX_TIME_IN_FRAMES - self.nplay_headless.sim.frame
        ) / MAX_TIME_IN_FRAMES

        ninja_state = self.nplay_headless.get_ninja_state()
        entity_states_full = self.nplay_headless.get_entity_states()

        # Use first 4 dimensions which contain critical entity counts
        entity_states = (
            entity_states_full[:4]
            if len(entity_states_full) >= 4
            else np.pad(entity_states_full, (0, 4 - len(entity_states_full)))
        )

        game_state = np.concatenate([ninja_state, entity_states])

        # Get entity states for PBRS hazard detection
        entity_states_raw = self.nplay_headless.get_entity_states()

        # Get actual entity objects for tracking critical entities
        from ..constants.entity_types import EntityType

        entities = []
        if hasattr(self.nplay_headless.sim, "entity_dic"):
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

        obs = {
            "screen": self.render(),
            "game_state": game_state,
            "player_dead": self.nplay_headless.ninja_has_died(),
            "player_won": self.nplay_headless.ninja_has_won(),
            "player_x": self.nplay_headless.ninja_position()[0],
            "player_y": self.nplay_headless.ninja_position()[1],
            "switch_activated": self.nplay_headless.exit_switch_activated(),
            "switch_x": self.nplay_headless.exit_switch_position()[0],
            "switch_y": self.nplay_headless.exit_switch_position()[1],
            "exit_door_x": self.nplay_headless.exit_door_position()[0],
            "exit_door_y": self.nplay_headless.exit_door_position()[1],
            "time_remaining": time_remaining,
            "sim_frame": self.nplay_headless.sim.frame,
            "doors_opened": self.nplay_headless.get_doors_opened(),
            "entity_states": entity_states_raw,  # For PBRS hazard detection
            "entities": entities,  # Entity objects (mines, locked doors, switches)
            "locked_doors": locked_doors,  # For hierarchical navigation
            "locked_door_switches": locked_door_switches,  # For hierarchical navigation
        }

        return obs

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

        # If player won, output current map name and total reward
        if player_won:
            map_to_display = self.map_loader.get_map_display_name()
            print(
                f"\n---\nPlayer won on map: {map_to_display} on frame {self.nplay_headless.sim.frame}\n---\n"
            )

        # Check truncation using our truncation checker
        ninja_x, ninja_y = self.nplay_headless.ninja_position()
        should_truncate, reason = self.truncation_checker.update(ninja_x, ninja_y)
        if should_truncate and self.enable_logging:
            print(f"Episode terminated due to time: {reason}")

        # We also terminate if the truncation state is reached, that way we can
        # learn from the episode, since our time remaining is in our observation
        return terminated or should_truncate, False, player_won

    def _calculate_reward(self, curr_obs, prev_obs):
        """Calculate the reward for the environment."""
        return self.reward_calculator.calculate_reward(curr_obs, prev_obs)

    def _process_observation(self, obs):
        """Process the observation from the environment."""
        return self.observation_processor.process_observation(obs)

    def _reset_reward_calculator(self):
        """Reset the reward calculator."""
        self.reward_calculator.reset()

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

        return LevelData(
            tiles=tiles,
            entities=entities,
            level_id=f"level_{getattr(self.nplay_headless.sim, 'frame', 0)}",
        )

    @property
    def level_data(self) -> LevelData:
        """
        Get current level data for external access.

        Returns:
            LevelData object containing tiles and entities
        """
        return self._extract_level_data()

    @property
    def entities(self) -> list:
        """
        Get current entities for external access.

        Returns:
            List of entity dictionaries
        """
        return self.entity_extractor.extract_graph_entities()

    @property
    def current_map_name(self) -> str:
        """Get the current map name."""
        return self.map_loader.current_map_name

    def __getstate__(self):
        """Custom pickle method to handle non-picklable pygame objects and support vectorization."""
        state = self.__dict__.copy()

        # Remove the entire nplay_headless object as it contains pygame objects
        # It will be recreated when needed after unpickling
        if "nplay_headless" in state:
            # Store initialization parameters instead
            state["_nplay_headless_params"] = {
                "render_mode": getattr(self.nplay_headless, "render_mode", "rgb_array"),
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
            )

        # Mark that we need to reinitialize on next reset
        self._needs_reinit = True
