from gymnasium.spaces import box, Dict as SpacesDict
import numpy as np
from typing import Tuple, Optional
import os
import uuid
from .constants import (
    GAME_STATE_FEATURES_LIMITED_ENTITY_COUNT,
    GAME_STATE_FEATURES_ONLY_NINJA_AND_EXIT_AND_SWITCH,
    GAME_STATE_FEATURES_ONLY_NINJA_AND_EXIT_AND_SWITCH_RICH,
    TEMPORAL_FRAMES,
    PLAYER_FRAME_WIDTH,
    PLAYER_FRAME_HEIGHT,
    MAX_TIME_IN_FRAMES,
    RENDERED_VIEW_WIDTH,
    RENDERED_VIEW_HEIGHT,
)
from .reward_calculation.main_reward_calculator import RewardCalculator
from .observation_processor import ObservationProcessor
from .truncation_checker import TruncationChecker
from ..base_environment import BaseEnvironment


class BasicLevelNoGold(BaseEnvironment):
    """Custom Gym environment for the game N++.

    This environment provides a Gym interface for the N++ game, allowing reinforcement
    learning agents to learn to play levels. We use a headless version of the game
    to speed up training.

    Currently, we cycle through the official maps.
    """

    metadata = {"render.modes": ["human", "rgb_array"]}

    # Path to the nclone.nclone package directory, which contains the 'maps' directory.
    # Current file is in nclone/nclone/nclone_environments/basic_level_no_gold/
    _package_root_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )

    OFFICIAL_MAP_DATA_PATH = (
        os.path.join(_package_root_dir, "maps", "official") + os.sep
    )
    EVAL_MODE_MAP_DATA_PATH = os.path.join(_package_root_dir, "maps", "eval") + os.sep
    RANDOM_MAP_CHANCE = 0.5
    LIMIT_GAME_STATE_TO_NINJA_AND_EXIT_AND_SWITCH = True

    def __init__(
        self,
        render_mode: str = "rgb_array",
        enable_frame_stack: bool = True,
        enable_animation: bool = False,
        enable_logging: bool = False,
        enable_debug_overlay: bool = False,
        enable_short_episode_truncation: bool = False,
        seed: Optional[int] = None,
        eval_mode: bool = False,
        observation_profile: str = "rich",
        use_rich_game_state: Optional[
            bool
        ] = None,  # Deprecated: use observation_profile
        enable_pbrs: bool = True,
        pbrs_weights: Optional[dict] = None,
        pbrs_gamma: float = 0.99,
        custom_map_path: Optional[str] = None,
    ):
        """Initialize the environment."""
        super().__init__(
            render_mode=render_mode,
            enable_animation=enable_animation,
            enable_logging=enable_logging,
            enable_debug_overlay=enable_debug_overlay,
            seed=seed,
            custom_map_path=custom_map_path,
        )

        # Initialize observation processor
        self.observation_processor = ObservationProcessor(
            enable_frame_stack=enable_frame_stack
        )

        # Initialize reward calculator with PBRS configuration
        self.reward_calculator = RewardCalculator(
            enable_pbrs=enable_pbrs, pbrs_weights=pbrs_weights, pbrs_gamma=pbrs_gamma
        )

        # Initialize truncation checker
        self.truncation_checker = TruncationChecker(
            self, enable_short_episode_truncation=enable_short_episode_truncation
        )

        # Handle deprecated use_rich_game_state flag
        if use_rich_game_state is not None:
            import warnings

            warnings.warn(
                "use_rich_game_state is deprecated, use observation_profile instead",
                DeprecationWarning,
                stacklevel=2,
            )
            if use_rich_game_state:
                observation_profile = "rich"
            else:
                observation_profile = "minimal"

        # Store observation profile configuration
        if observation_profile not in ["minimal", "rich"]:
            raise ValueError(
                f"observation_profile must be 'minimal' or 'rich', got {observation_profile}"
            )
        self.observation_profile = observation_profile

        # Store all configuration flags for logging and debugging
        self.config_flags = {
            "render_mode": render_mode,
            "enable_frame_stack": enable_frame_stack,
            "enable_animation": enable_animation,
            "enable_logging": enable_logging,
            "enable_debug_overlay": enable_debug_overlay,
            "enable_short_episode_truncation": enable_short_episode_truncation,
            "eval_mode": eval_mode,
            "observation_profile": observation_profile,
            "enable_pbrs": enable_pbrs,
            "pbrs_weights": pbrs_weights,
            "pbrs_gamma": pbrs_gamma,
        }
        self.use_rich_features = observation_profile == "rich"

        # Initialize observation space as a Dict space with player_frame, base_frame, and game_state
        player_frame_channels = TEMPORAL_FRAMES if enable_frame_stack else 1

        # Select game state feature count based on profile
        if self.LIMIT_GAME_STATE_TO_NINJA_AND_EXIT_AND_SWITCH:
            if self.use_rich_features:
                game_state_channels = (
                    GAME_STATE_FEATURES_ONLY_NINJA_AND_EXIT_AND_SWITCH_RICH
                )
            else:
                game_state_channels = GAME_STATE_FEATURES_ONLY_NINJA_AND_EXIT_AND_SWITCH
        else:
            game_state_channels = GAME_STATE_FEATURES_LIMITED_ENTITY_COUNT
        self.observation_space = SpacesDict(
            {
                # Player-centered frame
                "player_frame": box.Box(
                    low=0,
                    high=255,
                    shape=(
                        PLAYER_FRAME_HEIGHT,
                        PLAYER_FRAME_WIDTH,
                        player_frame_channels,
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
                # Game state features
                "game_state": box.Box(
                    low=-1, high=1, shape=(game_state_channels,), dtype=np.float32
                ),
            }
        )

        # Load the first map
        if eval_mode:
            self.current_map_name = f"eval_map_{uuid.uuid4()}"
            self.random_map_type = self.rng.choice(
                [
                    "JUMP_REQUIRED",
                    "MAZE",
                ]
            )
            self.nplay_headless.load_random_map(self.random_map_type)
        else:
            # load random map
            self.current_map_name = f"random_map_{uuid.uuid4()}"
            self.random_map_type = self.rng.choice(
                [
                    "SIMPLE_HORIZONTAL_NO_BACKTRACK",
                    "JUMP_REQUIRED",
                    "MAZE",
                ]
            )
            self.nplay_headless.load_random_map(self.random_map_type)

        # Initialize map cycle index. We want our cycle to be deterministic, so we use a counter.
        self.map_cycle_length = len(os.listdir(self.OFFICIAL_MAP_DATA_PATH))
        self.map_cycle_index = 1
        self.mirror_map = False
        self.random_map_type = None
        self.eval_mode = eval_mode

    def _get_observation(self) -> np.ndarray:
        """Get the current observation from the game state."""
        # Calculate time remaining feature
        time_remaining = (
            MAX_TIME_IN_FRAMES - self.nplay_headless.sim.frame
        ) / MAX_TIME_IN_FRAMES

        ninja_state = self.nplay_headless.get_ninja_state(
            use_rich_features=self.use_rich_features
        )
        entity_states = self.nplay_headless.get_entity_states(
            only_one_exit_and_switch=self.LIMIT_GAME_STATE_TO_NINJA_AND_EXIT_AND_SWITCH,
            use_rich_features=self.use_rich_features,
        )
        game_state = np.concatenate([ninja_state, entity_states])

        # Get entity states for PBRS hazard detection
        entity_states_raw = self.nplay_headless.get_entity_states(
            only_one_exit_and_switch=False,  # Get all entities for hazard detection
            use_rich_features=self.use_rich_features,
        )

        return {
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
            "gold_collected": self.nplay_headless.get_gold_collected(),
            "doors_opened": self.nplay_headless.get_doors_opened(),
            "total_gold_available": self.nplay_headless.get_total_gold_available(),
            "entity_states": entity_states_raw,  # For PBRS hazard detection
        }

    def _load_map(self):
        """Loads the map specified by custom_map_path or follows original logic."""
        # If a custom map path is provided, use that instead of default behavior
        if self.custom_map_path:
            # Extract map name from path for display purposes
            map_name = os.path.basename(self.custom_map_path)
            if not map_name:  # Handle trailing slash case
                map_name = os.path.basename(os.path.dirname(self.custom_map_path))
            self.current_map_name = map_name
            self.random_map_type = None
            self.nplay_headless.load_map(self.custom_map_path)
            return

        # If we are in eval mode, we want to load the next map in the cycle
        if self.eval_mode:
            # Eval mode will load a random JUMP_REQUIRED or MAZE map
            self.random_map_type = self.rng.choice(
                [
                    "JUMP_REQUIRED",
                    "MAZE",
                ]
            )
            self.current_map_name = f"eval_map_{uuid.uuid4()}"
            self.nplay_headless.load_random_map(self.random_map_type)
            return

        # First, choose if we want to generate a random map, or load the next map in the cycle
        # We always want a random map for now, since we are testing the renderer
        # if self.rng.random() < 0.0:
        #     self.current_map_name = f"random_map_{uuid.uuid4()}"
        #     self.random_map_type = self.rng.choice([
        #         "SIMPLE_HORIZONTAL_NO_BACKTRACK",
        #         "JUMP_REQUIRED",
        #         "MAZE",
        #     ])
        #     self.nplay_headless.load_random_map(self.random_map_type)
        # else:
        #     self.random_map_type = None
        #     self.current_map_name = self.nplay_headless.load_random_official_map()
        # Load the map 'maps/doortest'
        self.current_map_name = "doortest"
        self.nplay_headless.load_map("nclone/test_maps/doortest")

    def _check_termination(self) -> Tuple[bool]:
        """Check if the episode should be terminated.

        Args:
            observation: Current observation array

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
            map_to_display = (
                self.current_map_name
                if self.random_map_type is None
                else f"Random {self.random_map_type}"
            )
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

    def step(self, action: int):
        """Execute one environment step with enhanced episode info."""
        # Call parent step method
        obs, reward, terminated, truncated, info = super().step(action)

        # Add configuration flags to episode info
        info.update(
            {
                "config_flags": self.config_flags.copy(),
                "observation_profile": self.observation_profile,
                "pbrs_enabled": self.config_flags["enable_pbrs"],
            }
        )

        # Add PBRS component rewards if available
        if hasattr(self.reward_calculator, "last_pbrs_components"):
            info["pbrs_components"] = self.reward_calculator.last_pbrs_components.copy()

        return obs, reward, terminated, truncated, info

    def _calculate_reward(self, curr_obs, prev_obs):
        """Calculate the reward for the environment."""
        return self.reward_calculator.calculate_reward(curr_obs, prev_obs)

    def _process_observation(self, obs):
        """Process the observation from the environment."""
        return self.observation_processor.process_observation(obs)

    def _reset_observation_processor(self):
        """Reset the observation processor."""
        self.observation_processor.reset()

    def _reset_reward_calculator(self):
        """Reset the reward calculator."""
        self.reward_calculator.reset()

    def _debug_info(self):
        # Start with any base debug info (navigation, graph, etc.)
        base_info = super()._debug_info()

        # Get current cell coordinates
        ninja_x, ninja_y = self.nplay_headless.ninja_position()

        info = {
            "frame": self.nplay_headless.sim.frame,
            "current_ep_reward": self.current_ep_reward,
            "current_map_name": self.current_map_name,
            "ninja_position": self.nplay_headless.ninja_position(),
            "ninja_velocity": self.nplay_headless.ninja_velocity(),
        }

        if self._exploration_debug_enabled:
            cell_x, cell_y = (
                self.reward_calculator.exploration_calculator._get_cell_coords(
                    ninja_x, ninja_y
                )
            )
            area_4x4_x = cell_x // 4
            area_4x4_y = cell_y // 4
            area_8x8_x = cell_x // 8
            area_8x8_y = cell_y // 8
            area_16x16_x = cell_x // 16
            area_16x16_y = cell_y // 16

            exploration_info = {
                "current_cell": (cell_x, cell_y),
                "current_4x4_area": (area_4x4_x, area_4x4_y),
                "current_8x8_area": (area_8x8_x, area_8x8_y),
                "current_16x16_area": (area_16x16_x, area_16x16_y),
                "visited_cells": self.reward_calculator.exploration_calculator.visited_cells,
                "visited_4x4": self.reward_calculator.exploration_calculator.visited_4x4,
                "visited_8x8": self.reward_calculator.exploration_calculator.visited_8x8,
                "visited_16x16": self.reward_calculator.exploration_calculator.visited_16x16,
                "visited_cells_count": np.sum(
                    self.reward_calculator.exploration_calculator.visited_cells
                ),
                "visited_4x4_count": np.sum(
                    self.reward_calculator.exploration_calculator.visited_4x4
                ),
                "visited_8x8_count": np.sum(
                    self.reward_calculator.exploration_calculator.visited_8x8
                ),
                "visited_16x16_count": np.sum(
                    self.reward_calculator.exploration_calculator.visited_16x16
                ),
            }
            info["exploration"] = exploration_info

        # Merge with base info if present
        if base_info:
            # Ensure we don't overwrite exploration key; place base keys at top level
            info.update(base_info)
        return info

    def reset(self, seed=None, options=None):
        """Reset the environment."""
        # Handle reinitialization after unpickling
        if hasattr(self, "_needs_reinit") and self._needs_reinit:
            # Reinitialize components that may have been affected by pickling
            if hasattr(self, "observation_processor"):
                self.observation_processor.reset()
            if hasattr(self, "reward_calculator"):
                self.reward_calculator.reset()
            self._needs_reinit = False

        # Reset truncation checker
        self.truncation_checker.reset()

        return super().reset(seed=seed, options=options)

    def __getstate__(self):
        """Custom pickle method to handle non-picklable pygame objects."""
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

        return state

    def __setstate__(self, state):
        """Custom unpickle method to restore the environment."""
        self.__dict__.update(state)

        # Recreate nplay_headless if it was removed during pickling
        if not hasattr(self, "nplay_headless") and hasattr(
            self, "_nplay_headless_params"
        ):
            from ...nplay_headless import NPlayHeadless

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

            # Mark that we need to reinitialize on next reset
            self._needs_reinit = True
