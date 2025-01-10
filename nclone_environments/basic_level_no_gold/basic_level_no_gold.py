import gymnasium
from gymnasium.spaces import discrete, box, Dict
import numpy as np
from typing import Tuple
import os
import uuid
from nclone_environments.basic_level_no_gold.constants import (
    GAME_STATE_FEATURES_ONLY_NINJA_AND_EXIT_AND_SWITCH,
    TEMPORAL_FRAMES,
    PLAYER_FRAME_WIDTH,
    PLAYER_FRAME_HEIGHT,
    MAX_TIME_IN_FRAMES,
    RENDERED_VIEW_WIDTH,
    RENDERED_VIEW_HEIGHT
)
from nclone_environments.basic_level_no_gold.reward_calculation.main_reward_calculator import RewardCalculator
from nclone_environments.basic_level_no_gold.observation_processor import ObservationProcessor
from nclone_environments.basic_level_no_gold.truncation_checker import TruncationChecker
from nclone_environments.base_environment import BaseEnvironment
from map_generation.map import Map
from map_augmentation.mirror_map import mirror_map_horizontally


class BasicLevelNoGold(BaseEnvironment):
    """Custom Gym environment for the game N++.

    This environment provides a Gym interface for the N++ game, allowing reinforcement
    learning agents to learn to play levels. We use a headless version of the game
    to speed up training.

    Currently, we cycle through the official maps.
    """
    metadata = {'render.modes': ['human', 'rgb_array']}

    OFFICIAL_MAP_DATA_PATH = "../nclone/maps/official/"

    def __init__(self, render_mode: str = 'rgb_array', enable_frame_stack: bool = True, enable_animation: bool = False, enable_logging: bool = False, enable_debug_overlay: bool = False):
        """Initialize the environment."""
        super().__init__(render_mode=render_mode,
                         enable_animation=enable_animation, enable_logging=enable_logging, enable_debug_overlay=enable_debug_overlay)

        # Initialize observation processor
        self.observation_processor = ObservationProcessor(
            enable_frame_stack=enable_frame_stack)

        # Initialize reward calculator
        self.reward_calculator = RewardCalculator()

        # Initialize truncation checker
        self.truncation_checker = TruncationChecker(self)

        # Initialize observation space as a Dict space with player_frame, base_frame, and game_state
        player_frame_channels = TEMPORAL_FRAMES if enable_frame_stack else 1
        self.observation_space = Dict({
            # Player-centered frame
            'player_frame': box.Box(
                low=0,
                high=255,
                shape=(PLAYER_FRAME_HEIGHT, PLAYER_FRAME_WIDTH,
                       player_frame_channels),
                dtype=np.uint8
            ),
            # Global view frame
            'global_view': box.Box(
                low=0,
                high=255,
                shape=(RENDERED_VIEW_HEIGHT, RENDERED_VIEW_WIDTH, 1),
                dtype=np.uint8
            ),
            # Game state features
            'game_state': box.Box(
                low=-1,
                high=1,
                shape=(GAME_STATE_FEATURES_ONLY_NINJA_AND_EXIT_AND_SWITCH,),
                dtype=np.float32
            )
        })

        # Load the first map
        map_name = os.listdir(self.OFFICIAL_MAP_DATA_PATH)[0]
        self.nplay_headless.load_map(self.OFFICIAL_MAP_DATA_PATH + map_name)
        self.current_map_name = map_name

        # Initialize map cycle index. We want our cycle to be deterministic, so we use a counter.
        self.map_cycle_length = len(os.listdir(self.OFFICIAL_MAP_DATA_PATH))
        # self.map_cycle_length = 1
        self.map_cycle_index = 1
        self.mirror_map = False

    def _get_observation(self) -> np.ndarray:
        """Get the current observation from the game state."""
        # Calculate time remaining feature
        time_remaining = (MAX_TIME_IN_FRAMES -
                          self.nplay_headless.sim.frame) / MAX_TIME_IN_FRAMES

        ninja_state = self.nplay_headless.get_ninja_state()
        entity_states = self.nplay_headless.get_entity_states(
            only_one_exit_and_switch=True)
        game_state = np.concatenate([ninja_state, entity_states])

        return {
            'screen': self.render(),
            'game_state': game_state,
            'player_dead': self.nplay_headless.ninja_has_died(),
            'player_won': self.nplay_headless.ninja_has_won(),
            'player_x': self.nplay_headless.ninja_position()[0],
            'player_y': self.nplay_headless.ninja_position()[1],
            'switch_activated': self.nplay_headless.exit_switch_activated(),
            'switch_x': self.nplay_headless.exit_switch_position()[0],
            'switch_y': self.nplay_headless.exit_switch_position()[1],
            'exit_door_x': self.nplay_headless.exit_door_position()[0],
            'exit_door_y': self.nplay_headless.exit_door_position()[1],
            'time_remaining': time_remaining,
            'sim_frame': self.nplay_headless.sim.frame,
            'gold_collected': self.nplay_headless.get_gold_collected(),
            'total_gold_available': self.nplay_headless.get_total_gold_available(),
        }

    def _load_map(self):
        """Loads the official map corresponding to the current map cycle index."""
        self.current_map_name = os.listdir(self.OFFICIAL_MAP_DATA_PATH)[
            self.map_cycle_index % self.map_cycle_length]
        with open(self.OFFICIAL_MAP_DATA_PATH + self.current_map_name, 'rb') as file:
            map_file_data = file.read()
        map_obj = Map.from_map_data(map_file_data)
        if self.mirror_map:
            map_obj = mirror_map_horizontally(map_obj)
        self.nplay_headless.load_map_from_map_data(map_obj.map_data())
        self.map_cycle_index += 1
        # If we've cycled through all maps, mirror the map
        # if self.map_cycle_index % self.map_cycle_length == 0:
        #     self.mirror_map = not self.mirror_map

    def _check_termination(self) -> Tuple[bool]:
        """Check if the episode should be terminated.

        Args:
            observation: Current observation array

        Returns:
            Tuple containing:
            - terminated: True if episode should be terminated, False otherwise
            - truncated: True if episode should be truncated, False otherwise
        """
        player_won = self.nplay_headless.ninja_has_won()
        player_dead = self.nplay_headless.ninja_has_died()
        terminated = player_won or player_dead

        # # Check truncation using our truncation checker
        # ninja_x, ninja_y = self.nplay_headless.ninja_position()
        # should_truncate, reason = self.truncation_checker.update(
        #     ninja_x, ninja_y)
        # if should_truncate and self.enable_logging:
        #     print(f"Episode terminated due to time: {reason}")

        # Check truncation using a simple frame check
        should_truncate = self.nplay_headless.sim.frame >= MAX_TIME_IN_FRAMES

        # We also terminate if the truncation state is reached, that way we can
        # learn from the episode, since our time remaining is in our observation
        return terminated or should_truncate, False, player_won

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
        return {
            'frame': self.nplay_headless.sim.frame,
            'current_map_name': self.current_map_name,
            'ninja_position': self.nplay_headless.ninja_position(),
            'ninja_velocity': self.nplay_headless.ninja_velocity(),
            'truncation_info': self.truncation_checker.get_debug_info()
        }

    def reset(self, seed=None, options=None):
        """Reset the environment."""
        # Reset truncation checker
        self.truncation_checker.reset()

        return super().reset(seed=seed, options=options)
