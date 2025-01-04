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
    MAX_TIME_IN_FRAMES_SMALL_LEVEL
)
from nclone_environments.basic_level_no_gold.reward_calculation.main_reward_calculator import RewardCalculator
from nclone_environments.basic_level_no_gold.observation_processor import ObservationProcessor
from nclone_environments.base_environment import BaseEnvironment


class BasicLevelNoGold(BaseEnvironment):
    """Custom Gym environment for the game N++.

    This environment provides a Gym interface for the N++ game, allowing reinforcement
    learning agents to learn to play levels. We use a headless version of the game
    to speed up training.
    """
    metadata = {'render.modes': ['human', 'rgb_array']}

    MAP_DATA_PATH = "../nclone/maps/map_di"

    def __init__(self, render_mode: str = 'rgb_array', enable_frame_stack: bool = True):
        """Initialize the environment."""
        super().__init__(render_mode=render_mode)

        # self.nplay_headless.load_random_map()
        self.nplay_headless.load_map(self.MAP_DATA_PATH)

        # Initialize observation processor
        self.observation_processor = ObservationProcessor(
            enable_frame_stack=enable_frame_stack)

        # Initialize reward calculator
        self.reward_calculator = RewardCalculator()

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
            # Game state features
            'game_state': box.Box(
                low=-1,
                high=1,
                shape=(GAME_STATE_FEATURES_ONLY_NINJA_AND_EXIT_AND_SWITCH,),
                dtype=np.float32
            )
        })

    def _get_observation(self) -> np.ndarray:
        """Get the current observation from the game state."""
        # Calculate time remaining feature
        time_remaining = (MAX_TIME_IN_FRAMES_SMALL_LEVEL -
                          self.nplay_headless.sim.frame) / MAX_TIME_IN_FRAMES_SMALL_LEVEL

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
        }

    def get_gold_collected(self):
        return self.nplay_headless.get_gold_collected()

    def _check_termination(self) -> Tuple[bool, bool]:
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

        # Check truncation
        # Truncation is when the current simulation frame is greater than cap
        truncated = self.nplay_headless.sim.frame > MAX_TIME_IN_FRAMES_SMALL_LEVEL

        return terminated, truncated, player_won

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
