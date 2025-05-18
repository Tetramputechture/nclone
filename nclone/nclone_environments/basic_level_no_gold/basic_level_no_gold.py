from gymnasium.spaces import box, Dict as SpacesDict
import numpy as np
from typing import Tuple, Optional
import os
import uuid
from .constants import (
    GAME_STATE_FEATURES_LIMITED_ENTITY_COUNT,
    GAME_STATE_FEATURES_ONLY_NINJA_AND_EXIT_AND_SWITCH,
    TEMPORAL_FRAMES,
    PLAYER_FRAME_WIDTH,
    PLAYER_FRAME_HEIGHT,
    MAX_TIME_IN_FRAMES,
    RENDERED_VIEW_WIDTH,
    RENDERED_VIEW_HEIGHT
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
    metadata = {'render.modes': ['human', 'rgb_array']}

    # Path to the nclone.nclone package directory, which contains the 'maps' directory.
    # Current file is in nclone/nclone/nclone_environments/basic_level_no_gold/
    _package_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    
    OFFICIAL_MAP_DATA_PATH = os.path.join(_package_root_dir, "maps", "official") + os.sep
    EVAL_MODE_MAP_DATA_PATH = os.path.join(_package_root_dir, "maps", "eval") + os.sep
    RANDOM_MAP_CHANCE = 0.5
    LIMIT_GAME_STATE_TO_NINJA_AND_EXIT_AND_SWITCH = True

    def __init__(self,
                 render_mode: str = 'rgb_array',
                 enable_frame_stack: bool = True,
                 enable_animation: bool = False,
                 enable_logging: bool = False,
                 enable_debug_overlay: bool = False,
                 enable_short_episode_truncation: bool = False,
                 seed: Optional[int] = None,
                 eval_mode: bool = False):
        """Initialize the environment."""
        super().__init__(render_mode=render_mode,
                         enable_animation=enable_animation,
                         enable_logging=enable_logging,
                         enable_debug_overlay=enable_debug_overlay,
                         seed=seed)

        # Initialize observation processor
        self.observation_processor = ObservationProcessor(
            enable_frame_stack=enable_frame_stack)

        # Initialize reward calculator
        self.reward_calculator = RewardCalculator()

        # Initialize truncation checker
        self.truncation_checker = TruncationChecker(self,
                                                    enable_short_episode_truncation=enable_short_episode_truncation)

        # Initialize observation space as a Dict space with player_frame, base_frame, and game_state
        player_frame_channels = TEMPORAL_FRAMES if enable_frame_stack else 1
        game_state_channels = GAME_STATE_FEATURES_ONLY_NINJA_AND_EXIT_AND_SWITCH if self.LIMIT_GAME_STATE_TO_NINJA_AND_EXIT_AND_SWITCH else GAME_STATE_FEATURES_LIMITED_ENTITY_COUNT
        self.observation_space = SpacesDict({
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
                shape=(game_state_channels,),
                dtype=np.float32
            )
        })

        # Load the first map
        if eval_mode:
            self.current_map_name = f"eval_map_{uuid.uuid4()}"
            self.random_map_type = self.rng.choice([
                "JUMP_REQUIRED",
                "MAZE",
            ])
            self.nplay_headless.load_random_map(self.random_map_type)
        else:
            # load random map
            self.current_map_name = f"random_map_{uuid.uuid4()}"
            self.random_map_type = self.rng.choice([
                "SIMPLE_HORIZONTAL_NO_BACKTRACK",
                "JUMP_REQUIRED",
                "MAZE",
            ])
            self.nplay_headless.load_random_map(self.random_map_type)

        # Initialize map cycle index. We want our cycle to be deterministic, so we use a counter.
        self.map_cycle_length = len(os.listdir(self.OFFICIAL_MAP_DATA_PATH))
        # self.map_cycle_length = 1
        self.map_cycle_index = 1
        self.mirror_map = False
        self.random_map_type = None
        self.eval_mode = eval_mode

    def _get_observation(self) -> np.ndarray:
        """Get the current observation from the game state."""
        # Calculate time remaining feature
        time_remaining = (MAX_TIME_IN_FRAMES -
                          self.nplay_headless.sim.frame) / MAX_TIME_IN_FRAMES

        ninja_state = self.nplay_headless.get_ninja_state()
        entity_states = self.nplay_headless.get_entity_states(
            only_one_exit_and_switch=self.LIMIT_GAME_STATE_TO_NINJA_AND_EXIT_AND_SWITCH)
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
            'doors_opened': self.nplay_headless.get_doors_opened(),
            'total_gold_available': self.nplay_headless.get_total_gold_available(),
        }

    def _load_map(self):
        """Loads the official map corresponding to the current map cycle index."""
        # If we are in eval mode, we want to load the next map in the cycle
        if self.eval_mode:
            # Eval mode will load a random JUMP_REQUIRED or MAZE map
            self.random_map_type = self.rng.choice([
                "JUMP_REQUIRED",
                "MAZE",
            ])
            self.current_map_name = f"eval_map_{uuid.uuid4()}"
            self.nplay_headless.load_random_map(self.random_map_type)
            return

        # First, choose if we want to generate a random map, or load the next map in the cycle
        # We always want a random map for now, since we are testing the renderer
        if self.rng.random() < 1:
            self.current_map_name = f"random_map_{uuid.uuid4()}"
            self.random_map_type = self.rng.choice([
                "SIMPLE_HORIZONTAL_NO_BACKTRACK",
                "JUMP_REQUIRED",
                "MAZE",
            ])
            self.nplay_headless.load_random_map(self.random_map_type)
        else:
            self.random_map_type = None
            self.current_map_name = self.nplay_headless.load_random_official_map()

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
            map_to_display = self.current_map_name if self.random_map_type is None else f"Random {self.random_map_type}"
            print(
                f"\n---\nPlayer won on map: {map_to_display} on frame {self.nplay_headless.sim.frame}\n---\n")

        # Check truncation using our truncation checker
        ninja_x, ninja_y = self.nplay_headless.ninja_position()
        should_truncate, reason = self.truncation_checker.update(
            ninja_x, ninja_y)
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

    def _reset_observation_processor(self):
        """Reset the observation processor."""
        self.observation_processor.reset()

    def _reset_reward_calculator(self):
        """Reset the reward calculator."""
        self.reward_calculator.reset()

    def _debug_info(self):
        # Get current cell coordinates
        ninja_x, ninja_y = self.nplay_headless.ninja_position()
        cell_x, cell_y = self.reward_calculator.exploration_calculator._get_cell_coords(
            ninja_x, ninja_y)
        area_4x4_x = cell_x // 4
        area_4x4_y = cell_y // 4
        area_8x8_x = cell_x // 8
        area_8x8_y = cell_y // 8
        area_16x16_x = cell_x // 16
        area_16x16_y = cell_y // 16

        return {
            'frame': self.nplay_headless.sim.frame,
            'current_ep_reward': self.current_ep_reward,
            'current_map_name': self.current_map_name,
            'ninja_position': self.nplay_headless.ninja_position(),
            'ninja_velocity': self.nplay_headless.ninja_velocity(),
            'exploration': {
                'current_cell': (cell_x, cell_y),
                'current_4x4_area': (area_4x4_x, area_4x4_y),
                'current_8x8_area': (area_8x8_x, area_8x8_y),
                'current_16x16_area': (area_16x16_x, area_16x16_y),
                'visited_cells_count': np.sum(self.reward_calculator.exploration_calculator.visited_cells),
                'visited_4x4_count': np.sum(self.reward_calculator.exploration_calculator.visited_4x4),
                'visited_8x8_count': np.sum(self.reward_calculator.exploration_calculator.visited_8x8),
                'visited_16x16_count': np.sum(self.reward_calculator.exploration_calculator.visited_16x16),
            }
        }

    def reset(self, seed=None, options=None):
        """Reset the environment."""
        # Reset truncation checker
        self.truncation_checker.reset()

        return super().reset(seed=seed, options=options)
