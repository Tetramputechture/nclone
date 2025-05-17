"""Base environment class that handles logic for loading maps, resetting the game, and rendering."""

import gymnasium
from gymnasium.spaces import discrete
import random
from typing import Tuple, Optional

from ..nplay_headless import NPlayHeadless

class BaseEnvironment(gymnasium.Env):
    """Base environment class that handles logic for loading maps, resetting the game, and rendering."""
    metadata = {'render.modes': ['human', 'rgb_array']}

    MAP_DATA_PATH = None

    def __init__(self,
                 render_mode: str = 'rgb_array',
                 enable_animation: bool = False,
                 enable_logging: bool = False,
                 enable_debug_overlay: bool = False,
                 seed: Optional[int] = None):
        """Initialize the environment."""
        super().__init__()

        self.render_mode = render_mode
        self.enable_animation = enable_animation
        self.enable_logging = enable_logging
        self.nplay_headless = NPlayHeadless(
            render_mode=render_mode,
            enable_animation=enable_animation,
            enable_logging=enable_logging,
            enable_debug_overlay=enable_debug_overlay,
            seed=seed)

        # Initialize action space
        self.action_space = discrete.Discrete(6)

        # Initialize RNG
        self.rng = random.Random(seed)

        # Track reward for the current episode
        self.current_ep_reward = 0

    def _actions_to_execute(self, action: int) -> Tuple[int, int]:
        """Execute the specified action using the game controller.

        Args:
            action (int): Action to execute (0-5)

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

        # Execute the new action
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
        """Execute one environment step with planning and visualization."""
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

        ep_info = {'is_success': player_won}

        return processed_obs, reward, terminated, truncated, ep_info

    def reset(self, seed=None, options=None):
        """Reset the environment with planning components and visualization."""
        # Reset observation processor
        self._reset_observation_processor()

        # Reset reward calculator
        self._reset_reward_calculator()

        # Reset episode reward
        self.current_ep_reward = 0

        # Reset level and load map
        self.nplay_headless.reset()
        self._load_map()

        # Get initial observation and process it
        initial_obs = self._get_observation()
        processed_obs = self._process_observation(initial_obs)

        return (processed_obs, {})

    def render(self):
        """Render the environment."""
        return self.nplay_headless.render(self._debug_info())

    def _debug_info(self):
        """Returns a dictionary containing debug information to be displayed on the screen."""
        return None

    def _load_map(self):
        """Loads the map."""
        raise NotImplementedError

    def _get_observation(self):
        """Gets the observation from the environment. Unique per each environment, must be implemented in child class."""
        raise NotImplementedError

    def _check_termination(self):
        """Checks if the environment has terminated. Unique per each environment, must be implemented in child class."""
        raise NotImplementedError

    def _calculate_reward(self, curr_obs, prev_obs):
        """Calculates the reward for the environment. Unique per each environment, must be implemented in child class."""
        raise NotImplementedError

    def _process_observation(self, obs):
        """Processes the observation from the environment. Unique per each environment, must be implemented in child class."""
        raise NotImplementedError

    def _reset_observation_processor(self):
        """Resets the observation processor. Unique per each environment, must be implemented in child class."""
        raise NotImplementedError

    def _reset_reward_calculator(self):
        """Resets the reward calculator. Unique per each environment, must be implemented in child class."""
        raise NotImplementedError
