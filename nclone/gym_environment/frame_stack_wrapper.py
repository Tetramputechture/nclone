"""
Frame stacking wrapper for temporal observation history.

This module implements frame stacking for both visual observations and game state vectors.
Frame stacking is a common technique in Deep RL to provide temporal information to the
policy network, enabling it to infer velocity, acceleration, and motion dynamics.

Key Features:
- Independent stacking for visual frames and game states
- Consistent augmentation across stacked frames (when enabled)
- Configurable stack sizes (2-12 frames)
- Zero or repeat padding for initial frames
- Memory-efficient implementation using deques

References:
- Mnih et al. (2015). "Human-level control through deep reinforcement learning." Nature 518, 529-533.
  https://doi.org/10.1038/nature14236
  Original DQN paper demonstrating 4-frame stacking for Atari games.

- Machado et al. (2018). "Revisiting the Arcade Learning Environment: Evaluation Protocols 
  and Open Problems for General Agents." Journal of Artificial Intelligence Research 61, 523-562.
  Comprehensive analysis of preprocessing techniques including frame stacking.

- Bellemare et al. (2013). "The Arcade Learning Environment: An Evaluation Platform for 
  General Agents." Journal of Artificial Intelligence Research 47, 253-279.
  Early work on frame history for temporal credit assignment.
"""

import numpy as np
from collections import deque
from typing import Dict, Any, Optional
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box, Dict as SpacesDict
import gymnasium as gym


class FrameStackWrapper(ObservationWrapper):
    """
    Wrapper for stacking consecutive observations to provide temporal information.
    
    Stacks visual frames (player_frame, global_view) and game_state independently
    with configurable stack sizes. This allows the policy to learn temporal patterns
    like velocity and acceleration.
    
    The wrapper handles:
    1. Visual frame stacking (player_frame, global_view) - captures visual motion
    2. Game state stacking - captures physics state evolution
    3. Consistent augmentation - when augmentation is applied, the same transform
       is used across all frames in the stack to maintain temporal coherence
    4. Initial padding - uses zero or repeat padding for the first few steps
    
    Example:
        With visual_stack_size=4, player_frame shape changes from:
        - Input: (84, 84, 1) - single grayscale frame
        - Output: (4, 84, 84, 1) - stack of 4 frames, oldest first
        
        With state_stack_size=4, game_state shape changes from:
        - Input: (26,) - single state vector
        - Output: (4, 26) - stack of 4 state vectors, oldest first
    
    Args:
        env: Environment to wrap
        visual_stack_size: Number of visual frames to stack (1-12)
        state_stack_size: Number of game states to stack (1-12)
        enable_visual_stacking: Whether to stack visual frames
        enable_state_stacking: Whether to stack game states
        padding_type: "zero" or "repeat" for initial frame padding
    """
    
    def __init__(
        self,
        env: gym.Env,
        visual_stack_size: int = 4,
        state_stack_size: int = 4,
        enable_visual_stacking: bool = True,
        enable_state_stacking: bool = True,
        padding_type: str = "zero",
    ):
        super().__init__(env)
        
        self.visual_stack_size = visual_stack_size
        self.state_stack_size = state_stack_size
        self.enable_visual_stacking = enable_visual_stacking
        self.enable_state_stacking = enable_state_stacking
        self.padding_type = padding_type
        
        # Validate configuration
        if visual_stack_size < 1 or visual_stack_size > 12:
            raise ValueError("visual_stack_size must be between 1 and 12")
        if state_stack_size < 1 or state_stack_size > 12:
            raise ValueError("state_stack_size must be between 1 and 12")
        if padding_type not in ["zero", "repeat"]:
            raise ValueError("padding_type must be 'zero' or 'repeat'")
        
        # Initialize frame buffers (deques for efficient FIFO)
        self.player_frame_buffer = deque(maxlen=visual_stack_size if enable_visual_stacking else 1)
        self.global_view_buffer = deque(maxlen=visual_stack_size if enable_visual_stacking else 1)
        self.game_state_buffer = deque(maxlen=state_stack_size if enable_state_stacking else 1)
        
        # Update observation space
        self.observation_space = self._build_stacked_observation_space()
        
    def _build_stacked_observation_space(self) -> SpacesDict:
        """Build observation space with stacked dimensions."""
        base_spaces = self.env.observation_space.spaces
        new_spaces = {}
        
        # Stack player_frame if enabled
        if "player_frame" in base_spaces:
            base_shape = base_spaces["player_frame"].shape  # (H, W, C)
            if self.enable_visual_stacking:
                # Stack along first dimension: (stack_size, H, W, C)
                new_shape = (self.visual_stack_size,) + base_shape
            else:
                new_shape = base_shape
            new_spaces["player_frame"] = Box(
                low=0, high=255, shape=new_shape, dtype=np.uint8
            )
        
        # Stack global_view if enabled
        if "global_view" in base_spaces:
            base_shape = base_spaces["global_view"].shape  # (H, W, C)
            if self.enable_visual_stacking:
                # Stack along first dimension: (stack_size, H, W, C)
                new_shape = (self.visual_stack_size,) + base_shape
            else:
                new_shape = base_shape
            new_spaces["global_view"] = Box(
                low=0, high=255, shape=new_shape, dtype=np.uint8
            )
        
        # Stack game_state if enabled
        if "game_state" in base_spaces:
            base_shape = base_spaces["game_state"].shape  # (state_dim,)
            if self.enable_state_stacking:
                # Stack along first dimension: (stack_size, state_dim)
                new_shape = (self.state_stack_size,) + base_shape
            else:
                new_shape = base_shape
            new_spaces["game_state"] = Box(
                low=-1, high=1, shape=new_shape, dtype=np.float32
            )
        
        # Copy other observation spaces unchanged
        for key, space in base_spaces.items():
            if key not in new_spaces:
                new_spaces[key] = space
        
        return SpacesDict(new_spaces)
    
    def _get_padding_value(self, obs_key: str, obs_value: np.ndarray) -> np.ndarray:
        """Get padding value based on padding_type."""
        if self.padding_type == "zero":
            return np.zeros_like(obs_value)
        else:  # repeat
            return obs_value.copy()
    
    def reset(self, **kwargs) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset environment and initialize frame stacks.
        
        Note: Initializes buffers with (stack_size - 1) padding frames,
        then calls observation() which appends the real first frame.
        This prevents the double-append bug where the first frame would
        appear twice in the initial stack.
        """
        obs, info = self.env.reset(**kwargs)
        
        # Initialize buffers with padding (stack_size - 1 frames)
        # The observation() call will add the real frame as the last element
        if "player_frame" in obs and self.enable_visual_stacking:
            padding = self._get_padding_value("player_frame", obs["player_frame"])
            self.player_frame_buffer.clear()
            for _ in range(self.visual_stack_size - 1):
                self.player_frame_buffer.append(padding)
        elif "player_frame" in obs:
            self.player_frame_buffer.clear()
        
        if "global_view" in obs and self.enable_visual_stacking:
            padding = self._get_padding_value("global_view", obs["global_view"])
            self.global_view_buffer.clear()
            for _ in range(self.visual_stack_size - 1):
                self.global_view_buffer.append(padding)
        elif "global_view" in obs:
            self.global_view_buffer.clear()
        
        if "game_state" in obs and self.enable_state_stacking:
            padding = self._get_padding_value("game_state", obs["game_state"])
            self.game_state_buffer.clear()
            for _ in range(self.state_stack_size - 1):
                self.game_state_buffer.append(padding)
        elif "game_state" in obs:
            self.game_state_buffer.clear()
        
        return self.observation(obs), info
    
    def observation(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform observation by stacking frames.
        
        Args:
            observation: Current observation dict from environment
            
        Returns:
            Observation dict with stacked frames
        """
        # Update buffers with new observations
        if "player_frame" in observation:
            self.player_frame_buffer.append(observation["player_frame"])
        
        if "global_view" in observation:
            self.global_view_buffer.append(observation["global_view"])
        
        if "game_state" in observation:
            self.game_state_buffer.append(observation["game_state"])
        
        # Build stacked observation
        stacked_obs = {}
        
        # Stack visual frames if enabled
        if "player_frame" in observation:
            if self.enable_visual_stacking and len(self.player_frame_buffer) > 0:
                # Stack frames along first dimension: (stack_size, H, W, C)
                stacked_obs["player_frame"] = np.stack(self.player_frame_buffer, axis=0)
            else:
                stacked_obs["player_frame"] = observation["player_frame"]
        
        if "global_view" in observation:
            if self.enable_visual_stacking and len(self.global_view_buffer) > 0:
                # Stack frames along first dimension: (stack_size, H, W, C)
                stacked_obs["global_view"] = np.stack(self.global_view_buffer, axis=0)
            else:
                stacked_obs["global_view"] = observation["global_view"]
        
        # Stack game state if enabled
        if "game_state" in observation:
            if self.enable_state_stacking and len(self.game_state_buffer) > 0:
                # Stack states along first dimension: (stack_size, state_dim)
                stacked_obs["game_state"] = np.stack(self.game_state_buffer, axis=0)
            else:
                stacked_obs["game_state"] = observation["game_state"]
        
        # Copy other observations unchanged
        for key, value in observation.items():
            if key not in stacked_obs:
                stacked_obs[key] = value
        
        return stacked_obs
