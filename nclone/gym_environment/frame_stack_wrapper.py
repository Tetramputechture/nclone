"""
Frame stacking wrapper for temporal observation history.

This module implements frame stacking for player frame visual observations and game state vectors.
Frame stacking is a common technique in Deep RL to provide temporal information to the
policy network, enabling it to infer velocity, acceleration, and motion dynamics.

Key Features:
- Independent stacking for player frames and game states
- Global view is NOT stacked (always single frame for spatial context)
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

    Stacks player_frame and game_state independently with configurable stack sizes.
    This allows the policy to learn temporal patterns like velocity and acceleration.

    IMPORTANT: Only player_frame is stacked. Global view remains a single frame to
    provide spatial context without temporal redundancy.

    The wrapper handles:
    1. Player frame stacking - captures visual motion and temporal patterns
    2. Game state stacking - captures physics state evolution
    3. Global view passthrough - NOT stacked, always single frame
    4. Consistent augmentation - when augmentation is applied, the same transform
       is used across all frames in the stack to maintain temporal coherence
    5. Initial padding - uses zero or repeat padding for the first few steps

    Example:
        With visual_stack_size=4, player_frame shape changes from:
        - Input: (84, 84, 1) - single grayscale frame
        - Output: (4, 84, 84, 1) - stack of 4 frames, oldest first

        Global view shape remains unchanged:
        - Input: (176, 100, 1) - single grayscale frame
        - Output: (176, 100, 1) - same single frame (NOT stacked)

        With state_stack_size=4, game_state shape changes from:
        - Input: (GAME_STATE_CHANNELS,) - single state vector
        - Output: (4, GAME_STATE_CHANNELS) - stack of 4 state vectors, oldest first

    Args:
        env: Environment to wrap
        visual_stack_size: Number of player frames to stack (1-12)
        state_stack_size: Number of game states to stack (1-12)
        enable_visual_stacking: Whether to stack player frames
        enable_state_stacking: Whether to stack game states
        padding_type: "zero" or "repeat" for initial frame padding
        enable_augmentation: Whether to apply visual augmentation to frames
        augmentation_config: Configuration dict for augmentation (p, intensity, disable_validation)
                           If None, uses default: {"p": 0.5, "intensity": "medium", "disable_validation": False}

    Note:
        When enable_augmentation=True and enable_visual_stacking=True, the SAME
        augmentation is applied to all frames in the stack to maintain temporal
        coherence. This prevents the critical bug where each frame gets a different
        random augmentation, breaking visual continuity between frames.
    """

    def __init__(
        self,
        env: gym.Env,
        visual_stack_size: int = 4,
        state_stack_size: int = 4,
        enable_visual_stacking: bool = True,
        enable_state_stacking: bool = True,
        padding_type: str = "zero",
        enable_augmentation: bool = False,
        augmentation_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(env)

        self.visual_stack_size = visual_stack_size
        self.state_stack_size = state_stack_size
        self.enable_visual_stacking = enable_visual_stacking
        self.enable_state_stacking = enable_state_stacking
        self.padding_type = padding_type

        # Augmentation settings - applied consistently across frame stack
        self.enable_augmentation = enable_augmentation
        self.augmentation_config = augmentation_config or {
            "p": 0.5,
            "intensity": "medium",
            "disable_validation": False,
        }

        # Validate configuration
        if visual_stack_size < 1 or visual_stack_size > 12:
            raise ValueError("visual_stack_size must be between 1 and 12")
        if state_stack_size < 1 or state_stack_size > 12:
            raise ValueError("state_stack_size must be between 1 and 12")
        if padding_type not in ["zero", "repeat"]:
            raise ValueError("padding_type must be 'zero' or 'repeat'")

        # Initialize frame buffers (deques for efficient FIFO)
        self.player_frame_buffer = deque(
            maxlen=visual_stack_size if enable_visual_stacking else 1
        )
        self.game_state_buffer = deque(
            maxlen=state_stack_size if enable_state_stacking else 1
        )

        # Pre-allocate stacking buffers to avoid np.stack() copies
        # These buffers are reused and updated in-place
        self._player_frame_stack_buffer = None
        self._game_state_stack_buffer = None
        self._initialized_buffers = False

        # Update observation space
        self.observation_space = self._build_stacked_observation_space()

    def _build_stacked_observation_space(self) -> SpacesDict:
        """Build observation space with stacked dimensions.

        Only player_frame is stacked when enable_visual_stacking=True.
        Global view always remains a single frame.
        """
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

        # Global view is NOT stacked - always single frame
        if "global_view" in base_spaces:
            new_spaces["global_view"] = base_spaces["global_view"]

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

        Global view is NOT buffered - it passes through as-is.
        """
        obs, info = self.env.reset(**kwargs)

        # Reset buffer initialization flag to reinitialize on first observation
        self._initialized_buffers = False

        # Initialize buffers with padding (stack_size - 1 frames)
        # The observation() call will add the real frame as the last element
        if "player_frame" in obs and self.enable_visual_stacking:
            padding = self._get_padding_value("player_frame", obs["player_frame"])
            self.player_frame_buffer.clear()
            for _ in range(self.visual_stack_size - 1):
                self.player_frame_buffer.append(padding)
        elif "player_frame" in obs:
            self.player_frame_buffer.clear()

        if "game_state" in obs and self.enable_state_stacking:
            padding = self._get_padding_value("game_state", obs["game_state"])
            self.game_state_buffer.clear()
            for _ in range(self.state_stack_size - 1):
                self.game_state_buffer.append(padding)
        elif "game_state" in obs:
            self.game_state_buffer.clear()

        return self.observation(obs), info

    def _reset_to_checkpoint_from_wrapper(
        self, checkpoint
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Forward checkpoint reset to underlying environment and reinitialize frame buffers.

        Called by GoExploreVecEnv to restore environment state from a checkpoint.
        After forwarding to the underlying env, reinitializes frame buffers similar
        to reset() so the frame stack starts fresh from the checkpoint position.

        Args:
            checkpoint: CheckpointEntry with action_sequence for replay

        Returns:
            Tuple of (stacked_observation, info_dict)
        """
        # Forward to underlying environment
        obs, info = self.env._reset_to_checkpoint_from_wrapper(checkpoint)

        # Reset buffer initialization flag to reinitialize on first observation
        self._initialized_buffers = False

        # Reinitialize buffers with padding (stack_size - 1 frames)
        # The observation() call will add the real frame as the last element
        if "player_frame" in obs and self.enable_visual_stacking:
            padding = self._get_padding_value("player_frame", obs["player_frame"])
            self.player_frame_buffer.clear()
            for _ in range(self.visual_stack_size - 1):
                self.player_frame_buffer.append(padding)
        elif "player_frame" in obs:
            self.player_frame_buffer.clear()

        if "game_state" in obs and self.enable_state_stacking:
            padding = self._get_padding_value("game_state", obs["game_state"])
            self.game_state_buffer.clear()
            for _ in range(self.state_stack_size - 1):
                self.game_state_buffer.append(padding)
        elif "game_state" in obs:
            self.game_state_buffer.clear()

        return self.observation(obs), info

    def _init_stack_buffers(self, observation: Dict[str, Any]) -> None:
        """Initialize pre-allocated stacking buffers from observation shapes."""
        if self._initialized_buffers:
            return

        # Initialize player frame stack buffer if needed
        if (
            self.enable_visual_stacking
            and "player_frame" in observation
            and self._player_frame_stack_buffer is None
        ):
            frame_shape = observation["player_frame"].shape  # (H, W, C)
            stack_shape = (self.visual_stack_size,) + frame_shape
            self._player_frame_stack_buffer = np.zeros(
                stack_shape, dtype=observation["player_frame"].dtype
            )

        # Initialize game state stack buffer if needed
        if (
            self.enable_state_stacking
            and "game_state" in observation
            and self._game_state_stack_buffer is None
        ):
            state_shape = observation["game_state"].shape  # (state_dim,)
            stack_shape = (self.state_stack_size,) + state_shape
            self._game_state_stack_buffer = np.zeros(
                stack_shape, dtype=observation["game_state"].dtype
            )

        self._initialized_buffers = True

    def _copy_frames_to_buffer(
        self, buffer: deque, target_buffer: np.ndarray
    ) -> np.ndarray:
        """Copy frames from deque to pre-allocated buffer without creating new array.

        Args:
            buffer: Deque containing frames
            target_buffer: Pre-allocated buffer to copy into

        Returns:
            Reference to target_buffer (same array, updated in-place)
        """
        for i, frame in enumerate(buffer):
            target_buffer[i] = frame
        return target_buffer

    def observation(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform observation by stacking frames.

        CRITICAL: When augmentation is enabled and visual stacking is enabled,
        applies the SAME augmentation to all frames in the stack to maintain
        temporal coherence. This prevents the bug where each frame gets a
        different random augmentation, breaking visual continuity.

        Only player_frame is stacked. Global view passes through unchanged.

        Args:
            observation: Current observation dict from environment

        Returns:
            Observation dict with stacked player frames (augmented if enabled)
        """
        # Initialize buffers on first observation
        self._init_stack_buffers(observation)

        # Update buffers with new observations
        if "player_frame" in observation:
            self.player_frame_buffer.append(observation["player_frame"])

        if "game_state" in observation:
            self.game_state_buffer.append(observation["game_state"])

        # Build stacked observation
        stacked_obs = {}

        # Stack player frame if enabled, with consistent augmentation
        if "player_frame" in observation:
            if (
                self.enable_visual_stacking
                and len(self.player_frame_buffer) > 0
                and self._player_frame_stack_buffer is not None
            ):
                # Copy frames to pre-allocated buffer (avoids np.stack() copy)
                stacked_frames = self._copy_frames_to_buffer(
                    self.player_frame_buffer, self._player_frame_stack_buffer
                )

                # Apply consistent augmentation to all frames in the stack
                if self.enable_augmentation:
                    stacked_frames = self._apply_consistent_augmentation(stacked_frames)
                else:
                    # If no augmentation, need to copy buffer to avoid mutation issues
                    # when observation is stored (e.g., in rollout buffers)
                    stacked_frames = stacked_frames.copy()

                stacked_obs["player_frame"] = stacked_frames
            else:
                frame = observation["player_frame"]
                # Apply augmentation to single frame
                if self.enable_augmentation:
                    frame = self._apply_augmentation_to_frame(frame)
                stacked_obs["player_frame"] = frame

        # Global view is NOT stacked - pass through as-is with optional augmentation
        if "global_view" in observation:
            frame = observation["global_view"]
            # Apply augmentation to single frame
            if self.enable_augmentation:
                frame = self._apply_augmentation_to_frame(frame)
            stacked_obs["global_view"] = frame

        # Stack game state if enabled
        if "game_state" in observation:
            if (
                self.enable_state_stacking
                and len(self.game_state_buffer) > 0
                and self._game_state_stack_buffer is not None
            ):
                # Copy states to pre-allocated buffer (avoids np.stack() copy)
                stacked_states = self._copy_frames_to_buffer(
                    self.game_state_buffer, self._game_state_stack_buffer
                )
                # Need to copy buffer to avoid mutation issues when observation is stored
                stacked_obs["game_state"] = stacked_states.copy()
            else:
                stacked_obs["game_state"] = observation["game_state"]

        # Copy other observations unchanged
        for key, value in observation.items():
            if key not in stacked_obs:
                stacked_obs[key] = value

        return stacked_obs

    def _apply_augmentation_to_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply augmentation to a single frame."""
        from .frame_augmentation import apply_augmentation

        return apply_augmentation(
            frame,
            p=self.augmentation_config.get("p", 0.5),
            intensity=self.augmentation_config.get("intensity", "medium"),
            disable_validation=self.augmentation_config.get(
                "disable_validation", False
            ),
            return_replay=False,
        )

    def _apply_consistent_augmentation(self, stacked_frames: np.ndarray) -> np.ndarray:
        """Apply the SAME augmentation to all frames in a stack.

        This is critical for maintaining temporal coherence. All frames must
        receive the same random transformation (rotation, crop, brightness, etc.)
        so that the stack represents a consistent visual sequence.

        Args:
            stacked_frames: Array of shape (stack_size, H, W, C)

        Returns:
            Augmented frames with same shape, all with consistent augmentation
        """
        from .frame_augmentation import (
            apply_augmentation,
            apply_augmentation_with_replay,
        )

        stack_size = stacked_frames.shape[0]

        # Apply augmentation to first frame and get replay data
        first_frame_aug, replay_data = apply_augmentation(
            stacked_frames[0],
            p=self.augmentation_config.get("p", 0.5),
            intensity=self.augmentation_config.get("intensity", "medium"),
            disable_validation=self.augmentation_config.get(
                "disable_validation", False
            ),
            return_replay=True,
        )

        # Apply the same augmentation to all other frames
        augmented_frames = [first_frame_aug]
        for i in range(1, stack_size):
            aug_frame = apply_augmentation_with_replay(
                stacked_frames[i],
                replay_data,
                p=self.augmentation_config.get("p", 0.5),
                intensity=self.augmentation_config.get("intensity", "medium"),
                disable_validation=self.augmentation_config.get(
                    "disable_validation", False
                ),
            )
            augmented_frames.append(aug_frame)

        # Stack back into array
        return np.stack(augmented_frames, axis=0)

    # Passthrough methods for route visualization (SubprocVecEnv env_method support)
    def get_route_visualization_tile_data(self):
        """Passthrough to underlying environment for route visualization."""
        return self.env.get_route_visualization_tile_data()

    def get_route_visualization_mine_data(self):
        """Passthrough to underlying environment for route visualization."""
        return self.env.get_route_visualization_mine_data()

    def get_route_visualization_locked_door_data(self):
        """Passthrough to underlying environment for route visualization."""
        return self.env.get_route_visualization_locked_door_data()

    def get_graph_data_for_visualization(self):
        """Passthrough to underlying environment for PBRS path visualization."""
        return self.env.get_graph_data_for_visualization()

    def get_level_data_for_visualization(self):
        """Passthrough to underlying environment for PBRS path visualization."""
        return self.env.get_level_data_for_visualization()

    def get_pbrs_path_for_visualization(self, start_pos, goal_pos, switch_activated):
        """Passthrough to underlying environment for PBRS path visualization.
        
        Args:
            start_pos: Agent's starting position (x, y)
            goal_pos: Goal position (x, y) - switch or exit
            switch_activated: Whether switch has been activated
            
        Returns:
            List of (x, y) node positions forming the path, or None if unreachable
        """
        return self.env.get_pbrs_path_for_visualization(
            start_pos, goal_pos, switch_activated
        )
