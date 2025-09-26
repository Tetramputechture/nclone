"""This class handles the processing of the game state into a format that can be used for training.

We get our formatted game observation from the NPlusPlus get_observation() method, which has these keys:

    'screen' -- the current frame of the game
    'game_state' -- the current game state
        - Concatenated array of ninja state and entity states (only exit and switch)
    'player_dead' -- whether the player has died
    'player_won' -- whether the player has won
    'player_x' -- the x position of the player
    'player_y' -- the y position of the player
    'switch_activated' -- whether the switch has been activated
    'switch_x' -- the x position of the switch
    'switch_y' -- the y position of the switch
    'exit_door_x' -- the x position of the exit door
    'exit_door_y' -- the y position of the exit door
    'time_remaining' -- the time remaining in the simulation before truncation

This class should handle returning:

- The frame centered on the player (PLAYER_FRAME_WIDTH x PLAYER_FRAME_HEIGHT)
  - This covers the player's movement in detail
- The global view (RENDERED_VIEW_WIDTH x RENDERED_VIEW_HEIGHT)
  - This covers the global view of the level at 1/6th resolution
- The game state
    - Ninja state (12 values)
        - Position normalized
        - Speed normalized
        - Airborn boolean
        - Walled boolean
        - Jump duration normalized
        - Applied gravity normalized
        - Applied drag normalized
        - Applied friction normalized
    - Exit and switch entity states
    - Time remaining
    - Vector between ninja and switch
    - Vector between ninja and exit door
"""

import numpy as np
from collections import deque
import cv2
from typing import Dict, Any, Tuple
from .constants import (
    PLAYER_FRAME_WIDTH,
    PLAYER_FRAME_HEIGHT,
    LEVEL_WIDTH,
    LEVEL_HEIGHT,
    TEMPORAL_FRAMES,
    RENDERED_VIEW_WIDTH,
    RENDERED_VIEW_HEIGHT,
)
from ..constants.physics_constants import MAX_HOR_SPEED
from .frame_augmentation import apply_consistent_augmentation, get_recommended_config


def resize_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize frame to desired dimensions with stable interpolation."""
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def stabilize_frame(frame: np.ndarray) -> np.ndarray:
    """Ensure frame has consistent properties for stable processing."""
    # Convert pygame.Surface to numpy array if needed
    if not isinstance(frame, np.ndarray):
        try:
            import pygame  # type: ignore

            if isinstance(frame, pygame.Surface):
                # pygame.surfarray.array3d returns shape (W, H, 3)
                frame = np.transpose(pygame.surfarray.array3d(frame), (1, 0, 2))
            else:
                frame = np.asarray(frame)
        except Exception:
            frame = np.asarray(frame)

    # If channel-first (C, H, W), move channels last
    if (
        isinstance(frame, np.ndarray)
        and frame.ndim == 3
        and frame.shape[0] in (1, 3, 4)
        and frame.shape[0] < min(frame.shape[1], frame.shape[2])
    ):
        frame = np.moveaxis(frame, 0, -1)

    # Ensure uint8 dtype
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)

    # Ensure proper shape (H, W, 1) for grayscale
    if len(frame.shape) == 2:
        frame = frame[..., np.newaxis]
    elif len(frame.shape) == 3:
        # Normalize channel count
        if frame.shape[-1] == 4:
            # Drop alpha if present
            frame = frame[..., :3]
        if frame.shape[-1] == 3:
            # Convert RGB to grayscale
            gray = (
                0.2989 * frame[..., 0] + 0.5870 * frame[..., 1] + 0.1140 * frame[..., 2]
            )
            frame = gray[..., np.newaxis].astype(np.uint8)
        elif frame.shape[-1] != 1:
            raise ValueError(f"Unexpected frame shape: {frame.shape}")

    return frame


def normalize_position(x: float, y: float) -> Tuple[float, float]:
    """Normalize position coordinates to [-1, 1] range."""
    return ((x / LEVEL_WIDTH) * 2 - 1, (y / LEVEL_HEIGHT) * 2 - 1)


def normalize_velocity(vx: float, vy: float) -> Tuple[float, float]:
    """Normalize velocity components to [-1, 1] range."""
    return (np.clip(vx / MAX_HOR_SPEED, -1, 1), np.clip(vy / MAX_HOR_SPEED, -1, 1))


def calculate_vector(
    from_x: float, from_y: float, to_x: float, to_y: float
) -> Tuple[float, float]:
    """Calculate normalized vector between two points."""
    dx = to_x - from_x
    dy = to_y - from_y
    magnitude = np.sqrt(dx * dx + dy * dy)
    if magnitude == 0:
        return (0.0, 0.0)
    return (dx / magnitude, dy / magnitude)


class ObservationProcessor:
    """Processes raw game observations into frame stacks and normalized feature vectors."""

    def __init__(
        self, 
        enable_augmentation: bool = True,
        augmentation_config: Dict[str, Any] = None
    ):
        self.enable_augmentation = enable_augmentation
        self.frame_history = deque(maxlen=TEMPORAL_FRAMES)
        
        # Set augmentation configuration optimized for platformer games
        if augmentation_config is None:
            self.augmentation_config = get_recommended_config("mid")
        else:
            self.augmentation_config = augmentation_config

    def frame_around_player(
        self, frame: np.ndarray, player_x: float, player_y: float
    ) -> np.ndarray:
        """Crop the frame to a rectangle centered on the player."""
        # Ensure frame stability
        player_frame = stabilize_frame(frame)

        # Calculate the starting and ending coordinates for the crop
        start_x = int(player_x - PLAYER_FRAME_WIDTH // 2)
        end_x = int(player_x + PLAYER_FRAME_WIDTH // 2)
        start_y = int(player_y - PLAYER_FRAME_HEIGHT // 2)
        end_y = int(player_y + PLAYER_FRAME_HEIGHT // 2)

        # Ensure the crop is within the frame boundaries
        start_x = max(0, start_x)
        end_x = min(frame.shape[0], end_x)
        start_y = max(0, start_y)
        end_y = min(frame.shape[1], end_y)

        # Get the cropped frame
        player_frame = player_frame[start_x:end_x, start_y:end_y]

        # Calculate padding needed on each side to reach target dimensions
        current_width = player_frame.shape[1]
        current_height = player_frame.shape[0]

        # Calculate padding amounts
        left_pad = (PLAYER_FRAME_WIDTH - current_width) // 2
        right_pad = PLAYER_FRAME_WIDTH - current_width - left_pad
        top_pad = (PLAYER_FRAME_HEIGHT - current_height) // 2
        bottom_pad = PLAYER_FRAME_HEIGHT - current_height - top_pad

        # Ensure no negative padding
        left_pad = max(0, left_pad)
        right_pad = max(0, right_pad)
        top_pad = max(0, top_pad)
        bottom_pad = max(0, bottom_pad)

        # Pad the frame
        player_frame = cv2.copyMakeBorder(
            player_frame,
            top_pad,
            bottom_pad,
            left_pad,
            right_pad,
            cv2.BORDER_CONSTANT,
            value=0,
        )

        if len(player_frame.shape) == 2:
            # Ensure we have a channel dimension
            player_frame = player_frame[..., np.newaxis]

        # Final size check and correction if needed
        if player_frame.shape[:2] != (PLAYER_FRAME_HEIGHT, PLAYER_FRAME_WIDTH):
            player_frame = cv2.resize(
                player_frame, (PLAYER_FRAME_WIDTH, PLAYER_FRAME_HEIGHT)
            )
            if len(player_frame.shape) == 2:
                player_frame = player_frame[..., np.newaxis]

        return player_frame

    def process_game_state(self, obs: Dict[str, Any]) -> np.ndarray:
        """Process game state into enhanced normalized feature vector."""
        # Extract the ninja state (now 30 features)
        ninja_state = obs["game_state"][:30] if len(obs["game_state"]) >= 30 else obs["game_state"]
        ninja_state = list(ninja_state)  # Convert to list for modification
        
        # Calculate entity proximity features (features 24-27 in ninja state)
        ninja_x, ninja_y = obs["player_x"], obs["player_y"]
        
        # Calculate nearest hazard distance (normalized by screen diagonal)
        screen_diagonal = (LEVEL_WIDTH**2 + LEVEL_HEIGHT**2) ** 0.5
        nearest_hazard_dist = screen_diagonal  # Start with max distance
        hazard_threat = 0.0
        
        # Check entity states for hazards (mines, drones, thwumps, etc.)
        if "entity_states" in obs and len(obs["entity_states"]) > 0:
            entity_states = obs["entity_states"]
            # This is a simplified approach - in a full implementation, 
            # we'd parse the entity states to find hazard positions
            # For now, use a placeholder calculation
            nearest_hazard_dist = min(nearest_hazard_dist, screen_diagonal * 0.5)
            hazard_threat = 0.1  # Low threat level as placeholder
        
        # Normalize hazard distance to [-1, 1]
        nearest_hazard_norm = (nearest_hazard_dist / screen_diagonal) * 2 - 1
        
        # Calculate nearest collectible distance
        switch_dist = ((ninja_x - obs["switch_x"])**2 + (ninja_y - obs["switch_y"])**2) ** 0.5
        nearest_collectible_norm = (switch_dist / screen_diagonal) * 2 - 1
        
        # Update ninja state with computed proximity features
        if len(ninja_state) >= 30:
            ninja_state[23] = nearest_hazard_norm  # Feature 24: nearest hazard distance
            ninja_state[24] = nearest_collectible_norm  # Feature 25: nearest collectible distance  
            ninja_state[25] = hazard_threat * 2 - 1  # Feature 26: hazard threat level (normalized to [-1,1])
            # Feature 27 (interaction cooldown) is already computed in get_ninja_state
        
        # Calculate level progress features (features 28-30 in ninja state)
        # Switch activation progress
        if obs.get("switch_activated", False):
            switch_progress = 1.0  # Switch is activated
        else:
            # Progress based on distance to switch (closer = higher progress)
            max_switch_dist = screen_diagonal
            switch_progress = 1.0 - (switch_dist / max_switch_dist)
            switch_progress = switch_progress * 2 - 1  # Normalize to [-1, 1]
        
        # Exit accessibility
        exit_accessibility = 1.0 if obs.get("switch_activated", False) else -1.0
        
        # Level completion progress (combination of switch and exit progress)
        if obs.get("switch_activated", False):
            exit_dist = ((ninja_x - obs["exit_door_x"])**2 + (ninja_y - obs["exit_door_y"])**2) ** 0.5
            exit_progress = 1.0 - (exit_dist / screen_diagonal)
            completion_progress = (switch_progress + exit_progress) / 2
        else:
            completion_progress = switch_progress * 0.5  # Only switch progress matters
        
        # Update ninja state with computed progress features
        if len(ninja_state) >= 30:
            ninja_state[27] = switch_progress  # Feature 28: switch activation progress
            ninja_state[28] = exit_accessibility  # Feature 29: exit accessibility
            ninja_state[29] = completion_progress  # Feature 30: level completion progress
        
        # Convert back to numpy array
        ninja_state = np.array(ninja_state, dtype=np.float32)
        
        # For backward compatibility, if we have entity states, include them
        # but the new design focuses on the enhanced ninja state
        if len(obs["game_state"]) > 30:
            entity_states = obs["game_state"][30:]
            processed_state = np.concatenate([ninja_state, entity_states])
        else:
            processed_state = ninja_state
        
        return processed_state

    def process_rendered_global_view(self, screen: np.ndarray) -> np.ndarray:
        """Process the rendered screen into a downsampled global view.

        Args:
            screen (np.ndarray): The rendered screen image with shape (H, W, C) or (H, W)

        Returns:
            np.ndarray: Downsampled grayscale view with shape (RENDERED_VIEW_HEIGHT, RENDERED_VIEW_WIDTH, 1)
            in the range 0-255
        """
        # Ensure frame stability and consistent format
        screen = stabilize_frame(screen)

        # Downsample using area interpolation for better quality
        downsampled = cv2.resize(
            screen,
            (RENDERED_VIEW_WIDTH, RENDERED_VIEW_HEIGHT),
            interpolation=cv2.INTER_AREA,
        )

        # Ensure we have a channel dimension
        if len(downsampled.shape) == 2:
            downsampled = downsampled[..., np.newaxis]

        return downsampled

    def process_observation(self, obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Process observation into frame stack, base map, and feature vectors."""
        # Ensure screen stability and consistent format
        screen = stabilize_frame(obs["screen"])

        # Process current frame using already grayscale screen
        player_frame = self.frame_around_player(
            screen, obs["player_x"], obs["player_y"]
        )

        result = {
            "game_state": self.process_game_state(obs),
            "global_view": self.process_rendered_global_view(screen),
            "reachability_features": obs["reachability_features"],
        }

        # Update frame history with cropped player frame instead of full frame
        self.frame_history.append(player_frame)

        # Fill frame history if needed
        while len(self.frame_history) < TEMPORAL_FRAMES:
            self.frame_history.append(player_frame)

        # Get player frames from history
        player_frames = []
        # Reverse to get [current, last, second_to_last]
        for frame in reversed(self.frame_history):
            # Ensure each frame has shape (H, W, 1)
            if len(frame.shape) == 2:
                frame = frame[..., np.newaxis]
            player_frames.append(frame)

        # Apply consistent augmentation across all frames if enabled
        if self.enable_augmentation:
            player_frames = apply_consistent_augmentation(
                player_frames,
                p=self.augmentation_config.get("p", 0.5),
                intensity=self.augmentation_config.get("intensity", "medium")
            )

        # Stack frames along channel dimension
        result["player_frame"] = np.concatenate(player_frames, axis=-1)

        # Verify we have exactly 3 channels
        assert result["player_frame"].shape[-1] == TEMPORAL_FRAMES, (
            f"Expected {TEMPORAL_FRAMES} channels, got {result['player_frame'].shape[-1]}"
        )

        return result

    def update_augmentation_config(self, training_stage: str = None, config: Dict[str, Any] = None) -> None:
        """Update augmentation configuration during training.
        
        Args:
            training_stage: One of "early", "mid", "late" for recommended configs
            config: Custom configuration dictionary
        """
        if config is not None:
            self.augmentation_config = config
        elif training_stage is not None:
            self.augmentation_config = get_recommended_config(training_stage)
    
    def reset(self) -> None:
        """Reset processor state."""
        if self.frame_history is not None:
            self.frame_history.clear()
