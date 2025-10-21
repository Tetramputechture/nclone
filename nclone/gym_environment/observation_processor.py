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
from typing import Dict, Any, Tuple, Optional
from .constants import (
    PLAYER_FRAME_WIDTH,
    PLAYER_FRAME_HEIGHT,
    LEVEL_WIDTH,
    LEVEL_HEIGHT,
    RENDERED_VIEW_WIDTH,
    RENDERED_VIEW_HEIGHT,
)
from ..constants.physics_constants import MAX_HOR_SPEED
from .frame_augmentation import apply_consistent_augmentation, get_recommended_config
from .mine_state_processor import MineStateProcessor

# Entity position array indices
# Format: [ninja_x, ninja_y, switch_x, switch_y, exit_x, exit_y]
NINJA_POS_IDX = 0
SWITCH_POS_IDX = 2
EXIT_POS_IDX = 4
ENTITY_POSITIONS_SIZE = 6


def resize_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize frame to desired dimensions with stable interpolation."""
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def stabilize_frame(frame: np.ndarray) -> np.ndarray:
    """Ensure frame has consistent properties for stable processing.

    OPTIMIZED: More efficient conversion with early exits and optimized grayscale conversion.
    """
    # Convert pygame.Surface to numpy array if needed
    if not isinstance(frame, np.ndarray):
        try:
            import pygame  # type: ignore

            if isinstance(frame, pygame.Surface):
                # OPTIMIZATION: Use pixels_array for direct memory access (fastest method)
                # This is faster than array3d/array2d and handles both RGB and grayscale
                try:
                    # Try pygame.surfarray.pixels2d for grayscale or pixels3d for RGB
                    if frame.get_bytesize() == 1:
                        # Grayscale surface - use pixels2d with view (no copy!)
                        frame_view = pygame.surfarray.pixels2d(frame)
                        # Transpose to (H, W) and add channel dimension
                        frame = np.transpose(frame_view, (1, 0))
                        # Must copy because pixels2d returns a view that locks the surface
                        frame = np.array(frame, copy=True, dtype=np.uint8)
                        frame = frame[..., np.newaxis]
                        return frame
                    else:
                        # RGB surface - use array3d and transpose
                        frame = np.transpose(pygame.surfarray.array3d(frame), (1, 0, 2))
                except:
                    # Fallback to array3d if pixels2d fails
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
            # OPTIMIZATION: Use cv2.cvtColor for faster RGB to grayscale conversion
            # This is ~2-3x faster than manual weighted sum
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            frame = frame[..., np.newaxis]
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


def compute_hazard_from_entity_states(
    entity_states: np.ndarray, ninja_x: float, ninja_y: float
) -> Tuple[float, float]:
    """
    Compute hazard proximity from flattened entity_states array.

    This is a fallback for when mine tracking is disabled. It extracts mine proximity
    information from the flattened entity_states array structure.

    Entity states structure (from nplay_headless.get_entity_states()):
    - For each entity type (TOGGLE_MINE first with MAX_COUNT=128):
        - [0]: count of this entity type (normalized by MAX_COUNT)
        - Then for each entity (MAX_COUNT slots):
            - [0]: active (0 or 1)
            - [1]: x_pos (normalized by SRCWIDTH)
            - [2]: y_pos (normalized by SRCHEIGHT)
            - [3]: type (normalized by 28)
            - [4]: distance to ninja (normalized by screen diagonal)
            - [5]: relative velocity magnitude (normalized by MAX_HOR_SPEED)

    Args:
        entity_states: Flattened entity states array from obs["entity_states"]
        ninja_x: Ninja x position in pixels
        ninja_y: Ninja y position in pixels

    Returns:
        Tuple of (nearest_hazard_distance_pixels, hazard_threat_score)
    """
    from ..constants.render_utils import SRCWIDTH, SRCHEIGHT

    if len(entity_states) < 1:
        return (float("inf"), 0.0)

    # Entity states structure:
    # First entity type is TOGGLE_MINE with MAX_COUNT=128
    # Format: [count, entity0_attr0, entity0_attr1, ..., entity0_attr5, entity1_attr0, ...]
    MAX_ATTRIBUTES = 6
    TOGGLE_MINE_MAX_COUNT = 128

    # Extract mine count (first value)
    mine_count_norm = entity_states[0]
    mine_count = int(mine_count_norm * TOGGLE_MINE_MAX_COUNT)

    if mine_count == 0:
        return (float("inf"), 0.0)

    # Screen diagonal for denormalization
    screen_diagonal = (SRCWIDTH**2 + SRCHEIGHT**2) ** 0.5

    # Find nearest active mine
    nearest_dist = float("inf")
    for mine_idx in range(min(mine_count, TOGGLE_MINE_MAX_COUNT)):
        # Calculate offset in entity_states array
        # offset = 1 (count) + mine_idx * 6 (attributes per mine)
        offset = 1 + mine_idx * MAX_ATTRIBUTES

        if offset + MAX_ATTRIBUTES > len(entity_states):
            break

        # Extract mine attributes
        active = entity_states[offset + 0]
        x_norm = entity_states[offset + 1]
        y_norm = entity_states[offset + 2]
        # type_norm = entity_states[offset + 3]  # Not needed
        dist_norm = entity_states[offset + 4]
        # rel_vel = entity_states[offset + 5]  # Not needed

        # Only consider active mines
        if active < 0.5:
            continue

        # Denormalize distance
        distance = dist_norm * screen_diagonal

        if distance < nearest_dist:
            nearest_dist = distance

    # Compute hazard threat based on proximity
    # Threat is high when close, low when far
    # Use exponential decay: threat = exp(-distance / decay_factor)
    if nearest_dist < float("inf"):
        # Decay factor of 100 pixels means threat drops to ~0.37 at 100px
        decay_factor = 100.0
        hazard_threat = np.exp(-nearest_dist / decay_factor)
        return (nearest_dist, float(hazard_threat))

    return (float("inf"), 0.0)


class ObservationProcessor:
    """Processes raw game observations into frame stacks and normalized feature vectors.

    Performance optimizations:
    - Frame stabilization called once per observation (67% reduction from baseline)
    - Augmentation validation can be disabled for ~12% performance boost
    - Augmentation pipelines are cached via functools.lru_cache
    """

    def __init__(
        self,
        enable_augmentation: bool = True,
        augmentation_config: Dict[str, Any] = None,
        enable_mine_tracking: bool = True,
        training_mode: bool = True,
    ):
        # OPTIMIZATION: Only enable augmentation during training
        self.enable_augmentation = enable_augmentation and training_mode
        self.training_mode = training_mode

        # Set augmentation configuration optimized for platformer games
        if augmentation_config is None:
            self.augmentation_config = get_recommended_config("mid")
        else:
            self.augmentation_config = augmentation_config

        # Add disable_validation flag if not present (defaults based on training_mode)
        if "disable_validation" not in self.augmentation_config:
            self.augmentation_config["disable_validation"] = training_mode

        # Initialize mine state processor
        self.enable_mine_tracking = enable_mine_tracking
        self.mine_processor = MineStateProcessor() if enable_mine_tracking else None

        # OPTIMIZATION: Cache for stabilized frames to avoid redundant conversions
        self._frame_cache = None
        self._frame_cache_id = None

    def frame_around_player(
        self, frame: np.ndarray, player_x: float, player_y: float
    ) -> np.ndarray:
        """Crop the frame to a rectangle centered on the player.

        Note: Frame should already be stabilized by the caller to avoid redundant processing.
        """
        # Frame is already stabilized by caller - use directly
        player_frame = frame

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
        # Extract the ninja state (now 26 features after redundancy removal)
        ninja_state = (
            obs["game_state"][:26]
            if len(obs["game_state"]) >= 26
            else obs["game_state"]
        )
        ninja_state = list(ninja_state)  # Convert to list for modification

        # Calculate entity proximity features (features 21-23 in ninja state)
        ninja_x, ninja_y = obs["player_x"], obs["player_y"]
        ninja_position = (ninja_x, ninja_y)

        # Update mine states if mine tracking is enabled
        if self.enable_mine_tracking and self.mine_processor is not None:
            entities = obs.get("entities", [])
            self.mine_processor.update_mine_states(entities)

        # Calculate nearest hazard distance (normalized by screen diagonal)
        screen_diagonal = (LEVEL_WIDTH**2 + LEVEL_HEIGHT**2) ** 0.5
        nearest_hazard_dist = screen_diagonal  # Start with max distance
        hazard_threat = 0.0

        # Use mine processor for accurate hazard detection
        if self.enable_mine_tracking and self.mine_processor is not None:
            nearest_mine = self.mine_processor.get_nearest_dangerous_mine(
                ninja_position
            )
            if nearest_mine is not None:
                nearest_hazard_dist = nearest_mine.distance_to(ninja_position)
                hazard_threat = self.mine_processor.get_mine_proximity_score(
                    ninja_position
                )
        elif "entity_states" in obs and len(obs["entity_states"]) > 0:
            # Fallback: compute hazard from entity_states array
            entity_states = obs["entity_states"]
            nearest_hazard_dist, hazard_threat = compute_hazard_from_entity_states(
                entity_states, ninja_x, ninja_y
            )

        # Normalize hazard distance to [-1, 1]
        nearest_hazard_norm = (nearest_hazard_dist / screen_diagonal) * 2 - 1

        # Calculate nearest collectible distance
        switch_dist = (
            (ninja_x - obs["switch_x"]) ** 2 + (ninja_y - obs["switch_y"]) ** 2
        ) ** 0.5
        nearest_collectible_norm = (switch_dist / screen_diagonal) * 2 - 1

        # Update ninja state with computed proximity features
        # New indices after redundancy removal (26-feature state)
        if len(ninja_state) >= 26:
            ninja_state[21] = nearest_hazard_norm  # Feature 21: nearest hazard distance
            ninja_state[22] = (
                nearest_collectible_norm  # Feature 22: nearest collectible distance
            )
            # REMOVED: hazard_threat_level (redundant - exponential decay of nearest_hazard)
            # Feature 23 (interaction cooldown) is already computed in get_ninja_state

        # Calculate level progress features (features 24-25 in ninja state)
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

        # REMOVED: completion_progress (redundant - computed from switch_progress and exit distance)

        # Update ninja state with computed progress features
        if len(ninja_state) >= 26:
            ninja_state[24] = switch_progress  # Feature 24: switch activation progress
            ninja_state[25] = exit_accessibility  # Feature 25: exit accessibility

        # Convert back to numpy array
        ninja_state = np.array(ninja_state, dtype=np.float32)

        # For backward compatibility, if we have entity states, include them
        # but the new design focuses on the enhanced ninja state
        if len(obs["game_state"]) > 26:
            entity_states = obs["game_state"][26:]
            processed_state = np.concatenate([ninja_state, entity_states])
        else:
            processed_state = ninja_state

        return processed_state

    def process_rendered_global_view(self, screen: np.ndarray) -> np.ndarray:
        """Process the rendered screen into a downsampled global view.

        Args:
            screen (np.ndarray): The rendered screen image with shape (H, W, C) or (H, W)
                Should already be stabilized by caller for performance.

        Returns:
            np.ndarray: Downsampled grayscale view with shape (RENDERED_VIEW_HEIGHT, RENDERED_VIEW_WIDTH, 1)
            in the range 0-255
        """
        # Screen is already stabilized by caller - use directly

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

        # Extract entity positions from raw observation
        # Normalize positions to [0, 1] range based on level dimensions
        ninja_x_norm = obs["player_x"] / LEVEL_WIDTH
        ninja_y_norm = obs["player_y"] / LEVEL_HEIGHT
        switch_x_norm = obs["switch_x"] / LEVEL_WIDTH
        switch_y_norm = obs["switch_y"] / LEVEL_HEIGHT
        exit_x_norm = obs["exit_door_x"] / LEVEL_WIDTH
        exit_y_norm = obs["exit_door_y"] / LEVEL_HEIGHT

        # Create entity positions array: [ninja_x, ninja_y, switch_x, switch_y, exit_x, exit_y]
        entity_positions = np.zeros(ENTITY_POSITIONS_SIZE, dtype=np.float32)
        entity_positions[NINJA_POS_IDX : NINJA_POS_IDX + 2] = [
            ninja_x_norm,
            ninja_y_norm,
        ]
        entity_positions[SWITCH_POS_IDX : SWITCH_POS_IDX + 2] = [
            switch_x_norm,
            switch_y_norm,
        ]
        entity_positions[EXIT_POS_IDX : EXIT_POS_IDX + 2] = [exit_x_norm, exit_y_norm]

        result = {
            "game_state": self.process_game_state(obs),
            "global_view": self.process_rendered_global_view(screen),
            "reachability_features": obs["reachability_features"],
            "entity_positions": entity_positions,  # Add actual entity positions
        }

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
                intensity=self.augmentation_config.get("intensity", "medium"),
                disable_validation=self.augmentation_config.get(
                    "disable_validation", False
                ),
            )

        # Stack frames along channel dimension
        result["player_frame"] = np.concatenate(player_frames, axis=-1)

        return result

    def update_augmentation_config(
        self, training_stage: str = None, config: Dict[str, Any] = None
    ) -> None:
        """Update augmentation configuration during training.

        Args:
            training_stage: One of "early", "mid", "late" for recommended configs
            config: Custom configuration dictionary
        """
        if config is not None:
            self.augmentation_config = config
        elif training_stage is not None:
            self.augmentation_config = get_recommended_config(training_stage)

    def get_mine_features(
        self,
        ninja_position: Tuple[float, float],
        target_position: Optional[Tuple[float, float]] = None,
        max_mines: int = 5,
    ) -> Optional[np.ndarray]:
        """
        Get mine feature array for current state.

        Args:
            ninja_position: Current ninja position (x, y)
            target_position: Optional target position for path blocking check
            max_mines: Maximum number of mines to include

        Returns:
            Feature array of shape (max_mines, 7) or None if tracking disabled
        """
        if not self.enable_mine_tracking or self.mine_processor is None:
            return None

        return self.mine_processor.get_mine_features(
            ninja_position, target_position, max_mines
        )

    def is_path_safe_from_mines(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
    ) -> bool:
        """
        Check if a path is safe from dangerous mines.

        Args:
            start: Starting position (x, y)
            end: Ending position (x, y)

        Returns:
            True if path is safe, or if mine tracking is disabled
        """
        if not self.enable_mine_tracking or self.mine_processor is None:
            return True

        return self.mine_processor.is_path_safe(start, end)

    def get_mine_stats(self) -> Optional[Dict[str, Any]]:
        """
        Get mine statistics for debugging.

        Returns:
            Dictionary with mine stats or None if tracking disabled
        """
        if not self.enable_mine_tracking or self.mine_processor is None:
            return None

        return self.mine_processor.get_summary_stats()

    def reset(self) -> None:
        """Reset processor state."""
        if self.frame_history is not None:
            self.frame_history.clear()

        if self.enable_mine_tracking and self.mine_processor is not None:
            self.mine_processor.reset()
