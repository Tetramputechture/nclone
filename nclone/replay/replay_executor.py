"""
Deterministic Replay Executor

Replays stored input sequences against map data to regenerate observations.
Leverages the completely deterministic nature of N++ physics simulation.
"""

from typing import Dict, List, Any, Optional
import numpy as np

from ..nplay_headless import NPlayHeadless
from ..gym_environment.observation_processor import ObservationProcessor


def map_input_to_action(input_byte: int) -> int:
    """
    Map N++ input byte (0-7) to discrete action (0-5).

    Inverse of map_action_to_input from gameplay_recorder.py
    """
    input_to_action_map = {
        0: 0,  # 000: NOOP
        1: 3,  # 001: JUMP
        2: 2,  # 010: RIGHT
        3: 5,  # 011: RIGHT+JUMP
        4: 1,  # 100: LEFT
        5: 4,  # 101: LEFT+JUMP
        6: 0,  # 110: Invalid (mapped to NOOP)
        7: 0,  # 111: Invalid (mapped to NOOP)
    }
    return input_to_action_map.get(input_byte, 0)


def decode_input_to_controls(input_byte: int) -> tuple:
    """
    Decode input byte to horizontal and jump controls.

    Returns:
        (horizontal, jump) where:
            horizontal: -1 (left), 0 (none), 1 (right)
            jump: 0 (no jump), 1 (jump)
    """
    jump = 1 if (input_byte & 0x01) else 0
    right = 1 if (input_byte & 0x02) else 0
    left = 1 if (input_byte & 0x04) else 0

    # Handle conflicting inputs
    if left and right:
        horizontal = 0
    elif left:
        horizontal = -1
    elif right:
        horizontal = 1
    else:
        horizontal = 0

    return horizontal, jump


class ReplayExecutor:
    """Executes replays deterministically to generate observations."""

    def __init__(
        self,
        observation_config: Optional[Dict[str, Any]] = None,
        render_mode: str = "grayscale_array",
    ):
        """Initialize replay executor.

        Args:
            observation_config: Configuration for observation processor
            render_mode: Rendering mode for environment
        """
        self.observation_config = observation_config or {}
        self.render_mode = render_mode

        # Create headless environment
        self.nplay_headless = NPlayHeadless(
            render_mode=render_mode,
            enable_animation=False,
            enable_logging=False,
            enable_debug_overlay=False,
            seed=42,  # Fixed seed for determinism
        )

        # Create observation processor
        self.obs_processor = ObservationProcessor(
            enable_augmentation=False,  # No augmentation for replay
        )

    def execute_replay(
        self,
        map_data: bytes,
        input_sequence: List[int],
    ) -> List[Dict[str, Any]]:
        """Execute a replay and generate observations for each frame.

        Args:
            map_data: Raw map data (1335 bytes)
            input_sequence: Input sequence (1 byte per frame)

        Returns:
            List of observations, one per frame
        """
        # Load map
        self.nplay_headless.load_map_from_map_data(list(map_data))

        observations = []

        # Execute each input frame
        for frame_idx, input_byte in enumerate(input_sequence):
            # Decode input to controls
            horizontal, jump = decode_input_to_controls(input_byte)

            # Execute one simulation step
            self.nplay_headless.tick(horizontal, jump)

            # Get raw observation
            raw_obs = self._get_raw_observation()

            # Process observation
            processed_obs = self.obs_processor.process_observation(raw_obs)

            # Get discrete action for this frame
            action = map_input_to_action(input_byte)

            # Store observation with action
            observations.append(
                {
                    "observation": processed_obs,
                    "action": action,
                    "frame": frame_idx,
                }
            )

        return observations

    def _get_raw_observation(self) -> Dict[str, Any]:
        """Get raw observation from environment.
        
        Returns a complete raw observation dict with all required keys
        that the ObservationProcessor expects.
        
        Required keys:
        - screen: rendered frame
        - player_x, player_y: ninja position
        - switch_x, switch_y: exit switch position
        - exit_door_x, exit_door_y: exit door position
        - switch_activated: whether switch has been activated
        - game_state: concatenated ninja and entity state (26+ floats)
        - reachability_features: 8-dimensional reachability vector
        - player_won: whether the player has won
        - player_dead: whether the player has died
        """
        # Render current frame
        screen = self.nplay_headless.render()

        # Get ninja position
        ninja_x, ninja_y = self.nplay_headless.ninja_position()

        # Get ninja state
        ninja = self.nplay_headless.sim.ninja

        # Get entity positions
        switch_x, switch_y = self._get_switch_position()
        exit_door_x, exit_door_y = self._get_exit_door_position()
        switch_activated = self._is_switch_activated()

        # Build game state array (26 features minimum)
        game_state = self._build_game_state(ninja, ninja_x, ninja_y)

        # Compute reachability features
        reachability_features = self._compute_reachability_features(
            ninja_x, ninja_y, switch_x, switch_y, exit_door_x, exit_door_y
        )

        # Get win/death state
        player_won = self.nplay_headless.ninja_has_won()
        player_dead = self.nplay_headless.sim.ninja.has_died()

        # Build complete raw observation
        obs = {
            "screen": screen,
            "player_x": ninja_x,
            "player_y": ninja_y,
            "switch_x": switch_x,
            "switch_y": switch_y,
            "exit_door_x": exit_door_x,
            "exit_door_y": exit_door_y,
            "switch_activated": switch_activated,
            "game_state": game_state,
            "reachability_features": reachability_features,
            "player_won": player_won,
            "player_dead": player_dead,
        }

        return obs

    def _get_switch_position(self) -> tuple:
        """Get exit switch position with fallback."""
        try:
            return self.nplay_headless.exit_switch_position()
        except:
            # Fallback to level center if no switch exists
            return (528.0, 300.0)

    def _get_exit_door_position(self) -> tuple:
        """Get exit door position with fallback."""
        try:
            return self.nplay_headless.exit_door_position()
        except:
            # Fallback to level center if no exit exists
            return (528.0, 300.0)

    def _is_switch_activated(self) -> bool:
        """Check if exit switch is activated."""
        try:
            return self.nplay_headless.exit_switch_activated()
        except:
            return False

    def _build_game_state(
        self, ninja, ninja_x: float, ninja_y: float
    ) -> np.ndarray:
        """Build the game state array matching the format expected by observation processor.
        
        The game state contains ninja physics state (12 values) + entity states.
        Minimum 26 features to match observation processor expectations.
        
        Ninja state (12 values):
        - Position (x, y) normalized
        - Speed (xspeed, yspeed) normalized
        - Airborn boolean
        - Walled boolean
        - Jump duration normalized
        - Applied gravity normalized
        - Applied drag normalized
        - Applied friction normalized
        - State (movement state enum)
        - Jump buffer normalized
        """
        from ..constants import (
            MAX_HOR_SPEED,
            GRAVITY_FALL,
            DRAG_REGULAR,
            FRICTION_GROUND,
            MAX_JUMP_DURATION,
        )

        # Normalize values
        x_norm = ninja_x / 1056.0  # LEVEL_WIDTH
        y_norm = ninja_y / 600.0  # LEVEL_HEIGHT
        xspeed_norm = ninja.xspeed / MAX_HOR_SPEED
        yspeed_norm = ninja.yspeed / 10.0  # Reasonable max vertical speed
        jump_duration_norm = ninja.jump_duration / MAX_JUMP_DURATION
        gravity_norm = ninja.applied_gravity / GRAVITY_FALL
        drag_norm = ninja.applied_drag / DRAG_REGULAR
        friction_norm = ninja.applied_friction / FRICTION_GROUND
        state_norm = ninja.state / 5.0  # States are 0-5
        jump_buffer_norm = max(ninja.jump_buffer, 0) / 5.0  # Buffer is -1 to 5

        # Build ninja state array (12 values)
        ninja_state = np.array(
            [
                x_norm,
                y_norm,
                xspeed_norm,
                yspeed_norm,
                float(not ninja.airborn),  # on_ground
                float(ninja.walled),
                jump_duration_norm,
                gravity_norm,
                drag_norm,
                friction_norm,
                state_norm,
                jump_buffer_norm,
            ],
            dtype=np.float32,
        )

        # Pad to minimum 26 features (12 ninja + 14 padding for entity states)
        # The observation processor will extract what it needs
        padding = np.zeros(14, dtype=np.float32)
        game_state = np.concatenate([ninja_state, padding])

        return game_state

    def _compute_reachability_features(
        self,
        ninja_x: float,
        ninja_y: float,
        switch_x: float,
        switch_y: float,
        exit_door_x: float,
        exit_door_y: float,
    ) -> np.ndarray:
        """
        Compute 8-dimensional reachability features.

        Features:
        1. Reachable area ratio (simplified: always 0.5 as we don't do flood fill)
        2. Distance to nearest switch (normalized, inverted so closer = higher)
        3. Distance to exit (normalized, inverted so closer = higher)
        4. Reachable switches count (0 or 1 based on switch state)
        5. Reachable hazards count (count of nearby mines, normalized)
        6. Connectivity score (simplified: always 0.5)
        7. Exit reachable flag (based on switch activation)
        8. Switch-to-exit path exists (based on switch activation)
        """
        features = np.zeros(8, dtype=np.float32)

        # Feature 1: Reachable area ratio (simplified, not computed in replay mode)
        features[0] = 0.5

        # Feature 2: Distance to nearest switch (normalized, inverted)
        max_distance = np.sqrt(1056**2 + 600**2)  # Max level diagonal
        dist_to_switch = np.sqrt(
            (ninja_x - switch_x) ** 2 + (ninja_y - switch_y) ** 2
        )
        features[1] = 1.0 - min(dist_to_switch / max_distance, 1.0)

        # Feature 3: Distance to exit (normalized, inverted)
        dist_to_exit = np.sqrt(
            (ninja_x - exit_door_x) ** 2 + (ninja_y - exit_door_y) ** 2
        )
        features[2] = 1.0 - min(dist_to_exit / max_distance, 1.0)

        # Feature 4: Reachable switches count (0 or 1 based on activation state)
        switch_activated = self._is_switch_activated()
        features[3] = 1.0 if not switch_activated else 0.0  # 1 if still needs activation

        # Feature 5: Reachable hazards count (count of nearby mines, normalized)
        try:
            mines = self.nplay_headless.mines()
            # Count mines within reasonable distance (e.g., 200 pixels)
            nearby_mines = sum(
                1
                for mine in mines
                if np.sqrt((ninja_x - mine.xpos) ** 2 + (ninja_y - mine.ypos) ** 2)
                < 200
            )
            features[4] = min(nearby_mines / 10.0, 1.0)  # Normalize by expected max
        except:
            features[4] = 0.0

        # Feature 6: Connectivity score (simplified, not computed in replay mode)
        features[5] = 0.5

        # Feature 7: Exit reachable flag (based on switch activation)
        features[6] = 1.0 if switch_activated else 0.0

        # Feature 8: Switch-to-exit path exists (same as exit reachable)
        features[7] = features[6]

        return features

    def close(self):
        """Clean up resources."""
        if hasattr(self, "nplay_headless"):
            del self.nplay_headless
