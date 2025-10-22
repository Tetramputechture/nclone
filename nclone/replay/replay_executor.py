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
        """Get raw observation from environment."""
        # Render current frame
        screen = self.nplay_headless.render()

        # Get ninja position
        ninja_x, ninja_y = self.nplay_headless.ninja_position()

        # Get ninja state
        ninja = self.nplay_headless.sim.ninja

        # Compute simplified reachability features from available data
        reachability_features = self._compute_reachability_features(ninja_x, ninja_y)

        # Build raw observation (similar to npp_environment.py)
        obs = {
            "screen": screen,
            "player_x": ninja_x,
            "player_y": ninja_y,
            "game_state": np.array(
                [
                    # Ninja physics state (simplified)
                    ninja_x / 1008,  # Normalized position
                    ninja_y / 552,
                    ninja.xspeed / 10.0,  # Normalized velocity
                    ninja.yspeed / 10.0,
                    float(ninja.on_ground),
                    float(ninja.wall_sliding),
                    # Add more state as needed...
                ]
                + [0.0] * 20,
                dtype=np.float32,
            ),  # Pad to 26 features
            "reachability_features": reachability_features,
        }

        return obs

    def _compute_reachability_features(
        self, ninja_x: float, ninja_y: float
    ) -> np.ndarray:
        """
        Compute simplified 8-dimensional reachability features from available data.

        Features:
        1. Reachable area ratio (simplified: always 0.5 as we don't do flood fill)
        2. Distance to nearest switch (normalized)
        3. Distance to exit (normalized)
        4. Reachable switches count (simplified: 0 or 1 based on switch state)
        5. Reachable hazards count (simplified: count of nearby mines)
        6. Connectivity score (simplified: always 0.5)
        7. Exit reachable flag (based on switch activation)
        8. Switch-to-exit path exists (based on switch activation)
        """
        features = np.zeros(8, dtype=np.float32)

        # Feature 1: Reachable area ratio (simplified, not computed in replay mode)
        features[0] = 0.5

        # Feature 2: Distance to nearest switch (normalized by max level distance)
        try:
            switch_x, switch_y = self.nplay_headless.exit_switch_position()
            dist_to_switch = np.sqrt(
                (ninja_x - switch_x) ** 2 + (ninja_y - switch_y) ** 2
            )
            max_distance = np.sqrt(1056**2 + 600**2)  # Max level diagonal
            features[1] = 1.0 - min(dist_to_switch / max_distance, 1.0)
        except:
            features[1] = 0.0

        # Feature 3: Distance to exit (normalized)
        try:
            exit_x, exit_y = self.nplay_headless.exit_door_position()
            dist_to_exit = np.sqrt((ninja_x - exit_x) ** 2 + (ninja_y - exit_y) ** 2)
            features[2] = 1.0 - min(dist_to_exit / max_distance, 1.0)
        except:
            features[2] = 0.0

        # Feature 4: Reachable switches count (0 or 1 based on activation state)
        try:
            switch_activated = self.nplay_headless.exit_switch_activated()
            features[3] = (
                1.0 if not switch_activated else 0.0
            )  # 1 if still needs activation
        except:
            features[3] = 0.0

        # Feature 5: Reachable hazards count (simplified: count of nearby mines)
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
        try:
            switch_activated = self.nplay_headless.exit_switch_activated()
            features[6] = 1.0 if switch_activated else 0.0
        except:
            features[6] = 0.0

        # Feature 8: Switch-to-exit path exists (same as exit reachable for simplified version)
        features[7] = features[6]

        return features

    def close(self):
        """Clean up resources."""
        if hasattr(self, "nplay_headless"):
            del self.nplay_headless
