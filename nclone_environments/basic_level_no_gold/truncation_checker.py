"""Truncation checker for our environment.

This module provides truncation checking functionality for the N++ environment.
It monitors player movement at different scales to detect when the player might be stuck:

1. Cell-level movement (24px): Checks if player is stuck in same cell for 100 frames
2. Block-level movement (96px/4x4): Checks if player is stuck in same 4x4 area for 500 frames
3. Medium-area movement (192px/8x8): Checks if player is stuck in same 8x8 area for 1000 frames
4. Large-area movement (384px/16x16): Checks if player is stuck in same 16x16 area for 2000 frames

The episode will be truncated if:
- The max amount of frames (20,000) has been reached
- The player is detected as stuck at any movement scale
"""

import numpy as np
from typing import Tuple, Set


class TruncationChecker:
    # Constants for movement checking
    CELL_SIZE = 24  # Size of a single cell in pixels
    BLOCK_SIZE_4x4 = CELL_SIZE * 4  # Size of a 4x4 cell block
    BLOCK_SIZE_8x8 = CELL_SIZE * 8  # Size of a 8x8 cell block
    BLOCK_SIZE_16x16 = CELL_SIZE * 16  # Size of a 16x16 cell block

    # Frame windows for different movement scales (scaled exponentially)
    CELL_MOVEMENT_WINDOW = 500  # Single cell (baseline)
    BLOCK_4x4_MOVEMENT_WINDOW = CELL_MOVEMENT_WINDOW * 5  # 500 frames for 4x4
    BLOCK_8x8_MOVEMENT_WINDOW = CELL_MOVEMENT_WINDOW * 10  # 1000 frames for 8x8
    BLOCK_16x16_MOVEMENT_WINDOW = CELL_MOVEMENT_WINDOW * 20  # 2000 frames for 16x16
    MAX_FRAMES = 20000
    SHORT_EPISODE_MAX_FRAMES = 2000

    # Movement check configurations
    MOVEMENT_CHECKS = [
        {
            'name': 'cell',
            'size': CELL_SIZE,
            'window': CELL_MOVEMENT_WINDOW,
            'description': 'cell for 100 frames'
        },
        {
            'name': '4x4 block',
            'size': BLOCK_SIZE_4x4,
            'window': BLOCK_4x4_MOVEMENT_WINDOW,
            'description': '4x4 block for 500 frames'
        },
        {
            'name': '8x8 block',
            'size': BLOCK_SIZE_8x8,
            'window': BLOCK_8x8_MOVEMENT_WINDOW,
            'description': '8x8 block for 1000 frames'
        },
        {
            'name': '16x16 block',
            'size': BLOCK_SIZE_16x16,
            'window': BLOCK_16x16_MOVEMENT_WINDOW,
            'description': '16x16 block for 2000 frames'
        }
    ]

    def __init__(self, env, enable_short_episode_truncation: bool = False):
        """Initialize the truncation checker.

        Args:
            env: The BasicLevelNoGold environment instance
        """
        self.env = env
        self.position_history = []  # List of (x, y) tuples
        self.enable_short_episode_truncation = enable_short_episode_truncation

    def _check_movement_at_scale(self, window_size: int, block_size: int, description: str) -> Tuple[bool, str]:
        """Check if the ninja is stuck at a particular movement scale.

        Args:
            window_size: Number of frames to look back
            block_size: Size of the area to check movement in
            description: Description of the movement scale for error message

        Returns:
            Tuple of (is_stuck: bool, reason: str)
        """
        if len(self.position_history) < window_size:
            return False, ""

        recent_positions = set()
        for px, py in self.position_history[-window_size:]:
            block_x = int(px // block_size)
            block_y = int(py // block_size)
            recent_positions.add((block_x, block_y))

        if len(recent_positions) == 1:
            return True, f"Stuck in same {description}"

        return False, ""

    def update(self, x: float, y: float) -> Tuple[bool, str]:
        """Update position history and check for stuck conditions.

        Args:
            x: Current x position of the ninja
            y: Current y position of the ninja

        Returns:
            Tuple of (should_truncate: bool, reason: str)
        """
        self.position_history.append((x, y))

        # Check frame limits
        if self.enable_short_episode_truncation and len(self.position_history) >= self.SHORT_EPISODE_MAX_FRAMES:
            return True, "Max frames reached"

        if len(self.position_history) >= self.MAX_FRAMES:
            return True, "Max frames reached"

        # Check movement at each scale
        # for check in self.MOVEMENT_CHECKS:
        #     is_stuck, reason = self._check_movement_at_scale(
        #         check['window'],
        #         check['size'],
        #         check['description']
        #     )
        #     if is_stuck:
        #         return True, reason

        return False, ""

    def reset(self):
        """Reset the truncation checker state."""
        self.position_history = []

    def get_debug_info(self) -> dict:
        """Get debug information about the current state of movement tracking.

        Returns:
            dict: Debug information containing:
                - position_history_length: Length of position history
                - movement_stats: Dict of movement stats at each scale
        """
        debug_info = {
            'movement_stats': {}
        }

        if len(self.position_history) > 0:
            current_pos = self.position_history[-1]

            # Add movement stats for each scale
            for check in self.MOVEMENT_CHECKS:
                window_size = min(check['window'], len(self.position_history))
                recent_positions = set()
                for px, py in self.position_history[-window_size:]:
                    block_x = int(px // check['size'])
                    block_y = int(py // check['size'])
                    recent_positions.add((block_x, block_y))

                debug_info['movement_stats'][check['name']] = {
                    'unique_areas_visited': len(recent_positions),
                    'window_size': window_size,
                    'current_area': (
                        int(current_pos[0] // check['size']),
                        int(current_pos[1] // check['size'])
                    )
                }

        return debug_info
