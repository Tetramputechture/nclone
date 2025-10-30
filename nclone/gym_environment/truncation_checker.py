"""Truncation checker for our environment.

This module provides truncation checking functionality for the N++ environment.
It monitors player movement at different scales to detect when the player might be stuck:



The episode will be truncated if:
- The max amount of frames (20,000) has been reached
"""

from typing import Tuple


class TruncationChecker:
    MAX_FRAMES = 5000
    SHORT_EPISODE_MAX_FRAMES = 2000

    def __init__(self, env, enable_short_episode_truncation: bool = False):
        """Initialize the truncation checker.

        Args:
            env: The NppEnvironment environment instance
        """
        self.env = env
        self.position_history = []  # List of (x, y) tuples
        self.enable_short_episode_truncation = enable_short_episode_truncation

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
        if (
            self.enable_short_episode_truncation
            and len(self.position_history) >= self.SHORT_EPISODE_MAX_FRAMES
        ):
            return True, "Max frames reached"

        if len(self.position_history) >= self.MAX_FRAMES:
            return True, "Max frames reached"

        return False, ""

    def reset(self):
        """Reset the truncation checker state."""
        self.position_history = []
