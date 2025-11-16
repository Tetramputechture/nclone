"""Truncation checker for our environment.

This module provides truncation checking functionality for the N++ environment.
It monitors player movement at different scales to detect when the player might be stuck:



The episode will be truncated if:
- The dynamically calculated max amount of frames has been reached (based on level complexity)
"""

from typing import Tuple

from .constants import MAX_TIME_IN_FRAMES  # Keep as fallback
from .truncation_calculator import calculate_truncation_limit


class TruncationChecker:
    def __init__(self, env):
        """Initialize the truncation checker.

        Args:
            env: The NppEnvironment environment instance
        """
        self.env = env
        self.position_history = []  # List of (x, y) tuples
        self.current_truncation_limit = MAX_TIME_IN_FRAMES  # fallback

    def set_level_truncation_limit(
        self, surface_area: float, reachable_mine_count: int
    ) -> int:
        """
        Set truncation limit for current level based on complexity.
        Called once per level load (cached).

        Args:
            surface_area: PBRS surface area (number of reachable nodes)
            reachable_mine_count: Number of reachable toggle mines

        Returns:
            Computed truncation limit in frames
        """
        self.current_truncation_limit = calculate_truncation_limit(
            surface_area, reachable_mine_count
        )
        return self.current_truncation_limit

    def update(self, x: float, y: float) -> Tuple[bool, str]:
        """Update position history and check for stuck conditions.

        Args:
            x: Current x position of the ninja
            y: Current y position of the ninja

        Returns:
            Tuple of (should_truncate: bool, reason: str)
        """
        self.position_history.append((x, y))

        if len(self.position_history) >= self.current_truncation_limit:
            return True, f"Max frames reached ({self.current_truncation_limit})"

        return False, ""

    def reset(self):
        """Reset position history but keep truncation limit (per-level cache)."""
        self.position_history = []
