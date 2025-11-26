"""Truncation checker for our environment.

This module provides truncation checking functionality for the N++ environment.
It monitors player movement at different scales to detect when the player might be stuck:



The episode will be truncated if:
- The dynamically calculated max amount of frames has been reached (based on level complexity)
"""


from .constants import MAX_TIME_IN_FRAMES  # Keep as fallback
from .truncation_calculator import calculate_truncation_limit


class TruncationChecker:
    def __init__(self, env):
        """Initialize the truncation checker.

        Args:
            env: The NppEnvironment environment instance
        """
        self.env = env
        self.positions_visited_count = 0
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

    def update_and_check_for_truncation(self, multiplier: float) -> bool:
        """Update position history and check for stuck conditions."""
        self.positions_visited_count += 1
        return self.positions_visited_count >= int(
            self.current_truncation_limit * multiplier
        )

    def reset(self):
        """Reset position history but keep truncation limit (per-level cache)."""
        self.positions_visited_count = 0
