"""Truncation checker for our environment.

This module provides truncation checking functionality for the N++ environment.
It monitors player movement at different scales to detect when the player might be stuck:



The episode will be truncated if:
- The dynamically calculated max amount of frames has been reached (based on level complexity)
"""

from .constants import MAX_TIME_IN_FRAMES  # Keep as fallback
from .truncation_calculator import calculate_truncation_limit


class TruncationChecker:
    def __init__(self, env, verbose: int = 0):
        """Initialize the truncation checker.

        Args:
            env: The NppEnvironment environment instance
        """
        self.env = env
        self.current_truncation_limit = MAX_TIME_IN_FRAMES  # fallback
        self.verbose = verbose

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
        """Check if episode should be truncated based on actual frame count.

        Uses the simulator's frame counter directly to ensure accurate truncation
        regardless of frame_skip settings.

        For checkpoint episodes, only counts frames executed after checkpoint replay,
        not the frames used to replay the checkpoint itself.
        """
        total_frames = self.env.nplay_headless.sim.frame
        replay_frames = getattr(self.env, "_checkpoint_replay_frame_count", 0)
        current_frame = total_frames - replay_frames
        effective_limit = int(self.current_truncation_limit * multiplier)
        should_truncate = current_frame >= effective_limit

        if self.verbose > 0:
            # DEBUG: Always log truncation checks when frame count is high
            # Log every 100 frames after 90% of limit to diagnose why truncation isn't triggering
            if current_frame >= effective_limit * 0.90 and current_frame % 100 < 4:
                import logging

                logger = logging.getLogger(__name__)
                logger.error(
                    f"[TRUNCATION CHECK] frame={current_frame}, limit={effective_limit}, "
                    f"total_frames={total_frames}, replay_frames={replay_frames}, "
                    f"base_limit={self.current_truncation_limit}, multiplier={multiplier}, "
                    f"should_truncate={should_truncate}, "
                    f"comparison={current_frame}>={effective_limit}={current_frame >= effective_limit}"
                )

        return should_truncate

    def reset(self):
        """Reset state for new episode. Frame count is reset by simulator."""
        pass
