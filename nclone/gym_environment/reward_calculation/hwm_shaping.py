"""High-Water-Mark A* Distance Shaping for N++ RL training.

Replaces classical PBRS with a simpler, non-negative shaping mechanism:
- Agent receives positive reward ONLY when achieving a new minimum A* distance
- Backtracking yields zero reward (not negative)
- Eliminates the non-monotonic penalty that punishes necessary momentum-building

Two-phase reward machine:
- Phase 0 (pre-switch): Track A* distance to exit switch
- Phase 1 (post-switch): Track A* distance to exit door
- Watermark resets on phase transition

Properties:
- Non-negative: shaping reward >= 0 always
- Bounded cumulative: total HWM reward <= 1.0 per phase (when normalized)
- Non-exploitable: cycling reduces->increases distance yields one reward then zeros
- History-dependent: watermark is episode state (not in observation)

Reference: Sonic OpenAI Retro Contest (2018) validated HWM on momentum-based platformers.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EpisodeRewardState:
    """Per-environment, per-episode state for HWM reward shaping.

    Maintains watermarks for both objectives (switch and exit) and handles
    phase transitions when the switch is activated.
    """

    watermark_switch: float = float("inf")
    watermark_exit: float = float("inf")
    current_phase: int = 0

    max_path_dist_switch: float = 1.0
    max_path_dist_exit: float = 1.0

    def reset(
        self,
        initial_dist_switch: float,
        max_dist_switch: float,
        initial_dist_exit: float,
        max_dist_exit: float,
    ) -> None:
        """Reset for a new episode.

        Args:
            initial_dist_switch: A* distance from spawn to switch
            max_dist_switch: Maximum possible A* distance to switch (for normalization)
            initial_dist_exit: A* distance from spawn to exit
            max_dist_exit: Maximum possible A* distance to exit (for normalization)
        """
        self.watermark_switch = initial_dist_switch
        self.watermark_exit = float("inf")
        self.current_phase = 0
        self.max_path_dist_switch = max(max_dist_switch, 1.0)
        self.max_path_dist_exit = max(max_dist_exit, 1.0)

    def step(
        self,
        dist_switch: float,
        dist_exit: float,
        switch_activated: bool,
    ) -> float:
        """Compute HWM shaping reward for one frame.

        Args:
            dist_switch: Current A* path distance to switch (pixels)
            dist_exit: Current A* path distance to exit (pixels)
            switch_activated: Whether the switch has been collected

        Returns:
            Normalized shaping reward in [0, 1]. Positive only when a new
            minimum distance is achieved; zero otherwise.
        """
        new_phase = 1 if switch_activated else 0

        if new_phase != self.current_phase:
            self.current_phase = new_phase
            self.watermark_exit = dist_exit
            return 0.0

        if self.current_phase == 0:
            if dist_switch < self.watermark_switch:
                progress = self.watermark_switch - dist_switch
                self.watermark_switch = dist_switch
                return progress / self.max_path_dist_switch
            return 0.0
        else:
            if dist_exit < self.watermark_exit:
                progress = self.watermark_exit - dist_exit
                self.watermark_exit = dist_exit
                return progress / self.max_path_dist_exit
            return 0.0

    @property
    def best_switch_distance(self) -> float:
        """Best (minimum) A* distance to switch achieved this episode."""
        return self.watermark_switch

    @property
    def best_exit_distance(self) -> float:
        """Best (minimum) A* distance to exit achieved this episode."""
        return self.watermark_exit
