"""Sparse reward calculator for SAC-Discrete training.

Replaces HWM/PBRS dense shaping with minimal sparse rewards.
The RFCL curriculum provides the learning gradient instead of dense shaping.

Reward constants:
    R_SWITCH = +1.0  (exit switch activated)
    R_EXIT   = +5.0  (reached exit door)
    R_DEATH  = -0.5  (ninja dies, any cause)
    R_STEP   = -0.001 (per primitive frame)
"""

R_SWITCH = 1.0
R_EXIT = 5.0
R_DEATH = -0.5
R_STEP = -0.001


class SparseRewardCalculator:
    """Minimal sparse reward function for SAC-Discrete.

    No distance shaping, no HWM, no RND. The RFCL curriculum provides
    the learning gradient via initial state distribution.
    """

    def compute_frame_reward(
        self,
        switch_just_activated: bool,
        player_won: bool,
        player_dead: bool,
    ) -> float:
        """Compute reward for a single primitive frame.

        Args:
            switch_just_activated: True if exit switch was collected this frame.
            player_won: True if exit door was reached this frame.
            player_dead: True if ninja died this frame.

        Returns:
            Scalar reward for this frame.
        """
        if player_won:
            return R_EXIT
        if player_dead:
            return R_DEATH
        if switch_just_activated:
            return R_SWITCH
        return R_STEP
