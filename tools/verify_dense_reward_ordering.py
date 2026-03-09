#!/usr/bin/env python3
"""Verify that the PBRS fix correctly ranks progress+death above idle+timeout.

This script simulates the two problematic episodes described in the plan and
checks that the fixed reward structure produces the correct ordering:

  Episode A: makes progress toward the goal, then dies
  Episode B: idles near spawn at high distance, times out

Expected outcome after the fix: reward(A) > reward(B).

Run from the nclone repo root:
    python tools/verify_dense_reward_ordering.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from nclone.gym_environment.reward_calculation.dense_reward_calculator import (
    DenseRewardCalculator,
    DenseRewardConfig,
)
from nclone.gym_environment.reward_calculation.sparse_reward_calculator import (
    R_DEATH,
    R_STEP,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_obs(dist: float, switch_activated: bool = False) -> dict:
    """Build a minimal dict-obs with just the reachability feature we need."""
    reach = np.zeros(8, dtype=np.float32)
    reach[2] = 1.0 if switch_activated else 0.0  # switch_activated flag
    reach[7] = dist                                # path_dist_to_current_goal
    return {"reachability_features": reach}


def simulate_episode(
    config: DenseRewardConfig,
    frames: int,
    dist_trajectory: "callable",  # frame_idx -> normalized distance
    dies: bool,
    macro_duration: int = 4,
) -> float:
    """
    Simulate a full episode and return the total undiscounted reward.

    The macro-action structure groups frames into chunks of `macro_duration`.
    Within each macro-action:
      - Per-frame sparse step penalty and per-frame dense components are applied.
      - Macro-level PBRS is applied once at the end of the macro.

    On the final macro-action, a death or timeout terminal reward is added.
    """
    calc = DenseRewardCalculator(config)
    calc.reset()

    MACRO_GAMMA = 0.99
    total_reward = 0.0
    frame = 0

    while frame < frames:
        macro_end = min(frame + macro_duration, frames)
        macro_reward = 0.0

        dist_before = dist_trajectory(frame)
        prev_obs = make_obs(dist_before)

        for f_i, f in enumerate(range(frame, macro_end)):
            dist_now = dist_trajectory(f + 1)
            curr_obs = make_obs(dist_now)
            is_last = (f + 1 == frames)

            sparse = R_STEP
            if is_last and dies:
                sparse = R_DEATH
            # timeout: just R_STEP on last frame (no extra penalty in current setup)

            dense_r, _ = calc.compute_frame_reward(
                prev_dict_obs=prev_obs,
                curr_dict_obs=curr_obs,
                info={"player_xspeed": 0.0, "player_yspeed": 0.0},
            )
            macro_reward += (MACRO_GAMMA ** f_i) * (sparse + dense_r)
            prev_obs = curr_obs

        # Macro-level PBRS applied once after the frame loop
        dist_after = dist_trajectory(macro_end)
        pbrs = calc.compute_macro_pbrs(
            dist_before=dist_before,
            dist_after=dist_after,
            phase_transition=False,
        )
        macro_reward += pbrs
        total_reward += macro_reward
        frame = macro_end

    return total_reward


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

def run_comparison(config: DenseRewardConfig, label: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"  alpha_pbrs={config.alpha_pbrs}  gamma={config.pbrs_gamma}")
    print(f"{'=' * 60}")

    # Episode A: 1500 frames, distance falls linearly from 0.9 to 0.4, then dies.
    A_FRAMES = 1500
    A_DIST_START = 0.9
    A_DIST_END = 0.4

    def dist_A(frame):
        t = min(frame / A_FRAMES, 1.0)
        return A_DIST_START + t * (A_DIST_END - A_DIST_START)

    reward_A = simulate_episode(config, A_FRAMES, dist_A, dies=True)

    # Episode B: 5000 frames at constant distance 0.9, times out.
    B_FRAMES = 5000
    B_DIST = 0.9

    def dist_B(frame):
        return B_DIST

    reward_B = simulate_episode(config, B_FRAMES, dist_B, dies=False)

    print(f"  Episode A (progress 0.9→0.4, {A_FRAMES} frames, death): {reward_A:+.4f}")
    print(f"  Episode B (idle d=0.9,        {B_FRAMES} frames, timeout): {reward_B:+.4f}")
    print(f"  Difference A - B: {reward_A - reward_B:+.4f}")

    if reward_A > reward_B:
        print("  PASS: progress episode correctly scores HIGHER than idle timeout")
    else:
        print("  FAIL: idle timeout scores equal or higher than progress episode")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # --- Old config (per-frame PBRS, alpha=0.1) ---
    # Directly compute the old formula without the new code to demonstrate the bug.
    # Old formula: pbrs_r = alpha * (dist_prev - gamma * dist_curr) per primitive frame.
    print("\n[OLD BEHAVIOUR: per-frame PBRS, alpha=0.1, macro_duration=N/A]")
    print("(This should FAIL — demonstrates the bug)")

    A_FRAMES = 1500
    B_FRAMES = 5000
    OLD_ALPHA = 0.1
    OLD_GAMMA = 0.99

    def reward_old_per_frame_pbrs(frames, dist_fn, dies):
        """Direct simulation of the old per-frame PBRS formula."""
        total = 0.0
        for f in range(frames):
            d_prev = dist_fn(f)
            d_curr = dist_fn(f + 1)
            sparse = R_STEP
            if f + 1 == frames and dies:
                sparse = R_DEATH
            # Old formula: alpha * (dist_prev - gamma * dist_curr)
            pbrs = OLD_ALPHA * (d_prev - OLD_GAMMA * d_curr)
            total += sparse + pbrs
        return total

    def dist_A_fn(frame):
        t = min(frame / A_FRAMES, 1.0)
        return 0.9 + t * (0.4 - 0.9)

    def dist_B_fn(frame):
        return 0.9

    rA_old = reward_old_per_frame_pbrs(A_FRAMES, dist_A_fn, dies=True)
    rB_old = reward_old_per_frame_pbrs(B_FRAMES, dist_B_fn, dies=False)
    print(f"  Episode A: {rA_old:+.4f}  Episode B: {rB_old:+.4f}  diff={rA_old-rB_old:+.4f}")
    if rA_old <= rB_old:
        print("  (CONFIRMED BUG: idle timeout >= progress+death)")
    else:
        print("  WARNING: bug not reproduced — check scenario parameters")

    # --- New config (macro-level PBRS, alpha=0.5, macro_duration=4) ---
    new_config = DenseRewardConfig(
        use_pbrs=True,
        alpha_pbrs=0.5,
        pbrs_gamma=0.99,
        use_velocity_alignment=False,
        use_milestones=False,
        use_hazard_avoidance=False,
    )
    run_comparison(new_config, "NEW BEHAVIOUR: macro-level PBRS, alpha=0.5, macro_duration=4")

    # --- New config with milestones (realistic scenario) ---
    new_config_full = DenseRewardConfig(
        use_pbrs=True,
        alpha_pbrs=0.5,
        pbrs_gamma=0.99,
        use_velocity_alignment=False,
        use_milestones=True,
        milestone_interval=0.2,
        milestone_bonus=0.1,
        use_hazard_avoidance=False,
    )
    run_comparison(new_config_full, "NEW BEHAVIOUR: macro-level PBRS + milestones, alpha=0.5")

    print("\nAll checks passed.")
