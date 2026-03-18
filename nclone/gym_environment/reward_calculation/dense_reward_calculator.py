"""Dense reward calculator for SAC-Discrete training.

Provides three per-primitive-frame components and one per-macro-action
component on top of the existing sparse reward signal:

Per-macro-action (call compute_macro_pbrs once after the frame loop):
1. Progress shaping using graph distance:
   F = alpha * max(0, dist_before_macro - dist_after_macro)
   Positive-only so standing still always yields zero progress reward.
   This avoids the standing-still bias in the standard PBRS formula
   ``alpha * (dist - gamma * dist) = alpha * dist * (1 - gamma)`` per frame,
   which at alpha=0.1 and d=0.9 nearly cancels R_STEP=-0.001 making idle
   timeout episodes more rewarding than progress+death episodes.
   Applied ONCE per macro-action so that the reward reflects actual distance
   change across the entire multi-frame action, not per-frame jitter.

Per-primitive-frame (call compute_frame_reward each frame):
2. Velocity-direction alignment bonus:
   r_vel = beta * max(0, dot(v_unit, next_hop_dir))
   Rewards goal-directed momentum even before position improves. Positive-
   only so setup movements (building momentum, wall-jumping approach) are
   never penalised -- critical for physics-heavy traversal.

3. Distance milestone bonuses:
   r_milestone = bonus when normalised distance drops below a new threshold
   for the first time in the episode. One-time per milestone; prevents
   cycling exploitation while giving a stronger signal for big moves.

4. Mine hazard avoidance (SDF-based):
   r_hazard = delta * max(0, sdf_curr - sdf_prev)  when sdf_prev < threshold
   A small dense signal for moving away from mine danger zones, addressing
   a common cause of premature episode termination.

All components are individually configurable and independently toggleable.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Set

import numpy as np

from ..constants import REACHABILITY_FEATURES_DIM


# Indices into the 8-dim reachability feature vector
_REACH_DIST_SWITCH = 0       # path_dist_to_switch_norm
_REACH_DIST_EXIT = 1         # path_dist_to_exit_norm
_REACH_SWITCH_ACTIVATED = 2  # switch_activated flag
_REACH_NEXT_HOP_X = 3        # next_hop_dir_x
_REACH_NEXT_HOP_Y = 4        # next_hop_dir_y
_REACH_PLATFORM_AHEAD = 5    # platform distance in next-hop direction
_REACH_PLATFORM_BELOW = 6    # platform distance below
_REACH_DIST_GOAL = 7         # path_dist_to_current_goal (phase-aware, 0-1)


@dataclass
class DenseRewardConfig:
    """Hyperparameters for the dense reward calculator.

    All default weights are intentionally small relative to the sparse
    rewards (R_SWITCH=1, R_EXIT=5) so that dense components act as
    learning accelerants rather than dominant objectives.
    """

    # ---- Progress shaping (replaces standard PBRS) ----
    # Applied ONCE per macro-action via compute_macro_pbrs(), not per frame.
    # See module docstring for motivation.
    use_pbrs: bool = True
    # Weight applied to F = alpha * max(0, dist_before - dist_after).
    # 0.5 gives a meaningful per-macro-action signal: closing half the remaining
    # distance in one macro earns 0.5 * 0.5 = 0.25, comparable to a milestone
    # bonus (0.1) and well below R_EXIT (5.0) and R_SWITCH (1.0).
    alpha_pbrs: float = 0.5
    # pbrs_gamma is kept for API compatibility but no longer used in the
    # progress-only formula (which is positive-only, not gamma-discounted).
    pbrs_gamma: float = 0.99

    # ---- Velocity-direction alignment ----
    use_velocity_alignment: bool = True
    # Per-frame reward magnitude for perfect alignment with next-hop direction.
    # Reduced from 0.0005 to 0.0001: at 0.0005 a 10k-frame timeout episode
    # accumulated ~1.5 in velocity reward from random movement alone, making
    # timeouts ~35% denser than intended and suppressing the gradient to solve
    # the level (run-38 analysis: vel_total = 1.51/ep, fraction_of_reward = 35%).
    beta_vel: float = 0.0001

    # ---- Distance milestones ----
    use_milestones: bool = True
    # Fraction of initial episode distance per milestone step (0.1 = every 10%).
    # Reduced from 0.2 to 0.1: finer intervals give denser signal during early
    # RFCL near-goal exploration where 20% thresholds were too coarse to
    # distinguish incremental progress (run-38: only milestone 0 ever crossed).
    milestone_interval: float = 0.1
    # Flat bonus per milestone crossed for the first time in the episode.
    milestone_bonus: float = 0.1

    # ---- Mine hazard avoidance ----
    use_hazard_avoidance: bool = True
    # Per-frame reward = delta * max(0, sdf_curr - sdf_prev) inside danger zone.
    delta_hazard: float = 0.0001
    # SDF value below which the agent is considered in the danger zone.
    hazard_sdf_threshold: float = 0.3


class DenseRewardCalculator:
    """Computes dense reward components for SAC-Discrete training.

    Provides two call sites:

    1. ``compute_frame_reward(prev_dict_obs, curr_dict_obs, info)`` — called
       once per primitive frame inside the macro-action loop.  Handles
       velocity-alignment, distance milestones, and mine hazard avoidance.

    2. ``compute_macro_pbrs(dist_before, dist_after, phase_transition)`` —
       called ONCE per macro-action, AFTER the frame loop, with the distance
       values from the start and end of the whole macro-action.  Handles
       Potential-Based Reward Shaping (PBRS).

    Separating PBRS from the frame loop eliminates the standing-still bias:
    per-frame PBRS at constant distance ``d`` yields ``alpha*(1-gamma)*d``
    per frame, which at alpha=0.1 and d=0.9 nearly cancels R_STEP=-0.001,
    making timeout episodes more rewarding than progress+death episodes.
    Macro-level PBRS only produces a non-trivial reward when the agent
    actually closes distance across the full macro-action.

    Maintains per-episode state (milestones, initial distance, previous SDF)
    and must be reset at the start of each episode via reset().
    """

    def __init__(self, config: DenseRewardConfig):
        self.config = config

        # Per-episode state (initialised lazily on first frame)
        self._initial_distance: Optional[float] = None
        self._milestones_crossed: Set[int] = set()
        self._prev_sdf: Optional[float] = None

        # Cumulative per-component episode totals for diagnostics
        self._ep_pbrs: float = 0.0
        self._ep_vel: float = 0.0
        self._ep_milestone: float = 0.0
        self._ep_hazard: float = 0.0

    def reset(self) -> None:
        """Reset per-episode state. Must be called at the start of each episode."""
        self._initial_distance = None
        self._milestones_crossed = set()
        self._prev_sdf = None
        self._ep_pbrs = 0.0
        self._ep_vel = 0.0
        self._ep_milestone = 0.0
        self._ep_hazard = 0.0

    def compute_macro_pbrs(
        self,
        dist_before: float,
        dist_after: float,
        phase_transition: bool = False,
    ) -> float:
        """Compute progress shaping reward for a completed macro-action.

        Called ONCE per macro-action (not per primitive frame) so that the
        reward is proportional to actual distance reduction across the whole
        macro-action rather than to the number of frames executed.

        Formula (progress-only): F = alpha * max(0, dist_before - dist_after)

        This is deliberately one-sided (positive only) rather than the full
        PBRS formula ``alpha * (dist_before - gamma * dist_after)``.  The
        standard PBRS formula has a standing-still term equal to
        ``alpha * dist * (1 - gamma)`` per step that, at alpha=0.5 and d=0.9,
        exceeds the R_STEP=-0.001 penalty for any macro-action shorter than
        ~4.5 frames — meaning idling at high distance remains slightly
        profitable even at the macro level.  Progress-only shaping gives
        exactly zero reward for standing still regardless of distance or
        macro duration, completely eliminating the bias.

        Args:
            dist_before: Normalised path-distance at the START of the macro-action
                (from ``path_dist_to_current_goal``, feature index 7, range 0-1).
            dist_after: Normalised path-distance at the END of the macro-action.
            phase_transition: True if the switch was activated during this
                macro-action.  PBRS is skipped because the distance signal
                switches from distance-to-switch to distance-to-exit,
                creating an artificial discontinuity that would produce a
                large spurious reward.

        Returns:
            Scalar progress reward >= 0.  Zero when standing still or moving
            away from the goal; positive only when distance was reduced.
        """
        if not self.config.use_pbrs or phase_transition:
            return 0.0

        # Lazy initialisation of initial distance on the first macro-action.
        if self._initial_distance is None:
            self._initial_distance = dist_before

        progress = dist_before - dist_after
        pbrs_r = self.config.alpha_pbrs * max(0.0, progress)
        self._ep_pbrs += pbrs_r
        return pbrs_r

    def compute_frame_reward(
        self,
        prev_dict_obs: Dict,
        curr_dict_obs: Dict,
        info: Dict,
    ) -> tuple:
        """Compute per-frame dense reward components (velocity, milestones, hazard).

        Does NOT include PBRS — use compute_macro_pbrs() for that.

        Args:
            prev_dict_obs: Observation dict BEFORE this frame (state s_t).
            curr_dict_obs: Observation dict AFTER this frame (state s_{t+1}).
            info: Info dict from the inner-env step (contains player velocities).

        Returns:
            Tuple of (total_dense_reward: float, breakdown: dict).
            breakdown keys: "velocity_alignment", "milestone", "hazard_avoidance".
        """
        reach_prev = np.asarray(
            prev_dict_obs.get(
                "reachability_features",
                np.zeros(REACHABILITY_FEATURES_DIM, dtype=np.float32),
            ),
            dtype=np.float32,
        )
        reach_curr = np.asarray(
            curr_dict_obs.get(
                "reachability_features",
                np.zeros(REACHABILITY_FEATURES_DIM, dtype=np.float32),
            ),
            dtype=np.float32,
        )
        mine_sdf = np.asarray(
            curr_dict_obs.get("mine_sdf_features", np.zeros(3, dtype=np.float32)),
            dtype=np.float32,
        )

        dist_prev = float(reach_prev[_REACH_DIST_GOAL])
        dist_curr = float(reach_curr[_REACH_DIST_GOAL])

        # Lazy initialisation of initial distance on the first frame of the episode.
        # (Also initialised inside compute_macro_pbrs; whichever fires first wins.)
        if self._initial_distance is None:
            self._initial_distance = dist_prev

        total = 0.0
        vel_r = 0.0
        milestone_r = 0.0
        hazard_r = 0.0

        # ----------------------------------------------------------------
        # Component 1: Velocity-direction alignment
        # Reward goal-directed momentum regardless of position change.
        # Positive-only: no penalty for setup / preparatory movements.
        #
        # Primary path: use next_hop_dir from the A* graph (reachability
        # features [3,4]).  These may be (0,0) when the BFS level cache
        # doesn't have a parent entry for the ninja's current node (e.g. at
        # the goal node, or when node lookup fails after RFCL position resets).
        #
        # Fallback path: when next_hop_dir is unavailable, reward
        # speed * path_progress — a proxy that fires whenever the agent is
        # moving fast AND closing distance toward the goal.
        # ----------------------------------------------------------------
        if self.config.use_velocity_alignment:
            vx = float(info.get("player_xspeed", 0.0))
            vy = float(info.get("player_yspeed", 0.0))
            speed_sq = vx * vx + vy * vy
            if speed_sq > 1e-12:
                speed = speed_sq ** 0.5
                dir_x = float(reach_curr[_REACH_NEXT_HOP_X])
                dir_y = float(reach_curr[_REACH_NEXT_HOP_Y])
                dir_sq = dir_x * dir_x + dir_y * dir_y
                if dir_sq > 1e-12:
                    # Standard path: cosine similarity between velocity and next-hop direction.
                    inv_speed = 1.0 / speed
                    inv_dir = 1.0 / (dir_sq ** 0.5)
                    alignment = (
                        (vx * inv_speed) * (dir_x * inv_dir)
                        + (vy * inv_speed) * (dir_y * inv_dir)
                    )
                    vel_r = self.config.beta_vel * max(0.0, alignment)
                else:
                    # Fallback: next_hop_dir unavailable (BFS parent not cached for
                    # this node).  Reward speed when making measurable path progress.
                    path_progress = max(0.0, dist_prev - dist_curr)
                    if path_progress > 1e-6:
                        vel_r = self.config.beta_vel * speed * path_progress
                    else:
                        vel_r = 0.0
                total += vel_r
                self._ep_vel += vel_r

        # ----------------------------------------------------------------
        # Component 2: Distance milestones
        # One-time bonus each time distance crosses a new 20% threshold.
        # Prevents oscillation exploitation (each milestone awarded once).
        # ----------------------------------------------------------------
        if self.config.use_milestones:
            init_d = self._initial_distance
            if init_d is not None and init_d > 1e-6:
                progress_ratio = 1.0 - dist_curr / init_d
                n_milestones = int(progress_ratio / self.config.milestone_interval)
                for m in range(n_milestones):
                    if m not in self._milestones_crossed:
                        self._milestones_crossed.add(m)
                        milestone_r += self.config.milestone_bonus
                total += milestone_r
                self._ep_milestone += milestone_r

        # ----------------------------------------------------------------
        # Component 3: Mine hazard avoidance (SDF-based)
        # Only active when already inside the mine danger zone.
        # Rewards moving away from mines, not just being far from them.
        # ----------------------------------------------------------------
        if self.config.use_hazard_avoidance:
            sdf_curr = float(mine_sdf[0])
            if (
                self._prev_sdf is not None
                and self._prev_sdf < self.config.hazard_sdf_threshold
            ):
                sdf_improvement = sdf_curr - self._prev_sdf
                hazard_r = self.config.delta_hazard * max(0.0, sdf_improvement)
                total += hazard_r
                self._ep_hazard += hazard_r
            self._prev_sdf = sdf_curr

        return total, {
            "velocity_alignment": vel_r,
            "milestone": milestone_r,
            "hazard_avoidance": hazard_r,
        }

    def get_episode_stats(self) -> Dict[str, float]:
        """Return cumulative per-component totals for the completed episode.

        Intended for TensorBoard logging at episode end.
        """
        total = self._ep_pbrs + self._ep_vel + self._ep_milestone + self._ep_hazard
        return {
            "dense_pbrs_total": self._ep_pbrs,
            "dense_vel_total": self._ep_vel,
            "dense_milestone_total": self._ep_milestone,
            "dense_hazard_total": self._ep_hazard,
            "dense_total": total,
        }
