"""Simplified primitive-level reward calculator using Phi grids + Reward Machine.

Implements the optimal PBRS setup from pbrs-plan.md:
- Precomputed Phi grids for O(1) potential lookup
- Explicit 3-state reward machine for task phase tracking
- Per-frame PBRS (precomputed VF path): F_t = Phi(s') - Phi(s)  (undiscounted)
- Total reward: r_total = r_sparse(RM) + alpha * F_t

Key invariants (precomputed 5D VF):
- Use actual Phi(s') at terminals; sparse rewards encode death/success/timeout.
- F_t telescopes to Phi(end)-Phi(start) (no gamma in F_t), so progress dominates
  over episode length; time_penalty_per_frame provides hurry incentive.
- Primitive-level F_t computed every physics frame
- Phase-correct targeting (before switch -> switch; after -> exit)

This calculator is designed for primitive-level application within
FiGAR-PPO, where rewards are computed every physics frame.
"""

import logging
from typing import Dict, Any, Optional, List

from .reward_machine import (
    RewardMachine,
    DEFAULT_R_SWITCH,
    DEFAULT_R_EXIT,
    DEFAULT_C_PHASE,
)
from .reward_constants import (
    PBRS_GAMMA,
    GLOBAL_REWARD_SCALE,
    TIME_PENALTY_PER_STEP,
)

logger = logging.getLogger(__name__)

# Default shaping scale (alpha in r_total = r_sparse + alpha * F_t)
# Start at 1.0; tune via sweep [0.1, 0.3, 0.5, 1.0]
DEFAULT_SHAPING_ALPHA = 1.0


class PrimitiveRewardCalculator:
    """Simplified reward calculator for primitive-level PBRS with reward machine.

    Computes per-frame rewards suitable for primitive-level GAE:
        r_t = r_sparse(RM transition) + alpha * F_t

    For precomputed_pbrs: F_t = Phi(s') - Phi(s) (full Phi from TwoPhaseValuePBRS).
    Otherwise: F_t = gamma * Phi(s') - Phi(s) with Phi from reward machine + grid.

    Phi_state comes from precomputed grids (O(1) lookup).
    Phi_phase comes from the reward machine state.

    Attributes:
        reward_machine: RewardMachine tracking task phase
        gamma: Discount factor for PBRS (must match PPO gamma)
        alpha: Shaping scale for PBRS reward
        prev_potential: Previous total potential (for F_t computation)
    """

    def __init__(
        self,
        gamma: float = PBRS_GAMMA,
        alpha: float = DEFAULT_SHAPING_ALPHA,
        r_switch: float = DEFAULT_R_SWITCH,
        r_exit: float = DEFAULT_R_EXIT,
        r_death: float = 0.0,
        r_death_mine: float = 0.0,
        time_penalty_per_frame: float = TIME_PENALTY_PER_STEP,
        c_phase: float = DEFAULT_C_PHASE,
        global_reward_scale: float = GLOBAL_REWARD_SCALE,
    ):
        """Initialize primitive reward calculator.

        Args:
            gamma: PBRS discount factor (MUST match PPO gamma)
            alpha: Shaping scale multiplier for F_t
            r_switch: Sparse reward for switch collection
            r_exit: Sparse reward for exit reached
            r_death: Sparse penalty applied on the death frame (typically negative)
            r_death_mine: Extra penalty when death_cause == 'mine' (e.g. -2.0)
            time_penalty_per_frame: Per-frame time penalty (aligns with TIME_PENALTY_PER_STEP
                so FiGAR-PPO receives same incentive as MainRewardCalculator path)
            c_phase: Phase potential bonus for switch-collected state
            global_reward_scale: Scale applied to all rewards for value stability
        """
        self.gamma = gamma
        self.alpha = alpha
        self.r_death = r_death
        self.r_death_mine = r_death_mine
        self.time_penalty_per_frame = time_penalty_per_frame
        self.global_reward_scale = global_reward_scale

        # Reward machine for task phase tracking
        self.reward_machine = RewardMachine(
            r_switch=r_switch,
            r_exit=r_exit,
            c_phase=c_phase,
        )

        # Previous potential for PBRS delta computation
        self.prev_potential: Optional[float] = None

        # Episode tracking for diagnostics
        self.steps_taken: int = 0
        self.episode_pbrs_rewards: List[float] = []
        self.episode_sparse_rewards: List[float] = []
        self.episode_forward_steps: int = 0
        self.episode_backtrack_steps: int = 0
        # Raw Φ deltas (pre-alpha) summed when graph progress is forward vs backward
        self._pbrs_raw_sum_forward: float = 0.0
        self._pbrs_raw_sum_backtrack: float = 0.0
        self.episode_terminal_reward: float = 0.0

        # Step-level components for TensorBoard logging
        self.last_components: Dict[str, Any] = {}

    def calculate_frame_reward(
        self,
        player_x: float,
        player_y: float,
        player_dead: bool,
        player_won: bool,
        switch_activated: bool,
        prev_switch_activated: bool,
        phi_grid_cache: Any,
        rust_phi_cache: Optional[Any] = None,
        is_truncated: bool = False,
        precomputed_pbrs: Optional[Any] = None,
        player_xspeed: Optional[float] = None,
        player_yspeed: Optional[float] = None,
        jump_duration_remaining: Optional[float] = None,
        contact_state: Optional[int] = None,
        death_cause: Optional[str] = None,
    ) -> float:
        """Calculate reward for a single physics frame.

        This is the core per-frame reward function. Call once per physics
        frame (not per action) for primitive-level PBRS.

        Args:
            player_x: Current player world x coordinate
            player_y: Current player world y coordinate
            player_dead: Whether player died this frame
            player_won: Whether player reached exit this frame
            switch_activated: Whether switch is currently activated
            prev_switch_activated: Whether switch was activated before this frame
            phi_grid_cache: LevelBasedPathDistanceCache with built Phi grids
            rust_phi_cache: Optional Rust LevelCache for O(1) get_phi (faster)
            is_truncated: Whether episode was truncated (timeout)
            precomputed_pbrs: Optional TwoPhaseValuePBRS for 5D velocity-aware potentials
            player_xspeed: Optional velocity x (for precomputed_pbrs)
            player_yspeed: Optional velocity y (for precomputed_pbrs)
            jump_duration_remaining: Optional normalized [0,1] jump remaining (for precomputed_pbrs)
            death_cause: Cause of death when player_dead (e.g. 'mine'); used for R_DEATH_MINE

        Returns:
            Total reward for this frame (scaled by global_reward_scale)
        """
        self.steps_taken += 1

        # Detect events for reward machine
        events = set()
        switch_just_activated = switch_activated and not prev_switch_activated
        if switch_just_activated:
            events.add('switch_collected')
        if player_won:
            events.add('exit_reached')

        # Step reward machine and get sparse reward
        sparse_reward = self.reward_machine.step(events)

        # Explicit death penalty (sparse, applied on the terminal death frame)
        if player_dead:
            sparse_reward += self.r_death
            if death_cause == "mine":
                sparse_reward += self.r_death_mine

        # Determine terminal state
        is_terminal = player_dead or player_won or is_truncated

        # Get potential: prefer 5D precomputed value function when available
        rm_state = self.reward_machine.state
        if precomputed_pbrs is not None:
            state_dict = {
                "player_x": player_x,
                "player_y": player_y,
                "player_xspeed": player_xspeed if player_xspeed is not None else 0.0,
                "player_yspeed": player_yspeed if player_yspeed is not None else 0.0,
                "jump_duration_remaining": (
                    jump_duration_remaining if jump_duration_remaining is not None else 0.0
                ),
                "switch_activated": switch_activated,
            }
            if contact_state is not None:
                state_dict["contact_state"] = contact_state
            phi_state = precomputed_pbrs.get_potential(state_dict)
        elif rust_phi_cache is not None:
            phi_state = rust_phi_cache.get_phi(
                player_x, player_y, rm_state
            )
        else:
            phi_state = phi_grid_cache.get_phi(player_x, player_y, rm_state)

        # Compute total potential
        # Precomputed path: phi_state is full Phi from SurfaceGraphPBRS (phase-aware).
        # Keep Phi at terminals so F_t does not inject a spurious +(-Phi) kick on death.
        # Grid path: reward machine applies terminal handling for policy invariance.
        if precomputed_pbrs is not None:
            current_potential = phi_state
        else:
            current_potential = self.reward_machine.get_total_potential(
                phi_state, is_terminal=is_terminal
            )

        prev_for_log = self.prev_potential

        # PBRS: undiscounted delta for precomputed VF (see module docstring).
        pbrs_reward = 0.0
        if self.prev_potential is not None:
            if precomputed_pbrs is not None:
                pbrs_reward = current_potential - self.prev_potential
            else:
                pbrs_reward = self.gamma * current_potential - self.prev_potential

            actual_change = current_potential - self.prev_potential
            if actual_change > 0.001:
                self.episode_forward_steps += 1
                self._pbrs_raw_sum_forward += pbrs_reward
            elif actual_change < -0.001:
                self.episode_backtrack_steps += 1
                self._pbrs_raw_sum_backtrack += pbrs_reward

        # Update previous potential for next frame
        if is_terminal:
            self.prev_potential = None
        else:
            self.prev_potential = current_potential

        # Compose total reward: sparse + PBRS shaping + per-frame time penalty
        # Time penalty discourages timeout-at-wall; must match MainRewardCalculator
        # semantics so FiGAR-PPO gets consistent signal regardless of reward path.
        total_reward = (
            sparse_reward
            + self.alpha * pbrs_reward
            + self.time_penalty_per_frame
        )

        # Track for diagnostics
        self.episode_pbrs_rewards.append(pbrs_reward)
        self.episode_sparse_rewards.append(sparse_reward)
        if sparse_reward > 0 or is_terminal:
            self.episode_terminal_reward += sparse_reward

        # Apply global reward scaling
        scaled_reward = total_reward * self.global_reward_scale

        # Store components for TensorBoard logging
        self.last_components = {
            "sparse_reward": sparse_reward,
            "pbrs_reward": pbrs_reward,
            "time_penalty": self.time_penalty_per_frame,
            "total_reward": total_reward,
            "scaled_reward": scaled_reward,
            "current_potential": current_potential,
            "prev_potential": prev_for_log if prev_for_log is not None else 0.0,
            "phi_state": phi_state,
            "phi_phase": self.reward_machine.get_phi_phase(),
            "rm_state": rm_state,
            "is_terminal": is_terminal,
            "alpha": self.alpha,
            "gamma": self.gamma,
        }

        return scaled_reward

    def calculate_death_reward(self) -> float:
        """Legacy hook; primitive path uses sparse death penalties in calculate_frame_reward."""
        return 0.0

    def reset(self) -> None:
        """Reset calculator state for new episode."""
        self.reward_machine.reset()
        self.prev_potential = None
        self.steps_taken = 0
        self.episode_pbrs_rewards = []
        self.episode_sparse_rewards = []
        self.episode_forward_steps = 0
        self.episode_backtrack_steps = 0
        self._pbrs_raw_sum_forward = 0.0
        self._pbrs_raw_sum_backtrack = 0.0
        self.episode_terminal_reward = 0.0
        self.last_components = {}

    def get_episode_stats(self) -> Dict[str, float]:
        """Get episode-level statistics for TensorBoard logging.

        Returns:
            Dictionary of episode statistics
        """
        total_pbrs = sum(self.episode_pbrs_rewards) if self.episode_pbrs_rewards else 0.0
        total_sparse = sum(self.episode_sparse_rewards) if self.episode_sparse_rewards else 0.0
        n = len(self.episode_pbrs_rewards)
        time_unscaled = n * self.time_penalty_per_frame
        total_unscaled = total_sparse + self.alpha * total_pbrs + time_unscaled
        scaled_return = total_unscaled * self.global_reward_scale

        fwd_n = max(self.episode_forward_steps, 1)
        bck_n = max(self.episode_backtrack_steps, 1)
        return {
            "episode/total_pbrs": total_pbrs,
            "episode/total_sparse": total_sparse,
            "episode/total_reward": total_unscaled,
            "episode/scaled_return_from_components": scaled_return,
            "episode/forward_steps": float(self.episode_forward_steps),
            "episode/backtrack_steps": float(self.episode_backtrack_steps),
            "episode/terminal_reward": self.episode_terminal_reward,
            "episode/steps": float(self.steps_taken),
            "episode/pbrs_mean": total_pbrs / max(self.steps_taken, 1),
            # On-policy VF audit: Φ delta should be positive on forward graph steps
            "episode/pbrs_mean_raw_on_forward_frames": self._pbrs_raw_sum_forward
            / fwd_n
            if self.episode_forward_steps > 0
            else 0.0,
            "episode/pbrs_mean_raw_on_backtrack_frames": self._pbrs_raw_sum_backtrack
            / bck_n
            if self.episode_backtrack_steps > 0
            else 0.0,
        }

    def get_episode_reward_metrics(self) -> Dict[str, Any]:
        """Episode metrics for PBRSLoggingCallback / EnhancedTensorBoardCallback.

        Keys mirror ``RewardCalculator.get_episode_reward_metrics`` but values
        come from primitive PBRS (sparse + alpha * F_t + per-frame time penalty),
        scaled consistently with :meth:`calculate_frame_reward`.
        """
        import numpy as np

        pbrs_list = self.episode_pbrs_rewards
        scale = self.global_reward_scale
        total_sparse_u = (
            sum(self.episode_sparse_rewards) if self.episode_sparse_rewards else 0.0
        )
        n_frames = len(pbrs_list)
        time_u = n_frames * self.time_penalty_per_frame

        per_frame_pbrs_scaled = [self.alpha * r * scale for r in pbrs_list] if pbrs_list else []
        pbrs_total_scaled = float(sum(per_frame_pbrs_scaled))
        time_penalty_total_scaled = time_u * scale
        sparse_total_scaled = total_sparse_u * scale

        hwm_mean = float(np.mean(per_frame_pbrs_scaled)) if per_frame_pbrs_scaled else 0.0
        hwm_std = float(np.std(per_frame_pbrs_scaled)) if per_frame_pbrs_scaled else 0.0

        nav = self.episode_forward_steps + self.episode_backtrack_steps
        forward_pct = (
            (self.episode_forward_steps / nav * 100.0) if nav > 0 else 0.0
        )

        lc = self.last_components or {}
        ratio = 0.0
        if abs(time_penalty_total_scaled) > 1e-12:
            ratio = pbrs_total_scaled / abs(time_penalty_total_scaled)

        return {
            "pbrs_total": pbrs_total_scaled,
            "hwm_total": pbrs_total_scaled,
            "time_penalty_total": time_penalty_total_scaled,
            "sparse_total_scaled": sparse_total_scaled,
            "terminal_reward": self.episode_terminal_reward * scale,
            "pbrs_to_penalty_ratio": ratio,
            "forward_steps": self.episode_forward_steps,
            "backtrack_steps": self.episode_backtrack_steps,
            "pbrs_mean": hwm_mean,
            "pbrs_std": hwm_std,
            "hwm_mean": hwm_mean,
            "hwm_std": hwm_std,
            "current_potential": lc.get("current_potential"),
            "prev_potential": lc.get("prev_potential"),
            "path_optimality": 0.0,
            "forward_progress_pct": forward_pct,
            "backtracking_pct": 100.0 - forward_pct if nav > 0 else 0.0,
            "episode_path_length": 0.0,
            "optimal_path_length": 0.0,
            "watermark_switch": 0.0,
            "watermark_exit": 0.0,
        }
