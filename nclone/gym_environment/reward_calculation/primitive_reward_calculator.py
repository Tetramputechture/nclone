"""Simplified primitive-level reward calculator using Phi grids + Reward Machine.

Implements the optimal PBRS setup from pbrs-plan.md:
- Precomputed Phi grids for O(1) potential lookup
- Explicit 3-state reward machine for task phase tracking
- Per-frame PBRS: F_t = gamma * Phi(s') - Phi(s)
- Total reward: r_total = r_sparse(RM) + alpha * F_t

Key invariants:
- Phi(terminal) = 0 always (death, timeout, success)
- Same gamma everywhere (PBRS, GAE, value function)
- Primitive-level F_t computed every physics frame
- Phase-correct targeting (before switch -> switch; after -> exit)

This calculator is designed for primitive-level application within
FiGAR-PPO, where rewards are computed every physics frame.
"""

import logging
from typing import Dict, Any, Optional, Tuple, List

from .reward_machine import (
    RewardMachine,
    DEFAULT_R_SWITCH,
    DEFAULT_R_EXIT,
    DEFAULT_C_PHASE,
)
from .reward_constants import PBRS_GAMMA, GLOBAL_REWARD_SCALE

logger = logging.getLogger(__name__)

# Default shaping scale (alpha in r_total = r_sparse + alpha * F_t)
# Start at 1.0; tune via sweep [0.1, 0.3, 0.5, 1.0]
DEFAULT_SHAPING_ALPHA = 1.0


class PrimitiveRewardCalculator:
    """Simplified reward calculator for primitive-level PBRS with reward machine.

    Computes per-frame rewards suitable for primitive-level GAE:
        r_t = r_sparse(RM transition) + alpha * F_t

    where F_t = gamma * Phi(s_{t+1}, u_{t+1}) - Phi(s_t, u_t)
    and Phi(s, u) = Phi_state(s, u) + Phi_phase(u)

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
        c_phase: float = DEFAULT_C_PHASE,
        global_reward_scale: float = GLOBAL_REWARD_SCALE,
    ):
        """Initialize primitive reward calculator.

        Args:
            gamma: PBRS discount factor (MUST match PPO gamma)
            alpha: Shaping scale multiplier for F_t
            r_switch: Sparse reward for switch collection
            r_exit: Sparse reward for exit reached
            c_phase: Phase potential bonus for switch-collected state
            global_reward_scale: Scale applied to all rewards for value stability
        """
        self.gamma = gamma
        self.alpha = alpha
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

        # Determine terminal state
        is_terminal = player_dead or player_won or is_truncated

        # Get position-based potential from Phi grid (prefer Rust when available)
        rm_state = self.reward_machine.state
        if rust_phi_cache is not None:
            phi_state = rust_phi_cache.get_phi(
                player_x, player_y, rm_state
            )
        else:
            phi_state = phi_grid_cache.get_phi(player_x, player_y, rm_state)

        # Compute total potential: Phi(s, u) = Phi_state + Phi_phase
        current_potential = self.reward_machine.get_total_potential(
            phi_state, is_terminal=is_terminal
        )

        # Compute PBRS shaping reward: F_t = gamma * Phi(s') - Phi(s)
        pbrs_reward = 0.0
        if self.prev_potential is not None:
            pbrs_reward = self.gamma * current_potential - self.prev_potential

            # Track forward/backtrack for diagnostics
            potential_change = current_potential - (self.prev_potential / self.gamma
                                                     if self.gamma != 0 else 0)
            actual_change = current_potential - self.prev_potential
            if actual_change > 0.001:
                self.episode_forward_steps += 1
            elif actual_change < -0.001:
                self.episode_backtrack_steps += 1

        # Update previous potential for next frame
        if is_terminal:
            # Reset potential tracking on terminal (next episode starts fresh)
            self.prev_potential = None
        else:
            self.prev_potential = current_potential

        # Compose total reward
        total_reward = sparse_reward + self.alpha * pbrs_reward

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
            "total_reward": total_reward,
            "scaled_reward": scaled_reward,
            "current_potential": current_potential,
            "prev_potential": self.prev_potential if self.prev_potential else 0.0,
            "phi_state": phi_state,
            "phi_phase": self.reward_machine.get_phi_phase(),
            "rm_state": rm_state,
            "is_terminal": is_terminal,
            "alpha": self.alpha,
            "gamma": self.gamma,
        }

        return scaled_reward

    def calculate_death_reward(self) -> float:
        """Calculate reward for death (terminal state).

        Death gets no sparse reward, but PBRS ensures Phi(terminal)=0,
        which naturally penalizes dying (potential drops to 0 from wherever
        the agent was).

        Returns:
            Scaled death reward (typically negative due to PBRS Phi drop to 0)
        """
        # The PBRS component already handles this through Phi(terminal)=0
        # No additional death penalty needed -- the potential drop IS the penalty
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
        self.episode_terminal_reward = 0.0
        self.last_components = {}

    def get_episode_stats(self) -> Dict[str, float]:
        """Get episode-level statistics for TensorBoard logging.

        Returns:
            Dictionary of episode statistics
        """
        total_pbrs = sum(self.episode_pbrs_rewards) if self.episode_pbrs_rewards else 0.0
        total_sparse = sum(self.episode_sparse_rewards) if self.episode_sparse_rewards else 0.0
        total_reward = total_pbrs * self.alpha + total_sparse

        return {
            "episode/total_pbrs": total_pbrs,
            "episode/total_sparse": total_sparse,
            "episode/total_reward": total_reward,
            "episode/forward_steps": self.episode_forward_steps,
            "episode/backtrack_steps": self.episode_backtrack_steps,
            "episode/terminal_reward": self.episode_terminal_reward,
            "episode/steps": self.steps_taken,
            "episode/pbrs_mean": total_pbrs / max(self.steps_taken, 1),
        }
