"""Centralized reward configuration with curriculum-aware component lifecycle.

This module provides a single source of truth for all reward component states,
managing which components are active and their weights based on training progress.
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class RewardConfig:
    """Single source of truth for reward component lifecycle.

    Manages curriculum-aware transitions:
    - Which components are active at each training stage
    - What their weights/scales are
    - When transitions occur based on training progress and performance

    Design Philosophy:
    - Clear hierarchy: Terminal > PBRS Shaping > Time Penalty (conditional)
    - Zero redundancy: Each component serves ONE purpose
    - Curriculum lifecycle: Components have clear enable/disable thresholds
    """

    # Training context (updated by trainer)
    total_timesteps: int = 10_000_000
    current_timesteps: int = 0
    recent_success_rate: float = 0.0

    # Curriculum phase thresholds
    EARLY_PHASE_END: int = 1_000_000  # 0-1M: Bootstrap navigation
    MID_PHASE_END: int = 3_000_000  # 1M-3M: Refine paths
    # Late phase: 3M+: Optimize speed

    # Performance threshold for enabling time pressure
    SUCCESS_THRESHOLD: float = 0.5  # 50% completion rate

    @property
    def training_phase(self) -> str:
        """Current training phase based on timesteps.

        Returns:
            'early': 0-1M steps (bootstrap learning)
            'mid': 1M-3M steps (path refinement)
            'late': 3M+ steps (speed optimization)
        """
        if self.current_timesteps < self.EARLY_PHASE_END:
            return "early"
        elif self.current_timesteps < self.MID_PHASE_END:
            return "mid"
        return "late"

    @property
    def pbrs_objective_weight(self) -> float:
        """Emergency curriculum for 0% success rate - PBRS must dominate death penalties.

        For 800px path with agent at 0% success:
        - Emergency (200×): Total PBRS ~125, death penalty -6, ratio 20:1
        - Agent can die 20 times and still break even on reaching goal
        - Enables learning through trial-and-error mine avoidance

        Gradually reduce as agent learns successful navigation.

        Returns:
            200.0 (0-5% success): EMERGENCY - PBRS >> death penalties for discovery
            100.0 (5-20% success): Strong guidance for early learning
            60.0 (20-40% success): Moderate guidance for mid learning
            40.0 (early phase): Normal early training
            30.0 (mid phase): Normal mid training
            20.0 (late phase): Normal late training
        """
        if self.recent_success_rate < 0.05:  # Discovery phase (0-5% success)
            return 200.0  # EMERGENCY: 5× normal, PBRS >> death penalties
        elif self.recent_success_rate < 0.20:  # Early learning (5-20% success)
            return 100.0  # Strong guidance, 2.5× normal
        elif self.recent_success_rate < 0.40:  # Mid learning (20-40% success)
            return 60.0  # Moderate guidance
        elif self.training_phase == "early":
            return 40.0  # Normal early training
        elif self.training_phase == "mid":
            return 30.0  # Normal mid training
        return 20.0  # Normal late training

    @property
    def time_penalty_per_step(self) -> float:
        """Curriculum-based time penalty - scales with success rate.

        UPDATED: Increased by 200-1000× to create meaningful efficiency pressure.
        Previous values were ~0.003% of episode reward (effectively irrelevant).

        Diagnostic showed agent barely moving (0.02px/action), creating weak PBRS.
        Time penalty now creates noticeable pressure to move faster and more directly.

        Analysis with 4-frame skip (600 frame typical episode):
        - Discovery (<5%):   -0.0002 × 600 = -0.12 total (0.6% of completion reward)
        - Early (5-30%):     -0.0005 × 600 = -0.30 total (1.5% of completion reward)
        - Mid (30-50%):      -0.001 × 600 = -0.60 total (3.0% of completion reward)
        - Refinement (>50%): -0.002 × 600 = -1.20 total (6.0% of completion reward)

        These create gradual time pressure without overwhelming PBRS guidance.

        Returns:
            -0.0002 (<5% success): 200× stronger, minimal but present
            -0.0005 (5-30% success): 500× stronger, light pressure
            -0.001 (30-50% success): 1000× stronger, moderate pressure
            -0.0004 (early phase): 100× stronger
            -0.0008 (mid phase): 100× stronger
            -0.001 (late phase): 100× stronger, efficiency incentive
        """
        if self.recent_success_rate < 0.05:  # Discovery phase (0-5% success)
            return -0.0002  # 400× stronger: -0.0008/action (4-frame skip)
        elif self.recent_success_rate < 0.30:  # Early learning (5-30% success)
            return -0.0005  # 500× stronger: -0.002/action
        elif self.recent_success_rate < 0.50:  # Mid learning (30-50%)
            return -0.001  # 500× stronger: -0.004/action
        elif self.training_phase == "early":
            return -0.0004  # 100× stronger
        elif self.training_phase == "mid":
            return -0.0008  # 100× stronger
        return -0.001  # 100× stronger: strong efficiency pressure

    @property
    def exploration_bonus(self) -> float:
        """Per-cell exploration bonus - DISABLED.

        Exploration is now handled by RND (Random Network Distillation) at the
        training level via RNDCallback. This provides better exploration signals:
        - RND rewards novelty based on neural network prediction error
        - Works across episodes (not just within-episode cell tracking)
        - Scales with long-chain exploration (50-200 step chains)

        Returns:
            0.0: Always disabled - RND handles exploration
        """
        return 0.0  # RND handles exploration at training level

    @property
    def revisit_penalty_weight(self) -> float:
        """Penalty weight for revisiting same position (oscillation deterrent).

        UPDATED: Changed from sqrt scaling to LINEAR scaling with 5× stronger weights
        to aggressively combat oscillation/meandering observed in trajectories.

        Previous sqrt scaling allowed 100+ revisits before strong penalty.
        Linear scaling reaches breakeven with exploration bonus at 2-3 visits.

        Penalty = -weight × visit_count (LINEAR, not sqrt) using 100-step sliding window.

        Examples with weight=0.015 (linear):
        - 2 visits: -0.030 penalty (breakeven with exploration bonus)
        - 5 visits: -0.075 penalty (strong deterrent)
        - 10 visits: -0.150 penalty (very strong deterrent against looping)
        - 20 visits: -0.300 penalty (extreme penalty for oscillation)

        This creates immediate pressure against revisiting, forcing more directed exploration.

        Returns:
            0.015 (0-20% success): Very strong deterrent (5× stronger than before)
            0.010 (20-40% success): Strong as agent improves
            0.005 (40%+ success): Moderate refinement penalty
        """
        if self.recent_success_rate < 0.20:
            return 0.015  # Very strong deterrent: 5× stronger, linear scaling
        elif self.recent_success_rate < 0.40:
            return 0.010  # Strong: 5× stronger
        return 0.005  # Moderate: 5× stronger

    @property
    def pbrs_normalization_scale(self) -> float:
        """DEPRECATED: Scale factor no longer used with direct path normalization.

        Kept for backward compatibility with code that passes this parameter,
        but it's ignored in the new direct normalization approach.
        Gradient strength is now controlled entirely by pbrs_objective_weight.

        Returns:
            1.0: Always 1.0 (no scaling effect)
        """
        return 1.0  # No longer used - gradient control via weights only

    def update(self, timesteps: int, success_rate: float) -> None:
        """Update configuration with current training metrics.

        Called by trainer at evaluation checkpoints to track progress
        and trigger phase transitions.

        Args:
            timesteps: Total timesteps trained so far
            success_rate: Recent evaluation success rate (0.0-1.0)
        """
        self.current_timesteps = timesteps
        self.recent_success_rate = success_rate

    def get_active_components(self) -> Dict[str, Any]:
        """Get current state of all reward components for logging.

        Returns:
            Dictionary with current phase, timesteps, and active component values
        """
        return {
            "phase": self.training_phase,
            "timesteps": self.current_timesteps,
            "success_rate": self.recent_success_rate,
            "pbrs_objective_weight": self.pbrs_objective_weight,
            "time_penalty_per_step": self.time_penalty_per_step,
            "exploration_bonus": self.exploration_bonus,
            "revisit_penalty_weight": self.revisit_penalty_weight,
            "pbrs_normalization_scale": self.pbrs_normalization_scale,
        }

    def __str__(self) -> str:
        """String representation for logging."""
        return (
            f"RewardConfig(phase={self.training_phase}, "
            f"timesteps={self.current_timesteps:,}, "
            f"success={self.recent_success_rate:.1%}, "
            f"pbrs_weight={self.pbrs_objective_weight:.1f}, "
            f"time_penalty={self.time_penalty_per_step:.4f})"
        )
