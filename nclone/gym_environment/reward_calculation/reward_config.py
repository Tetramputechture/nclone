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
        """Curriculum-scaled PBRS objective potential weight.

        Applied to the potential function Φ(s) in F(s,s') = γ * Φ(s') - Φ(s).
        Higher weights provide stronger gradients for learning efficient paths.

        With DIRECT path normalization (Φ = 1 - d/total_path):
        - Typical level: combined_path = 2000px
        - Typical movement: 1-8px per action (slow at start, faster when skilled)
        - Without weight: ΔΦ = 0.0005-0.004 per action (TOO WEAK!)
        - With weight 40: ΔΦ = 0.02-0.16 per action (GOOD RANGE)

        Target PBRS magnitude: 0.05-0.20 per action to dominate time penalty (-0.00032)
        and provide clear learning signal relative to death penalties (-4 to -7).

        Returns:
            60.0 (early): Very strong guidance for slow initial movement
            40.0 (mid): Strong guidance as agent learns to move faster
            20.0 (late): Moderate guidance for fine-tuning efficient paths
        """
        if self.training_phase == "early":
            return 60.0  # Strong signal even with 1-2px movement
        elif self.training_phase == "mid":
            return 40.0  # Balanced signal for 4-6px movement
        return 20.0  # Refined signal for 6-12px movement

    @property
    def time_penalty_per_step(self) -> float:
        """Per-step time penalty (enabled from start for baseline training).

        UPDATED: Reduced by 75% to prevent time penalty from overpowering PBRS signal.
        Previous configuration had time penalty equal to or stronger than PBRS rewards,
        causing agent to minimize steps rather than follow PBRS gradients.

        With frame skip (4 frames), this becomes -0.00032 per action, allowing PBRS
        signals (0.02-0.05) to dominate with a 62-156× ratio instead of <1× ratio.

        Time penalty provides efficiency incentive without overwhelming PBRS gradients.

        Returns:
            -0.00008 (all phases): Very light time pressure, PBRS-dominated learning
        """
        # REBALANCED: Reduced by 75% (from -0.0003 to -0.00008)
        # With 4-frame skip: -0.00008 × 4 = -0.00032 per action (was -0.0012)
        # Target: PBRS reward magnitude >> time penalty for gradient-based learning
        return -0.00008  # Was -0.0003 (75% reduction)

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
