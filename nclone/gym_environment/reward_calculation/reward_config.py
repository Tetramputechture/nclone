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

        Reduced from original values to make room for physics discovery rewards.

        Returns:
            1.5 (early): Reduced guidance (~3.75 accumulated per episode)
            0.75 (mid): Moderate guidance (~1.875 accumulated)
            0.375 (late): Light shaping (~0.9375 accumulated)
        """
        if self.training_phase == "early":
            return 1.5  # Reduced from 2.0 (25% reduction)
        elif self.training_phase == "mid":
            return 0.75  # Reduced from 1.0 (25% reduction)
        return 0.375  # Reduced from 0.5 (25% reduction)

    @property
    def enable_physics_discovery(self) -> bool:
        """Enable physics discovery rewards for exploration.

        Physics discovery provides alternative exploration signals to reduce
        over-reliance on PBRS guidance.

        Returns:
            True: Physics discovery rewards are active
        """
        return True

    @property
    def physics_discovery_weight(self) -> float:
        """Total weight for physics discovery rewards.

        Reduced to prevent reward exploitation while still encouraging exploration.

        Returns:
            0.1: Conservative weight for physics discovery components
        """
        return 0.02

    @property
    def time_penalty_per_step(self) -> float:
        """Per-step time penalty (disabled until completion learned).

        DISABLED during early training to prevent punishment during learning.
        Only enabled after agent achieves >50% completion rate.

        Returns:
            0.0 (early or low success): No efficiency pressure
            -0.005 (mid, high success): Moderate efficiency incentive
            -0.01 (late, high success): Full speed optimization
        """
        # Always disabled during early training
        if self.training_phase == "early":
            return 0.0  # NO punishment during learning

        # Only enable if agent has learned to complete levels
        if self.recent_success_rate < self.SUCCESS_THRESHOLD:
            return 0.0  # Still learning completion, no speed pressure

        # Gradually increase efficiency pressure
        if self.training_phase == "mid":
            return -0.005  # Moderate efficiency pressure (5x stronger)
        return -0.01  # Full speed optimization (3.3x stronger)

    @property
    def pbrs_normalization_scale(self) -> float:
        """Scale factor for PBRS distance normalization.

        Reduces normalization early for stronger gradients on large levels,
        gradually increases to full normalization as policy improves.

        Returns:
            0.5 (early): Half normalization = 2x gradient strength
            0.75 (mid): 3/4 normalization = 1.33x gradient strength
            1.0 (late): Full normalization
        """
        if self.training_phase == "early":
            return 0.5  # 2x gradient strength
        elif self.training_phase == "mid":
            return 0.75  # 1.33x gradient strength
        return 1.0  # Full normalization

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
            "enable_physics_discovery": self.enable_physics_discovery,
            "physics_discovery_weight": self.physics_discovery_weight,
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
