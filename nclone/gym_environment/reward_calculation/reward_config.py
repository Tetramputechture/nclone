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

        NOTE: Weights increased to compensate for realistic small movements (6-8px typical)
        with frame skip. With acceleration physics, ninja moves ~6-8px per action (4 frames),
        not the theoretical max of ~13px. This requires stronger PBRS weights to maintain
        sufficient learning signal relative to time penalty.

        Returns:
            5.0 (early): Strong guidance for initial learning with small movements
            3.0 (mid): Moderate guidance as policy improves
            1.5 (late): Light shaping for fine-tuning
        """
        if self.training_phase == "early":
            return 5.0  # Increased from 3.0 to account for realistic movement physics
        elif self.training_phase == "mid":
            return 3.0  # Increased from 2.0 for stronger mid-phase gradients
        return 1.5  # Increased from 1.0 for continued guidance

    @property
    def enable_physics_discovery(self) -> bool:
        """Enable physics discovery rewards for exploration.

        DISABLED: Physics discovery adds noise to the path distance objective.
        With proper PBRS providing dense guidance, physics discovery is not needed.

        Returns:
            False: Physics discovery rewards are disabled for clean path focus
        """
        return False

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
        """Per-step time penalty (enabled from start for baseline training).

        UPDATED: Further reduced based on realistic movement analysis (6-8px per action).
        With frame skip (4 frames), this becomes -0.0012 per action, allowing PBRS
        signals (0.025-0.030) to dominate with a 20-25× ratio.

        Time penalty provides efficiency incentive without overwhelming PBRS gradients.

        Returns:
            -0.0003 (all phases): Light time pressure optimized for frame skip
        """
        # BASELINE MODE: Immediate time pressure from start
        # Reduced from -0.0005 to -0.0003 (40% reduction) for stronger PBRS/penalty ratio
        # With 4-frame skip: -0.0003 × 4 = -0.0012 per action
        return -0.0003  # Was -0.0005 (40% reduction)

    @property
    def pbrs_normalization_scale(self) -> float:
        """Scale factor for PBRS distance normalization.

        Reduces normalization early for stronger gradients on large levels,
        gradually increases to full normalization as policy improves.

        Returns:
            0.3 (early): 30% normalization = 3.3x gradient strength (was 0.5)
            0.5 (mid): Half normalization = 2x gradient strength (was 0.75)
            1.0 (late): Full normalization
        """
        if self.training_phase == "early":
            return 0.3  # 3.3x gradient strength (was 0.5)
        elif self.training_phase == "mid":
            return 0.5  # 2x gradient strength (was 0.75)
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
