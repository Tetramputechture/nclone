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
        """Curriculum-based PBRS weight - scaled to keep returns manageable.

        IMPORTANT: Weights reduced to prevent accumulated PBRS from dominating
        terminal rewards. Previous weights (30.0 in discovery) created -0.3 penalty
        per backward step, accumulating to -60+ over oscillating paths.

        Max episode return = PBRS_weight + completion(50) + switch(5) ≈ weight + 55

        With these reduced weights:
        - Discovery (<5%): 15 + 55 = 70 max return (stable for value function)
        - Early learning: 12 + 55 = 67 max return
        - Late training: 4 + 55 = 59 max return

        Death penalties (-10 to -18) remain significant relative to PBRS rewards.

        Returns:
            15.0 (0-5% success): Strong guidance, less punishing for oscillation
            12.0 (5-20% success): Proportional reduction for early learning
            10.0 (20-40% success): Moderate guidance for mid learning
            8.0 (early phase): Reduced for stability
            6.0 (mid phase): Reduced for stability
            4.0 (late phase): Light shaping maintained
        """
        if self.recent_success_rate < 0.05:  # Discovery phase (0-5% success)
            return 15.0  # Strong guidance, less punishing for oscillation
        elif self.recent_success_rate < 0.20:  # Early learning (5-20% success)
            return 12.0  # Proportional reduction
        elif self.recent_success_rate < 0.40:  # Mid learning (20-40% success)
            return 10.0  # Moderate guidance
        elif self.training_phase == "early":
            return 8.0  # Reduced for stability
        elif self.training_phase == "mid":
            return 6.0  # Reduced for stability
        return 4.0  # Light shaping maintained

    @property
    def time_penalty_per_step(self) -> float:
        """Curriculum-based time penalty - scales with success rate.

        UPDATED: Increased by 20-25× to aggressively combat oscillation/stuck behavior.
        Previous values were too weak to deter jumping in place near spawn.

        Strong time pressure makes staying still very costly, forcing exploration.

        Analysis with 4-frame skip (600 frame typical episode):
        - Discovery (<5%):   -0.005 × 600 = -3.0 total (15% of completion reward)
        - Early (5-30%):     -0.01 × 600 = -6.0 total (30% of completion reward)
        - Mid (30-50%):      -0.02 × 600 = -12.0 total (60% of completion reward)
        - Refinement (>50%): -0.02 × 600 = -12.0 total (60% of completion reward)

        With 4-frame skip: -0.02 to -0.08 per action (strong pressure to move efficiently).

        Returns:
            -0.005 (<5% success): Strong pressure even during discovery
            -0.01 (5-30% success): Very strong pressure for early learning
            -0.02 (30%+ success): Maximum pressure for efficiency
        """
        if self.recent_success_rate < 0.05:  # Discovery phase (0-5% success)
            return -0.005  # 25× stronger: -0.02/action (4-frame skip)
        elif self.recent_success_rate < 0.30:  # Early learning (5-30% success)
            return -0.01  # 20× stronger: -0.04/action
        elif self.recent_success_rate < 0.50:  # Mid learning (30-50%)
            return -0.02  # 20× stronger: -0.08/action
        elif self.training_phase == "early":
            return -0.01  # Strong pressure
        elif self.training_phase == "mid":
            return -0.015  # Stronger pressure
        return -0.02  # Maximum efficiency pressure

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

        INCREASED (5x): Previous weight of 0.02 was clearly insufficient.
        TensorBoard analysis showed 99% position revisit rate with only -0.09 total penalty,
        which was not deterring oscillation behavior.

        Uses LOGARITHMIC scaling: -weight × log(1 + visit_count)
        Combined with per-step cap of -0.5 for bounded worst-case.

        Examples with weight=0.10 (logarithmic scaling):
        - 2 visits: -0.10 × log(3) = -0.11 penalty
        - 5 visits: -0.10 × log(6) = -0.18 penalty
        - 10 visits: -0.10 × log(11) = -0.24 penalty
        - 50 visits: -0.10 × log(51) = -0.39 penalty (capped at -0.5)

        Max accumulated revisit penalty per episode: ~-75 (enough to deter oscillation)

        Returns:
            0.10 (0-20% success): Strong deterrent with log scaling
            0.075 (20-40% success): Moderate as agent improves
            0.05 (40%+ success): Light refinement penalty
        """
        if self.recent_success_rate < 0.20:
            return 0.10  # Strong deterrent with logarithmic scaling (5x increase)
        elif self.recent_success_rate < 0.40:
            return 0.075  # Moderate as agent improves
        return 0.05  # Light refinement penalty

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
