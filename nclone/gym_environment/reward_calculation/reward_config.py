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

    # Waypoint system configuration (always enabled in unified system)
    path_waypoint_progress_spacing: float = (
        100.0  # Spacing for progress checkpoints (pixels)
    )
    path_waypoint_min_angle: float = 45.0  # Minimum angle for turn detection (degrees)
    path_waypoint_cluster_radius: float = 40.0  # Clustering radius (pixels)

    # Curriculum phase thresholds
    # UPDATED 2025-12-07: Success-rate based progression with minimum timestep gates
    # This accounts for complex levels that take longer to learn
    MIN_TIMESTEPS_FOR_MID: int = 500_000  # Minimum steps before mid phase
    MIN_TIMESTEPS_FOR_LATE: int = 1_500_000  # Minimum steps before late phase

    # Success rate thresholds for phase transitions
    SUCCESS_THRESHOLD_MID: float = 0.15  # 15% success to enter mid phase
    SUCCESS_THRESHOLD_LATE: float = 0.40  # 40% success to enter late phase

    @property
    def training_phase(self) -> str:
        """Current training phase based on success rate (with minimum timestep gates).

        UPDATED 2025-12-07: Primary driver is success_rate, not timesteps.
        This allows complex levels to stay in early/mid phases longer until
        agent demonstrates actual learning.

        Timesteps act as minimum gates to prevent premature transitions
        (e.g., can't reach "late" phase before 1.5M steps even with lucky early success).

        Returns:
            'early': <15% success OR <500K steps (bootstrap navigation)
            'mid': 15-40% success AND >500K steps (path refinement)
            'late': >40% success AND >1.5M steps (speed optimization)
        """
        # Early phase: Low success OR haven't trained minimum steps
        if (
            self.recent_success_rate < self.SUCCESS_THRESHOLD_MID
            or self.current_timesteps < self.MIN_TIMESTEPS_FOR_MID
        ):
            return "early"

        # Mid phase: Moderate success AND passed minimum gate
        if (
            self.recent_success_rate < self.SUCCESS_THRESHOLD_LATE
            or self.current_timesteps < self.MIN_TIMESTEPS_FOR_LATE
        ):
            return "mid"

        # Late phase: High success AND sufficient training
        return "late"

    @property
    def pbrs_objective_weight(self) -> float:
        """Success-rate-based PBRS weight - scaled to keep returns manageable.

        UPDATED 2025-12-07: Fully success-rate driven (no timestep fallbacks).
        Complex levels progress based on actual learning, not arbitrary time thresholds.

        IMPORTANT: Weights reduced to prevent accumulated PBRS from dominating
        terminal rewards. Previous weights (30.0 in discovery) created -0.3 penalty
        per backward step, accumulating to -60+ over oscillating paths.

        Max episode return = PBRS_weight + completion(50) + switch(5) ≈ weight + 55

        With these reduced weights:
        - Discovery (<5%): 15 + 55 = 70 max return (stable for value function)
        - Early learning (5-20%): 12 + 55 = 67 max return
        - Mid learning (20-40%): 10 + 55 = 65 max return
        - Advanced (40-60%): 6 + 55 = 61 max return
        - Mastery (60%+): 4 + 55 = 59 max return

        Death penalties (-10 to -18) remain significant relative to PBRS rewards.

        Returns:
            20.0 (0-5% success): Very strong guidance for discovery (long-horizon)
            15.0 (5-20% success): Strong guidance for early learning
            12.0 (20-40% success): Moderate guidance for mid learning
            8.0 (40-60% success): Reduced for advanced learning
            5.0 (60%+ success): Light shaping for mastery
        """
        if self.recent_success_rate < 0.05:  # Discovery phase (0-5% success)
            return 20.0  # Very strong guidance for long-horizon discovery
        elif self.recent_success_rate < 0.20:  # Early learning (5-20% success)
            return 15.0  # Strong guidance
        elif self.recent_success_rate < 0.40:  # Mid learning (20-40% success)
            return 12.0  # Moderate guidance
        elif self.recent_success_rate < 0.60:  # Advanced learning (40-60%)
            return 8.0  # Reduced guidance
        return 5.0  # Light shaping for mastery (60%+)

    @property
    def time_penalty_per_step(self) -> float:
        """Success-rate-based time penalty - scales with agent competence.

        UPDATED 2025-12-07: Fully success-rate driven (no timestep fallbacks).
        Penalty increases as agent masters the level, creating efficiency pressure.

        UPDATED: Increased by 20-25× to aggressively combat oscillation/stuck behavior.
        Previous values were too weak to deter jumping in place near spawn.

        Strong time pressure makes staying still very costly, forcing exploration.

        Analysis with 4-frame skip (150 actions typical episode):
        - Discovery (<5%):   -0.002 × 150 = -0.3 total (0.6% of completion reward)
        - Early (5-30%):     -0.01 × 150 = -1.5 total (3% of completion reward)
        - Mid (30-50%):      -0.02 × 150 = -3.0 total (6% of completion reward)
        - Advanced (50-70%): -0.025 × 150 = -3.75 total (7.5% of completion reward)
        - Mastery (70%+):    -0.03 × 150 = -4.5 total (9% of completion reward)

        With 4-frame skip: -0.008 to -0.12 per action (strong pressure to move efficiently).

        Returns:
            -0.002 (<5% success): Minimal pressure for discovery
            -0.01 (5-30% success): Moderate pressure for early learning
            -0.02 (30-50% success): Strong pressure for mid learning
            -0.025 (50-70% success): Stronger pressure for advanced learning
            -0.03 (70%+ success): Maximum pressure for mastery
        """
        if self.recent_success_rate < 0.05:  # Discovery phase (0-5% success)
            return -0.002  # Minimal pressure - allow exploration time
        elif self.recent_success_rate < 0.30:  # Early learning (5-30% success)
            return -0.01  # Moderate pressure
        elif self.recent_success_rate < 0.50:  # Mid learning (30-50%)
            return -0.02  # Strong pressure
        elif self.recent_success_rate < 0.70:  # Advanced learning (50-70%)
            return -0.025  # Stronger pressure
        return -0.03  # Maximum efficiency pressure (70%+)

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

    @property
    def velocity_alignment_weight(self) -> float:
        """Curriculum-adaptive velocity alignment weight for path direction guidance.

        Velocity alignment provides continuous direction signal between discrete graph nodes,
        critical for learning inflection points where agent must go "wrong" direction first
        (e.g., LEFT when goal is RIGHT).

        UPDATED: Quadrupled weights to fix zero euclidean_alignment in TensorBoard.

        Stronger early when agent is learning basic navigation, weaker late when
        agent has mastered path direction and needs refinement only.

        Analysis with PBRS weight and velocity alignment:
        - Early (20.0 PBRS + 8.0 vel): Velocity is 40% of total signal (was 10%)
        - Mid (10.0 PBRS + 3.0 vel): Velocity is 30% of signal (was 5%)
        - Late (4.0 PBRS + 1.0 vel): Velocity is 25% of signal (was 2.5%)

        This ensures strong direction learning to overcome inflection point failures.

        Returns:
            8.0 (0-15% success): Very strong direction learning (was 2.0, 4x increase)
            3.0 (15-40% success): Strong guidance (was 0.5, 6x increase)
            1.0 (40%+ success): Moderate guidance (was 0.1, 10x increase)
        """
        if self.recent_success_rate < 0.15:  # Early phase
            return 8.0  # Was 2.0 → 4x stronger for inflection points
        elif self.recent_success_rate < 0.40:  # Mid phase
            return 3.0  # Was 0.5 → 6x stronger refinement
        return 1.0  # Was 0.1 → 10x light guidance

    @property
    def mine_hazard_cost_multiplier(self) -> float:
        """Curriculum-adaptive mine hazard cost multiplier for pathfinding.

        Controls how expensive paths near deadly mines are during A* pathfinding.
        This shapes the PBRS gradient field to guide agents along safer routes.

        UPDATED: Tripled costs for stronger mine avoidance to prevent inflection point deaths.

        Returns:
            25.0 (0-15% success): Strong early avoidance (was 5.0, 5x increase)
            40.0 (15-40% success): Very strong avoidance (was 15.0, ~3x increase)
            60.0 (40%+ success): Extreme avoidance for mastery (was 25.0, ~2.5x increase)
        """
        if self.recent_success_rate < 0.15:  # Early phase - discovery
            return 25.0  # Was 5.0 → 5x stronger for mine field detours
        elif self.recent_success_rate < 0.40:  # Mid phase - learning
            return 40.0  # Was 15.0 → ~3x stronger avoidance
        return 60.0  # Was 25.0 → ~2.5x strict safety

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
            "velocity_alignment_weight": self.velocity_alignment_weight,
            "mine_hazard_cost_multiplier": self.mine_hazard_cost_multiplier,
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
