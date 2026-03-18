"""Centralized reward configuration for HWM A* reward pipeline.

Simplified from the original curriculum-adaptive PBRS configuration.
Fixed sparse reward magnitudes (no curriculum scaling). The only
adaptive components are entropy coefficient and RND beta, which
follow a 3-phase schedule based on training progress.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional

from .reward_constants import (
    R_SWITCH,
    R_EXIT,
    R_DEATH,
    R_DEATH_MINE,
    HWM_ALPHA,
    TIME_PENALTY_PER_STEP,
    TRUNCATION_PENALTY,
)


@dataclass
class RewardConfig:
    """Single source of truth for reward component lifecycle.

    HWM pipeline uses fixed reward magnitudes:
    - R_SWITCH = 1.0 (switch activation)
    - R_EXIT = 5.0 (level completion)
    - R_DEATH = -0.5 (any death)
    - HWM_ALPHA = 1.0 (shaping weight, constant)

    Adaptive components (3-phase schedule):
    - entropy_coefficient: 0.05 -> 0.02 -> 0.01
    - rnd_beta: managed externally by ScheduleCallback
    """

    total_timesteps: int = 10_000_000
    current_timesteps: int = 0
    recent_success_rate: float = 0.0

    _curriculum_stage: int = 0
    _num_curriculum_stages: int = 1

    SUCCESS_THRESHOLD_MID: float = 0.15
    SUCCESS_THRESHOLD_LATE: float = 0.40

    # Optuna overrides (kept for hyperparameter search compatibility)
    _death_penalty_override: Optional[float] = None
    _pbrs_objective_weight_override: Optional[float] = None
    _time_penalty_per_step_override: Optional[float] = None
    _level_completion_reward_override: Optional[float] = None
    _velocity_alignment_weight_override: Optional[float] = None
    _mine_hazard_cost_multiplier_override: Optional[float] = None
    _revisit_penalty_weight_override: Optional[float] = None
    _entropy_coefficient_override: Optional[float] = None
    _hwm_alpha_override: Optional[float] = None

    @property
    def training_phase(self) -> str:
        """Current training phase based on success rate.

        Returns:
            'early': <15% success
            'mid': 15-40% success
            'late': >40% success
        """
        if self.recent_success_rate < self.SUCCESS_THRESHOLD_MID:
            return "early"
        if self.recent_success_rate < self.SUCCESS_THRESHOLD_LATE:
            return "mid"
        return "late"

    @property
    def death_penalty(self) -> float:
        """Fixed death penalty per spec."""
        if self._death_penalty_override is not None:
            return self._death_penalty_override
        return R_DEATH

    @property
    def mine_death_penalty(self) -> float:
        """Extra penalty applied on top of death_penalty when death_cause == 'mine'."""
        return R_DEATH_MINE

    @property
    def truncation_penalty(self) -> float:
        """Terminal penalty when episode ends by timeout (truncation)."""
        return TRUNCATION_PENALTY

    @property
    def level_completion_reward(self) -> float:
        """Fixed level completion reward per spec."""
        if self._level_completion_reward_override is not None:
            return self._level_completion_reward_override
        return R_EXIT

    @property
    def switch_activation_reward(self) -> float:
        """Fixed switch activation reward per spec."""
        return R_SWITCH

    @property
    def hwm_alpha(self) -> float:
        """HWM shaping weight. Zero when HWM is disabled (e.g. value-function-only mode)."""
        if self._hwm_alpha_override is not None:
            return self._hwm_alpha_override
        return HWM_ALPHA

    @property
    def entropy_coefficient(self) -> float:
        """3-phase entropy schedule per spec Section 7.

        Schedule:
        - Phase 1 (early, <15% success): 0.05 (explore)
        - Phase 2 (mid, 15-40% success): 0.02 (refine)
        - Phase 3 (late, >40% success): 0.01 (exploit)

        Returns:
            Entropy coefficient for PPO.
        """
        if self._entropy_coefficient_override is not None:
            return self._entropy_coefficient_override

        phase = self.training_phase
        if phase == "early":
            return 0.05
        elif phase == "mid":
            return 0.02
        return 0.01

    # ---- Deprecated properties (kept for backward compatibility) ----

    @property
    def pbrs_objective_weight(self) -> float:
        """DEPRECATED: HWM uses alpha instead of PBRS weight."""
        if self._pbrs_objective_weight_override is not None:
            return self._pbrs_objective_weight_override
        return HWM_ALPHA

    @property
    def time_penalty_per_step(self) -> float:
        """Per-step time penalty to discourage backtracking/aimless wandering.

        Uses TIME_PENALTY_PER_STEP from constants (-0.001). At 500 steps this
        costs -0.5 (same as R_DEATH), making inaction as costly as dying.
        Set via _time_penalty_per_step_override for Optuna/ablation studies.
        """
        if self._time_penalty_per_step_override is not None:
            return self._time_penalty_per_step_override
        return TIME_PENALTY_PER_STEP

    @property
    def exploration_bonus(self) -> float:
        """DEPRECATED: RND handles exploration."""
        return 0.0

    @property
    def revisit_penalty_weight(self) -> float:
        """DEPRECATED: HWM handles backtracking via watermark."""
        if self._revisit_penalty_weight_override is not None:
            return self._revisit_penalty_weight_override
        return 0.0

    @property
    def pbrs_normalization_scale(self) -> float:
        """DEPRECATED."""
        return 1.0

    def get_curriculum_pbrs_scale(self, curriculum_stage: int, num_stages: int) -> float:
        """DEPRECATED."""
        return 1.0

    @property
    def waypoint_action_diversity_bonus(self) -> float:
        """DEPRECATED."""
        return 0.0

    @property
    def completion_approach_bonus_weight(self) -> float:
        """DEPRECATED."""
        return 0.0

    @property
    def velocity_alignment_weight(self) -> float:
        """DEPRECATED."""
        if self._velocity_alignment_weight_override is not None:
            return self._velocity_alignment_weight_override
        return 0.0

    @property
    def mine_hazard_cost_multiplier(self) -> float:
        """Mine hazard cost for A* pathfinding (not a reward parameter)."""
        if self._mine_hazard_cost_multiplier_override is not None:
            return self._mine_hazard_cost_multiplier_override
        return 2.0

    @property
    def survival_bonus_per_100_frames(self) -> float:
        """DEPRECATED."""
        return 0.0

    @property
    def action_masking_mode(self) -> str:
        """Curriculum-aware action masking mode (preserved from original)."""
        curriculum_stage = getattr(self, "_curriculum_stage", 0)
        num_stages = getattr(self, "_num_curriculum_stages", 1)
        curriculum_progress = curriculum_stage / max(num_stages - 1, 1)

        if curriculum_progress < 0.3 or self.recent_success_rate < 0.15:
            return "hard"
        return "none"

    @property
    def logit_bias_strength(self) -> float:
        """Logit bias strength for soft masking mode."""
        if self.action_masking_mode != "soft":
            return 0.0
        t = (self.recent_success_rate - 0.15) / 0.25
        t = min(max(t, 0.0), 1.0)
        return 2.0 - 1.5 * t

    def set_curriculum_stage(self, stage: int, num_stages: int) -> None:
        """Update curriculum stage for action masking mode calculation."""
        self._curriculum_stage = stage
        self._num_curriculum_stages = num_stages

    def update(self, timesteps: int, success_rate: float) -> None:
        """Update configuration with current training metrics."""
        self.current_timesteps = timesteps
        self.recent_success_rate = success_rate

    def get_active_components(self) -> Dict[str, Any]:
        """Get current state of all reward components for logging."""
        return {
            "phase": self.training_phase,
            "timesteps": self.current_timesteps,
            "success_rate": self.recent_success_rate,
            "death_penalty": self.death_penalty,
            "level_completion_reward": self.level_completion_reward,
            "switch_activation_reward": self.switch_activation_reward,
            "hwm_alpha": self.hwm_alpha,
            "entropy_coefficient": self.entropy_coefficient,
            "action_masking_mode": self.action_masking_mode,
            "logit_bias_strength": self.logit_bias_strength,
        }

    def __str__(self) -> str:
        return (
            f"RewardConfig(phase={self.training_phase}, "
            f"timesteps={self.current_timesteps:,}, "
            f"success={self.recent_success_rate:.1%}, "
            f"hwm_alpha={self.hwm_alpha:.1f}, "
            f"death={self.death_penalty:.2f}, "
            f"entropy={self.entropy_coefficient:.3f})"
        )


@dataclass
class GoalCurriculumConfig:
    """Configuration for sliding window goal curriculum learning."""

    enabled: bool = True
    stage_distance_interval: float = 100.0
    advancement_threshold: float = 0.80
    rolling_window: int = 5000
    time_based_escalation_steps: int = 5_000_000

    # DEPRECATED: kept for backwards compatibility
    progress_stages: tuple = (0.2, 0.4, 0.6, 0.8, 1.0)
    virtual_goal_radius: float = 15.0
