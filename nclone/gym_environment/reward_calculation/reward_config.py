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
    R_LOCKED_DOOR_SWITCH,
    R_DEATH,
    R_DEATH_MINE,
    R_GOLD,
    R_ALL_GOLD,
    GOLD_FRAME_VALUE,
    HWM_ALPHA,
    TIME_PENALTY_PER_STEP,
    TRUNCATION_PENALTY,
)


@dataclass
class RewardConfig:
    """Single source of truth for reward component lifecycle.

    HWM pipeline uses fixed reward magnitudes:
    - R_SWITCH = 0.5 (switch activation / phase bonus)
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

    # --- NPP-14 / NPP-45: BSRS blend with HWM (A* distance surrogate) ---
    # When enable_bsrs_blend is True, HWM dense shaping is scaled by beta_blend(timesteps)
    # and the trainer adds (1-beta) * eta_bsrs * (gamma * V(s') - V(s)) on each step.
    enable_bsrs_blend: bool = False
    eta_bsrs: float = 0.005  # Theorem-compatible with gamma=0.99: (1-gamma)/(1+gamma) ≈ 0.005025
    bsrs_gamma: float = 0.99  # Must match PPO gamma; used in BSRS term only
    bsrs_blend_horizon_fraction: float = 0.7  # beta goes 1→0 linearly over this fraction of total_timesteps

    # --- Gold collection (docs/GOLD_COLLECTION_HANDOFF.md) ---
    # Off by default for a clean A/B against the completion-only pipeline. When
    # True, the calculator pays a per-gold pickup reward calibrated so each gold
    # is worth GOLD_FRAME_VALUE (120) frames of saved time — the exchange rate is
    # tied to TIME_PENALTY_PER_STEP (see reward_constants). Gold is a soft
    # objective: it never gates completion. No all-gold bonus by default (uniform
    # valuation); set _all_gold_bonus_override for the collect-everything variant.
    enable_gold_rewards: bool = False
    _gold_reward_per_piece_override: Optional[float] = None
    _all_gold_bonus_override: Optional[float] = None

    # --- Locked-door switches (docs/ENTITY_AWARE_LATTICE_PLANNER_HANDOFF.md) ---
    # When True, activating a NECESSARY locked-door switch (one the planner's
    # door-dependency solver marks as required for completion) pays a one-shot
    # milestone bonus, teaching the agent to route through gating switches.
    # Opening a non-necessary door pays nothing (no detour incentive).
    enable_locked_door_rewards: bool = False
    _locked_door_switch_reward_override: Optional[float] = None

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
    def locked_door_switch_reward(self) -> float:
        """Per-necessary-locked-door-switch activation bonus (only used when
        enable_locked_door_rewards)."""
        if self._locked_door_switch_reward_override is not None:
            return self._locked_door_switch_reward_override
        return R_LOCKED_DOOR_SWITCH

    @property
    def gold_reward_per_piece(self) -> float:
        """Per-gold sparse pickup reward (only used when enable_gold_rewards)."""
        if self._gold_reward_per_piece_override is not None:
            return self._gold_reward_per_piece_override
        return R_GOLD

    @property
    def all_gold_bonus(self) -> float:
        """One-shot bonus for collecting all gold (only used when enable_gold_rewards).

        Default 0.0 under the uniform 'each gold = GOLD_FRAME_VALUE frames'
        objective. Set ``_all_gold_bonus_override`` for a collect-everything A/B.
        """
        if self._all_gold_bonus_override is not None:
            return self._all_gold_bonus_override
        return R_ALL_GOLD

    @property
    def gold_frame_value(self) -> float:
        """Frames of saved time each collected gold is worth (the exchange rate).

        Single source of truth shared with the planner and the lattice gold
        potential: the orienteering rule is 'collect a gold iff its detour costs
        fewer than this many frames'. ``R_GOLD`` is calibrated to it.
        """
        return GOLD_FRAME_VALUE

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

    def update(self, timesteps: int, success_rate: float) -> None:
        """Update configuration with current training metrics."""
        self.current_timesteps = timesteps
        self.recent_success_rate = success_rate

    def sync_training_timesteps(self, timesteps: int) -> None:
        """Sync global training timestep counter (for beta_blend) without success-rate churn."""
        self.current_timesteps = int(timesteps)

    def beta_blend(self) -> float:
        """Anneal A* (HWM) weight vs BSRS: 1.0 early → 0.0 after horizon fraction of training.

        Returns 1.0 when BSRS blend is disabled (full HWM scale).
        """
        if not self.enable_bsrs_blend:
            return 1.0
        horizon = max(int(self.total_timesteps * self.bsrs_blend_horizon_fraction), 1)
        t = int(self.current_timesteps)
        if t >= horizon:
            return 0.0
        return 1.0 - (float(t) / float(horizon))

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
            "enable_bsrs_blend": self.enable_bsrs_blend,
            "eta_bsrs": self.eta_bsrs,
            "bsrs_gamma": self.bsrs_gamma,
            "beta_blend": self.beta_blend(),
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
