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
    LATTICE_PBRS_WEIGHT,
    LATTICE_PBRS_GAMMA,
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

    # --- Lattice / free-space PBRS (reward_constants LATTICE_PBRS_* block) ---
    # When True, a true policy-invariant potential-based shaping term
    # F = pbrs_weight * (pbrs_gamma * Phi(s') - Phi(s)), with
    # Phi(s) = -normalized cost-to-completion from the env's reachability source,
    # REPLACES the HWM ratchet + backtrack penalty as the sole dense extrinsic
    # signal. Requires the env's ReachabilityConfig.use_lattice_potential so the
    # substituted (lattice/free-space) cost-to-completion reaches the obs
    # distances; the trainer wires that automatically when this is set.
    enable_lattice_pbrs: bool = False
    _pbrs_weight_override: Optional[float] = None
    _pbrs_gamma_override: Optional[float] = None
    # Fixed reference for the PBRS potential normalization (D0 = cost-to-completion
    # denominator), in the same units as the active cost-to-go source. Default None
    # = per-EPISODE D0 (the spawn cost-to-completion), which makes the per-step
    # shaping magnitude scale as 1/route-length: on a long route (e.g. the tight
    # mine-morph rungs, D0~4438) a forward step earns ~6x less than on a short one
    # (D0~763), so the dense gradient fades exactly where it is needed most. Set a
    # FIXED reference (e.g. the hardest rung's D0) so per-step magnitude is
    # ROUTE-LENGTH-INVARIANT across a geometry-morphing curriculum — each pixel of
    # progress pays the same reward on every rung (the total return then scales with
    # route length, which the value head absorbs). Still policy-invariant within an
    # episode (a constant rescale of a state-only potential). See the
    # geometry-morphing curriculum docs.
    _pbrs_d0_ref_override: Optional[float] = None
    # Fraction of the terminal PBRS potential paid on DEATH (a true terminal).
    #   1.0 = strict policy-invariant: death pays -w*Phi(s_last) so shaping
    #         telescopes to a constant +w for every terminal (no approach gradient
    #         survives death -> on a hard exploration level the agent collapses to
    #         a fast-death local optimum, empirically observed).
    #   0.0 = exploration-friendly (DEFAULT): death pays NO terminal potential, so
    #         the dense approach reward w*(Phi(s_last)+1) the agent earned on the
    #         way IS BANKED -> dying nearer the goal is strictly better than dying
    #         at spawn, giving a per-episode gradient that rewards reaching deeper.
    #         Not policy-invariant, but a desired exploration bias when the agent
    #         cannot otherwise leave spawn. (Win always pays the FULL terminal
    #         potential, so completion's +w bonus is unaffected.)
    pbrs_death_clawback: float = 0.0

    # --- Goal-distance reverse curriculum ---
    # Move the GOAL closer to spawn early, then outward. The "near goal" is a
    # cost-to-go THRESHOLD on the (reachable) free-space path: the episode SUCCEEDS
    # (pays the exit reward + terminates) as soon as the agent's cost-to-completion
    # drops to <= C_spawn*(1-goal_distance_fraction), where C_spawn is the spawn
    # cost-to-completion. goal_distance_fraction=1.0 -> true goal only (off);
    # f=0.3 -> succeed after covering 30% of the path. Annealed low->1.0 as the
    # agent learns. Bootstraps long-horizon (FAR_OBJECTIVE) + makes the sparse
    # terminal reachable + shortens early episodes (helps value stability).
    enable_goal_distance_curriculum: bool = False
    goal_distance_fraction: float = 1.0

    # --- Per-leg waypoint PBRS (docs/LONG_HORIZON_DISTANCE_HANDOFF.md) ---
    # When True, a per-leg potential term F = pbrs_weight*(pbrs_gamma*Phi(s') - Phi(s))
    # with Phi(s) = -clamp(waypoint_leg_distance / waypoint_leg_scale, 0, 1) shapes
    # progress toward the CURRENT local waypoint (from a WaypointTracker, fed into obs
    # as ``waypoint_leg_distance`` + ``waypoint_index``). The potential basis is REBASED
    # every time ``waypoint_index`` advances, so each leg is short and the per-step
    # gradient is ROUTE-LENGTH-INVARIANT -- the long-horizon distance-dropoff fix
    # (decompose one far goal into a chain of near subgoals). Mutually exclusive with the
    # HWM ratchet and the lattice/free-space PBRS: ``hwm_alpha`` is forced to 0 and the
    # backtrack penalty is skipped (the per-leg term already penalizes regress
    # symmetrically). Reuses ``pbrs_weight`` / ``pbrs_gamma``.
    enable_waypoint_pbrs: bool = False
    waypoint_leg_scale: float = 96.0  # px; denominator for the per-leg potential
    # Explicit per-waypoint REACH BONUS: paid each time the WaypointTracker cursor
    # advances forward (monotonic; the tracker never retreats so it cannot be
    # farmed). Unlike the policy-invariant per-leg PBRS, this is a concrete dense
    # progress reward on the long return leg -- exactly where the agent times out.
    # 0.0 = off (pure PBRS). A densified route is ~48px/leg, so e.g. 0.05 pays
    # ~0.05 every 48px of forward route progress, route-length-invariant.
    waypoint_reach_bonus: float = 0.0

    # --- Frontier watermark bonus (long-horizon "loop"-level commitment) ---
    # On long loop levels the agent times out wandering a safe basin instead of
    # committing to the long climb to the goal: the dense free-space cost-to-go
    # PBRS is correct but its per-step gradient over 600-1100px is too weak to
    # overcome the discount. When True, a small one-sided bonus pays the agent
    # ONLY when it reaches a strictly new closest-approach (a new minimum
    # free-space cost-to-go) to the current live objective this episode -- the
    # switch pre-activation, the exit post-activation. The bonus is
    # ``frontier_bonus_weight * (best_prev - now)`` (positive, proportional to the
    # improvement); a non-improving step pays nothing, so basin-wandering earns 0
    # while monotonic progress around the loop is rewarded. It REUSES the same
    # phase-aware cost-to-go (``current_goal_dist``) the PBRS potential is built
    # from -- no new distance source -- so the two signals stay consistent. The
    # episode watermark is reset on episode reset and REBASED (no spurious bonus)
    # when the live objective changes switch->exit, exactly like the waypoint
    # rebase. Off by default; additive (does not change behavior when off).
    # ``frontier_bonus_weight`` is in the same reward units as the per-step PBRS
    # term and is intentionally MUCH smaller than the sparse switch/exit rewards.
    enable_frontier_bonus: bool = False
    frontier_bonus_weight: float = 0.02

    # --- Dense mine-hazard avoidance PBRS (T3/T4 mine-death fix) ---
    # On long mine levels the agent dies threading the mines (~65% of T3/T4 FAR
    # episodes) because the ONLY mine signal in the reward is the TERMINAL death
    # penalty (-3.5) — there is no per-step gradient to steer AROUND a mine, and
    # with pbrs_death_clawback=0 a deep death even BANKS the approach reward, so
    # "die deeper" out-scores "die at spawn". This adds a policy-invariant PBRS
    # term on the normalized nearest-deadly-mine distance d in [0,1] (1=safe,
    # 0=touching), read from game_state: Phi_haz = d, F = w*(Phi' - Phi) with
    # gamma=1 (pure telescoping -> exactly 0 when no mine is near, so no survival
    # drift). Moving away from a mine pays +, moving toward pays - — a dense
    # avoidance gradient the agent can act on BEFORE dying. Deliberately a SHAPING
    # gradient, NOT a bigger terminal penalty (run-220/221 showed a larger death
    # penalty causes passive risk-aversion). Requires the policy to SEE mines
    # locally (HazardEncoder spatial readout) to be exploitable. Off by default.
    enable_hazard_pbrs: bool = False
    hazard_pbrs_weight: float = 0.5

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
        """HWM shaping weight. Zero when HWM is disabled (e.g. value-function-only mode).

        Forced to 0 when ``enable_lattice_pbrs`` or ``enable_waypoint_pbrs`` is
        set: those dense PBRS terms REPLACE the HWM ratchet, never stack with it.
        """
        if self.enable_lattice_pbrs or self.enable_waypoint_pbrs:
            return 0.0
        if self._hwm_alpha_override is not None:
            return self._hwm_alpha_override
        return HWM_ALPHA

    @property
    def pbrs_weight(self) -> float:
        """Weight on the lattice/free-space PBRS term (only used when
        ``enable_lattice_pbrs``)."""
        if self._pbrs_weight_override is not None:
            return self._pbrs_weight_override
        return LATTICE_PBRS_WEIGHT

    @property
    def pbrs_gamma(self) -> float:
        """Discount inside the PBRS shaping term (only used when
        ``enable_lattice_pbrs``). Match the PPO gamma for strict invariance."""
        if self._pbrs_gamma_override is not None:
            return self._pbrs_gamma_override
        return LATTICE_PBRS_GAMMA

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
            # PBRS shaping knobs (consumed by npp-rl PBRSLoggingCallback for the
            # curriculum/* TB panel). These are @property values, surfaced here.
            "pbrs_objective_weight": self.pbrs_objective_weight,
            "pbrs_normalization_scale": self.pbrs_normalization_scale,
            "time_penalty_per_step": self.time_penalty_per_step,
            "hwm_alpha": self.hwm_alpha,
            "entropy_coefficient": self.entropy_coefficient,
            "action_masking_mode": self.action_masking_mode,
            "logit_bias_strength": self.logit_bias_strength,
            "enable_bsrs_blend": self.enable_bsrs_blend,
            "eta_bsrs": self.eta_bsrs,
            "bsrs_gamma": self.bsrs_gamma,
            "beta_blend": self.beta_blend(),
            "enable_lattice_pbrs": self.enable_lattice_pbrs,
            "enable_waypoint_pbrs": self.enable_waypoint_pbrs,
            "waypoint_leg_scale": self.waypoint_leg_scale,
            "pbrs_weight": self.pbrs_weight,
            "pbrs_gamma": self.pbrs_gamma,
            # Fixed PBRS D0 reference (None = per-episode spawn D0). Surfaced so the
            # geometry-morphing curriculum's route-length-invariant calibration is
            # visible in TB.
            "pbrs_d0_ref": self._pbrs_d0_ref_override,
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
