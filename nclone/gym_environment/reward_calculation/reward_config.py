"""Centralized reward configuration with curriculum-aware component lifecycle.

This module provides a single source of truth for all reward component states,
managing which components are active and their weights based on training progress.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional


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
    recent_success_rate: float = (
        0.0  # Curriculum stage success rate (when curriculum active)
    )

    # Waypoint system configuration (DISABLED 2025-12-25)
    # Waypoints no longer provide reward bonuses - PBRS handles path following
    enable_path_waypoints: bool = (
        False  # Disabled - PBRS provides distance reduction signal
    )
    path_waypoint_progress_spacing: float = 35.0  # Kept for visualization only
    path_waypoint_min_angle: float = 35.0  # Kept for visualization only
    path_waypoint_cluster_radius: float = 20.0  # Kept for visualization only

    # Curriculum phase thresholds
    # UPDATED 2025-12-07: Success-rate based progression with minimum timestep gates
    # This accounts for complex levels that take longer to learn
    MIN_TIMESTEPS_FOR_MID: int = 500_000  # Minimum steps before mid phase
    MIN_TIMESTEPS_FOR_LATE: int = 1_500_000  # Minimum steps before late phase

    # Success rate thresholds for phase transitions
    SUCCESS_THRESHOLD_MID: float = 0.15  # 15% success to enter mid phase
    SUCCESS_THRESHOLD_LATE: float = 0.40  # 40% success to enter late phase

    # Optuna override fields (None = use curriculum-computed values)
    # These allow hyperparameter optimization to test different reward balances
    _death_penalty_override: Optional[float] = None
    _pbrs_objective_weight_override: Optional[float] = None
    _time_penalty_per_step_override: Optional[float] = None
    _level_completion_reward_override: Optional[float] = None
    _velocity_alignment_weight_override: Optional[float] = None
    _mine_hazard_cost_multiplier_override: Optional[float] = None
    _revisit_penalty_weight_override: Optional[float] = None

    @property
    def training_phase(self) -> str:
        """Current training phase based on curriculum-aware success rate.

        UPDATED 2025-12-27: When goal curriculum is active, uses curriculum stage success rate
        (agent reaching curriculum-adjusted goals at current stage).
        When curriculum inactive, uses overall level completion rate.

        Success rate thresholds:
        - Early phase: <15% curriculum success
        - Mid phase: 15-40% curriculum success
        - Late phase: >40% curriculum success

        This ensures reward config phases progress based on agent competence at
        current curriculum difficulty, not full level mastery.

        Returns:
            'early': <15% success (bootstrap navigation)
            'mid': 15-40% success (path refinement)
            'late': >40% success (speed optimization)
        """
        # Early phase: Low success OR haven't trained minimum steps
        if self.recent_success_rate < self.SUCCESS_THRESHOLD_MID:
            return "early"

        # Mid phase: Moderate success AND passed minimum gate
        if self.recent_success_rate < self.SUCCESS_THRESHOLD_LATE:
            return "mid"

        # Late phase: High success AND sufficient training
        return "late"

    @property
    def pbrs_objective_weight(self) -> float:
        """Success-rate-based PBRS weight - scaled to keep returns manageable.

        UPDATED 2025-12-07: Fully success-rate driven (no timestep fallbacks).
        Complex levels progress based on actual learning, not arbitrary time thresholds.

        UPDATED 2025-12-18: Reduced discovery phase weight from 80.0 to 40.0 for balanced
        value learning and signal strength. Initial reduction to 20.0 was too conservative -
        the gradient signal was drowning in entropy noise (entropy increased during training).

        40.0 provides strong directional guidance while maintaining value stability:
        - Strong enough to overcome entropy noise (3x entropy coefficient)
        - Weak enough for stable value learning (2x lower than original 80.0)
        - Allows 12px movement to give meaningful reward (~0.05 scaled)

        Max episode return = PBRS_weight + completion(50) + switch(40) ≈ weight + 90

        With these weights:
        - Discovery (<5%): 40 + 90 = 130 max return (balanced strength and stability)
        - Early learning (5-20%): 15 + 90 = 105 max return
        - Mid learning (20-40%): 12 + 90 = 102 max return
        - Advanced (40-60%): 8 + 90 = 98 max return
        - Mastery (60%+): 5 + 90 = 95 max return

        Death penalty of -8.0 in discovery phase still makes progress worthwhile.

        Returns:
            15.0 (0-5% success): Balanced guidance allowing exploration mistakes
            15.0 (5-20% success): Strong guidance for early learning
            12.0 (20-40% success): Moderate guidance for mid learning
            8.0 (40-60% success): Reduced for advanced learning
            5.0 (60%+ success): Light shaping for mastery
        """
        # Check for Optuna override first
        if self._pbrs_objective_weight_override is not None:
            return self._pbrs_objective_weight_override

        if self.recent_success_rate < 0.05:  # Discovery phase (0-5% success)
            return 50.0
        elif self.recent_success_rate < 0.20:  # Early learning (5-20% success)
            return 15.0  # Strong guidance
        elif self.recent_success_rate < 0.40:  # Mid learning (20-40% success)
            return 12.0  # Moderate guidance
        elif self.recent_success_rate < 0.60:  # Advanced learning (40-60%)
            return 8.0  # Reduced guidance
        return 5.0  # Light shaping for mastery (60%+)

    @property
    def time_penalty_per_step(self) -> float:
        """Time penalty DISABLED - γ<1 PBRS provides unified time pressure.

        UPDATED 2025-12-27: Removed separate time penalty in favor of unified design
        where PBRS_GAMMA < 1.0 creates natural time pressure and anti-oscillation.

        With γ=0.99 PBRS:
        - Forward progress: Positive reward (progress toward goal)
        - Staying still: -0.01×Φ(current) per step (implicit time penalty)
        - Backtracking: Negative reward (loses progress + 1% discount)
        - Oscillation: GUARANTEED LOSS (can never break even)

        Benefits of unified signal:
        1. Maintains PBRS policy invariance (γ_PBRS = γ_PPO = 0.99)
        2. Single coherent objective (no conflicting signals)
        3. Natural anti-oscillation (mathematically impossible to exploit)
        4. Self-tuning urgency (scales with potential magnitude)

        Returns:
            0.0 (always - time pressure now handled by γ<1 PBRS)
        """
        # Allow Optuna override for hyperparameter search (backwards compatibility)
        if self._time_penalty_per_step_override is not None:
            return self._time_penalty_per_step_override

        # Time penalty disabled - γ<1 PBRS provides unified time pressure
        return 0.0

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

        SIMPLIFIED 2025-12-15: Disabled - PBRS already penalizes returning to
        higher-distance states (backtracking), making explicit revisit penalties redundant.

        Oscillation (moving but staying at same path distance) receives zero PBRS reward,
        while accumulating time penalties. This naturally deters unproductive behavior.

        Returns:
            0.0: Always disabled - PBRS handles revisit penalties via potential decrease
        """
        # Check for Optuna override first
        if self._revisit_penalty_weight_override is not None:
            return self._revisit_penalty_weight_override

        return 0.0  # SIMPLIFIED: PBRS already handles this

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
    def waypoint_action_diversity_bonus(self) -> float:
        """Bonus for trying different actions near critical waypoints.

        ADDED 2025-12-20: Encourages action exploration at inflection points to
        discover optimal action sequences (e.g., RIGHT × N → JUMP+LEFT at turn).

        With low base entropy (0.03), the policy becomes deterministic. This bonus
        provides localized exploration at waypoints without disrupting overall learning.

        Applied within 30px of sharp turn waypoints (>60° curvature) when:
        - Action differs from the dominant action in recent history
        - Helps discover transitions like "stop going RIGHT, do JUMP+LEFT"

        Returns:
            0.3 (<5% success): Moderate bonus for action discovery
            0.15 (5-20% success): Reduced as agent learns action sequences
            0.0 (>20% success): Disabled when agent demonstrates competence
        """
        return 0.0

    @property
    def completion_approach_bonus_weight(self) -> float:
        """DISABLED 2025-12-25: PBRS provides sufficient gradient near goal.

        The completion approach bonus attempted to provide an exponential gradient
        in the final 50px. However, PBRS already provides stronger gradient as the
        agent gets closer (same denominator, decreasing numerator = increasing
        gradient per pixel moved).

        The natural PBRS gradient is sufficient to guide completion without
        additional shaping.

        Returns:
            0.0: Always disabled - PBRS provides sufficient gradient near goal
        """
        return 0.0  # Disabled - PBRS provides sufficient gradient

    @property
    def velocity_alignment_weight(self) -> float:
        """DISABLED 2025-12-25: PBRS gradient provides directional signal.

        Velocity alignment bonuses are redundant with PBRS, which already rewards
        moving along the optimal path through potential changes. The PBRS gradient
        field naturally guides the agent in the correct direction.

        Returns:
            0.0: Always disabled - focus on PBRS distance reduction
        """
        # Check for Optuna override first (allow override for experiments)
        if self._velocity_alignment_weight_override is not None:
            return self._velocity_alignment_weight_override

        return 0.0  # Disabled - PBRS provides directional signal

    @property
    def mine_hazard_cost_multiplier(self) -> float:
        """Mine hazard cost multiplier for pathfinding (constant).

        SIMPLIFIED 2025-12-15: Removed curriculum scaling. Use constant multiplier
        throughout training. The mine proximity cost in A* pathfinding shapes PBRS
        gradient to naturally guide agents along safer paths.

        This is a pathfinding parameter, not a reward parameter. It affects which
        paths the A* algorithm considers "optimal", which then affects the PBRS
        gradient field. The agent learns mine avoidance through PBRS, not through
        explicit reward modulation.

        Returns:
            2.0: Low but meaningful avoidance without excessive detours
        """
        # Check for Optuna override first
        if self._mine_hazard_cost_multiplier_override is not None:
            return self._mine_hazard_cost_multiplier_override

        return 2.0  # Low but meaningful: avoids mines without forcing inefficient paths

    @property
    def death_penalty(self) -> float:
        """Curriculum-adaptive death penalty scaled with agent competence.

        Implements graduated penalty system that:
        1. Breaks "progress+death" local minima in discovery phase (-8)
        2. Encourages forward progress over safe camping (10% extra progress justifies risk)
        3. Reduces penalty as agent demonstrates mine avoidance (->0)
        4. Enables bold risk-taking once competent (0 at >40% success)

        Penalty scaling ensures forward progress is ALWAYS better than camping:
        - Discovery: 50% risky progress > 16% safe camping (see REWARD_BALANCE_VERIFICATION.md)
        - Requires ~20% progress to justify death risk (breakeven analysis)
        - Completion always strongly preferred over partial progress

        Combined with strong PBRS (40 in discovery), this creates hierarchy:
        - Complete level (40 PBRS + 50 completion + 40 switch) = +130 = +13.0 scaled (BEST)
        - Switch + death (30 PBRS + 40 switch - 8 penalty) = +62 = +6.2 scaled (good milestone)
        - 50% progress + death (20 PBRS - 8 penalty) = +12 = +1.2 scaled (acceptable risk)
        - Camp 16% + timeout (6.4 PBRS - 1.5 time) = +4.9 = +0.49 scaled (poor)
        - Stagnate <15% + timeout (PBRS - 20 penalty - 1.5 time) = negative (WORST)

        Returns:
            -15.0 (<5% success): Strong deterrent making death more costly than oscillation
            -6.0 (5-20% success): Lighter deterrent as skills improve
            -3.0 (20-40% success): Minimal deterrent, agent showing competence
            0.0 (>40% success): Zero penalty for advanced play
        """
        # Check for Optuna override first
        if self._death_penalty_override is not None:
            return self._death_penalty_override

        if self.recent_success_rate < 0.05:  # Discovery phase
            return -30.0  # Strong penalty to discourage "die quickly" strategy
        elif self.recent_success_rate < 0.20:  # Early learning
            return -10.0  # Lighter deterrent
        elif self.recent_success_rate < 0.40:  # Mid learning
            return -5.0  # Minimal deterrent, agent showing competence
        return -2.0  # Advanced: encourage bold risk-taking

    @property
    def level_completion_reward(self) -> float:
        """Level completion reward (constant or overridden by Optuna).

        Returns:
            Override value if set by Optuna, else 50.0 (standard completion reward)
        """
        if self._level_completion_reward_override is not None:
            return self._level_completion_reward_override
        return 50.0  # Standard from reward_constants.LEVEL_COMPLETION_REWARD

    @property
    def survival_bonus_per_100_frames(self) -> float:
        """Survival bonus to encourage longer episodes and counter quick-death exploit.

        DISABLED 2025-12-20: Removed because it was making timeouts attractive.
        With survival bonus, agent learned to "get close and timeout" instead of completing.

        The combination of:
        - Higher death penalty (-15)
        - Zero time penalty in discovery
        - RND exploration bonuses

        Already provides sufficient incentive to survive without explicit survival rewards.

        Returns:
            0.0: Always disabled
        """
        return 0.0  # DISABLED: Was encouraging "timeout at 99%" behavior

    @property
    def entropy_coefficient(self) -> float:
        """Adaptive entropy coefficient based on learning progress.

        UPDATED 2025-12-27: Reduced values to prevent entropy saturation.
        Previous training run showed ent_coef=0.1 caused 99.65% of maximum entropy
        (essentially uniform random policy). Adaptive system now stays in safe 0.01-0.02 range.

        Strategy:
        - Start low (0.01) for focused learning from BC pretraining
        - MODEST boost to 0.02 when stuck (<5% success after 1M steps)
        - Drop back to 0.01 as success improves

        Benefits of conservative entropy:
        - Allows PBRS gradient signal to guide policy (not drowned by exploration noise)
        - Maintains task-relevant action distribution (not uniform random)
        - Still permits exploration through RND intrinsic motivation
        - Compatible with unified γ=0.99 PBRS design

        Returns:
            0.01 (< 1M steps): Initial learning with BC pretraining guidance
            0.02 (<5% success, >1M steps): Modest boost when stuck
            0.01 (>5% success): Standard exploitation throughout
        """
        if self.current_timesteps < 1_000_000:
            return 0.01  # Low initial - trust BC pretraining
        elif self.recent_success_rate < 0.05:
            return 0.02  # Modest boost when stuck (not too high!)
        return 0.01  # Standard exploitation

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
            "death_penalty": self.death_penalty,
            "time_penalty_per_step": self.time_penalty_per_step,
            "completion_approach_bonus_weight": self.completion_approach_bonus_weight,
            "entropy_coefficient": self.entropy_coefficient,  # NEW: Adaptive entropy
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
            f"death_penalty={self.death_penalty:.1f}, "
            f"vel_align={self.velocity_alignment_weight:.2f})"
        )


@dataclass
class GoalCurriculumConfig:
    """Configuration for sliding window goal curriculum learning.

    Sliding window approach: Both switch and exit slide forward along the combined
    optimal path (spawn → original_switch → original_exit), maintaining a fixed
    distance interval between them. This teaches successive subsections of the
    trajectory with smooth continuous progression.

    This maintains observation-reward consistency (Markov property) since all
    observations read directly from entity positions.
    """

    # Enable/disable goal curriculum
    enabled: bool = False

    # Distance interval between switch and exit, and advancement step size (pixels)
    # With 150px: agent learns 150px subsections, each requiring ~25 steps at 6px/step
    stage_distance_interval: float = 150.0

    # Success rate threshold to advance to next stage (0.0-1.0)
    # When rolling completion rate exceeds this threshold, both entities advance
    advancement_threshold: float = 0.50

    # Rolling window size for success rate calculation (episodes)
    rolling_window: int = 100

    # DEPRECATED: Kept for backwards compatibility
    # Sliding window model uses stage_distance_interval instead
    progress_stages: tuple = (0.25, 0.50, 0.75, 1.0)
    virtual_goal_radius: float = 15.0
