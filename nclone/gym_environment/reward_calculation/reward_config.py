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

    # Waypoint system configuration (SIMPLIFIED 2026-01-03)
    # Provides guidance through non-linear paths where PBRS alone fails
    # Simplified to uniform 12px spacing along optimal path (matches sub-node spacing)
    enable_path_waypoints: bool = True  # Re-enabled for non-linear path guidance
    path_waypoint_progress_spacing: float = (
        12.0  # Sub-node spacing for uniform coarse rewards
    )
    path_waypoint_cluster_radius: float = 10.0  # Matches collection radius

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

        UPDATED 2025-12-29: Flattened weight curve to maintain stronger gradient signal
        in advanced phase. Previous steep drop-off (50→5) caused slow convergence at high
        success rates as dense shaping signal was drowned out by sparse terminal rewards.

        New schedule maintains meaningful PBRS throughout training:
        - Discovery: 40.0 (reduced from 50.0 for stability)
        - Early: 25.0 (increased from 15.0 for stronger learning)
        - Mid: 18.0 (increased from 12.0 to maintain gradient)
        - Advanced: 12.0 (increased from 8.0 for continued optimization)
        - Mastery: 8.0 (increased from 5.0 to preserve gradient signal)

        Max episode return = PBRS_weight + completion(200) + switch(100) ≈ weight + 300

        With new weights:
        - Discovery (<5%): 40 + 300 = 340 max return
        - Early learning (5-20%): 25 + 300 = 325 max return
        - Mid learning (20-40%): 18 + 300 = 318 max return
        - Advanced (40-60%): 12 + 300 = 312 max return
        - Mastery (60%+): 8 + 300 = 308 max return

        Flatter curve ensures PBRS remains influential for policy gradient even at high success rates.

        Returns:
            40.0 (0-5% success): Balanced guidance for discovery
            25.0 (5-20% success): Stronger early learning signal
            18.0 (20-40% success): Maintained gradient for mid learning
            12.0 (40-60% success): Still meaningful for advanced optimization
            8.0 (60%+ success): Preserved gradient for mastery
        """
        # Check for Optuna override first
        if self._pbrs_objective_weight_override is not None:
            return self._pbrs_objective_weight_override

        if self.recent_success_rate < 0.05:  # Discovery phase (0-5% success)
            return 40.0  # Slightly reduced for stability
        elif self.recent_success_rate < 0.20:  # Early learning (5-20% success)
            return 25.0  # Stronger early learning
        elif self.recent_success_rate < 0.40:  # Mid learning (20-40% success)
            return 18.0  # Maintain gradient strength
        elif self.recent_success_rate < 0.60:  # Advanced learning (40-60%)
            return 12.0  # Still meaningful
        return 8.0  # Keep gradient signal for optimization (60%+)

    @property
    def time_penalty_per_step(self) -> float:
        """Light time penalty for speed optimization in advanced phases.

        UPDATED 2025-12-29: Re-enabled light explicit time penalty for advanced phase
        to accelerate convergence. While γ=0.99 PBRS provides implicit time pressure
        (-0.01×Φ per step), this is too weak in practice at high success rates.

        Curriculum-adaptive schedule:
        - Early (<20% success): 0.0 (no time pressure during learning)
        - Mid (20-40%): -0.001/step (very light - ~0.5 over 500 steps)
        - Advanced (40-60%): -0.002/step (moderate - ~1.0 over 500 steps)
        - Mastery (60%+): -0.003/step (speed optimization - ~1.5 over 500 steps)

        At -0.003/step over 500 steps = -1.5 unscaled = -0.15 scaled.
        This is ~15% of PBRS reward (8-12 range), creating urgency without
        overwhelming the primary objective signal.

        Benefits:
        - Accelerates convergence in advanced phase
        - Encourages efficient paths without penalizing learning
        - Complements γ<1 PBRS without violating policy invariance
        - Scales with training progress

        Returns:
            0.0 (<20% success): No time pressure during learning
            -0.001 (20-40%): Very light pressure for mid learning
            -0.002 (40-60%): Moderate pressure for advanced optimization
            -0.003 (60%+): Speed optimization for mastery
        """
        # Allow Optuna override for hyperparameter search (backwards compatibility)
        if self._time_penalty_per_step_override is not None:
            return self._time_penalty_per_step_override

        # UPDATED 2025-12-29: Re-enable light time penalty for advanced phase optimization
        # While γ=0.99 PBRS provides implicit time pressure, it's too weak in practice
        # Light explicit penalty accelerates convergence without overwhelming primary signal
        if self.recent_success_rate < 0.20:
            return 0.0  # No time pressure during learning
        elif self.recent_success_rate < 0.40:
            return -0.001  # Very light pressure
        elif self.recent_success_rate < 0.60:
            return -0.002  # Moderate pressure
        return -0.003  # Speed optimization phase

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
        """Base PBRS normalization scale (curriculum-adjusted in reward calculator).

        Base scale factor for PBRS potential normalization. The actual scale is
        curriculum-aware and computed in the reward calculator based on curriculum
        stage to maintain gradient strength over longer paths at later stages.

        Returns:
            1.0: Base scale (adjusted by curriculum stage at runtime)
        """
        return 1.0  # Base scale - curriculum adjustment applied in reward calculator

    def get_curriculum_pbrs_scale(
        self, curriculum_stage: int, num_stages: int
    ) -> float:
        """DEPRECATED (2026-01-04): Gradient scaling now handled automatically in PBRS calculator.

        This method is no longer used. Gradient scaling is now computed automatically
        in PBRSCalculator.calculate_combined_potential() based on the actual path length,
        using sqrt scaling: gradient_scale = sqrt(path_length / 200px).

        This provides more accurate and consistent per-step gradients across all path
        lengths (50px to 2000px) without requiring curriculum stage information.

        See: pbrs_potentials.py - PBRSCalculator.calculate_combined_potential()
             reward_constants.py - PBRS_GRADIENT_REFERENCE_DISTANCE, PBRS_GRADIENT_SCALE_CAP

        Args:
            curriculum_stage: Current curriculum stage (0 to num_stages-1) [UNUSED]
            num_stages: Total number of curriculum stages [UNUSED]

        Returns:
            1.0: Always returns base scale (deprecated, no longer applied)
        """
        # DEPRECATED: Return 1.0 to avoid breaking existing code that calls this method
        # The actual gradient scaling is now computed in the PBRS calculator
        return 1.0

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

        CRITICAL UPDATE 2026-01-03: With increased waypoint rewards (+8 × ~15 = +120 max) and
        PBRS distance reduction, discovery/early phases must heavily penalize death to prevent
        learning "doomed trajectory" patterns where agent gets close to goal but dies.

        Problem scenario at 0% success rate: "Fast fail" strategy (jump directly, reduce
        distance quickly, die) vs "Waypoint collection" (follow non-linear path, collect waypoints):
            Fast fail: +15 PBRS (quick distance) - 80 death = -65 (CLEARLY BAD)
            Waypoint strategy: +30 PBRS + 96 waypoints (80%) - 80 death = +46 (GOOD!)
            This creates strong preference for following waypoint guidance.

        Discovery phase analysis with -80 penalty and +8 waypoint rewards:
        - Complete: +40 PBRS + 120 waypoints + 100 switch + 200 completion = +460 (BEST)
        - 80% waypoints + death: +32 PBRS + 96 waypoints - 80 death = +48 (decent progress)
        - 50% waypoints + death: +20 PBRS + 60 waypoints - 80 death = +0 (neutral)
        - Fast fail (quick death): +15 PBRS - 80 death = -65 (strongly discouraged)

        Implements graduated penalty system that:
        1. Breaks "doomed trajectory" local minima in discovery (-80)
        2. Maintains completion preference in early learning (-50)
        3. Allows tactical risk-taking in mid-level (-20)
        4. Prevents waypoint farming in advanced (-8)
        5. Maintains strategic awareness in mastery (-6)

        Returns:
            -80.0 (<5% success): Prevents "doomed trajectory" exploitation
            -50.0 (5-20% success): Strong completion preference
            -20.0 (20-40% success): Balanced risk-taking
            -8.0 (40-60% success): Prevents waypoint farming
            -6.0 (60%+ success): Strategic risk consideration
        """
        # Check for Optuna override first
        if self._death_penalty_override is not None:
            return self._death_penalty_override

        if self.recent_success_rate < 0.05:  # Discovery phase
            return -80.0  # Prevents "doomed trajectory" patterns (get close + die)
        elif self.recent_success_rate < 0.20:  # Early learning
            return -50.0  # Strong completion preference while learning
        elif self.recent_success_rate < 0.40:  # Mid-level learning
            return -20.0  # Balancing speed vs safety
        elif self.recent_success_rate < 0.60:  # Advanced learning
            return -8.0  # Prevents "collect waypoints + die" exploitation
        return (
            -6.0
        )  # Mastery: maintains strategic risk-taking, prevents waypoint farming

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
        - Zero time penalty in discoveryR
        - RND exploration bonuses

        Already provides sufficient incentive to survive without explicit survival rewards.

        Returns:
            0.0: Always disabled
        """
        return 0.0  # DISABLED: Was encouraging "timeout at 99%" behavior

    @property
    def entropy_coefficient(self) -> float:
        """Adaptive entropy coefficient based on learning progress.

        UPDATED 2026-01-03: Increased for multi-sequence momentum learning with GRU.
        Previous 0.01-0.02 range was too conservative for exploring action sequence
        variations (e.g., RIGHT×3→JUMP vs RIGHT×4→JUMP timing differences).

        Strategy for multi-sequence levels:
        - Discovery phase: Higher entropy (0.03) to explore action sequences
        - Early learning: Moderate entropy (0.025) for sequence refinement
        - Advanced: Reduced entropy (0.02) for convergence on optimal sequences

        GRU benefits from action diversity:
        - Needs to observe multiple jump timing variations to learn momentum dependencies
        - 0.025-0.03 allows structured exploration without becoming random
        - Still much lower than problematic 0.1 (which caused 99.65% max entropy)

        Compatible with:
        - RND intrinsic motivation (state exploration)
        - PBRS gradient signals (path following)
        - GRU temporal learning (sequence memory)

        Returns:
            0.03 (<5% success): Discovery - explore action sequence variations
            0.025 (5-20% success): Early learning - refine momentum sequences
            0.02 (>20% success): Advanced - converge on optimal sequences
        """
        if self.recent_success_rate < 0.05:
            return 0.03  # Discovery: explore action sequences for multi-jump chains
        elif self.recent_success_rate < 0.20:
            return 0.025  # Early learning: refine sequences with GRU memory
        return 0.02  # Advanced: converge on optimal momentum patterns

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
    enabled: bool = True

    # Distance interval between switch and exit, and advancement step size (pixels)
    # With 150px: agent learns 150px subsections, each requiring ~25 steps at 6px/step
    stage_distance_interval: float = 100.0

    # Success rate threshold to advance to next stage (0.0-1.0)
    # When rolling completion rate exceeds this threshold, both entities advance
    advancement_threshold: float = 0.80

    # Rolling window size for success rate calculation (episodes)
    rolling_window: int = 500

    # DEPRECATED: Kept for backwards compatibility
    # Sliding window model uses stage_distance_interval instead
    progress_stages: tuple = (0.2, 0.4, 0.6, 0.8, 1.0)
    virtual_goal_radius: float = 15.0
