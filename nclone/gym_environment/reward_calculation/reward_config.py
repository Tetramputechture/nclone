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

    # Curriculum stage tracking (for action masking mode)
    # ADDED 2026-01-05: Used by action_masking_mode property
    _curriculum_stage: int = 0
    _num_curriculum_stages: int = 1

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
    _entropy_coefficient_override: Optional[float] = None

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

        UPDATED 2026-02-03: Flattened late-stage weights to maintain strong PBRS signal
        throughout training. PBRS is now the sole dense shaping signal (waypoints removed).
        Agent needs strong PBRS gradient to discover shortcuts even at high success rates (60%+).

        Schedule maintains strong PBRS throughout training:
        - Discovery: 40.0 (baseline for learning)
        - Early: 30.0 (strong early learning)
        - Mid: 25.0 (maintained gradient)
        - Advanced: 20.0 (continued optimization)
        - Mastery: 15.0 (preserved optimization signal)

        Max episode return = PBRS_weight + completion(200) + switch(100) ≈ weight + 300

        With current weights:
        - Discovery (<5%): 40 + 300 = 340 max return
        - Early learning (5-20%): 30 + 300 = 330 max return
        - Mid learning (20-40%): 25 + 300 = 325 max return
        - Advanced (40-60%): 20 + 300 = 320 max return
        - Mastery (60%+): 15 + 300 = 315 max return

        Returns:
            40.0 (0-5% success): Balanced guidance for discovery
            30.0 (5-20% success): Stronger early learning signal
            25.0 (20-40% success): Strong gradient for mid learning
            20.0 (40-60% success): Continued meaningful optimization signal
            15.0 (60%+ success): Preserved strong gradient for mastery-level optimization
        """
        # Check for Optuna override first
        if self._pbrs_objective_weight_override is not None:
            return self._pbrs_objective_weight_override

        if self.recent_success_rate < 0.05:  # Discovery phase (0-5% success)
            return 40.0  # Baseline for learning
        elif self.recent_success_rate < 0.20:  # Early learning (5-20% success)
            return 30.0  # Stronger early learning (increased from 25.0)
        elif self.recent_success_rate < 0.40:  # Mid learning (20-40% success)
            return 25.0  # Strong gradient for mid learning (increased from 18.0)
        elif self.recent_success_rate < 0.60:  # Advanced learning (40-60%)
            return 20.0  # Continued optimization (increased from 12.0)
        return 15.0  # Preserve strong signal for mastery (increased from 8.0)

    @property
    def time_penalty_per_step(self) -> float:
        """Strengthened time penalty for path efficiency optimization.

        UPDATED 2026-02-03: Significantly increased step penalty (10-20x) to make "fewer steps"
        a first-class optimization target alongside completion. Previous values (-0.001 to -0.003)
        were too weak - acting as "subtle afterthoughts" rather than meaningful signals.
        Strong step penalty makes efficiency explicit rather than implicit.

        Curriculum-adaptive schedule (10-20x increase from previous):
        - Discovery (<5% success): 0.0 (no pressure during initial learning)
        - Early (5-20%): -0.01/step (light pressure as competence develops)
        - Mid (20-40%): -0.02/step (moderate pressure for path refinement)
        - Advanced (40-60%): -0.03/step (strong pressure for optimization)
        - Mastery (60%+): -0.05/step (maximum speed optimization)

        Impact analysis (500-step episode):
        - Previous: -0.5 to -1.5 unscaled (-0.05 to -0.15 scaled)
        - New: -5 to -25 unscaled (-0.5 to -2.5 scaled)
        - Creates strong incentive to find shorter paths (200-300 steps vs 500 steps)

        This makes step count a primary signal rather than afterthought, aligning with
        the explicit goal of "find shortest path" rather than just "complete level".

        Returns:
            0.0 (<5% success): No time pressure during discovery
            -0.01 (5-20% success): Light pressure for early learning
            -0.02 (20-40% success): Moderate pressure for mid learning
            -0.03 (40-60% success): Strong pressure for advanced optimization
            -0.05 (60%+ success): Maximum speed optimization for mastery
        """
        # Allow Optuna override for hyperparameter search (backwards compatibility)
        if self._time_penalty_per_step_override is not None:
            return self._time_penalty_per_step_override

        # UPDATED 2026-02-03: Strengthened step penalty to make efficiency first-class signal
        # Pure PBRS + step penalty creates clear optimization objective
        if self.recent_success_rate < 0.05:
            return 0.0  # No time pressure during discovery
        elif self.recent_success_rate < 0.20:
            return -0.01  # Light pressure for early learning
        elif self.recent_success_rate < 0.40:
            return -0.02  # Moderate pressure for mid learning
        elif self.recent_success_rate < 0.60:
            return -0.03  # Strong pressure for advanced optimization
        return -0.05  # Maximum speed optimization for mastery

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
        """DISABLED: Action exploration bonus removed with waypoints.

        Returns:
            0.0: Always disabled
        """
        return 0.0

    @property
    def completion_approach_bonus_weight(self) -> float:
        """DISABLED 2025-12-25: PBRS provides sufficient gradient near goal.

        Returns:
            0.0: Always disabled - PBRS provides sufficient gradient
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

        Death penalties strongly deter learning "doomed trajectory" patterns where agent gets
        close to goal but dies, especially critical in discovery/early phases.

        Discovery phase analysis with pure PBRS:
        - Complete: +40 PBRS + 100 switch + 200 completion = +340 (BEST)
        - 80% progress + death: +32 PBRS - 80 death = -48 (bad)
        - 50% progress + death: +20 PBRS - 80 death = -60 (worse)
        - Fast fail (quick death): +4 PBRS - 80 death = -76 (worst)

        This creates clear hierarchy: completion >> progress, encouraging agents to
        learn robust policies rather than risky "almost complete" trajectories.

        Implements graduated penalty system that:
        1. Breaks "doomed trajectory" local minima in discovery (-80)
        2. Maintains completion preference in early learning (-50)
        3. Allows tactical risk-taking in mid-level (-20)
        4. Reduces to light penalty as mastery develops (-8, -6)

        Returns:
            Base penalties (success-rate scaled):
                -80.0 (<5% success): Prevents "doomed trajectory" exploitation
                -50.0 (5-20% success): Strong completion preference
                -20.0 (20-40% success): Balanced risk-taking
                -8.0 (40-60% success): Light penalty for advanced agents
                -6.0 (60%+ success): Minimal penalty at mastery
        """
        # Check for Optuna override first
        if self._death_penalty_override is not None:
            return self._death_penalty_override

        # Base penalty from success rate
        if self.recent_success_rate < 0.05:  # Discovery phase
            base_penalty = (
                -80.0
            )  # Prevents "doomed trajectory" patterns (get close + die)
        elif self.recent_success_rate < 0.20:  # Early learning
            base_penalty = -50.0  # Strong completion preference while learning
        elif self.recent_success_rate < 0.40:  # Mid-level learning
            base_penalty = -20.0  # Balancing speed vs safety
        elif self.recent_success_rate < 0.60:  # Advanced learning
            base_penalty = -8.0  # Light penalty for advanced agents
        else:
            base_penalty = -6.0  # Mastery: maintains strategic risk-taking

        return base_penalty

    @property
    def level_completion_reward(self) -> float:
        """Curriculum-adaptive level completion reward.

        UPDATED 2026-01-21: Scales completion reward with curriculum difficulty to
        maintain consistent returns across stages. Early curriculum stages have much
        shorter paths (5-10x shorter), making completion proportionally easier.

        Scaling Strategy:
        - Stage 0 (early curriculum): 0.5x base reward (100 vs 200)
        - Final stage (full level): 1.0x base reward (200)
        - Linear interpolation between stages

        This prevents early stages from providing disproportionately high returns
        for easier completions, while maintaining strong completion preference
        throughout training.

        Rationale:
        - Early stage (100px path): Easier to complete → lower reward (100)
        - Late stage (1000px path): Harder to complete → full reward (200)
        - Maintains ~2:1 completion-to-death ratio across all stages

        Returns:
            Curriculum-scaled completion reward (100-200 unscaled)
        """
        # Check for Optuna override first
        if self._level_completion_reward_override is not None:
            return self._level_completion_reward_override

        BASE_COMPLETION_REWARD = 200.0

        # No curriculum or single-stage: use full reward
        if self._num_curriculum_stages <= 1:
            return BASE_COMPLETION_REWARD

        # Calculate curriculum progress (0.0 at stage 0, 1.0 at final stage)
        curriculum_progress = self._curriculum_stage / max(
            self._num_curriculum_stages - 1, 1
        )

        # Scale from 0.5x at stage 0 to 1.0x at final stage
        # This accounts for proportionally easier completion on shorter paths
        scale = 0.5 + 0.5 * curriculum_progress

        return BASE_COMPLETION_REWARD * scale

    @property
    def switch_activation_reward(self) -> float:
        """Curriculum-adaptive switch activation reward.

        ADDED 2026-01-21: Scales switch reward with curriculum difficulty to
        maintain consistent returns across stages. Early curriculum stages have
        switch positioned much closer to spawn (5-10x closer).

        Scaling Strategy:
        - Stage 0 (early curriculum): 0.5x base reward (50 vs 100)
        - Final stage (full level): 1.0x base reward (100)
        - Linear interpolation between stages

        This maintains the critical 50% milestone relationship with completion
        reward across all curriculum stages: switch = 0.5 × completion.

        Rationale:
        - Early stage: Reaching switch is easier → proportionally lower reward
        - Late stage: Reaching switch is harder → full milestone reward
        - Preserves completion > switch > death hierarchy throughout

        Returns:
            Curriculum-scaled switch activation reward (50-100 unscaled)
        """
        BASE_SWITCH_REWARD = 100.0

        # No curriculum or single-stage: use full reward
        if self._num_curriculum_stages <= 1:
            return BASE_SWITCH_REWARD

        # Calculate curriculum progress (0.0 at stage 0, 1.0 at final stage)
        curriculum_progress = self._curriculum_stage / max(
            self._num_curriculum_stages - 1, 1
        )

        # Scale from 0.5x at stage 0 to 1.0x at final stage
        # Matches completion reward scaling to preserve relative importance
        scale = 0.5 + 0.5 * curriculum_progress

        return BASE_SWITCH_REWARD * scale

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
        """Adaptive entropy coefficient based on learning progress and curriculum stage.

        UPDATED 2026-01-03: Increased for multi-sequence momentum learning with GRU.
        Previous 0.01-0.02 range was too conservative for exploring action sequence
        variations (e.g., RIGHT×3→JUMP vs RIGHT×4→JUMP timing differences).

        UPDATED 2026-01-06: Added HPO override support for fair trial comparison.
        When HPO override is set, it takes precedence over curriculum-adaptive values.

        UPDATED 2026-01-12: Added curriculum-stage-aware entropy boost for late stages.
        At stage 7+ on complex non-linear paths, agent needs sustained exploration of
        action sequence timing variations even if success rate is moderate. This prevents
        premature convergence on suboptimal momentum patterns.

        Strategy for multi-sequence levels:
        - Discovery phase: Higher entropy (0.03) to explore action sequences
        - Early learning: Moderate entropy (0.025) for sequence refinement
        - Late curriculum (stage 7+): Boosted entropy (0.04) for complex sequences
        - Advanced (otherwise): Reduced entropy (0.02) for convergence

        GRU benefits from action diversity:
        - Needs to observe multiple jump timing variations to learn momentum dependencies
        - 0.025-0.04 allows structured exploration without becoming random
        - Still much lower than problematic 0.1 (which caused 99.65% max entropy)

        Compatible with:
        - RND intrinsic motivation (state exploration)
        - PBRS gradient signals (path following)
        - GRU temporal learning (sequence memory)

        Returns:
            Override value if set by HPO, else curriculum-adaptive:
            0.03 (<5% success): Discovery - explore action sequence variations
            0.04 (stage 7+ AND <60% success): Late curriculum - sequence refinement
            0.025 (5-20% success): Early learning - refine momentum sequences
            0.02 (>20% success): Advanced - converge on optimal sequences
        """
        # Check for HPO override first (for fair hyperparameter comparison)
        if self._entropy_coefficient_override is not None:
            return self._entropy_coefficient_override

        curriculum_stage = getattr(self, "_curriculum_stage", 0)

        # STAGE 8+ COMMITMENT PHASE: Dynamically reduce entropy as success improves
        # Addresses success-spike-then-crash pattern by preventing exploration from
        # disrupting discovered patterns once agent shows consistent success.
        # This "commitment phase" allows agent to solidify learned behavior.
        #
        # TIER 2 FIX (2026-02-02): More aggressive convergence at late stages with success
        # Analysis: Tuned run had 0.96 entropy at stage 9 (too high), preventing commitment
        # to optimal patterns. Lower entropy forces convergence once good patterns discovered.
        if curriculum_stage >= 9 and self.recent_success_rate >= 0.10:
            return 0.012  # Strong commitment - force convergence on optimal patterns
        elif curriculum_stage >= 8 and self.recent_success_rate >= 0.05:
            return 0.015  # Moderate commitment - reduce exploration significantly
        elif curriculum_stage >= 8:
            if self.recent_success_rate >= 0.15:
                return 0.018  # Exploitation - solidify learned pattern (original)
            elif self.recent_success_rate >= 0.05:
                return 0.022  # Transitioning - moderate exploration (original)
            else:
                return 0.028  # Discovery - still exploring

        # LATE CURRICULUM BOOST: At stage 7, maintain higher entropy for complex sequences
        # This prevents policy from becoming overly deterministic when exploring long
        # non-linear paths requiring precise momentum-based jump timing variations
        if curriculum_stage >= 7 and self.recent_success_rate < 0.60:
            return 0.04  # Boosted entropy for late-stage sequence refinement

        # Standard curriculum-adaptive (default behavior)
        if self.recent_success_rate < 0.05:
            return 0.03  # Discovery: explore action sequences for multi-jump chains
        elif self.recent_success_rate < 0.20:
            return 0.025  # Early learning: refine sequences with GRU memory
        return 0.02  # Advanced: converge on optimal momentum patterns

    @property
    def safety_critic_weight(self) -> float:
        """Curriculum-adaptive safety critic contribution weight.

        Modulates both auxiliary training loss and inference logit penalties
        based on agent competence. During early training (<5% success), reduces
        safety constraints to enable riskier exploration and discovery.

        Strategy:
        - Discovery (<5%): 0.1x weight - minimal safety constraints for exploration
        - Early learning (5-20%): 0.5x weight - gradual safety introduction
        - Mid learning (20-40%): 0.75x weight - balanced risk/safety
        - Advanced (>40%): 1.0x weight - full safety critic influence

        Applied to:
        1. Training: Auxiliary loss weight (base 0.1 * this multiplier)
        2. Inference: Logit penalty strength (-20/-10/-5 * this multiplier)

        Returns:
            0.1 (<5% success): Minimal safety - enable risky exploration
            0.5 (5-20% success): Moderate safety - learning fundamentals
            0.75 (20-40% success): Strong safety - refinement
            1.0 (>40% success): Full safety - optimization
        """
        if self.recent_success_rate < 0.05:  # Discovery phase
            return 0.1  # Very low - enable risky exploration
        elif self.recent_success_rate < 0.20:  # Early learning
            return 0.5  # Moderate increase as competence grows
        elif self.recent_success_rate < 0.40:  # Mid learning
            return 0.75  # Further increase for refinement
        return 1.0  # Full strength for advanced optimization

    @property
    def action_masking_mode(self) -> str:
        """Curriculum-aware action masking mode for annealed guidance.

        ADDED 2026-01-05: Implements annealed masking schedule that transitions from
        hard masking (early) through soft logit biasing (mid) to no guidance (late).

        Uses dual criteria (success rate AND curriculum stage) to prevent premature
        removal of guidance when curriculum just advanced but success temporarily dropped.

        Mode transitions:
        - Hard masking: Early curriculum (<30% progress) OR low success (<15%)
          * Eliminates obviously wrong actions on trivial straight paths
          * Focuses learning on physics and timing, not basic direction
          * Applied in Ninja.get_valid_action_mask() for path-direction masking

        - Soft biasing: Mid curriculum (30-70% progress) OR moderate success (15-40%)
          * Adds logit bias toward heuristic actions without hard constraints
          * Preserves exploration while guiding toward promising actions
          * Applied in policy forward pass via compute_heuristic_logit_bias()

        - No guidance: Late curriculum (>70% progress) AND high success (>40%)
          * Policy must be fully autonomous and robust
          * Generalizes beyond scaffolded training distribution
          * No masking or biasing applied

        Design Philosophy:
        - Use OR for hard/soft (guidance when either condition met)
        - Use AND for none (full autonomy only when both conditions met)
        - This ensures guidance persists through curriculum difficulty spikes

        Returns:
            'hard': Hard action masking for trivial paths
            'soft': Soft logit biasing toward heuristic actions
            'none': No masking or biasing
        """
        # Calculate curriculum progress (avoid division by zero)
        # Note: curriculum_stage and num_stages come from goal curriculum
        curriculum_stage = getattr(self, "_curriculum_stage", 0)
        num_stages = getattr(self, "_num_curriculum_stages", 1)
        curriculum_progress = curriculum_stage / max(num_stages - 1, 1)

        # Hard masking: Early curriculum OR low success
        if curriculum_progress < 0.3 or self.recent_success_rate < 0.15:
            return "hard"

        # No guidance: Late curriculum AND high success
        return "none"

    @property
    def logit_bias_strength(self) -> float:
        """Logit bias strength for soft masking mode.

        ADDED 2026-01-05: Controls strength of heuristic logit bias during soft mode.
        Decays linearly as success rate increases to create smooth transition.

        Strength schedule:
        - 15% success: 2.0 (strong guidance at soft mode start)
        - 27.5% success: 1.25 (moderate guidance at midpoint)
        - 40% success: 0.5 (light guidance before transition to none)

        Returns:
            Bias strength multiplier (0.0 if not in soft mode)
        """
        if self.action_masking_mode != "soft":
            return 0.0

        # Linear decay from 2.0 to 0.5 as success rate increases 15%->40%
        t = (self.recent_success_rate - 0.15) / 0.25
        t = min(max(t, 0.0), 1.0)  # Clamp to [0, 1]
        return 2.0 - 1.5 * t  # 2.0 at t=0, 0.5 at t=1

    def set_curriculum_stage(self, stage: int, num_stages: int) -> None:
        """Update curriculum stage for action masking mode calculation.

        Called by GoalCurriculumCallback when curriculum stage advances.

        Args:
            stage: Current curriculum stage index (0 to num_stages-1)
            num_stages: Total number of curriculum stages
        """
        self._curriculum_stage = stage
        self._num_curriculum_stages = num_stages

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
            "safety_critic_weight": self.safety_critic_weight,  # NEW: Adaptive safety
            "exploration_bonus": self.exploration_bonus,
            "revisit_penalty_weight": self.revisit_penalty_weight,
            "pbrs_normalization_scale": self.pbrs_normalization_scale,
            "velocity_alignment_weight": self.velocity_alignment_weight,
            "mine_hazard_cost_multiplier": self.mine_hazard_cost_multiplier,
            "action_masking_mode": self.action_masking_mode,  # NEW: Annealed action masking
            "logit_bias_strength": self.logit_bias_strength,  # NEW: Soft guidance strength
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
    rolling_window: int = 5000

    # DEPRECATED: Kept for backwards compatibility
    # Sliding window model uses stage_distance_interval instead
    progress_stages: tuple = (0.2, 0.4, 0.6, 0.8, 1.0)
    virtual_goal_radius: float = 15.0
