"""PBRS reward constants with oscillation penalty for N++ RL training.

Centralized reward system with clear hierarchy:
1. Terminal rewards - Define task success/failure
2. PBRS shaping: F(s,s') = γ * Φ(s') - Φ(s) - Dense path guidance with oscillation penalty
3. Time penalty - Efficiency pressure (curriculum-managed, frame-skip aware)

PBRS with γ=0.99 implementation:
- Dense reward signal at every step based on geometric path distance potential
- Automatic backtracking penalties (potential decrease + 1% discount)
- Active oscillation penalties (0 potential change - 1% accumulated per wasted step)
- Temporal preference for efficient paths (shorter episodes = higher cumulative PBRS)
- Near policy-invariance (99% of gradient preserved, Ng et al. 1999)
- Markov property (no episode history dependencies)

Oscillation Penalty Mechanism (γ=0.99):
- Forward 12px: +0.99*ΔΦ ≈ +0.15 reward (progress outweighs discount)
- Stay still: -0.01*Φ_current ≈ -0.0075 penalty (wasted step)
- Backward 12px: -0.99*ΔΦ ≈ -0.15 penalty (regress + discount)
- Oscillate A↔B: -0.01*(Φ_A + Φ_B) per cycle (accumulated negative bias)
- 100-step vs 10-step path: differs by ~0.5 cumulative PBRS (efficiency incentive)

Frame Skip Integration:
- Reward calculated ONCE per action (not per frame) for 75% computational savings
- PBRS telescopes: γ^n*Φ(final) - Φ(initial) with accumulated -(1-γ)*Σ bias
- Time penalty scales by frames_executed for correct per-action magnitude
- With 4-frame skip: typical movement is 6-8px (not max speed), requiring stronger weights

References:
- Ng et al. (1999): "Policy Invariance Under Reward Transformations" (PBRS theory)
- Sutton & Barto (2018): "Reinforcement Learning: An Introduction" (reward scaling)
"""

# =============================================================================
# GLOBAL REWARD SCALING (Value Function Stability)
# =============================================================================
# Scale ALL rewards by this factor to reduce value loss magnitudes.
#
# WHY THIS HELPS:
# - TensorBoard showed value_loss ~76, episode rewards -500 to -800
# - Large returns cause high variance in value function targets
# - Scaling by 0.1 (divide by 10) reduces returns to -50 to -80 range
# - This makes value function easier to learn (lower MSE targets)
#
# CRITICAL: This does NOT change learning dynamics because:
# - All rewards are scaled uniformly (relative magnitudes preserved)
# - Optimal policy is unchanged (argmax over scaled rewards = argmax over original)
# - PPO's advantage normalization already handles scale (mean/std normalization)
# - The ratio of positive to negative rewards stays the same
#
# WHAT CHANGES:
# - Value estimates: -4 to -123 → -0.4 to -12.3
# - Value loss: ~76 → ~0.76
# - Episode rewards (logged): -500 → -50
#
# WHAT STAYS THE SAME:
# - Relative importance: completion(+50) >> death(-12) >> time_penalty(-0.02)
# - Learning signal quality: same gradients, just scaled
# - Convergence rate: PPO is scale-invariant due to advantage normalization
GLOBAL_REWARD_SCALE = 0.1  # Divide all rewards by 10

# CRITICAL: This scale must be applied to ALL reward components including:
# - Terminal rewards (completion, death, switch) - applied in main_reward_calculator.py
# - PBRS shaping rewards - applied in main_reward_calculator.py
# - Time penalties - applied in main_reward_calculator.py
# - RND intrinsic rewards - applied in rnd_callback.py
# Inconsistent scaling causes reward imbalance and policy collapse.


# =============================================================================
# TERMINAL REWARD CONSTANTS (Always Active)
# =============================================================================
# Terminal rewards are the primary learning signals that define task completion
#
# REWARD HIERARCHY (critical for correct learning):
#   Success (+105) >> Timeout+Progress (-1) >> Progress+Death (-8) >> Quick Death (-15)
#
# UPDATED 2025-12-20: Rebalanced to prevent both "die quickly" and "wander forever" exploits
#   1. PBRS weight reduced (60→15) to allow exploration mistakes
#   2. Death penalty increased (-8→-15) to make quick death unattractive
#   3. Very light time penalty (-0.0005) provides mild completion urgency without survival tax
#   4. RND exploration bonuses (+0.3 to +1.5) make new state discovery rewarding
#   5. Velocity alignment (+1.0) guides agent along curved optimal paths
#
# Example with discovery phase (PBRS weight=15, death penalty=-15, time=-0.0005):
#   - Complete: +15 (PBRS) + 50 (completion) + 40 (switch) = +105 (BEST)
#   - 99% + timeout: +14.85 (PBRS) - 0.25 (time) + approach bonus = ~+15 (good!)
#   - 50% progress + death: +7.5 (PBRS) - 15 (death) - 0.125 (time) = -7.6 (risky)
#   - Quick death (200f): -15 (death) - 0.3 (PBRS) - 0.025 (time) = -15.3 (WORST)
#
# The math guarantees: Completion > Exploration > Death
# Light time penalty prevents "wander forever" without creating "die quickly" incentive

# Level completion reward - primary learning signal
# Rationale: Increased from 20.0 to 50.0 (2.5×) to ensure terminal rewards
# dominate over accumulated shaping penalties. With reduced PBRS weights (15.0 max)
# and reduced revisit penalties, this ensures successful episodes have higher
# rewards than failed episodes.
LEVEL_COMPLETION_REWARD = 200.0

# Death penalty - CURRICULUM-ADAPTIVE (scaled with agent competence)
#
# UPDATED 2025-12-17: Reintroduced with curriculum scaling to fix local minima.
#
# Problem identified at 5M steps:
# - Agent stuck at <5% success making 50% progress then dying to same mine
# - With PBRS weight=80 and death penalty=0, "progress+death" gives +4.0 reward
# - This is better than safer alternatives, creating stable local optimum
# - Agent never learns mine avoidance because risky strategy is optimal
#
# Solution: Curriculum-based penalties that break early-phase local minima:
# - Discovery (<5%): -15 penalty makes risky death less attractive than completion
# - Early (5-20%): -10 penalty continues to favor safe completion
# - Mid (20-40%): -5 penalty reduces as agent shows mine avoidance competence
# - Advanced (>40%): 0 penalty encourages risk-taking in advanced play
#
# This preserves original intent (encourage bold play at high skill) while
# preventing the "rush to death" local minimum that traps early learning.
#
# NOTE: These are BASE values. Actual penalties calculated in RewardConfig
# based on real-time success rate tracking.

# Death penalties are now managed by RewardConfig.death_penalty property
# These constants serve as reference values for the curriculum
DEATH_PENALTY_DISCOVERY = (
    -8.0
)  # <5% success: balanced deterrent (10% extra progress needed)
DEATH_PENALTY_EARLY = -6.0  # 5-20% success: lighter deterrent (7.5% extra progress)
DEATH_PENALTY_MID = -3.0  # 20-40% success: mild deterrent (3.75% extra progress)
DEATH_PENALTY_ADVANCED = 0.0  # >40% success: encourage risk-taking

# Legacy constants for backward compatibility (use RewardConfig instead)
DEATH_PENALTY = -8.0  # Default to discovery phase
IMPACT_DEATH_PENALTY = -8.0
HAZARD_DEATH_PENALTY = -8.0

# Timeout/truncation penalty (episode time limit exceeded)
# UPDATED 2025-12-25: Stagnation penalty REMOVED - redundant and confusing
#
# History:
# - 2025-12-14: Changed from -10.0 to 0.0 to fix RND policy collapse
# - 2025-12-15: Added progress-gated penalty (-20) for stagnation
# - 2025-12-25: REMOVED stagnation penalty - focus on core components
#
# Rationale for removal:
# 1. Redundant with time penalty: Time penalty already discourages long episodes
#    (-0.0005 to -0.015 per step = -0.25 to -1.5 over typical 2000 frame timeout)
#
# 2. Buggy implementation: Penalized legitimate high progress (97%) when path was
#    inefficient, creating confusing/contradictory learning signals
#
# 3. Violates design goal: Adds complexity beyond core objectives (completion,
#    path distance, speed). Arbitrary threshold checks (15%) don't align with
#    continuous PBRS gradient field.
#
# 4. Natural hierarchy already works:
#    - Complete: +15 PBRS + 110 terminal - 0.25 time = +124.75 (best)
#    - Timeout at 97%: +14.55 PBRS - 0.25 time = +14.30 (good)
#    - Timeout at 15%: +2.25 PBRS - 0.25 time = +2.00 (poor but natural)
#    - Death at 50%: +7.5 PBRS - 15 death - 0.125 time = -7.6 (worst)
#
# The reward structure naturally discourages stagnation without explicit penalties.
# Let PBRS (distance reduction) + time penalty (speed) + terminal rewards (completion)
# provide clear, interpretable learning signals.
TIMEOUT_PENALTY = 0.0  # No penalty for truncation (let core components handle it)
STAGNATION_TIMEOUT_PENALTY = (
    -20.0
)  # DEPRECATED (no longer applied, kept for compatibility)
STAGNATION_PROGRESS_THRESHOLD = (
    0.15  # DEPRECATED (no longer used, kept for compatibility)
)

# Legacy constant for backward compatibility (maps to hazard death)
# Note: Also reduced from -15 to -12 to encourage progress over oscillation
MINE_DEATH_PENALTY = HAZARD_DEATH_PENALTY

# Switch activation reward
# Rationale: CRITICAL MILESTONE reward (80% of completion) - ensures routes that
# reach the switch are significantly more valuable than routes that die halfway.
# This is the key differentiator between "almost" and "success".
#
# UPDATED 2025-12-17: Increased from 30.0 to 40.0 to make switch a major milestone.
# UPDATED 2025-12-18: Rebalanced for PBRS weight=40 (reduced from 80, increased from 20)
#
# With discovery PBRS weight=40:
# - 50% progress + death (no switch): +20 PBRS - 8 death = +12 (good risk)
# - Reach switch + death: +30 PBRS + 40 switch - 8 death = +62 (5.2x better!)
#
# This creates clear milestone hierarchy:
# - Complete (50 + 40 switch + 40 PBRS) = 130 (best)
# - Switch + death (40 switch + 30 PBRS - 8 death) = 62 (excellent milestone)
# - Half progress + death (20 PBRS - 8 death) = 12 (acceptable bold exploration)
# - Camp 16% = 6.4 PBRS - 1.5 time = 4.9 (poor, low value)
# - Stagnate <15% = PBRS - 20 penalty - 1.5 time ≈ -15 (worst)
#
# Agent learns: "Reaching switch is a major achievement worth pursuing even with death risk"
SWITCH_ACTIVATION_REWARD = 100.0


# =============================================================================
# PBRS CONSTANTS (Curriculum-Managed)
# =============================================================================
# Potential-Based Reward Shaping following Ng et al. (1999)
# F(s,s') = γ * Φ(s') - Φ(s) ensures policy invariance

# PBRS discount factor
# γ=0.99 provides unified signal: progress toward goal WITH implicit time pressure.
#
# CRITICAL: Must match PPO gamma for PBRS policy invariance (Ng et al. 1999).
# Policy invariance guarantee only holds when γ_PBRS = γ_PPO.
#
# Mathematical justification for γ=0.99:
# - Forward progress (A→B): F = 0.99·Φ(B) - Φ(A) (positive if B closer to goal)
# - Backtrack (B→A): F = 0.99·Φ(A) - Φ(B) (negative, loses progress + 1% discount)
# - Oscillation (A→B→A): Net = -0.01·(Φ(A)+Φ(B)) (GUARANTEED LOSS, prevents oscillation)
# - Stay still: F = -0.01·Φ(current) (small negative, implicit time pressure)
#
# Why γ=0.99 is superior to γ=1.0 + time_penalty:
# 1. UNIFIED SIGNAL: Single coherent objective (no conflicting pressures)
# 2. NATURAL ANTI-BACKTRACKING: Oscillation always costs more than staying still
# 3. POLICY INVARIANCE: Matches PPO gamma for theoretical guarantees
# 4. SIMPLER DEBUGGING: One reward component instead of two competing signals
# 5. SELF-TUNING URGENCY: Time pressure scales with potential magnitude
#
# The 1% discount per step creates implicit time pressure:
# - At potential Φ=0.5 (mid-progress): staying still costs -0.005/step
# - At potential Φ=0.1 (near goal): staying still costs -0.001/step
# This scales naturally with game progress without separate time penalty.
PBRS_GAMMA = 0.99

# NOTE: PBRS objective weight is now managed by RewardConfig with curriculum scaling:
# - Early training (0-1M steps): weight = 15.0 (strong guidance for small movements)
# - Mid training (1M-3M steps): weight = 10.0 (moderate guidance)
# - Late training (3M+ steps): weight = 5.0 (light shaping)
# Weights SIGNIFICANTLY increased (5×) to provide stronger gradients for long-horizon tasks.
# With frame skip (6-8px typical movement), this ensures meaningful PBRS signal.
# Kept here for backwards compatibility; RewardConfig overrides during training.
PBRS_OBJECTIVE_WEIGHT = (
    2.5  # Static default (increased from 0.3, not used during curriculum training)
)

# PBRS scaling for switch and exit phases
# Scale of 1.0 provides effective gradients while keeping shaping < terminal rewards
PBRS_SWITCH_DISTANCE_SCALE = 1.0
PBRS_EXIT_DISTANCE_SCALE = 1.0

# DEPRECATED: Path normalization factor (no longer used)
# The actual normalization uses direct path distance: Φ(s) = 1 - (distance / combined_path_distance)
# This constant exists for historical reasons but is NOT applied in the calculation.
# See PBRS_CALCULATION_VERIFICATION.md for details on the actual normalization formula.
#
# Actual normalization (in pbrs_potentials.py):
#   effective_normalization = max(800.0, combined_path_distance)
#   potential = 1.0 - (distance / effective_normalization)
#
# This gives clean, interpretable gradients:
# - With combined_path=1000px, weight=20: 12px forward = +0.24 potential = +0.16 PBRS
# - Gradient strength controlled entirely by objective_weight (20.0 in discovery)
PBRS_PATH_NORMALIZATION_FACTOR = (
    1.0  # DEPRECATED - not used in actual calculation, kept for backward compatibility
)

# Displacement gate threshold for PBRS
# DISABLED: Set to 0.0 to allow PBRS from first step.
# Previous 10px threshold created a "reward trap" near spawn:
# - Typical movement is 0.5-3px per action, requiring 3-20 actions to escape
# - No PBRS in neutral zone made oscillation appear optimal
# - True PBRS is policy-invariant: backtracking naturally penalized via potential decrease
PBRS_DISPLACEMENT_THRESHOLD = 0.0  # pixels (disabled)


# =============================================================================
# DEPRECATED CONSTANTS (Simplified Reward System)
# =============================================================================
# SIMPLIFIED 2025-12-15: The following components have been removed as they are
# redundant with PBRS. PBRS already provides:
# - Zero reward for stationary behavior (potential unchanged)
# - Zero net reward for oscillation (returning to same position)
# - Negative reward for backtracking (returning to higher-distance states)
# - Directional gradient (F(s,s') rewards distance reduction in any direction)
#
# These constants are kept for reference but no longer used in reward calculation.

# DEPRECATED: Ineffective action penalty (PBRS gives 0 for no movement)
INEFFECTIVE_ACTION_THRESHOLD = 0.35  # DEPRECATED
INEFFECTIVE_ACTION_PENALTY = -0.05  # DEPRECATED

# DEPRECATED: Oscillation detection (PBRS gives 0 net reward for oscillation)
OSCILLATION_WINDOW = 10  # DEPRECATED
NET_DISPLACEMENT_THRESHOLD = 8.0  # DEPRECATED
OSCILLATION_PENALTY = -0.05  # DEPRECATED

# DEPRECATED: Velocity alignment (PBRS gradient provides directional signal)
VELOCITY_ALIGNMENT_MIN_SPEED = 0.5  # DEPRECATED
VELOCITY_ALIGNMENT_WEIGHT = 0.15  # DEPRECATED
VELOCITY_ALIGNMENT_WEIGHT_RATIO = 0.0  # DEPRECATED


# =============================================================================
# DISABLED AUXILIARY COMPONENTS (2025-12-25)
# =============================================================================
# The following auxiliary shaping components have been disabled to focus learning
# on core objectives: level completion and speed optimization.
#
# DISABLED COMPONENTS:
# - Path waypoint bonuses: Redundant with PBRS distance reduction
#   * Waypoints provided discrete rewards for reaching turns/inflection points
#   * PBRS already rewards moving along optimal path through potential changes
#
# - Velocity alignment bonuses: PBRS gradient provides directional signal
#   * Attempted to reward moving in the "correct" direction
#   * PBRS naturally guides direction through distance reduction rewards
#
# - Exit direction bonus: Not needed with pure PBRS
#   * Post-waypoint guidance to continue in optimal direction
#   * Redundant with PBRS continuous gradient field
#
# - Completion approach bonus: PBRS provides sufficient gradient near goal
#   * Quadratic bonus in final 50px to encourage completion
#   * PBRS gradient naturally intensifies as distance decreases
#
# Rationale: These components added complexity without improving learning.
# PBRS (Potential-Based Reward Shaping) with shortest path distance already
# provides dense, continuous gradients that guide the agent toward completion.
# Removing auxiliary components simplifies the reward structure and allows the
# agent to focus on the core objective: reducing path distance to goal.
#
# Active reward components:
# 1. Terminal rewards: Completion (+50), Switch (+60), Death (curriculum -15 to 0)
# 2. PBRS: F(s,s') = γ * Φ(s') - Φ(s) for path distance reduction
# 3. Time penalty: Small curriculum-based penalty (-0.0005 to -0.015) for speed


# =============================================================================
# PATHFINDING HAZARD AVOIDANCE (for PBRS-based safe navigation)
# =============================================================================
# Applied during A* pathfinding to make paths near deadly hazards more expensive
# This makes PBRS naturally guide agent along safer paths while preserving policy invariance

# Mine hazard avoidance radius (pixels)
# Rationale: Paths within this distance of deadly mines incur cost penalty
# - Deadly toggle mines have radius ~4px (state 0)
# - Ninja has radius 10px
MINE_HAZARD_RADIUS = 48.0  # pixels (increased from 50.0)

# Mine hazard cost multiplier for A* pathfinding
# SIMPLIFIED 2025-12-15: Constant value (was curriculum-adaptive 50-90)
#
# This multiplier increases edge costs for paths within MINE_HAZARD_RADIUS (75px)
# of deadly mines during A* pathfinding. This shapes the PBRS gradient field to
# naturally guide agents along safer routes.
#
# Rationale for 5.0:
# - Low but meaningful: 5x path cost near mines provides avoidance signal
# - Doesn't force excessive detours when mines are on/near optimal path
# - Allows agent to take calculated risks when necessary
# - Balanced with average 6px movement per 4 frames for proportional gradient
#
# This is a pathfinding parameter, not a reward parameter. It affects which paths
# are considered "optimal", which then determines PBRS gradient direction.
MINE_HAZARD_COST_MULTIPLIER = (
    1.1  # Low but meaningful avoidance without excessive detours
)

# Only penalize deadly mines (state 0), not safe mines (state 1)
MINE_PENALIZE_DEADLY_ONLY = True


# =============================================================================
# REMOVED COMPONENTS (No Longer Used)
# =============================================================================
# The following components have been removed in favor of focused PBRS:
#
# REMOVED: Discrete achievement bonuses (violated policy invariance, replaced with dense PBRS)
# REMOVED: Episode length normalization (violated Markov property)
# REMOVED: Per-episode reward caps (violated Markov property)
# REMOVED: Physics discovery rewards (disabled for clean path focus)
# REMOVED: Exploration rewards (PBRS provides via potential gradients)
# REMOVED: Progress bonuses (redundant with PBRS)
# REMOVED: Backtrack penalties (PBRS handles automatically via potential decrease)
# REMOVED: Stagnation penalties (redundant with truncation)
# REMOVED: NOOP penalties (let PBRS guide, don't punish stillness)
# REMOVED: Buffer bonuses (gameplay mechanic, not learning signal)
# REMOVED: Momentum rewards (physics-based, not reward-based)
# REMOVED: Hazard proximity PBRS (death penalty is clearer)
# REMOVED: Impact risk PBRS (death penalty is clearer)
# REMOVED: Completion bonus (time penalty provides efficiency incentive)
#
# SIMPLIFIED 2025-12-15 (Focused PBRS):
# REMOVED: RND intrinsic rewards (exploration unnecessary with complete gradient field)
# REMOVED: Go-Explore checkpoints (gradient field has no local minima to escape)
# REMOVED: Ineffective action penalties (PBRS gives 0 for no movement)
# REMOVED: Oscillation penalties (PBRS gives 0 net reward for oscillation)
# REMOVED: Revisit penalties (PBRS penalizes returning to higher-distance states)
# REMOVED: Velocity alignment bonuses (PBRS gradient provides directional signal)
# REMOVED: Waypoint bonuses (PBRS rewards reaching any path position that reduces distance)
#
# Total: ~31 redundant components removed for focused distance-reduction objective
