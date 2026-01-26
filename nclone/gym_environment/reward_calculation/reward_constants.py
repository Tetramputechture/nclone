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
# UPDATED 2026-01-21: Set to 1.0 to rely on PPO's native advantage normalization.
#
# RATIONALE FOR REMOVING 0.1 SCALING:
# 1. PPO already normalizes advantages by (advantage - mean) / std per batch
# 2. The 0.1 multiplier added unnecessary complexity and made TensorBoard harder to interpret
# 3. With curriculum-scaled terminal rewards, raw values are now more consistent (100-200 range)
# 4. Value function capacity increased (256 width) can handle slightly larger targets
# 5. May have interfered with RND intrinsic reward scaling (different magnitude order)
#
# WHAT THIS CHANGES:
# - Episode rewards (logged): now show actual values (50-200 vs 5-20)
# - Value estimates: larger magnitude (10-20 vs 1-2)
# - Value loss: proportionally larger (but relative to targets, so same difficulty)
#
# WHAT STAYS THE SAME:
# - Learning dynamics: PPO is scale-invariant due to advantage normalization
# - Optimal policy: unchanged (argmax doesn't depend on reward scale)
# - Relative importance: completion(200) >> switch(100) >> death(-80/-50/-20)
# - Convergence rate: determined by advantage gradients, not raw reward magnitude
#
# If value function training becomes unstable, consider:
# - Increasing vf_coef (e.g., 1.5 → 2.0) to strengthen value learning
# - Using separate value network architecture (code change required)
# - Reverting to 0.1 scaling as fallback
#
# REVERTED 2026-01-24: Analysis of failed training run revealed value function explosion:
# - Value loss mean: 390.7 (should be <100)
# - Value loss max: 91,614 (catastrophic spike)
# - Coefficient of variation: 9.11 (extreme instability)
# Root cause: Value network calibrated for 0.1 scale, 10x jump caused miscalibration.
# Advantages became garbage, policy couldn't learn (entropy stuck at 1.066, no convergence).
#
# Path forward: Use proven 0.1 scale. Future scaling must be gradual: 0.1→0.3→0.5→1.0
# with fine-tuning at each step to let value function adapt.
GLOBAL_REWARD_SCALE = 0.1  # Proven scale - value function calibrated for this

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
#   Success (+340) >> High Progress+Timeout (+8) >> Progress+Death (+2) >> Quick Death (-30)
#
# UPDATED 2025-12-29: Rebalanced for faster convergence in advanced phase (40%+ success)
#   1. PBRS weight curve flattened (50→8 instead of 50→5) to maintain gradient signal
#   2. Death penalty kept meaningful in advanced phase (-4 to -3 instead of -2)
#   3. Time penalty re-enabled for speed optimization (-0.001 to -0.003 by phase)
#   4. Stronger PBRS throughout training accelerates policy convergence
#
# Example with DISCOVERY phase (PBRS=40, death=-30, time=0):
#   - Complete: +40 (PBRS) + 200 (completion) + 100 (switch) = +340 (BEST)
#   - 90% + timeout: +36 (PBRS) + 100 (switch) = +136 (good milestone)
#   - 50% progress + death: +20 (PBRS) - 30 (death) = -10 (bad - need >75% to justify)
#   - Quick death (10% progress): +4 (PBRS) - 30 (death) = -26 (WORST)
#
# Example with ADVANCED phase (PBRS=12, death=-4, time=-0.002/step):
#   - Complete: +12 (PBRS) + 200 (completion) + 100 (switch) - 1.0 (time) = +311 (BEST)
#   - 90% + timeout: +10.8 (PBRS) + 100 (switch) - 1.0 (time) = +109.8 (good)
#   - 50% progress + death: +6 (PBRS) - 4 (death) - 0.5 (time) = +1.5 (risky but acceptable)
#   - Quick death (10% progress): +1.2 (PBRS) - 4 (death) - 0.2 (time) = -3.0 (bad)
#
# The math guarantees: Completion > Progress > Death, with meaningful tradeoffs preserved

# Level completion reward - primary learning signal
# Rationale: Set at 200.0 to ensure terminal rewards strongly dominate over
# accumulated shaping signals. With PBRS weights ranging from 40 (discovery) to
# 8 (mastery), completion reward is 5-25× larger than maximum PBRS contribution,
# ensuring successful episodes are always strongly preferred over failed episodes
# while maintaining meaningful dense gradient signal throughout training.
LEVEL_COMPLETION_REWARD = 200.0

# Death penalty - CURRICULUM-ADAPTIVE (scaled with agent competence)
#
# UPDATED 2025-12-29: Maintained meaningful penalty in advanced phase for faster convergence.
#
# Previous approach of dropping to -2 at 40%+ success made death effectively costless
# (only 1% of completion reward), eliminating risk/reward tradeoffs and slowing convergence.
# Analysis showed this contributed to slow optimization in advanced training phase.
#
# New curriculum-based penalties maintain strategic risk consideration throughout:
# - Discovery (<5%): -30 penalty strongly deters "die quickly" strategy
# - Early (5-20%): -10 penalty continues favoring safe completion during learning
# - Mid (20-40%): -5 penalty reduces as agent demonstrates competence
# - Advanced (40-60%): -4 penalty maintains meaningful risk (requires ~33% progress to justify)
# - Mastery (60%+): -3 penalty preserves strategic consideration (requires ~25% progress)
#
# This balances encouraging bold play with maintaining clear incentive structures.
# With PBRS weights of 8-12 in advanced phase, agent must achieve substantial progress
# to justify death risk, creating meaningful optimization pressure.
#
# NOTE: These are BASE values. Actual penalties calculated in RewardConfig
# based on real-time success rate tracking.

# Death penalties are now managed by RewardConfig.death_penalty property
# These constants serve as reference values for the curriculum
DEATH_PENALTY_DISCOVERY = -30.0  # <5% success: strong deterrent against quick death
DEATH_PENALTY_EARLY = -10.0  # 5-20% success: lighter deterrent as skills improve
DEATH_PENALTY_MID = -5.0  # 20-40% success: minimal deterrent, agent showing competence
DEATH_PENALTY_ADVANCED = -4.0  # 40-60% success: still meaningful risk consideration
DEATH_PENALTY_MASTERY = -3.0  # 60%+ success: maintains strategic risk-taking

# Legacy constants for backward compatibility (use RewardConfig instead)
DEATH_PENALTY = -30.0  # Default to discovery phase
IMPACT_DEATH_PENALTY = -30.0
HAZARD_DEATH_PENALTY = -30.0

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
# Rationale: CRITICAL MILESTONE reward (50% of completion) - ensures routes that
# reach the switch are significantly more valuable than routes that die halfway.
# This is the key differentiator between "almost" and "success".
#
# UPDATED 2025-12-29: Rebalanced examples for new PBRS weight curve (40→8 by phase)
#
# With discovery phase (PBRS=40, death=-30, time=0):
# - 50% progress + death (no switch): +20 PBRS - 30 death = -10 (bad - need 75%+ to justify)
# - Reach switch + death: +30 PBRS + 100 switch - 30 death = +100 (5x better!)
# - Complete: +40 PBRS + 100 switch + 200 completion = +340 (best)
#
# With advanced phase (PBRS=12, death=-4, time=-0.002/step):
# - 50% progress + death: +6 PBRS - 4 death - 0.5 time = +1.5 (risky but acceptable)
# - Reach switch + death: +9 PBRS + 100 switch - 4 death - 0.75 time = +104.25 (excellent!)
# - Complete: +12 PBRS + 100 switch + 200 completion - 1.0 time = +311 (best)
#
# This creates clear milestone hierarchy where reaching switch is always a major achievement
# worth pursuing, while maintaining meaningful risk/reward tradeoffs throughout training.
# Agent learns: "Reaching switch is valuable even with death risk, but completion is best"
SWITCH_ACTIVATION_REWARD = 100.0

# Milestone rewards for curriculum-based progress
# ADDED 2026-01-02: Distance-based milestones replace pre-computed waypoints
# Rewards are given when shortest path distance decreases by >= SUB_GOAL_SPACING
#
# Design Philosophy:
# - Distance-based: No pre-computation needed, uses PBRS path distances
# - Curriculum-aware: Automatically uses curriculum goal positions
# - One-time per milestone: Distance threshold prevents re-collection
# - Reward hierarchy: Milestones (30) < Switch (100) < Completion (200)
#
# Example progression (500px total path, 48px spacing):
# - Start: 500px from goal
# - Reduce to 452px: +30 reward (first milestone)
# - Reduce to 404px: +30 reward (second milestone)
# - Continue every 48px reduction
# - Typical episode: 10-15 milestones = 300-450 total reward
#
# Benefits over pre-computed waypoints:
# - No path recomputation when curriculum changes
# - Rewards any path that makes progress (not just optimal)
# - Simpler implementation (~500 lines removed)
# - Uses existing cached distances (no extra overhead)
SUB_GOAL_REWARD_PROGRESS = 30.0  # Distance milestone reward (3x baseline)
SUB_GOAL_SPACING = 48.0  # Distance reduction threshold for milestone (pixels)

# Waypoint collection rewards (RE-ENABLED 2026-01-03, ENHANCED 2026-01-09)
# Provides guidance through non-linear optimal paths where agent must move away
# from goal first (e.g., drop down to ramp, then jump up-right to exit).
#
# CRITICAL INSIGHT: Pure PBRS with path distance is INSUFFICIENT for detours because:
# 1. Shortcuts temporarily reduce path distance (positive PBRS signal)
# 2. Death penalty comes too late (agent already learned wrong pattern)
# 3. Sequential waypoints provide IMMEDIATE, UNAMBIGUOUS guidance for correct path
#
# SEQUENTIAL COLLECTION BONUS (NEW 2026-01-09):
# - Base reward for any waypoint collection: 0.5
# - Sequential bonus for in-order collection: base × (1.0 + streak × 0.1)
# - Out-of-order collection: base × 0.3 (discourages shortcuts)
# - Collection radius: 18.0 pixels (1.5 × sub-node size for reliable collection)
#
# Example rewards (base = 1.5, INCREASED 2026-01-10 to combat oscillation):
# - Waypoint 0 (in-sequence, streak=1): 1.5 × 1.2 = 1.80
# - Waypoint 1 (in-sequence, streak=2): 1.5 × 1.4 = 2.10
# - Waypoint 2 (in-sequence, streak=3): 1.5 × 1.6 = 2.40
# - Waypoint 5 (skipped 3,4, out-of-order): 1.5 × 0.3 = 0.45
#
# With ~30 waypoints collected in-sequence = ~60 total bonus (strong path-following incentive)
# INCREASED: Stronger waypoint rewards help break oscillation local minima by providing
# clear gradient toward progress. Agent gets immediate positive feedback for following
# the correct path, making exploration more directed.
# REVERTED 2026-01-24: Back to 3 since GLOBAL_REWARD_SCALE reverted to 0.1
# The 30 value was correct for 1.0 scale but not needed with 0.1 scale.
WAYPOINT_BASE_BONUS = 3  # Base reward for waypoint collection (3x increase from 0.5)
WAYPOINT_COLLECTION_RADIUS = (
    18.0  # pixels - 1.5 × sub-node size for reliable collection
)
WAYPOINT_SEQUENCE_MULTIPLIER = (
    0.2  # Streak bonus per consecutive in-order collection (2x increase from 0.1)
)
WAYPOINT_OUT_OF_ORDER_SCALE = 0.3  # Reduced reward for out-of-order collection

# Legacy constants (deprecated, kept for compatibility)
WAYPOINT_COLLECTION_REWARD = 8.0  # DEPRECATED - use WAYPOINT_BASE_BONUS instead
WAYPOINT_PROGRESS_SCALE_BASE = 2.0  # DEPRECATED - use sequential system instead


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
# UPDATED 2025-12-29: Flattened weight curve to maintain gradient signal in advanced phase
# - Discovery (0-5%): weight = 40.0 (strong but stable guidance)
# - Early (5-20%): weight = 25.0 (stronger learning signal)
# - Mid (20-40%): weight = 18.0 (maintained gradient strength)
# - Advanced (40-60%): weight = 12.0 (still meaningful for optimization)
# - Mastery (60%+): weight = 8.0 (preserved gradient signal)
# Flatter curve prevents dense shaping signal from being drowned out by terminal rewards,
# accelerating convergence in advanced training phase. With frame skip (6-8px typical movement),
# this ensures meaningful PBRS signal throughout training.
# Kept here for backwards compatibility; RewardConfig overrides during training.
PBRS_OBJECTIVE_WEIGHT = (
    2.5  # Static default (increased from 0.3, not used during curriculum training)
)

# PBRS scaling for switch and exit phases
# Scale of 1.0 provides effective gradients while keeping shaping < terminal rewards
PBRS_SWITCH_DISTANCE_SCALE = 1.0
PBRS_EXIT_DISTANCE_SCALE = 1.0

# PBRS Gradient Scaling (maintains consistent per-step signal across path lengths)
# ADDED 2026-01-04: Adaptive weight scaling to maintain effective gradients on long paths
#
# Problem: Linear normalization Φ(d) = 1 - d/k causes gradient decay with path length:
# - 200px path: 12px movement = 0.06 potential change (strong signal)
# - 2000px path: 12px movement = 0.006 potential change (10x weaker!)
#
# Solution: Scale PBRS weight by sqrt(path_length / reference) to compensate:
# - 200px: scale = 1.0x (baseline)
# - 500px: scale = 1.58x (moderate boost)
# - 1000px: scale = 2.24x (significant boost)
# - 2000px: scale = 3.16x (strong boost)
# - 3000px: scale = 3.87x (late curriculum stages)
#
# UPDATED 2026-01-12: Increased cap from 5.0 to 7.0 for late-stage curriculum training.
# At stage 8+ on complex levels (e.g., "006 ramp jumping"), combined path distances
# can reach 1000-1200+ pixels. The 7.0 cap allows sqrt(1200/200) = 2.45x scaling
# with headroom, maintaining strong per-step gradients on long non-linear paths while
# still preventing instability from unbounded scaling on extreme outliers.
PBRS_GRADIENT_REFERENCE_DISTANCE = 200.0  # Reference path length for base gradient
PBRS_GRADIENT_SCALE_CAP = 7.0  # Maximum scaling factor (increased for late curriculum)

# DEPRECATED: Path normalization factor (no longer used)
# The actual normalization uses LINEAR potential: Φ(d) = 1 - d/k
# where k is the characteristic distance (phase-dependent, see below).
# This constant exists for historical reasons but is NOT applied in the calculation.
#
# Actual normalization (in pbrs_potentials.py):
#   UPDATED 2026-01-03: SWITCHED FROM HYPERBOLIC TO LINEAR for uniform gradient strength
#   FIXED 2026-01-09: Post-switch phase uses combined_distance to prevent clamping
#
# Pre-switch phase (Φ ∈ [0, 0.5]):
#   k = max(200.0, spawn_to_switch_distance)
#   potential_raw = max(0, 1.0 - distance / k)
#   potential = 0.5 * potential_raw
#
# Post-switch phase (Φ ∈ [0.5, 1.0]):
#   k = max(50.0, 1.5 * (spawn_to_switch_distance + switch_to_exit_distance))  # FIXED: 1.5x combined
#   potential_raw = max(0, 1.0 - distance / k)
#   potential = 0.5 + 0.5 * potential_raw
#   1.5x multiplier accounts for alternative paths and exploration beyond optimal envelope
#
# WHY COMBINED DISTANCE FOR POST-SWITCH:
# Agent can be anywhere after switch activation (often near spawn due to detours).
# Using switch_to_exit_distance alone caused clamping when agent was far from exit:
#   - Agent at spawn (800px from exit), switch_to_exit = 200px
#   - potential_raw = 1.0 - (800/200) = -3.0 → clamped to 0.0
#   - Result: Zero gradient, causing oscillation/meandering
# Combined distance ensures valid potential from any reachable position.
#
# Linear potential provides UNIFORM gradients (each part of path matters equally):
#   dΦ/dd = -1/k everywhere (constant!)
#
# Fallback defaults:
#   - spawn_to_switch_distance: 1000.0 (fallback for missing data)
#   - switch_to_exit_distance: 1000.0 (fallback for missing data)
#   - MIN_SCALE: 50.0 (reduced from 200 to support short curriculum paths, min 24px)
#
# Gradient examples with weight=40 (discovery phase):
# - Small level (k=500px): 12px forward = +0.012 potential = +0.48 PBRS (strong!)
# - Large level (k=2000px): 12px forward = +0.003 potential = +0.12 PBRS (appropriate)
# - Gradient UNIFORM across entire path (same at spawn, halfway, and near goal)
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
# DISABLED AUXILIARY COMPONENTS (2025-12-25, UPDATED 2026-01-09)
# =============================================================================
# The following auxiliary shaping components have been disabled to focus learning
# on core objectives: level completion and speed optimization.
#
# DISABLED COMPONENTS:
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
# RE-ENABLED COMPONENTS (2026-01-09):
# - Sequential waypoint bonuses: CRITICAL for non-linear paths with detours
#   * CORRECTED UNDERSTANDING: Pure PBRS fails on detours because shortcuts
#     temporarily reduce path distance before causing death
#   * Sequential bonuses provide IMMEDIATE positive signal for correct routing
#   * Not redundant - they solve the "shortcut then die" local minimum
#   * See reward_constants.py lines 228-254 for sequential bonus design
#
# Active reward components:
# 1. Terminal rewards: Completion (+200), Switch (+100), Death (curriculum -30 to -3)
# 2. PBRS: F(s,s') = γ * Φ(s') - Φ(s) for path distance reduction
# 3. Sequential waypoint bonuses: Immediate guidance for non-linear paths (RE-ENABLED)
# 4. Time penalty: Small curriculum-based penalty (-0.0005 to -0.015) for speed


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
#
# RE-ENABLED 2026-01-09:
# RESTORED: Sequential waypoint bonuses (CRITICAL for non-linear detour paths)
#   * Original removal was based on incorrect assumption that PBRS alone suffices
#   * Detour paths create "shortcut then die" local minimum that PBRS cannot solve
#   * Sequential bonuses provide immediate positive signal for correct routing
#
# Total: ~30 redundant components removed for focused distance-reduction objective
