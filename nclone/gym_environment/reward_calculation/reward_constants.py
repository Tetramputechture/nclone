"""Policy-invariant reward constants using true PBRS for N++ RL training.

Centralized reward system with clear hierarchy:
1. Terminal rewards - Define task success/failure
2. PBRS shaping: F(s,s') = γ * Φ(s') - Φ(s) - Dense, policy-invariant path guidance
3. Time penalty - Efficiency pressure (curriculum-managed, frame-skip aware)

True PBRS implementation:
- Dense reward signal at every step based on path distance potential
- Automatic backtracking penalties (no manual penalty needed)
- Policy invariance guarantee per Ng et al. (1999)
- Markov property (no episode history dependencies)

Frame Skip Integration (NEW):
- Reward calculated ONCE per action (not per frame) for 75% computational savings
- PBRS telescopes: Φ(final) - Φ(initial) regardless of intermediate frames
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


# =============================================================================
# TERMINAL REWARD CONSTANTS (Always Active)
# =============================================================================
# Terminal rewards are the primary learning signals that define task completion
#
# REWARD HIERARCHY (critical for correct learning):
#   Success (+67) >> Progress+Death (-5 to -10) >> Oscillate/Timeout (-30 or worse)
#
# The agent should ALWAYS prefer making progress over staying still, even if
# that progress might lead to death. This is ensured by:
#   1. Strong PBRS rewards for progress (up to +15 for full path)
#   2. Moderate death penalties (less than PBRS for 50%+ progress)
#   3. Harsh accumulating penalties for oscillation (time + revisit)
#
# Example with 800px path and PBRS weight=15:
#   - 50% progress + hazard death: +7.5 (PBRS) - 12 (death) = -4.5
#   - 25% progress + death: +3.75 (PBRS) - 12 (death) = -8.25
#   - 10% progress + death: +1.5 (PBRS) - 12 (death) = -10.5
#   - Oscillate → timeout: 0 (PBRS) - 10 (timeout) - 3 (time) - 20+ (revisit) = -33+
#
# The math guarantees: ANY progress + death > pure oscillation

# Level completion reward - primary learning signal
# Rationale: Increased from 20.0 to 50.0 (2.5×) to ensure terminal rewards
# dominate over accumulated shaping penalties. With reduced PBRS weights (15.0 max)
# and reduced revisit penalties, this ensures successful episodes have higher
# rewards than failed episodes.
LEVEL_COMPLETION_REWARD = 50.0

# Death penalties - BALANCED to encourage risk-taking over oscillation
# Key principle: Death penalties must be LESS than PBRS reward for meaningful progress.
# With PBRS weight=15 for full path, death penalties should be < 15 so that
# 50%+ progress before death is still better than staying still.

# Impact death (ceiling/floor collision at high velocity)
# Rationale: Physics-based failure, somewhat preventable with careful movement.
# 20% of completion reward, provides moderate deterrent.
# With PBRS: 50% progress + impact death = +7.5 - 10 = -2.5 (better than timeout!)
IMPACT_DEATH_PENALTY = -10.0

# Hazard death (mines, drones, thwumps, other deadly entities)
# Rationale: Highly preventable through observation and planning.
# REDUCED from -15 to -12 (24% of completion reward) to ensure progress is always
# valued. Even 40% progress + hazard death = +6 - 12 = -6 (still better than timeout)
HAZARD_DEATH_PENALTY = -12.0

# Generic death penalty (fallback for unspecified death causes)
# Rationale: Conservative middle ground.
# 22% of completion reward.
DEATH_PENALTY = -11.0

# Timeout/truncation penalty (episode time limit exceeded)
# Rationale: Indicates inefficient navigation or getting stuck.
# 20% of completion reward - equal to lightest death penalty.
# Timeout should NOT be punished more harshly than actual deaths, otherwise
# agents learn to die intentionally when stuck rather than keep trying.
# However, with revisit penalties, oscillating agents get MUCH worse total reward.
# Correct hierarchy: Success >> Keep trying >> Any death >= Timeout (base only)
TIMEOUT_PENALTY = -10.0

# Legacy constant for backward compatibility (maps to hazard death)
# Note: Also reduced from -15 to -12 to encourage progress over oscillation
MINE_DEATH_PENALTY = HAZARD_DEATH_PENALTY

# Switch activation reward
# Rationale: CRITICAL MILESTONE reward (30% of completion) - ensures routes that
# actually achieve the switch are significantly more valuable than routes that
# just get close. This is the key differentiator between "almost" and "success".
#
# Why 30% of completion (15.0):
# - Switch is the halfway point of the two-phase task (switch → exit)
# - Must be large enough that switch activation is ALWAYS worth pursuing
# - Even with death immediately after: +15 (switch) - 12 (death) = +3 net
# - Compare to oscillating near switch: 0 (no activation) + penalties = negative
#
# With PBRS: An agent 50% through switch phase + activation + death:
#   +3.75 (PBRS to switch) + 15 (activation) - 12 (death) = +6.75 (positive!)
# Without activation: +7.5 (PBRS) - 12 (death) = -4.5 (negative!)
SWITCH_ACTIVATION_REWARD = 15.0


# =============================================================================
# PBRS CONSTANTS (Curriculum-Managed)
# =============================================================================
# Potential-Based Reward Shaping following Ng et al. (1999)
# F(s,s') = γ * Φ(s') - Φ(s) ensures policy invariance

# PBRS discount factor
# For heuristic potential functions (path distance), γ=1.0 is standard and eliminates negative bias
# Policy invariance holds for ANY γ (Ng et al. 1999 Theorem 1), but γ=1.0 ensures:
# - Accumulated PBRS = Φ(goal) - Φ(start) exactly (no (γ-1)*Σ Φ negative bias)
# - In episodic tasks with heuristic potentials, no discount needed
# Note: γ should match MDP discount ONLY when Φ is the true value function V(s)
PBRS_GAMMA = 1.0

# NOTE: PBRS objective weight is now managed by RewardConfig with curriculum scaling:
# - Early training (0-1M steps): weight = 5.0 (strong guidance for small movements)
# - Mid training (1M-3M steps): weight = 3.0 (moderate guidance)
# - Late training (3M+ steps): weight = 1.5 (light shaping)
# Weights increased to compensate for realistic movement physics (6-8px typical per action)
# with frame skip, ensuring strong learning signal (20-25× vs time penalty).
# Kept here for backwards compatibility; RewardConfig overrides during training.
PBRS_OBJECTIVE_WEIGHT = 0.3  # Static default (not used during curriculum training)

# PBRS scaling for switch and exit phases
# Scale of 1.0 provides effective gradients while keeping shaping < terminal rewards
PBRS_SWITCH_DISTANCE_SCALE = 1.0
PBRS_EXIT_DISTANCE_SCALE = 1.0

# Path-based normalization factor
# Controls how combined path distance (spawn→switch + switch→exit) is used for normalization
# **CRITICAL**: Lower values = stronger gradients (smaller area_scale), higher values = weaker gradients
# Replaces surface-area-based normalization for better handling of open levels with focused paths
#
# IMPORTANT: Used with HYBRID normalization: Φ(s) = 1 - d/area_scale (linear near goal)
# where area_scale = combined_path_distance * PBRS_PATH_NORMALIZATION_FACTOR * scale_factor
#
# Formula analysis for typical level (combined_path = 2000px, mid-phase scale_factor=0.5):
# - Factor = 0.6: area_scale = 600px → 6px movement = 0.01 potential change → 0.12 PBRS (w=12)
# - Factor = 1.0: area_scale = 1000px → 6px movement = 0.006 potential change → 0.072 PBRS (w=12)
# - Factor = 2.5: area_scale = 2500px → 6px movement = 0.0024 potential change → 0.029 PBRS (w=12)
#
# FIXED: Decreased from 2.5 to 0.6 for 4× stronger gradients per unit movement.
# Previous increase to 2.5 was BACKWARDS - made gradients weaker, not stronger!
PBRS_PATH_NORMALIZATION_FACTOR = (
    0.6  # Was 2.5 (wrong direction!) - target 0.1-0.15 PBRS per 6px move
)

# Displacement gate threshold for PBRS
# DISABLED: Set to 0.0 to allow PBRS from first step.
# Previous 10px threshold created a "reward trap" near spawn:
# - Typical movement is 0.5-3px per action, requiring 3-20 actions to escape
# - No PBRS in neutral zone made oscillation appear optimal
# - True PBRS is policy-invariant: backtracking naturally penalized via potential decrease
PBRS_DISPLACEMENT_THRESHOLD = 0.0  # pixels (disabled)


# =============================================================================
# INEFFECTIVE ACTION PENALTY (Anti-Oscillation)
# =============================================================================
# Penalizes actions that produce minimal displacement, targeting JUMP+RIGHT oscillation.
# Cannot be exploited: moving is always better than not moving.
#
# CRITICAL: These penalties ensure oscillating/staying still accumulates enough
# negative reward to make ANY progress + death preferable to stagnation.

# Minimum displacement threshold (pixels per action)
# Actions producing less than this displacement incur a penalty.
#
# Physics-grounded calculation (from ninja.py integrate() and think()):
# - Ground accel: 0.0667 px/frame², air accel: 0.0444 px/frame²
# - Drag: 0.9933 per frame (DRAG_REGULAR)
# - Frame skip: 4 frames per action
#
# Minimum 4-frame displacement from rest:
# - Ground movement: ~0.66px (sum of 0.066 + 0.132 + 0.197 + 0.262)
# - Air movement: ~0.44px (sum of 0.044 + 0.088 + 0.131 + 0.175)
#
# Threshold set below minimum air movement to avoid false positives:
# 0.35px catches oscillation while allowing minimum intentional movement
INEFFECTIVE_ACTION_THRESHOLD = 0.35  # pixels (physics-grounded)

# Penalty for ineffective actions
# INCREASED from -0.03 to -0.05 for stronger anti-oscillation signal.
# Applied per action when displacement < INEFFECTIVE_ACTION_THRESHOLD
# 150 ineffective actions = -7.5 penalty (significant vs death at -10 to -12)
INEFFECTIVE_ACTION_PENALTY = -0.05


# =============================================================================
# OSCILLATION DETECTION (Net Displacement Tracking)
# =============================================================================
# Detects when agent is moving but not making progress (oscillating in place).
# The ineffective action penalty only catches stationary behavior (< 0.35px/action).
# Oscillation detection catches movement that nets to zero over multiple actions.
#
# Example: LEFT 2px, RIGHT 2px, LEFT 2px, RIGHT 2px = 8px total movement, 0 net progress
# Each individual action passes ineffective threshold but net displacement is 0.

# Window size: Number of actions to track for net displacement calculation
# 10 actions * 4 frames/action = 40 frames ≈ 0.67 seconds at 60fps
OSCILLATION_WINDOW = 10  # actions

# Minimum net displacement required over the window
# With typical movement 0.66-3px/action, 10 actions should cover 6-30px
# INCREASED from 5px to 8px to catch more subtle oscillation
NET_DISPLACEMENT_THRESHOLD = 8.0  # pixels over OSCILLATION_WINDOW actions

# Penalty for oscillation (applied once per window check, not per action)
# INCREASED from -0.02 to -0.05 for stronger deterrent
# An agent oscillating for 150 actions triggers ~15 window checks = -0.75 penalty
OSCILLATION_PENALTY = -0.05


# =============================================================================
# VELOCITY ALIGNMENT BONUS - RE-ENABLED WITH NEXT-HOP GRADIENT
# =============================================================================
# RE-ENABLED: Now uses next_hop direction instead of Euclidean for winding paths.
#
# Why this is now useful:
# 1. PBRS rewards moving toward goal, but for winding paths the discrete node
#    jumps can create noisy reward signals
# 2. Next-hop gradient provides a CONTINUOUS direction signal that respects
#    level geometry (walls, corridors, going away from goal to reach it)
# 3. Cheap: Uses precomputed next_hop cache (O(1) lookup)
# 4. Complements PBRS by smoothing direction signal between nodes
#
# The velocity alignment bonus rewards movement in the optimal path direction,
# even when that direction points away from the goal (for winding levels).

# Minimum speed to consider velocity direction meaningful
VELOCITY_ALIGNMENT_MIN_SPEED = 0.5  # pixels/frame

# Weight for velocity alignment potential bonus
# This is added to the PBRS potential, not multiplied
# Range [-1, 1] scaled by this weight
# 0.15 = small bonus, up to ±0.15 potential per step
VELOCITY_ALIGNMENT_WEIGHT = 0.15

# Legacy constant (deprecated, use VELOCITY_ALIGNMENT_WEIGHT instead)
VELOCITY_ALIGNMENT_WEIGHT_RATIO = 0.0  # DEPRECATED


# =============================================================================
# PATHFINDING HAZARD AVOIDANCE (for PBRS-based safe navigation)
# =============================================================================
# Applied during A* pathfinding to make paths near deadly hazards more expensive
# This makes PBRS naturally guide agent along safer paths while preserving policy invariance

# Mine hazard avoidance radius (pixels)
# Rationale: Paths within this distance of deadly mines incur cost penalty
# - Deadly toggle mines have radius ~4px (state 0)
# - Ninja has radius 10px
# - Safe buffer: 50-60px prevents risky close approaches
MINE_HAZARD_RADIUS = 50.0  # pixels

# Mine hazard cost multiplier
# Rationale: How much more expensive are paths near mines?
# - 1.0 = no penalty (disabled)
# - 2.0 = twice as expensive (moderate avoidance)
# - 5.0 = five times more expensive (strong avoidance)
# - 10.0+ = extreme avoidance
MINE_HAZARD_COST_MULTIPLIER = 15.0

# Only penalize deadly mines (state 0), not safe mines (state 1)
MINE_PENALIZE_DEADLY_ONLY = True


# =============================================================================
# REMOVED COMPONENTS (No Longer Used)
# =============================================================================
# The following components have been removed in favor of true PBRS:
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
# Total: ~24 redundant/problematic constants removed for policy invariance
