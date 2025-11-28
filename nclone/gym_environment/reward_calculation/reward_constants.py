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
# TERMINAL REWARD CONSTANTS (Always Active)
# =============================================================================
# Terminal rewards are the primary learning signals that define task completion

# Level completion reward - primary learning signal
# Rationale: Large reward (20.0) strongly incentivizes level completion.
# PBRS shaping provides dense guidance at every step (F(s,s') formula) to guide HOW
# to reach the goal. Terminal reward defines WHAT success means.
LEVEL_COMPLETION_REWARD = 20.0

# Death penalties - REBALANCED to match new PBRS magnitude
# With PBRS weights of 6.0-20.0, PBRS can contribute ±6 to ±20 per episode.
# Death penalties must be significant relative to this to deter risky behavior.

# Impact death (ceiling/floor collision at high velocity)
# Rationale: Physics-based failure, somewhat preventable with careful movement.
# 20% of completion reward, provides moderate deterrent.
IMPACT_DEATH_PENALTY = -4.0

# Hazard death (mines, drones, thwumps, other deadly entities)
# Rationale: Highly preventable through observation and planning.
# 30% of completion reward, strong deterrent for reckless navigation.
HAZARD_DEATH_PENALTY = -6.0

# Generic death penalty (fallback for unspecified death causes)
# Rationale: Conservative middle ground between impact and hazard.
# 25% of completion reward.
DEATH_PENALTY = -5.0

# Timeout/truncation penalty (episode time limit exceeded)
# Rationale: Indicates inefficient navigation or getting stuck.
# 35% of completion reward - strongest penalty to discourage wasting time.
# Timeout is entirely preventable with efficient pathing.
TIMEOUT_PENALTY = -7.0

# Legacy constant for backward compatibility (maps to hazard death)
MINE_DEATH_PENALTY = HAZARD_DEATH_PENALTY

# Switch activation reward
# Rationale: Milestone reward (10% of completion) provides intermediate signal
# for two-phase task (switch → exit).
SWITCH_ACTIVATION_REWARD = 2.0


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


# =============================================================================
# PATHFINDING HAZARD AVOIDANCE (for PBRS-based safe navigation)
# =============================================================================
# Applied during A* pathfinding to make paths near deadly hazards more expensive
# This makes PBRS naturally guide agent along safer paths while preserving policy invariance

# Mine hazard avoidance radius (pixels)
# Rationale: Paths within this distance of deadly mines incur cost penalty
# - Deadly toggle mines have radius ~4px (state 0)
# - Ninja has radius 10px
# - Safe buffer: 30-50px prevents risky close approaches
MINE_HAZARD_RADIUS = 40.0  # pixels

# Mine hazard cost multiplier
# Rationale: How much more expensive are paths near mines?
# - 1.0 = no penalty (disabled)
# - 2.0 = twice as expensive (moderate avoidance)
# - 5.0 = five times more expensive (strong avoidance)
# - 10.0 = ten times more expensive (extreme avoidance)
# Start with 3.0 for balanced risk/reward tradeoff
MINE_HAZARD_COST_MULTIPLIER = 10.0

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
