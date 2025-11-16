"""Centralized reward constants and configuration for N++ RL training.

This module defines all reward-related constants following RL best practices:
1. All magic numbers are eliminated and replaced with named constants
2. Each constant includes documentation explaining its purpose and rationale
3. Constants are grouped by reward component for clarity
4. Values are based on research and empirical tuning

References:
- Ng et al. (1999): "Policy Invariance Under Reward Transformations" (PBRS theory)
- Sutton & Barto (2018): "Reinforcement Learning: An Introduction" (reward scaling)
"""

from ..constants import LEVEL_DIAGONAL


# =============================================================================
# TERMINAL REWARD CONSTANTS
# =============================================================================
# Terminal rewards are the primary learning signals that define task completion

# Level completion reward - primary learning signal
# Rationale: Large enough (20.0) to strongly incentivize efficient, shortest-time completion.
# PBRS provides dense shaping rewards (~1.5 accumulated over episode) to guide HOW to reach
# the goal, while the completion reward motivates COMPLETING it quickly. With completion
# reward at 20.0, agents have strong incentive to optimize for speed rather than maximizing
# per-step rewards. Even slow completions (10k steps) result in positive returns
# (20.0 - 1.0 = +19.0), but fast completions are significantly more rewarding.
LEVEL_COMPLETION_REWARD = 20.0

# Death penalty
# Rationale: Moderate negative reward (-2.0) discourages death without dominating learning.
# Increased to 10% of completion reward (was 5%) to better discourage death while still
# allowing learning. Also applied to truncation (timeout) to strongly discourage it.
# Moderate penalties prevent overly conservative behavior while guiding toward completion.
DEATH_PENALTY = -2.0

# Mine death penalty
# Rationale: Stronger penalty (-3.0) for mine deaths to encourage hazard avoidance.
# Increased to 15% of completion reward (was 12.5%). Mine deaths are preventable through
# better path planning, so they deserve stronger negative signal than other death types.
MINE_DEATH_PENALTY = -3.0


# =============================================================================
# OBJECTIVE MILESTONE REWARDS
# =============================================================================
# Intermediate rewards for achieving subgoals

# Switch activation reward
# Rationale: Milestone reward (10% of completion) provides intermediate signal
# without creating local optima that distract from the ultimate goal.
SWITCH_ACTIVATION_REWARD = 2.0


# =============================================================================
# TIME-BASED REWARDS
# =============================================================================
# Per-step rewards that encourage efficiency

# Time penalty per step (default/fixed mode)
# Rationale: Strong efficiency incentive creates 3-4 point difference between
# fast and slow completion. UPDATED to -0.003 per step (300x increase):
#   Fast completion (400 steps): -1.2 total
#   Slow completion (1500 steps): -4.5 total
#   Difference: 3.3 points, strongly incentivizes speed
# PBRS provides gradient for completion, time penalty provides pressure for efficiency.
# The combination enables both general learning AND speed optimization.
# Terminal reward (+20.0) still dominates, maintaining learning stability.
TIME_PENALTY_PER_STEP = -0.003  # was -0.00001 (300x increase)

# Progressive time penalty schedule (for speed optimization)
# Rationale: Allows early exploration while increasing pressure for efficiency
# over episode duration. Use if fixed penalty prevents learning.
# - Early phase: minimal penalty, encourage exploration
# - Middle phase: moderate penalty, find solutions
# - Late phase: high penalty, optimize routes
TIME_PENALTY_EARLY = -0.001  # Steps 0-30%: exploration phase (20x increase)
TIME_PENALTY_MIDDLE = -0.003  # Steps 30-70%: solution phase (15x increase)
TIME_PENALTY_LATE = -0.006  # Steps 70-100%: optimization phase (12x increase)

# Phase thresholds (as fraction of max episode length)
TIME_PENALTY_EARLY_THRESHOLD = 0.3  # 30% of episode
TIME_PENALTY_LATE_THRESHOLD = 0.7  # 70% of episode

# Completion time bonus (for fine-tuning speed optimization)
# Rationale: Explicitly rewards fast completion without punishing slow solutions.
# Bonus linearly decreases from maximum to zero as completion time increases.
# Compatible with curriculum: train for completion first, then fine-tune for speed.
COMPLETION_TIME_BONUS_MAX = 2.0  # Maximum bonus (instant completion)
COMPLETION_TIME_TARGET = (
    5000  # Target steps for full bonus (adjust per level difficulty)
)

# NOOP action penalty
# Rationale: Moderate penalty (-0.05) discourages standing still without overwhelming
# other reward signals. Encourages active exploration and movement toward objectives.
# Without this, agents can exploit doing nothing to avoid negative outcomes.
# UPDATED: Increased from -0.02 to -0.05 to exceed exploration rewards and prevent exploitation.
NOOP_ACTION_PENALTY = -0.05

# Invalid masked action penalty (should rarely trigger if masking works)
# Rationale: Large penalty (-0.1) for selecting masked actions. This should
# almost never occur if action masking is implemented correctly in policy.
MASKED_ACTION_PENALTY = -0.1

# Ineffective action penalty (post-action detection)
# Rationale: Moderate penalty (-0.05) for actions that produce no position change.
# This catches cases where horizontal input has mechanical effects (wall sliding)
# but doesn't produce movement. UPDATED: Increased from -0.02 to -0.05 to exceed exploration rewards.
INEFFECTIVE_ACTION_PENALTY = -0.05


# =============================================================================
# MOMENTUM PRESERVATION REWARDS
# =============================================================================
# Rewards for maintaining high velocity (core N++ gameplay mechanic)

# Momentum bonus per step
# Rationale: Continuous bonus encourages maintaining high speed without overwhelming
# terminal rewards. Over 5000 steps at max speed, this yields ~5.0 total momentum
# bonus, roughly 25% of completion reward (20.0).
# UPDATED 2025-11-08: Increased 5x from 0.0002 to 0.001 based on analysis showing
# momentum rewards were too weak to influence policy. Higher bonus encourages
# speed-running behavior characteristic of expert N++ play.
MOMENTUM_BONUS_PER_STEP = 0.001  # was 0.0002

# Momentum efficiency threshold
# Rationale: 80% of MAX_HOR_SPEED (2.666 px/frame) provides a reasonable threshold
# that encourages near-maximum speed without being too strict. Allows for slight
# speed variations while still rewarding high-speed play.
MOMENTUM_EFFICIENCY_THRESHOLD = 0.8  # 80% of MAX_HOR_SPEED

# =============================================================================
# BUFFER UTILIZATION REWARDS
# =============================================================================
# Rewards for successful buffer-based jumps (frame-perfect execution)

# Buffer usage bonus
# Rationale: Reward for frame-perfect buffer execution. Encourages precise timing
# characteristic of expert N++ play.
# UPDATED 2025-11-08: Increased 2x from 0.05 to 0.1 to better reward skilled
# movement. Frame-perfect execution should be clearly rewarded.
# UPDATED: Reduced from 0.1 to 0.02 to align with step-level rewards and prevent perverse incentives.
BUFFER_USAGE_BONUS = 0.02  # Reduced from 0.1 to align with step-level rewards


# =============================================================================
# EXPLORATION REWARD CONSTANTS
# =============================================================================
# Multi-scale spatial exploration rewards following count-based methods
# (Bellemare et al., 2016: "Unifying Count-Based Exploration")

# Grid cell dimensions
# Rationale: Based on N++ level structure (42x23 cells of 24 pixels each).
# Slightly larger to account for margins.
EXPLORATION_GRID_WIDTH = 44
EXPLORATION_GRID_HEIGHT = 25
EXPLORATION_CELL_SIZE = 24.0  # pixels

# Exploration rewards at different spatial scales
# DISABLED: PBRS provides exploration through potential gradients. Explicit exploration
# conflicts with "shortest path" objective and can encourage wandering instead of
# efficient completion. Removing 0.02/step that could encourage non-optimal behavior.
# Keep exploration calculator for diagnostic metrics but set rewards to 0.

# Single cell (24x24 pixels) - finest granularity
EXPLORATION_CELL_REWARD = 0.0  # DISABLED

# Medium area (4x4 cells = 96x96 pixels) - room-sized regions
EXPLORATION_AREA_4X4_REWARD = 0.0  # DISABLED

# Large area (8x8 cells = 192x192 pixels) - section-sized regions
EXPLORATION_AREA_8X8_REWARD = 0.0  # DISABLED

# Very large area (16x16 cells = 384x384 pixels) - major regions
EXPLORATION_AREA_16X16_REWARD = 0.0  # DISABLED

# Exploration decay configuration
# Rationale: Reduce exploration rewards as episode progresses to prioritize speed
# optimization. Early exploration helps discover goals, but later in episode
# agent should focus on efficient completion rather than continued exploration.
EXPLORATION_DECAY_ENABLED = True
EXPLORATION_DECAY_START_STEP = 500  # Start decay after this many steps
EXPLORATION_DECAY_END_STEP = 2000  # Full decay by this step
EXPLORATION_MIN_SCALE = 0.2  # Minimum scale factor (20% of original)


# =============================================================================
# POTENTIAL-BASED REWARD SHAPING (PBRS) CONSTANTS
# =============================================================================
# Following Ng et al. (1999) theory: F(s,s') = γ * Φ(s') - Φ(s)
# Ensures policy invariance while providing dense reward signal

# PBRS discount factor
# According to Ng et al. (1999), F(s,s') = γ * Φ(s') - Φ(s) requires the same
# γ as the RL algorithm to maintain optimal policy invariance.
# MUST match PPO gamma for PBRS policy invariance guarantee.
PBRS_GAMMA = 0.995

# PBRS component weights (default configuration)
# Rationale: Weights control relative importance of different potential functions.
# Tuned through empirical evaluation to balance competing objectives.

# Objective distance potential weight
# Rationale: Reduced to maintain 20:1 terminal/dense reward ratio (Sutton & Barto, 2018).
# At 0.3 weight, PBRS accumulates ~0.75-1.5 over episode (3.75-7.5% of terminal).
# This provides sufficient gradient for learning without overwhelming terminal rewards.
# Terminal rewards (±20.0) remain the dominant learning signal.
# UPDATED: Reduced 6.67x from 2.0 to 0.3 to restore proper hierarchy.
PBRS_OBJECTIVE_WEIGHT = 0.3  # Reduced 6.67x from 2.0

# Hazard proximity potential weight
# Rationale: Safety hint weight provides hazard awareness without overwhelming
# objective-seeking behavior. Reduced proportionally to maintain 40% ratio to objective.
# At 0.12 weight, hazard avoidance guides routing while objective potential dominates.
# UPDATED: Reduced 6.67x from 0.8 to 0.12 to maintain ratio with objective weight.
PBRS_HAZARD_WEIGHT = 0.12  # Reduced 6.67x from 0.8 (maintains 40% ratio)

# Impact risk potential weight
# Rationale: Impact awareness weight encourages safer movement without dominating.
# Reduced proportionally to maintain 30% ratio to objective weight.
# UPDATED: Reduced 6.67x from 0.6 to 0.09 to maintain ratio with objective weight.
PBRS_IMPACT_WEIGHT = 0.09  # Reduced 6.67x from 0.6 (maintains 30% ratio)

# Exploration potential weight
# Rationale: Combines with explicit exploration rewards for better coverage.
# UPDATED 2025-11-08: Increased 3x from 0.2 to 0.6 to encourage spatial exploration.
# UPDATED 2025-11-08 (Tier 1): Reduced 50% from 0.6 to 0.3 to reduce excessive
# wandering and focus agent on direct paths to objectives. Objective weight (4.5)
# now dominates by 15x instead of 7.5x, providing clearer directional signal.
PBRS_EXPLORATION_WEIGHT = 0.3  # was 0.6 (Tier 1 path efficiency reduction)

# PBRS scaling for switch and exit phases
# Rationale: Scale of 1.0 ensures PBRS rewards are effective and guide learning.
# With γ=0.995, moving closer (increasing potential) always yields positive reward:
# F(s,s') = γ * Φ(s') - Φ(s) > 0 when Φ(s') > Φ(s)/γ ≈ Φ(s)
# Previous value (0.5) was too conservative, making PBRS rewards too small to effectively guide learning.
# PBRS rewards need to be large enough to provide meaningful gradient while still being
# smaller than terminal rewards (20.0 completion, 2.0 switch activation). PBRS provides
# dense shaping rewards (~1.5 accumulated) to guide HOW to reach the goal, while terminal
# rewards motivate COMPLETING it efficiently.
PBRS_SWITCH_DISTANCE_SCALE = 1.0
PBRS_EXIT_DISTANCE_SCALE = 1.0


# =============================================================================
# PBRS POTENTIAL FUNCTION NORMALIZATION CONSTANTS
# =============================================================================
# Constants for normalizing potential functions to comparable scales

# Level dimensions for distance normalization
# Rationale: N++ levels are 1056x600 pixels (42x23 cells of 24 pixels).
# Level diagonal used for normalizing distances to [0, 1] range.
# These values are imported from constants.py to maintain single source of truth.

# Maximum ninja velocity estimate (pixels/frame)
# Rationale: Approximate maximum velocity (~10 px/frame) for normalizing
# velocity-based potentials. Based on N++ physics constants.
PBRS_MAX_VELOCITY = 10.0

# Hazard danger radius (pixels)
# Rationale: Distance (50 pixels) within which hazards significantly affect potential.
# Chosen to be roughly twice ninja radius to provide early warning.
PBRS_HAZARD_DANGER_RADIUS = 50.0

# Exploration visit threshold (pixels)
# Rationale: Positions within 25 pixels considered "visited" for exploration potential.
# Chosen to be roughly one cell size for efficient memory usage.
PBRS_EXPLORATION_VISIT_THRESHOLD = 25.0

# Exploration radius for novelty detection (pixels)
# Rationale: Distance (30 pixels) defining "unexplored" regions for exploration potential.
# Slightly larger than visit threshold to encourage broad coverage.
PBRS_EXPLORATION_RADIUS = 30.0

# Fallback distance scale for PBRS adaptive normalization
# Rationale: Used when adaptive scaling cannot be computed (e.g., no reachable nodes).
# Adaptive scaling computes maximum reachable distance per level using BFS flood fill,
# but falls back to LEVEL_DIAGONAL if no reachable area found or max distance is 0.
# This ensures proper normalization even in edge cases.
# NOTE: Actual normalization scale is computed dynamically per level based on reachable area.
PBRS_FALLBACK_DISTANCE_SCALE = (
    LEVEL_DIAGONAL  # Fallback when adaptive scaling unavailable
)


# =============================================================================
# PATH EFFICIENCY REWARDS (Tier 1)
# =============================================================================
# Rewards and penalties for efficient path planning and progress tracking
# Added 2025-11-08 to address inefficient movement patterns (looping, backtracking)

# Progress tracking and backtracking detection
# Rationale: Tracks best PATH distance achieved to each objective and penalizes
# significant regression. Uses graph-based shortest path distances (not Euclidean)
# to respect level geometry and obstacles. Encourages monotonic progress toward
# objectives while allowing necessary corrections.
BACKTRACK_THRESHOLD_DISTANCE = 20.0  # path distance units (graph-based)
BACKTRACK_PENALTY_SCALE = 0.00003  # Conservative penalty per unit distance
PROGRESS_BONUS_SCALE = 0.0  # Disabled - redundant with PBRS objective potential
STAGNATION_THRESHOLD = 75  # frames without progress before penalty (reduced from 150)
STAGNATION_PENALTY_PER_FRAME = 0.0001  # Increased from 0.00003 for stronger signal
PROGRESS_CHECK_THRESHOLD = 5.0  # Minimum improvement to count as progress (path units)
