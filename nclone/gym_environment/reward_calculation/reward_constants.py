"""Centralized reward constants and configuration for N++ RL training.

This module defines all reward-related constants following RL best practices:
1. All magic numbers are eliminated and replaced with named constants
2. Each constant includes documentation explaining its purpose and rationale
3. Constants are grouped by reward component for clarity
4. Values are based on research and empirical tuning

References:
- Ng et al. (1999): "Policy Invariance Under Reward Transformations" (PBRS theory)
- Pathak et al. (2017): "Curiosity-driven Exploration" (ICM)
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
# Rationale: Moderate negative reward (-1.0) discourages death without dominating learning.
# Kept proportional to completion reward (5% of completion).
# Too large penalties can lead to overly conservative behavior.
DEATH_PENALTY = -1.0


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
# Rationale: Encourages efficiency without overwhelming terminal rewards.
# At -0.0001 per step, even episodes at max length (20k steps) maintain positive
# returns when successful: +10.0 completion - 2.0 penalty = +8.0.
TIME_PENALTY_PER_STEP = -0.0001

# Progressive time penalty schedule (for speed optimization)
# Rationale: Allows early exploration while increasing pressure for efficiency
# over episode duration. Phased approach supports curriculum learning:
# - Early phase: minimal penalty, encourage exploration
# - Middle phase: moderate penalty, find solutions
# - Late phase: high penalty, optimize routes
TIME_PENALTY_EARLY = -0.00005  # Steps 0-30%: exploration phase
TIME_PENALTY_MIDDLE = -0.0002  # Steps 30-70%: solution phase
TIME_PENALTY_LATE = -0.0005  # Steps 70-100%: optimization phase

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
# Rationale: Small penalty (-0.01) discourages standing still without overwhelming
# other reward signals. Encourages active exploration and movement toward objectives.
# Without this, agents can exploit doing nothing to avoid negative outcomes.
NOOP_ACTION_PENALTY = -0.02

# Invalid masked action penalty (should rarely trigger if masking works)
# Rationale: Large penalty (-0.1) for selecting masked actions. This should
# almost never occur if action masking is implemented correctly in policy.
MASKED_ACTION_PENALTY = -0.1

# Ineffective action penalty (post-action detection)
# Rationale: Moderate penalty (-0.02) for actions that produce no position change.
# This catches cases where horizontal input has mechanical effects (wall sliding)
# but doesn't produce movement. Kept moderate since mechanical effects may be valuable.
INEFFECTIVE_ACTION_PENALTY = -0.02


# =============================================================================
# MOMENTUM PRESERVATION REWARDS
# =============================================================================
# Rewards for maintaining high velocity (core N++ gameplay mechanic)

# Momentum bonus per step
# Rationale: Small continuous bonus (0.0002) encourages maintaining high speed
# without overwhelming terminal rewards. Over 5000 steps at max speed, this yields
# ~1.0 total momentum bonus, roughly 5% of completion reward (20.0). Kept small to
# encourage speed without creating perverse incentives for longer episodes.
MOMENTUM_BONUS_PER_STEP = 0.0002

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
# Rationale: Moderate reward (0.05) for frame-perfect buffer execution, roughly
# 5% of switch activation reward (1.0). Encourages precise timing without dominating
# other reward signals.
BUFFER_USAGE_BONUS = 0.05


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
# Rationale: Multi-scale rewards encourage both fine-grained and broad exploration.
# Total maximum per-step exploration reward = 0.04

# Single cell (24x24 pixels) - finest granularity
EXPLORATION_CELL_REWARD = 0.005

# Medium area (4x4 cells = 96x96 pixels) - room-sized regions
EXPLORATION_AREA_4X4_REWARD = 0.005

# Large area (8x8 cells = 192x192 pixels) - section-sized regions
EXPLORATION_AREA_8X8_REWARD = 0.005

# Very large area (16x16 cells = 384x384 pixels) - major regions
EXPLORATION_AREA_16X16_REWARD = 0.005


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
# Rationale: Primary weight (1.0) for distance to switch/exit objectives.
# This is the main shaping signal for task completion.
PBRS_OBJECTIVE_WEIGHT = 1.5

# Hazard proximity potential weight
# Rationale: Minimal weight (0.04) provides subtle safety hints without distracting
# from objectives. After fixing double-weighting bug, reduced from 0.2 to restore
# original effective magnitude. Objective (1.5) dominates by 37.5x.
PBRS_HAZARD_WEIGHT = 0.04

# Impact risk potential weight
# Rationale: Minimal weight (0.04) for impact awareness without conservatism.
# After fixing double-weighting bug, reduced from 0.2 to restore original
# effective magnitude. Objective (1.5) dominates by 37.5x. Agent prioritizes
# speed and completion over impact avoidance.
PBRS_IMPACT_WEIGHT = 0.04

# Exploration potential weight
# Rationale: Combines with explicit exploration rewards for better coverage.
PBRS_EXPLORATION_WEIGHT = 0.2

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
# INTRINSIC MOTIVATION CONSTANTS (for npp-rl integration)
# =============================================================================
# Constants for ICM-based curiosity and intrinsic rewards
# (Pathak et al., 2017: "Curiosity-driven Exploration by Self-supervised Prediction")

# ICM reward combination weight (alpha)
# Rationale: Weight (0.1) for combining intrinsic and extrinsic rewards.
# Formula: total_reward = extrinsic + alpha * intrinsic
# Set to 10% to provide exploration boost without overwhelming task rewards.
ICM_ALPHA = 0.1

# ICM intrinsic reward clip value
# Rationale: Maximum intrinsic reward (1.0) to prevent instability from
# large prediction errors during early training.
ICM_REWARD_CLIP = 1.0

# ICM loss function weights
# Rationale: Standard ICM weights from Pathak et al. (2017).
# Forward model (0.9) dominates for curiosity signal.
# Inverse model (0.1) provides auxiliary learning objective.
ICM_FORWARD_LOSS_WEIGHT = 0.9
ICM_INVERSE_LOSS_WEIGHT = 0.1

# ICM learning rate
# Rationale: Moderate learning rate (1e-3) for ICM network training.
# Higher than policy network (3e-4) as ICM learns faster auxiliary task.
ICM_LEARNING_RATE = 1e-3
