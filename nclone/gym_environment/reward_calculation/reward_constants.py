"""HWM A* reward constants for N++ RL training.

Reward system based on High-Water-Mark (HWM) A* distance shaping:
1. Sparse terminal rewards - Define task success/failure (fixed values)
2. HWM shaping: Non-negative reward for achieving new minimum A* distance
3. RND intrinsic rewards (handled on training side, not here)

HWM Shaping Properties:
- Non-negative: reward >= 0 always (backtracking yields zero, not negative)
- Bounded: cumulative HWM reward <= 1.0 per phase (normalized by max distance)
- Non-exploitable: cycling distance down->up->down yields reward once then zeros
- History-dependent: watermark is episode state

Reference: REWARD_PIPELINE_SPEC.md (Section 5)
"""

# =============================================================================
# GLOBAL REWARD SCALING
# =============================================================================
# Set to 1.0 for HWM pipeline. The smaller reward magnitudes (R_EXIT=5.0 vs
# old LEVEL_COMPLETION_REWARD=200.0) make global scaling unnecessary.
# PPO's advantage normalization handles scale invariance.
GLOBAL_REWARD_SCALE = 1.0

# =============================================================================
# HWM SPARSE TERMINAL REWARDS (Fixed Values)
# =============================================================================
# No curriculum scaling - fixed magnitudes per spec Section 5.
# Hierarchy: R_EXIT(5.0) >> R_SWITCH(1.0) >> HWM_max(3.0) >> |R_DEATH|(0.5)
#
# With HWM_ALPHA=3.0 and normalized HWM in [0, 1]:
#   Complete: R_SWITCH(1.0) + HWM_switch(~3.0) + R_EXIT(5.0) + HWM_exit(~3.0) = ~12.0
#   Switch + death: R_SWITCH(1.0) + HWM_switch(~1.0) + R_DEATH(-0.5) = ~1.5
#   Quick death: HWM(~0.1) + R_DEATH(-0.5) = ~-0.4
#
# R_DEATH reverted to -0.5 (from -1.5): run-221 showed -1.5 caused passive risk-aversion
# (37% NOOP, 95% mine deaths vs 61% with -0.5 in run-220). The higher death penalty
# dominated HWM rewards by 1500:1, making exploration irrational.
# HWM_ALPHA increased to 3.0 (from 1.0): ratio death/HWM at stage-0 is now 28:1 (was 83:1),
# making the learning signal substantially stronger relative to the death risk.
# TIME_PENALTY increased to -0.001 (from -0.0005): with R_DEATH=-0.5, standing still for
# 500 steps costs -0.5 (same as one death), creating a strong incentive to act efficiently.
R_SWITCH = 1.0
R_EXIT = 5.0
R_DEATH = -0.5

# HWM shaping weight (constant throughout training per spec Section 7)
HWM_ALPHA = 3.0

# Per-step time penalty (applied in RewardConfig.time_penalty_per_step)
# -0.001/step: at 500 steps = -0.5 (same as R_DEATH), making inaction as costly as dying.
# Incentivizes efficient paths while allowing momentum-building backtracking.
TIME_PENALTY_PER_STEP = -0.001

# Per-step backtracking penalty applied when the A* distance to the current goal increases.
# This discourages aimless wandering away from the objective. Run-221-2 showed 80-93%
# backtracking across all stages despite HWM shaping rewarding forward progress.
# The HWM reward gives 0 for backtracking (non-negative), but this makes backtracking
# actively negative. Magnitude -0.0002 is 20% of TIME_PENALTY_PER_STEP -- a nudge
# not a hard penalty, preserving the agent's ability to backtrack for momentum.
BACKTRACK_PENALTY = -0.0002

# =============================================================================
# PATHFINDING CONFIGURATION
# =============================================================================
# Geometric A* only - no physics-based costs for HWM pipeline.
PBRS_USE_PHYSICS_COSTS = False

# Mine hazard avoidance for A* pathfinding (affects path shape, not reward)
MINE_HAZARD_RADIUS = 48.0
MINE_HAZARD_COST_MULTIPLIER = 1.1
MINE_PENALIZE_DEADLY_ONLY = True

# =============================================================================
# DEPRECATED CONSTANTS (kept for backward compatibility)
# =============================================================================
# These are no longer used in the HWM pipeline but kept so that any
# remaining references don't cause ImportError.

LEVEL_COMPLETION_REWARD = 200.0  # DEPRECATED: use R_EXIT
SWITCH_ACTIVATION_REWARD = 100.0  # DEPRECATED: use R_SWITCH
DEATH_PENALTY = -30.0  # DEPRECATED: use R_DEATH
IMPACT_DEATH_PENALTY = -30.0  # DEPRECATED
HAZARD_DEATH_PENALTY = -30.0  # DEPRECATED
MINE_DEATH_PENALTY = -30.0  # DEPRECATED
TIMEOUT_PENALTY = 0.0  # DEPRECATED
STAGNATION_TIMEOUT_PENALTY = -20.0  # DEPRECATED
STAGNATION_PROGRESS_THRESHOLD = 0.15  # DEPRECATED

DEATH_PENALTY_DISCOVERY = -30.0  # DEPRECATED
DEATH_PENALTY_EARLY = -10.0  # DEPRECATED
DEATH_PENALTY_MID = -5.0  # DEPRECATED
DEATH_PENALTY_ADVANCED = -4.0  # DEPRECATED
DEATH_PENALTY_MASTERY = -3.0  # DEPRECATED

PBRS_GAMMA = 0.99  # DEPRECATED: HWM doesn't use PBRS gamma
PBRS_OBJECTIVE_WEIGHT = 2.5  # DEPRECATED
PBRS_SWITCH_DISTANCE_SCALE = 1.0  # DEPRECATED
PBRS_EXIT_DISTANCE_SCALE = 1.0  # DEPRECATED
PBRS_GRADIENT_REFERENCE_DISTANCE = 200.0  # DEPRECATED
PBRS_GRADIENT_SCALE_CAP = 7.0  # DEPRECATED
PBRS_PATH_NORMALIZATION_FACTOR = 1.0  # DEPRECATED
PBRS_DISPLACEMENT_THRESHOLD = 0.0  # DEPRECATED
PBRS_VELOCITY_BUCKET_SCALE = 2.0  # DEPRECATED
PBRS_GEOMETRIC_FALLBACK_WEIGHT = 0.2  # DEPRECATED
PBRS_MIN_PHYSICS_SCALE = 50.0  # DEPRECATED

SUB_GOAL_REWARD_PROGRESS = 30.0  # DEPRECATED
SUB_GOAL_SPACING = 48.0  # DEPRECATED

# =============================================================================
# DENSE REWARD DEFAULTS (SAC DenseRewardCalculator)
# =============================================================================
# These defaults mirror DenseRewardConfig to serve as a central reference.
# Override via DenseRewardConfig fields in SACConfig.

# Progress shaping weight: F = alpha_pbrs * max(0, dist_before_macro - dist_after_macro)
# Applied ONCE per macro-action (not per frame) — see DenseRewardCalculator.compute_macro_pbrs().
DENSE_ALPHA_PBRS = 0.5
# Discount used in PBRS formula (should match SACConfig.gamma = 0.99)
DENSE_PBRS_GAMMA = 0.99

# Per-frame velocity-alignment reward = beta * max(0, dot(v_unit, hop_dir))
# Reduced from 0.0005: see DenseRewardConfig.beta_vel for rationale.
DENSE_BETA_VEL = 0.0001

# Distance milestone: bonus per interval crossed for the first time per episode
# Reduced from 0.2 to 0.1: see DenseRewardConfig.milestone_interval for rationale.
DENSE_MILESTONE_INTERVAL = 0.1   # 10% distance reduction per milestone
DENSE_MILESTONE_BONUS = 0.1      # flat reward per milestone

# Mine hazard avoidance: delta * max(0, sdf_curr - sdf_prev) inside danger zone
DENSE_DELTA_HAZARD = 0.0001
DENSE_HAZARD_SDF_THRESHOLD = 0.3  # SDF below this activates hazard shaping

INEFFECTIVE_ACTION_THRESHOLD = 0.35  # DEPRECATED
INEFFECTIVE_ACTION_PENALTY = -0.05  # DEPRECATED
OSCILLATION_WINDOW = 10  # DEPRECATED
NET_DISPLACEMENT_THRESHOLD = 8.0  # DEPRECATED
OSCILLATION_PENALTY = -0.05  # DEPRECATED
VELOCITY_ALIGNMENT_MIN_SPEED = 0.5  # DEPRECATED
VELOCITY_ALIGNMENT_WEIGHT = 0.15  # DEPRECATED
VELOCITY_ALIGNMENT_WEIGHT_RATIO = 0.0  # DEPRECATED
