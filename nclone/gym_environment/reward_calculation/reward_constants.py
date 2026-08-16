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
# Hierarchy: R_EXIT(5.0) >> HWM_max(3.0) >> R_SWITCH(0.5) >> |mine death|(3.5)
#
# With HWM_ALPHA=3.0 and normalized HWM in [0, 1]:
#   Complete: R_SWITCH(0.5) + HWM_switch(~3.0) + R_EXIT(5.0) + HWM_exit(~3.0) = ~11.5
#   Switch + death: R_SWITCH(0.5) + HWM_switch(~1.0) + R_DEATH(-0.5) = ~1.0
#   Quick death: HWM(~0.1) + R_DEATH(-0.5) = ~-0.4
#
# R_DEATH reverted to -0.5 (from -1.5): run-221 showed -1.5 caused passive risk-aversion
# (37% NOOP, 95% mine deaths vs 61% with -0.5 in run-220). The higher death penalty
# dominated HWM rewards by 1500:1, making exploration irrational.
# HWM_ALPHA increased to 3.0 (from 1.0): ratio death/HWM at stage-0 is now 28:1 (was 83:1),
# making the learning signal substantially stronger relative to the death risk.
# TIME_PENALTY increased to -0.001 (from -0.0005): with R_DEATH=-0.5, standing still for
# 500 steps costs -0.5 (same as one death), creating a strong incentive to act efficiently.
R_SWITCH = 0.5  # L1.5 phase-transition bonus (C_PHASE); replaces prior 1.0 sparse switch reward
# Raised 5.0 -> 10.0 for COMPLETION DOMINANCE on long levels. Under PBRS (not HWM)
# the persistent un-farmable shaping an agent can bank WITHOUT winning is ~PBRS
# terminal (w*cost_spawn/D0 ≈ 1.8) + waypoint-reach (~1.3) ≈ 3.1; with the old
# R_EXIT=5 the completion margin over "approach-and-time-out" was thin, and an
# over-scaled frontier bonus (now D0-normalized) could exceed it entirely. A
# +10 win gives a clear, unconditional preference for finishing over any bounded
# shaping the agent can accrue by merely approaching the goal.
R_EXIT = 10.0
# Per-NECESSARY-locked-door-switch activation bonus. Each necessary locked-door
# switch is a required sub-goal on the path to completion (the planner's
# door-dependency solver decides necessity), so it earns a milestone bonus like
# the exit switch — teaching the agent to route through gating switches. Only
# NECESSARY doors are rewarded (opening an irrelevant door pays nothing) so the
# signal never encourages detours. Gated by RewardConfig.enable_locked_door_rewards.
R_LOCKED_DOOR_SWITCH = 0.5
# Ceiling on the TOTAL locked-door milestone an episode can bank, spread evenly
# over the level's necessary doors (each pays min(R_LOCKED_DOOR_SWITCH, cap/k)).
#
# Without this the milestone breaks completion dominance on the tail of the
# corpus. Measured over all 808 official locked-door levels: 294 need at least
# one door, p50 = 2, but the tail runs to 71 -- and 0.5 * 71 = 35.5 against
# R_EXIT = 10.0. Nine levels (3.1% of the door-needing set) pay more for opening
# doors than for finishing, and 24 (8.2%) pay more than half of it. Locked doors
# are IRREVERSIBLE, so on those levels "open every door, then run out the clock"
# is not merely competitive with winning, it strictly dominates it -- the agent
# banks the whole payout and then declines the death risk of the exit leg.
#
# 2.0 = 20% of R_EXIT. It is deliberately below the ~3.1 of persistent
# un-farmable shaping already bankable without a win (see R_EXIT's note), so the
# two together stay well clear of 10.0. k <= 4 covers 225 of the 294 levels and
# is unaffected; only the tail is compressed. Per-door value is constant within
# an episode (k is a static property of the map), so this adds no
# non-stationarity to the signal.
R_LOCKED_DOOR_TOTAL_CAP = 2.0
R_DEATH = -0.5

# Extra penalty for mine deaths. Applied when death_cause == 'mine'. Total mine
# death penalty = R_DEATH + R_DEATH_MINE.
# Raised to -3.0 (total -3.5 with R_DEATH) for FiGAR + precomputed PBRS: a weak mine
# penalty let the agent prefer quick mine deaths (low time cost) over longer runs
# that reach further toward the goal; stronger mine loss breaks that local minimum.
R_DEATH_MINE = -3.0

# HWM shaping weight (constant throughout training per spec Section 7)
HWM_ALPHA = 3.0

# =============================================================================
# LATTICE / FREE-SPACE PBRS (gated by RewardConfig.enable_lattice_pbrs)
# =============================================================================
# A true, policy-invariant potential-based shaping term that REPLACES the HWM
# ratchet + backtrack penalty when enabled. The potential is the negative
# normalized cost-to-completion supplied by the env's reachability source
# (surface lattice d_comb where available, else the mine-aware free-space grid):
#
#     Phi(s) = -clamp(completion_cost(s) / D0, 0, 1)        in [-1, 0]
#     F(s, s') = w * (gamma * Phi(s') - Phi(s))             added each step
#
# D0 is the spawn cost-to-completion (the existing optimal_path_length), so
# Phi(spawn) = -1 and Phi(complete) = 0. Unlike HWM, F is symmetric (rewards
# progress AND penalizes regress every frame) and telescopes to w*(Phi_T - Phi_0)
# over an episode, giving a dense per-frame gradient without the ratchet's blind
# spot. Weight matches HWM_ALPHA so the dense scale is comparable on day one.
LATTICE_PBRS_WEIGHT = 3.0
# Discount used INSIDE the shaping term. Default 1.0 (NOT the PPO gamma): for a
# heuristic cost-to-go potential, gamma=1.0 gives pure telescoping F = w*(Phi'-Phi)
# = a clean per-step progress delta, exactly the codebase convention for heuristic
# potentials (see PBRSConfig.pbrs_gamma docs). gamma<1 instead adds a constant
# (gamma-1)*Phi survival drift (e.g. +0.03/step at spawn with w=3, gamma=0.99 —
# empirically observed), which rewards stalling near spawn. Policy invariance
# holds for ANY gamma in the episodic setting, so 1.0 is the clean choice.
LATTICE_PBRS_GAMMA = 1.0

# =============================================================================
# TIME PENALTY (the "fewest frames" pressure — gold value is calibrated to it)
# =============================================================================
# Per-step time penalty (applied in RewardConfig.time_penalty_per_step and primitive PBRS path).
# Reduced from -0.011 to -0.002: with precomputed PBRS (alpha=5), per-frame progress shaping
# must not be dominated by time cost, or the agent rationally prefers short episodes / quick
# deaths. Truncation (-1.5) and mine penalties still discourage passive timeout vs progress.
# NOTE: R_GOLD below is derived from this value — changing it rescales the gold
# exchange rate automatically. Keep this definition ABOVE the gold block.
TIME_PENALTY_PER_STEP = -0.002

# =============================================================================
# GOLD COLLECTION REWARDS (gated by RewardConfig.enable_gold_rewards)
# =============================================================================
# Objective reframe (docs/GOLD_COLLECTION_HANDOFF.md): the goal is completion in
# the fewest frames, and **each gold piece is declared worth GOLD_FRAME_VALUE
# frames of saved time**. We CALIBRATE the per-gold pickup reward to the per-frame
# time penalty so the exchange is mathematically exact — collecting a gold pays
# back precisely the time penalty of a GOLD_FRAME_VALUE-frame detour:
#
#     a detour of D frames costs        D * |TIME_PENALTY_PER_STEP|
#     collecting one gold pays   GOLD_FRAME_VALUE * |TIME_PENALTY_PER_STEP|
#     => break-even exactly at D == GOLD_FRAME_VALUE frames
#
# So the orienteering rule "collect a gold iff its detour < GOLD_FRAME_VALUE
# frames" emerges from the reward arithmetic instead of being hand-tuned. This is
# the single source of truth for the exchange rate; the planner and the lattice
# gold potential read the same GOLD_FRAME_VALUE.
#
# Order-agnostic: reward is paid on the gold_collected delta regardless of which
# gold or in what order, so opportunistic cluster grabs are never penalized.
GOLD_FRAME_VALUE = 120.0  # frames of time saved per gold collected
R_GOLD = GOLD_FRAME_VALUE * abs(TIME_PENALTY_PER_STEP)  # = 0.24 at -0.002/frame
# Uniform valuation => NO all-gold one-shot bonus: a spike on the final gold would
# make it worth far more than GOLD_FRAME_VALUE frames and distort the optimum.
# Kept as a constant (default 0.0) only for backward-compat / optional
# collect-everything A/B (RewardConfig._all_gold_bonus_override).
R_ALL_GOLD = 0.0

# Per-step backtracking penalty applied when the A* distance to the current goal increases.
# This discourages aimless wandering away from the objective. Run-221-2 showed 80-93%
# backtracking across all stages despite HWM shaping rewarding forward progress.
# The HWM reward gives 0 for backtracking (non-negative), but this makes backtracking
# actively negative. Magnitude 20% of TIME_PENALTY_PER_STEP -- a nudge.
BACKTRACK_PENALTY = -0.0004  # ~20% of -0.002

# Terminal penalty when episode ends by truncation (timeout). Applied to discourage
# the agent from hugging the wall until timeout instead of attempting progress.
# -1.5 makes timeout comparable to mine death (-2.5) so the agent is incentivized
# to try navigating around mines rather than passively timing out.
TRUNCATION_PENALTY = -1.5

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
