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

# Death penalty
# Rationale: Moderate negative reward (-2.0) discourages death without dominating learning.
# 10% of completion reward balances learning signal strength.
DEATH_PENALTY = -2.0

# Mine death penalty
# Rationale: Stronger penalty (-3.0) for preventable mine deaths.
# 15% of completion reward emphasizes the importance of hazard avoidance.
MINE_DEATH_PENALTY = -3.0

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
# Lower values = stronger gradients, higher values = weaker gradients
# Replaces surface-area-based normalization for better handling of open levels with focused paths
#
# IMPORTANT: Used with NON-LINEAR normalization: Φ(s) = 1 / (1 + distance/area_scale)
# where area_scale = combined_path_distance * PBRS_PATH_NORMALIZATION_FACTOR * scale_factor
#
# Non-linear normalization ensures gradients at ALL distances (no "dead zone" at far distances)
# This fixes the issue where linear capping (min(1.0, distance/area_scale)) caused
# potential=0 beyond certain distances, resulting in zero PBRS rewards for most steps.
#
# Increased from 0.8 to 1.5 to provide stronger potentials and gradients:
# - With scale_factor=0.3 (early training): area_scale = 0.45 * combined_path (was 0.24)
# - Potentials will be in ~0.4-0.6 range instead of 0.2-0.4 range
# - Stronger per-step PBRS rewards for better learning signal
PBRS_PATH_NORMALIZATION_FACTOR = 1.5  # Tunable: 0.5-2.0 range


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
