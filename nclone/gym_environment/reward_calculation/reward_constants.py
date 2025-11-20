"""Simplified, focused reward constants for N++ RL training.

Centralized reward system with clear hierarchy:
1. Terminal rewards - Define task success/failure
2. PBRS objective potential - Policy-invariant guidance (curriculum-managed)
3. Time penalty - Efficiency pressure (curriculum-managed)

All redundant/confusing components removed. Curriculum logic managed by RewardConfig.

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
# PBRS provides dense guidance (~2-5 accumulated depending on training phase) to
# guide HOW to reach the goal. Terminal reward defines WHAT success means.
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
# MUST match PPO gamma for policy invariance guarantee
PBRS_GAMMA = 0.995

# NOTE: PBRS objective weight is now managed by RewardConfig with curriculum scaling:
# - Early training (0-1M steps): weight = 2.0 (strong guidance)
# - Mid training (1M-3M steps): weight = 1.0 (moderate guidance)
# - Late training (3M+ steps): weight = 0.5 (light shaping)
# Kept here for backwards compatibility; RewardConfig overrides during training.
PBRS_OBJECTIVE_WEIGHT = 0.3  # Static default (not used during curriculum training)

# PBRS scaling for switch and exit phases
# Scale of 1.0 provides effective gradients while keeping shaping < terminal rewards
PBRS_SWITCH_DISTANCE_SCALE = 1.0
PBRS_EXIT_DISTANCE_SCALE = 1.0


# =============================================================================
# REMOVED COMPONENTS (No Longer Used)
# =============================================================================
# The following components have been removed as redundant or confusing:
#
# REMOVED: Time penalties (now managed by RewardConfig with curriculum awareness)
# REMOVED: Exploration rewards (PBRS provides via potential gradients)
# REMOVED: Progress bonuses (redundant with PBRS)
# REMOVED: Backtrack penalties (confusing, PBRS handles naturally)
# REMOVED: Stagnation penalties (redundant with truncation)
# REMOVED: NOOP penalties (let PBRS guide, don't punish stillness)
# REMOVED: Buffer bonuses (gameplay mechanic, not learning signal)
# REMOVED: Momentum rewards (physics-based, not reward-based)
# REMOVED: Hazard proximity PBRS (death penalty is clearer)
# REMOVED: Impact risk PBRS (death penalty is clearer)
# REMOVED: Completion bonus (time penalty provides efficiency incentive)
#
# Total: ~20 redundant constants removed for clarity
