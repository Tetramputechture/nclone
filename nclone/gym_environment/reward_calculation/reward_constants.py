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

from typing import Dict, Any


# =============================================================================
# TERMINAL REWARD CONSTANTS
# =============================================================================
# Terminal rewards are the primary learning signals that define task completion

# Level completion reward - primary learning signal
# Rationale: Large enough (10.0) to dominate cumulative time penalty, ensuring
# successful episodes always yield positive returns. Even slow completions
# (10k steps) result in positive reward (10.0 - 1.0 = +9.0).
LEVEL_COMPLETION_REWARD = 10.0

# Death penalty
# Rationale: Moderate negative reward (-0.5) discourages death without dominating learning.
# Kept proportional to new completion reward (5% of completion).
# Too large penalties can lead to overly conservative behavior.
DEATH_PENALTY = -0.5


# =============================================================================
# OBJECTIVE MILESTONE REWARDS
# =============================================================================
# Intermediate rewards for achieving subgoals

# Switch activation reward
# Rationale: Milestone reward (10% of completion) provides intermediate signal
# without creating local optima that distract from the ultimate goal.
SWITCH_ACTIVATION_REWARD = 1.0


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
TIME_PENALTY_EARLY = -0.00005     # Steps 0-30%: exploration phase
TIME_PENALTY_MIDDLE = -0.0002     # Steps 30-70%: solution phase  
TIME_PENALTY_LATE = -0.0005       # Steps 70-100%: optimization phase

# Phase thresholds (as fraction of max episode length)
TIME_PENALTY_EARLY_THRESHOLD = 0.3    # 30% of episode
TIME_PENALTY_LATE_THRESHOLD = 0.7     # 70% of episode

# Completion time bonus (for fine-tuning speed optimization)
# Rationale: Explicitly rewards fast completion without punishing slow solutions.
# Bonus linearly decreases from maximum to zero as completion time increases.
# Compatible with curriculum: train for completion first, then fine-tune for speed.
COMPLETION_TIME_BONUS_MAX = 2.0      # Maximum bonus (instant completion)
COMPLETION_TIME_TARGET = 5000        # Target steps for full bonus (adjust per level difficulty)


# =============================================================================
# NAVIGATION REWARD SHAPING CONSTANTS
# =============================================================================
# Constants for potential-based reward shaping (Ng et al., 1999)
# These provide dense reward signals without changing the optimal policy

# Distance improvement scale for navigation rewards
# Rationale: Provides dense feedback about progress toward objectives.
# Formula: reward = distance_improvement * DISTANCE_IMPROVEMENT_SCALE
# Example: Moving 100 pixels closer = 0.1 reward
NAVIGATION_DISTANCE_IMPROVEMENT_SCALE = 0.001

# Minimum distance threshold for proximity bonus (pixels)
# Rationale: Provides small extra reward when within 20 pixels of objective.
# Encourages precise navigation in final approach phase.
NAVIGATION_MIN_DISTANCE_THRESHOLD = 20.0

# Potential-based shaping scale
# Rationale: Scale factor (0.0005) for continuous potential-based shaping.
# Keeps shaping rewards smaller than main terminal rewards while providing
# meaningful gradient for learning.
NAVIGATION_POTENTIAL_SCALE = 0.0005


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
EXPLORATION_CELL_REWARD = 0.01

# Medium area (4x4 cells = 96x96 pixels) - room-sized regions
EXPLORATION_AREA_4X4_REWARD = 0.01

# Large area (8x8 cells = 192x192 pixels) - section-sized regions
EXPLORATION_AREA_8X8_REWARD = 0.01

# Very large area (16x16 cells = 384x384 pixels) - major regions
EXPLORATION_AREA_16X16_REWARD = 0.01


# =============================================================================
# POTENTIAL-BASED REWARD SHAPING (PBRS) CONSTANTS
# =============================================================================
# Following Ng et al. (1999) theory: F(s,s') = γ * Φ(s') - Φ(s)
# Ensures policy invariance while providing dense reward signal

# PBRS discount factor
# Rationale: MUST match PPO gamma (0.999) for PBRS policy invariance guarantee.
# According to Ng et al. (1999), F(s,s') = γ * Φ(s') - Φ(s) requires the same
# γ as the RL algorithm to maintain optimal policy invariance.
# CRITICAL: If changing PPO gamma, this MUST be updated to match!
PBRS_GAMMA = 0.999

# PBRS component weights (default configuration)
# Rationale: Weights control relative importance of different potential functions.
# Tuned through empirical evaluation to balance competing objectives.

# Objective distance potential weight
# Rationale: Primary weight (1.0) for distance to switch/exit objectives.
# This is the main shaping signal for task completion.
PBRS_OBJECTIVE_WEIGHT = 1.0

# Hazard proximity potential weight
# Rationale: Small weight provides safety signal without making agent too conservative.
PBRS_HAZARD_WEIGHT = 0.1

# Impact risk potential weight
# Rationale: Keep at 0.0 for completion-focused training.
# Focus on speed and completion rather than impact avoidance.
PBRS_IMPACT_WEIGHT = 0.0

# Exploration potential weight
# Rationale: Combines with explicit exploration rewards for better coverage.
PBRS_EXPLORATION_WEIGHT = 0.2

# PBRS scaling for switch and exit phases
# Rationale: Provides meaningful navigation signal without dominating terminal rewards.
PBRS_SWITCH_DISTANCE_SCALE = 0.5
PBRS_EXIT_DISTANCE_SCALE = 0.5


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


# =============================================================================
# REWARD CONFIGURATION PRESETS
# =============================================================================
# Pre-configured reward settings for common training scenarios

def get_completion_focused_config() -> Dict[str, Any]:
    """Get reward configuration optimized for level completion.
    
    This configuration prioritizes:
    - Fast level completion (high terminal rewards)
    - Efficient solutions (time penalties)
    - Minimal safety constraints
    
    Best for: Initial training, speed-running objectives
    
    Returns:
        dict: Reward configuration parameters
    """
    return {
        # Terminal rewards
        "level_completion_reward": LEVEL_COMPLETION_REWARD,
        "death_penalty": DEATH_PENALTY,
        "switch_activation_reward": SWITCH_ACTIVATION_REWARD,
        "time_penalty": TIME_PENALTY_PER_STEP,
        
        # PBRS configuration
        "enable_pbrs": True,
        "pbrs_gamma": PBRS_GAMMA,
        "pbrs_weights": {
            "objective_weight": PBRS_OBJECTIVE_WEIGHT,
            "hazard_weight": 0.0,  # Disabled for completion focus
            "impact_weight": 0.0,  # Disabled for completion focus
            "exploration_weight": 0.0,  # Use explicit exploration instead
        },
        
        # Exploration configuration
        "enable_exploration_rewards": True,
        "exploration_scales": {
            "cell_reward": EXPLORATION_CELL_REWARD,
            "area_4x4_reward": EXPLORATION_AREA_4X4_REWARD,
            "area_8x8_reward": EXPLORATION_AREA_8X8_REWARD,
            "area_16x16_reward": EXPLORATION_AREA_16X16_REWARD,
        }
    }


def get_safe_navigation_config() -> Dict[str, Any]:
    """Get reward configuration for safe, cautious navigation.
    
    This configuration prioritizes:
    - Level completion with safety constraints
    - Hazard avoidance
    - Impact risk minimization
    
    Best for: Deployment scenarios, safety-critical applications
    
    Returns:
        dict: Reward configuration parameters
    """
    return {
        # Terminal rewards (same as completion-focused)
        "level_completion_reward": LEVEL_COMPLETION_REWARD,
        "death_penalty": DEATH_PENALTY * 2.0,  # Double penalty for safety
        "switch_activation_reward": SWITCH_ACTIVATION_REWARD,
        "time_penalty": TIME_PENALTY_PER_STEP * 0.5,  # Reduced time pressure
        
        # PBRS configuration (with safety potentials enabled)
        "enable_pbrs": True,
        "pbrs_gamma": PBRS_GAMMA,
        "pbrs_weights": {
            "objective_weight": PBRS_OBJECTIVE_WEIGHT,
            "hazard_weight": 0.5,  # Enable hazard avoidance
            "impact_weight": 0.3,  # Enable impact risk avoidance
            "exploration_weight": 0.0,
        },
        
        # Exploration configuration (reduced for safety)
        "enable_exploration_rewards": True,
        "exploration_scales": {
            "cell_reward": EXPLORATION_CELL_REWARD * 0.5,
            "area_4x4_reward": EXPLORATION_AREA_4X4_REWARD * 0.5,
            "area_8x8_reward": EXPLORATION_AREA_8X8_REWARD * 0.5,
            "area_16x16_reward": EXPLORATION_AREA_16X16_REWARD * 0.5,
        }
    }


def get_exploration_focused_config() -> Dict[str, Any]:
    """Get reward configuration for maximum exploration.
    
    This configuration prioritizes:
    - Comprehensive map coverage
    - Discovery of all areas
    - Balanced with completion objectives
    
    Best for: Curriculum learning, discovery phases
    
    Returns:
        dict: Reward configuration parameters
    """
    return {
        # Terminal rewards (reduced to encourage exploration over speed)
        "level_completion_reward": LEVEL_COMPLETION_REWARD,
        "death_penalty": DEATH_PENALTY * 0.5,  # Reduced to encourage risk-taking
        "switch_activation_reward": SWITCH_ACTIVATION_REWARD,
        "time_penalty": TIME_PENALTY_PER_STEP * 0.1,  # Minimal time pressure
        
        # PBRS configuration
        "enable_pbrs": True,
        "pbrs_gamma": PBRS_GAMMA,
        "pbrs_weights": {
            "objective_weight": PBRS_OBJECTIVE_WEIGHT * 0.5,  # Reduced objective focus
            "hazard_weight": 0.0,
            "impact_weight": 0.0,
            "exploration_weight": 0.5,  # Enable PBRS exploration
        },
        
        # Exploration configuration (boosted)
        "enable_exploration_rewards": True,
        "exploration_scales": {
            "cell_reward": EXPLORATION_CELL_REWARD * 3.0,
            "area_4x4_reward": EXPLORATION_AREA_4X4_REWARD * 3.0,
            "area_8x8_reward": EXPLORATION_AREA_8X8_REWARD * 3.0,
            "area_16x16_reward": EXPLORATION_AREA_16X16_REWARD * 3.0,
        }
    }


def get_minimal_shaping_config() -> Dict[str, Any]:
    """Get minimal reward configuration with only terminal rewards.
    
    This configuration includes:
    - Only terminal rewards (completion, death)
    - No reward shaping or exploration bonuses
    - Pure sparse reward signal
    
    Best for: Baseline comparisons, studying sparse reward learning
    
    Returns:
        dict: Reward configuration parameters
    """
    return {
        # Terminal rewards only
        "level_completion_reward": LEVEL_COMPLETION_REWARD,
        "death_penalty": DEATH_PENALTY,
        "switch_activation_reward": SWITCH_ACTIVATION_REWARD,
        "time_penalty": 0.0,  # No time penalty
        
        # All shaping disabled
        "enable_pbrs": False,
        "pbrs_gamma": PBRS_GAMMA,
        "pbrs_weights": {
            "objective_weight": 0.0,
            "hazard_weight": 0.0,
            "impact_weight": 0.0,
            "exploration_weight": 0.0,
        },
        
        # Exploration disabled
        "enable_exploration_rewards": False,
        "exploration_scales": {
            "cell_reward": 0.0,
            "area_4x4_reward": 0.0,
            "area_8x8_reward": 0.0,
            "area_16x16_reward": 0.0,
        }
    }


def get_speed_optimized_config() -> Dict[str, Any]:
    """Get reward configuration for learning optimal, efficient routes.
    
    This configuration encourages fast level completion while maintaining
    completion as the primary goal. Designed for fine-tuning after initial
    training has achieved reliable completion.
    
    Key features:
    - Progressive time penalty (increasing pressure over episode)
    - Completion time bonus (rewards fast solutions)
    - Strong navigation shaping (efficient routing)
    - Minimal exploration (assumes agent knows how to complete)
    
    Training curriculum:
    1. Initial training: Use completion_focused_config
    2. Fine-tuning: Switch to speed_optimized_config
    
    Best for: Speedrunning, route optimization, fine-tuning
    
    Returns:
        dict: Reward configuration parameters
    """
    return {
        # Terminal rewards
        "level_completion_reward": LEVEL_COMPLETION_REWARD,
        "death_penalty": DEATH_PENALTY,
        "switch_activation_reward": SWITCH_ACTIVATION_REWARD,
        
        # Progressive time penalty
        "time_penalty_mode": "progressive",  # vs "fixed"
        "time_penalty_early": TIME_PENALTY_EARLY,
        "time_penalty_middle": TIME_PENALTY_MIDDLE,
        "time_penalty_late": TIME_PENALTY_LATE,
        "time_penalty_early_threshold": TIME_PENALTY_EARLY_THRESHOLD,
        "time_penalty_late_threshold": TIME_PENALTY_LATE_THRESHOLD,
        
        # Completion time bonus
        "enable_completion_bonus": True,
        "completion_bonus_max": COMPLETION_TIME_BONUS_MAX,
        "completion_bonus_target": COMPLETION_TIME_TARGET,
        
        # PBRS (focus on efficient navigation)
        "enable_pbrs": True,
        "pbrs_gamma": PBRS_GAMMA,
        "pbrs_weights": {
            "objective_weight": PBRS_OBJECTIVE_WEIGHT * 1.5,  # Stronger nav signal
            "hazard_weight": 0.0,  # Speed over safety
            "impact_weight": 0.0,  # Speed over caution
            "exploration_weight": 0.0,  # Minimal exploration in fine-tuning
        },
        
        # Reduced exploration (agent already knows how to complete)
        "enable_exploration_rewards": False,
        "exploration_scales": {
            "cell_reward": 0.0,
            "area_4x4_reward": 0.0,
            "area_8x8_reward": 0.0,
            "area_16x16_reward": 0.0,
        }
    }


# =============================================================================
# REWARD VALIDATION AND SANITY CHECKS
# =============================================================================

def validate_reward_config(config: Dict[str, Any]) -> bool:
    """Validate reward configuration for common pitfalls.
    
    Checks for:
    - Terminal rewards dominating time penalties
    - PBRS weights in reasonable ranges
    - Exploration rewards not overwhelming primary objectives
    
    Args:
        config: Reward configuration dictionary
        
    Returns:
        bool: True if configuration passes validation
        
    Raises:
        ValueError: If configuration has critical issues
    """
    max_episode_steps = 20000  # N++ default
    
    # Check terminal reward magnitudes
    completion = config.get("level_completion_reward", LEVEL_COMPLETION_REWARD)
    death = config.get("death_penalty", DEATH_PENALTY)
    time_penalty = config.get("time_penalty", TIME_PENALTY_PER_STEP)
    
    # Terminal rewards should dominate cumulative time penalties
    max_time_penalty = abs(time_penalty) * max_episode_steps
    if completion < max_time_penalty:
        raise ValueError(
            f"Completion reward ({completion}) is smaller than maximum "
            f"cumulative time penalty ({max_time_penalty}). This can cause "
            f"the agent to optimize for death over completion."
        )
    
    # Death penalty should be meaningful but not overwhelming
    if abs(death) > abs(completion):
        print(
            "WARNING: Death penalty magnitude exceeds completion reward. "
            "This may lead to overly conservative behavior."
        )
    
    # PBRS weights should be moderate
    pbrs_weights = config.get("pbrs_weights", {})
    for key, value in pbrs_weights.items():
        if value > 5.0:
            print(
                f"WARNING: PBRS weight '{key}' = {value} is unusually large. "
                f"This may overwhelm terminal rewards."
            )
    
    # Exploration rewards should be modest
    exploration_scales = config.get("exploration_scales", {})
    total_exploration = sum(exploration_scales.values())
    if total_exploration > abs(time_penalty):
        print(
            f"WARNING: Total exploration reward ({total_exploration}) exceeds "
            f"time penalty magnitude ({abs(time_penalty)}). This may "
            f"encourage aimless wandering."
        )
    
    return True


def print_reward_summary(config: Dict[str, Any]) -> None:
    """Print human-readable summary of reward configuration.
    
    Args:
        config: Reward configuration dictionary
    """
    print("=" * 70)
    print("REWARD CONFIGURATION SUMMARY")
    print("=" * 70)
    
    print("\nTerminal Rewards:")
    print(f"  Level Completion: +{config.get('level_completion_reward', 0)}")
    print(f"  Death Penalty:    {config.get('death_penalty', 0)}")
    print(f"  Switch Activation: +{config.get('switch_activation_reward', 0)}")
    print(f"  Time Penalty:     {config.get('time_penalty', 0)} per step")
    
    print("\nPBRS Configuration:")
    print(f"  Enabled: {config.get('enable_pbrs', False)}")
    if config.get('enable_pbrs', False):
        weights = config.get('pbrs_weights', {})
        print(f"  Objective Weight:    {weights.get('objective_weight', 0)}")
        print(f"  Hazard Weight:       {weights.get('hazard_weight', 0)}")
        print(f"  Impact Weight:       {weights.get('impact_weight', 0)}")
        print(f"  Exploration Weight:  {weights.get('exploration_weight', 0)}")
    
    print("\nExploration Rewards:")
    print(f"  Enabled: {config.get('enable_exploration_rewards', False)}")
    if config.get('enable_exploration_rewards', False):
        scales = config.get('exploration_scales', {})
        print(f"  Cell (24x24):        {scales.get('cell_reward', 0)}")
        print(f"  Area (96x96):        {scales.get('area_4x4_reward', 0)}")
        print(f"  Area (192x192):      {scales.get('area_8x8_reward', 0)}")
        print(f"  Area (384x384):      {scales.get('area_16x16_reward', 0)}")
    
    print("=" * 70)
