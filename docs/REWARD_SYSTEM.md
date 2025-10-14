# N++ Reward System Documentation

## Overview

The nclone reward system implements a sophisticated multi-component reward structure designed for deep reinforcement learning training. The system follows established RL best practices and is grounded in reward shaping theory.

**Key Design Principles:**
1. **No Magic Numbers**: All reward constants defined in `reward_constants.py` with full documentation
2. **Policy Invariance**: Uses Potential-Based Reward Shaping (PBRS) to provide dense rewards without changing optimal policy
3. **Multi-Scale Learning**: Combines terminal rewards, milestone rewards, and dense shaping signals
4. **Theoretical Foundation**: Based on peer-reviewed research (Ng et al. 1999, Pathak et al. 2017, Bellemare et al. 2016)
5. **Production Ready**: Includes validation, presets, and comprehensive testing

## Table of Contents

- [Reward Components](#reward-components)
- [Reward Constants](#reward-constants)
- [Configuration Presets](#configuration-presets)
- [Theoretical Foundation](#theoretical-foundation)
- [Usage Examples](#usage-examples)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Reward Components

The reward system consists of multiple components that work together to provide effective learning signals:

### 1. Terminal Rewards

Terminal rewards provide the primary learning signal for task success/failure.

| Event | Reward | Rationale |
|-------|--------|-----------|
| **Level Completion** | +1.0 | Large positive reward for achieving the ultimate goal |
| **Death** | -0.5 | Moderate penalty discourages death without overwhelming learning |
| **Switch Activation** | +0.1 | Intermediate milestone reward (10% of completion) |

**Design Rationale:**
- Completion reward (1.0) dominates cumulative penalties over full episodes
- Death penalty (0.5 × completion) maintains focus on positive objectives
- Too large penalties lead to overly conservative behavior
- Switch reward (0.1) provides intermediate signal without creating local optima

### 2. Time-Based Penalties

Encourage efficient solutions by penalizing step count.

| Penalty | Value | Rationale |
|---------|-------|-----------|
| **Time Penalty per Step** | -0.01 | Small negative reward for each timestep |

**Design Rationale:**
- Noticeable over typical episodes (20,000 steps max = -200 total penalty)
- Smaller than terminal rewards to avoid overwhelming primary objectives
- Consistent with OpenAI SpinningUp recommendations

### 3. Navigation Rewards (Distance-Based Shaping)

Provide dense gradient for learning by rewarding progress toward objectives.

| Component | Scale | Description |
|-----------|-------|-------------|
| **Distance Improvement** | 0.0001 | Reward for moving closer to current objective |
| **Proximity Bonus** | 0.0005 | Extra reward when within 20 pixels of objective |
| **Potential-Based Shaping** | 0.0005 | Continuous PBRS-based navigation signal |

**Design Rationale:**
- Distance improvement: Moving 100 pixels closer = 0.01 reward (≈ 1 timestep saved)
- Proximity bonus: Encourages precise navigation in final approach
- PBRS ensures policy invariance while providing meaningful gradient

### 4. Exploration Rewards (Multi-Scale Spatial Coverage)

Reward discovering new areas using count-based exploration methods.

| Scale | Area Size | Reward | Description |
|-------|-----------|--------|-------------|
| **Cell** | 24×24 px | 0.001 | Fine-grained exploration |
| **Medium Area** | 96×96 px | 0.001 | Room-sized regions |
| **Large Area** | 192×192 px | 0.001 | Section-sized regions |
| **Major Region** | 384×384 px | 0.001 | Major level regions |

**Design Rationale:**
- Multi-scale approach encourages both thorough and broad exploration
- Total max per-step exploration (0.004) < time penalty (0.01)
- Ensures exploration doesn't dominate time efficiency
- Based on Bellemare et al. (2016) count-based exploration

### 5. Potential-Based Reward Shaping (PBRS)

Theory-grounded reward shaping that maintains policy invariance.

**Formula:** `F(s,s') = γ * Φ(s') - Φ(s')`

| Potential | Weight | Description |
|-----------|--------|-------------|
| **Objective Distance** | 1.0 | Distance to switch/exit (primary) |
| **Hazard Proximity** | 0.0* | Proximity to dangerous mines |
| **Impact Risk** | 0.0* | High-velocity collision risk |
| **Exploration** | 0.0* | State visit novelty |

*Default disabled for completion-focused training. Can be enabled for safety-focused scenarios.

**Design Rationale:**
- Based on Ng et al. (1999) policy invariance theory
- Provides dense gradient without changing optimal policy
- Configurable weights allow task-specific tuning
- Normalized potentials ensure consistent scaling

## Reward Constants

All reward constants are centralized in `reward_constants.py` to eliminate magic numbers and provide clear documentation.

### Terminal Reward Constants

```python
LEVEL_COMPLETION_REWARD = 1.0      # Primary success signal
DEATH_PENALTY = -0.5                # Moderate death discouragement
SWITCH_ACTIVATION_REWARD = 0.1      # Intermediate milestone
TIME_PENALTY_PER_STEP = -0.01       # Efficiency encouragement
```

### Navigation Constants

```python
NAVIGATION_DISTANCE_IMPROVEMENT_SCALE = 0.0001  # Distance progress scaling
NAVIGATION_MIN_DISTANCE_THRESHOLD = 20.0         # Proximity bonus threshold (px)
NAVIGATION_POTENTIAL_SCALE = 0.0005              # PBRS shaping scale
```

### Exploration Constants

```python
EXPLORATION_GRID_WIDTH = 44          # Grid cells (horizontal)
EXPLORATION_GRID_HEIGHT = 25         # Grid cells (vertical)
EXPLORATION_CELL_SIZE = 24.0         # Pixels per cell
EXPLORATION_CELL_REWARD = 0.001      # Single cell reward
EXPLORATION_AREA_4X4_REWARD = 0.001  # Medium area reward
EXPLORATION_AREA_8X8_REWARD = 0.001  # Large area reward
EXPLORATION_AREA_16X16_REWARD = 0.001 # Major region reward
```

### PBRS Constants

```python
PBRS_GAMMA = 0.99                     # Discount factor for PBRS
PBRS_OBJECTIVE_WEIGHT = 1.0           # Objective distance weight
PBRS_HAZARD_WEIGHT = 0.0              # Hazard proximity weight (disabled)
PBRS_IMPACT_WEIGHT = 0.0              # Impact risk weight (disabled)
PBRS_EXPLORATION_WEIGHT = 0.0         # Exploration weight (disabled)
PBRS_SWITCH_DISTANCE_SCALE = 0.05     # Switch phase scaling
PBRS_EXIT_DISTANCE_SCALE = 0.05       # Exit phase scaling
```

### Intrinsic Motivation Constants (ICM)

For integration with npp-rl curiosity-driven exploration:

```python
ICM_ALPHA = 0.1                       # Intrinsic/extrinsic combination weight
ICM_REWARD_CLIP = 1.0                 # Maximum intrinsic reward
ICM_FORWARD_LOSS_WEIGHT = 0.9         # Forward model loss weight
ICM_INVERSE_LOSS_WEIGHT = 0.1         # Inverse model loss weight
ICM_LEARNING_RATE = 1e-3              # ICM network learning rate
```

## Configuration Presets

Pre-configured reward settings for common training scenarios.

### 1. Completion-Focused (Default)

**Best for:** Initial training, speed-running objectives

```python
from nclone.gym_environment.reward_calculation.reward_constants import (
    get_completion_focused_config
)

config = get_completion_focused_config()
reward_calculator = RewardCalculator(
    enable_pbrs=config["enable_pbrs"],
    pbrs_weights=config["pbrs_weights"],
    pbrs_gamma=config["pbrs_gamma"]
)
```

**Characteristics:**
- Maximizes level completion speed
- High terminal rewards, efficient time penalties
- Minimal safety constraints
- Objective-focused PBRS only

### 2. Safe Navigation

**Best for:** Deployment scenarios, safety-critical applications

```python
from nclone.gym_environment.reward_calculation.reward_constants import (
    get_safe_navigation_config
)

config = get_safe_navigation_config()
reward_calculator = RewardCalculator(
    enable_pbrs=config["enable_pbrs"],
    pbrs_weights=config["pbrs_weights"],
    pbrs_gamma=config["pbrs_gamma"]
)
```

**Characteristics:**
- Level completion with safety constraints
- Double death penalty (-1.0)
- Hazard avoidance enabled (weight 0.5)
- Impact risk avoidance enabled (weight 0.3)
- Reduced time pressure (0.5×)

### 3. Exploration-Focused

**Best for:** Curriculum learning, discovery phases

```python
from nclone.gym_environment.reward_calculation.reward_constants import (
    get_exploration_focused_config
)

config = get_exploration_focused_config()
reward_calculator = RewardCalculator(
    enable_pbrs=config["enable_pbrs"],
    pbrs_weights=config["pbrs_weights"],
    pbrs_gamma=config["pbrs_gamma"]
)
```

**Characteristics:**
- Comprehensive map coverage priority
- Boosted exploration rewards (3×)
- Reduced death penalty (0.5×)
- Minimal time pressure (0.1×)
- Exploration PBRS enabled

### 4. Minimal Shaping (Sparse Rewards)

**Best for:** Baseline comparisons, pure RL research

```python
from nclone.gym_environment.reward_calculation.reward_constants import (
    get_minimal_shaping_config
)

config = get_minimal_shaping_config()
reward_calculator = RewardCalculator(
    enable_pbrs=config["enable_pbrs"],
    pbrs_weights=config["pbrs_weights"],
    pbrs_gamma=config["pbrs_gamma"]
)
```

**Characteristics:**
- Terminal rewards only (completion, death, switch)
- No time penalties
- No reward shaping or exploration bonuses
- Pure sparse reward signal

## Theoretical Foundation

### Potential-Based Reward Shaping (PBRS)

**Reference:** Ng, A.Y., Harada, D., and Russell, S. (1999). "Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping."

**Key Theorem:** Given a potential function Φ(s), the shaped reward:
```
F(s,s') = γ * Φ(s') - Φ(s)
```
preserves the optimal policy while providing dense reward signals.

**Proof Sketch:**
The shaped return for an episode is:
```
G^Φ = Σ γ^i (r_i + F(s_i, s_{i+1}))
    = Σ γ^i r_i + Σ γ^{i+1} Φ(s_{i+1}) - Σ γ^i Φ(s_i)
    = G + γ Φ(s_∞) - Φ(s_0)
    = G - Φ(s_0)    [assuming γ Φ(s_∞) = 0]
```

Since the shaped return differs from the original only by a constant (the initial potential), the optimal policy is unchanged.

**Implications:**
- We can safely add dense PBRS rewards for faster learning
- The optimal policy remains theoretically guaranteed
- Poor potential functions slow learning but don't break optimality

### Count-Based Exploration

**Reference:** Bellemare, M. et al. (2016). "Unifying Count-Based Exploration and Intrinsic Motivation."

**Principle:** Reward visiting novel states with frequency-based bonuses:
```
r_exploration(s) = β / √count(s)
```

**Our Implementation:**
- Multi-scale grid-based counting (24px, 96px, 192px, 384px cells)
- Binary visit tracking (visited/unvisited)
- Resets on switch activation to encourage re-exploration

**Benefits:**
- Simple and computationally efficient
- Encourages broad coverage
- Works well in procedurally generated environments

### Intrinsic Curiosity Module (ICM)

**Reference:** Pathak, D. et al. (2017). "Curiosity-driven Exploration by Self-supervised Prediction."

**Principle:** Reward prediction errors in learned dynamics models:
```
r_intrinsic = η * ||φ(s') - f_forward(φ(s), a)||²
```

**Implementation** (in npp-rl):
- Forward model predicts next state features from current state + action
- Inverse model predicts action from state transition
- Prediction error provides curiosity bonus
- Combined with extrinsic rewards: `r_total = r_ext + α * r_int`

**Benefits:**
- Effective in sparse-reward environments
- Learns task-relevant features
- Scales to high-dimensional observations

## Usage Examples

### Basic Usage

```python
from nclone.gym_environment.reward_calculation import RewardCalculator

# Create calculator with default (completion-focused) settings
calculator = RewardCalculator(
    enable_pbrs=True,
    pbrs_gamma=0.99
)

# Calculate reward for state transition
reward = calculator.calculate_reward(current_obs, previous_obs)

# Get detailed reward breakdown
components = calculator.get_reward_components(current_obs)
print(f"Navigation reward: {components.get('navigation', 0)}")
print(f"Exploration reward: {components.get('exploration', 0)}")
print(f"PBRS reward: {components.get('pbrs', 0)}")
```

### Custom Configuration

```python
from nclone.gym_environment.reward_calculation import RewardCalculator

# Create custom reward configuration
custom_weights = {
    "objective_weight": 0.8,    # Slightly reduced objective focus
    "hazard_weight": 0.3,        # Enable moderate hazard avoidance
    "impact_weight": 0.2,        # Enable light impact avoidance
    "exploration_weight": 0.1,   # Small exploration bonus
}

calculator = RewardCalculator(
    enable_pbrs=True,
    pbrs_weights=custom_weights,
    pbrs_gamma=0.99
)
```

### Validating Configuration

```python
from nclone.gym_environment.reward_calculation.reward_constants import (
    get_completion_focused_config,
    validate_reward_config,
    print_reward_summary
)

# Get preset configuration
config = get_completion_focused_config()

# Validate for common issues
try:
    validate_reward_config(config)
    print("Configuration valid!")
except ValueError as e:
    print(f"Configuration error: {e}")

# Print human-readable summary
print_reward_summary(config)
```

### Integration with Environment

```python
from nclone import NPPEnvironment
from nclone.gym_environment.reward_calculation import RewardCalculator
from nclone.gym_environment.reward_calculation.reward_constants import (
    get_completion_focused_config
)

# Create environment
env = NPPEnvironment(
    render_mode="rgb_array",
    dataset_dir="datasets/train"
)

# Create custom reward calculator
config = get_completion_focused_config()
reward_calculator = RewardCalculator(
    enable_pbrs=config["enable_pbrs"],
    pbrs_weights=config["pbrs_weights"],
    pbrs_gamma=config["pbrs_gamma"]
)

# Training loop with custom rewards
obs, info = env.reset()
for _ in range(1000):
    action = policy(obs)
    next_obs, base_reward, terminated, truncated, info = env.step(action)
    
    # Override with custom reward (if needed)
    # custom_reward = reward_calculator.calculate_reward(next_obs, obs)
    
    # Train agent...
    
    if terminated or truncated:
        reward_calculator.reset()  # Reset for new episode
        obs, info = env.reset()
    else:
        obs = next_obs
```

## Best Practices

### 1. Start with Default Configuration

Begin training with the completion-focused preset:
- Well-tested and balanced
- Suitable for most use cases
- Provides good baseline performance

### 2. Validate Before Training

Always validate custom configurations:
```python
from nclone.gym_environment.reward_calculation.reward_constants import (
    validate_reward_config, print_reward_summary
)

# Validate configuration
validate_reward_config(config)

# Print summary to verify settings
print_reward_summary(config)
```

### 3. Monitor Reward Components

Track individual reward components during training:
```python
# Log component breakdown
components = calculator.last_pbrs_components
logger.log("navigation_reward", components["navigation_reward"])
logger.log("exploration_reward", components["exploration_reward"])
logger.log("pbrs_reward", components["pbrs_reward"])
logger.log("total_reward", components["total_reward"])
```

### 4. Tune Incrementally

When customizing rewards:
1. Start from a working preset
2. Change one component at a time
3. Validate after each change
4. Monitor training stability

### 5. Consider Curriculum Learning

Use different configurations at different training stages:
```python
# Stage 1: Exploration phase (1M steps)
config_explore = get_exploration_focused_config()

# Stage 2: Completion phase (5M steps)
config_complete = get_completion_focused_config()

# Stage 3: Safety refinement (2M steps)
config_safe = get_safe_navigation_config()
```

### 6. Avoid Common Pitfalls

**❌ Don't:**
- Make terminal rewards smaller than cumulative time penalties
- Use death penalties larger than completion rewards
- Create exploration rewards larger than time penalties
- Modify constants without documenting rationale

**✅ Do:**
- Use provided validation functions
- Document all custom configurations
- Monitor reward component distributions
- Test thoroughly before long training runs

## Troubleshooting

### Problem: Agent learns to die quickly

**Symptoms:**
- Negative cumulative rewards
- Very short episodes
- Agent seeks death

**Causes:**
- Time penalty too large relative to completion reward
- Death penalty too small

**Solutions:**
```python
# Increase completion reward or decrease time penalty
config["level_completion_reward"] = 2.0  # Double completion reward
# OR
config["time_penalty"] = -0.005  # Halve time penalty

# Validate
validate_reward_config(config)
```

### Problem: Agent wanders aimlessly

**Symptoms:**
- Long episodes without progress
- High exploration rewards
- Low completion rate

**Causes:**
- Exploration rewards too large
- Objective weights too small
- Time penalty too small

**Solutions:**
```python
# Reduce exploration or increase objective focus
config["exploration_scales"] = {
    "cell_reward": 0.0005,  # Halve exploration rewards
    # ...
}
config["pbrs_weights"]["objective_weight"] = 2.0  # Double objective weight
config["time_penalty"] = -0.02  # Double time penalty
```

### Problem: Overly conservative behavior

**Symptoms:**
- Agent avoids risky but optimal paths
- Very slow but safe navigation
- Low death rate but poor completion rate

**Causes:**
- Death penalty too large
- Hazard/impact weights too high

**Solutions:**
```python
# Reduce safety constraints
config["death_penalty"] = -0.25  # Reduce death penalty
config["pbrs_weights"]["hazard_weight"] = 0.0  # Disable hazard avoidance
config["pbrs_weights"]["impact_weight"] = 0.0  # Disable impact avoidance
```

### Problem: Unstable training

**Symptoms:**
- Reward spikes and crashes
- Policy oscillations
- Diverging losses

**Causes:**
- Reward scale too large
- Conflicting reward components
- Missing reward clipping

**Solutions:**
```python
# Scale down all rewards proportionally
config["level_completion_reward"] = 0.1
config["death_penalty"] = -0.05
config["time_penalty"] = -0.001
# ... scale all components by 0.1

# Add reward clipping in training loop
reward = np.clip(reward, -1.0, 1.0)
```

### Problem: Slow convergence

**Symptoms:**
- Low learning rate
- Minimal improvement over time
- Sparse reward signals

**Causes:**
- PBRS disabled or misconfigured
- Exploration rewards too small
- Poor potential functions

**Solutions:**
```python
# Enable/enhance reward shaping
config["enable_pbrs"] = True
config["pbrs_weights"]["objective_weight"] = 1.5  # Increase shaping

# Boost exploration
config["exploration_scales"] = {
    key: value * 2.0 for key, value in config["exploration_scales"].items()
}
```

## References

1. **Ng, A.Y., Harada, D., and Russell, S. (1999).** "Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping." *ICML 1999*.

2. **Pathak, D., Agrawal, P., Efros, A.A., and Darrell, T. (2017).** "Curiosity-driven Exploration by Self-supervised Prediction." *ICML 2017*.

3. **Bellemare, M., Srinivasan, S., Ostrovski, G., Schaul, T., Saxton, D., and Munos, R. (2016).** "Unifying Count-Based Exploration and Intrinsic Motivation." *NIPS 2016*.

4. **Sutton, R.S. and Barto, A.G. (2018).** "Reinforcement Learning: An Introduction." *MIT Press*.

5. **Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. (2017).** "Proximal Policy Optimization Algorithms." *arXiv:1707.06347*.

6. **OpenAI Spinning Up.** "Introduction to RL." https://spinningup.openai.com/

## Contributing

When modifying the reward system:

1. **Update reward_constants.py** with new constants and full documentation
2. **Add validation** for new parameters in `validate_reward_config()`
3. **Create tests** in `tests/test_reward_calculation.py`
4. **Update this documentation** with rationale and examples
5. **Cite research** backing design decisions

For questions or suggestions, please open an issue on GitHub.
