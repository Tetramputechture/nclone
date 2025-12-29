# Temporal Exploration Solutions for Action Sequence Discovery

## Problem Analysis

**Observed Behavior** (from route at step 3767040):
```
Steps 0-40:   Action=RIGHT (learned well, makes progress)
Step 40:      Critical junction - needs JUMP+LEFT
Step 40+:     Agent keeps doing RIGHT (overconfident)
Result:       Dies to mines at 47 steps
```

**Root Cause**: **Temporal Exploitation Trap**
- Policy learned early sequence (RIGHT × 40) very well
- High confidence in this sequence → low entropy → no exploration
- Fails to discover the critical action change (JUMP+LEFT)
- Gets stuck in local minimum

---

## Solution 1: Go-Explore Checkpointing (BEST for this problem) 🎯

### What It Does
Your codebase already has Go-Explore! It:
1. Saves checkpoints at positions with best cumulative reward
2. Stores full action sequence to reach checkpoint
3. Can reset episodes TO that checkpoint
4. Explores from there with fresh entropy

### How It Solves Your Problem
```
Episode 1:   Steps 0-40 (RIGHT × 40) → Checkpoint saved!
Episode 2:   Replay steps 0-40 → Start at junction → Explore with entropy
Episode 3:   Replay steps 0-40 → Try JUMP+LEFT → Discover new path!
```

**This is EXACTLY designed for your use case!**

### Configuration
```bash
--enable-go-explore \
--checkpoint-selection-strategy ucb  # UCB balances exploitation/exploration
```

### Parameters in Code
```python
# From go_explore_callback.py
checkpoint_reward_threshold: 0.1  # Save when reward improves by 0.1 scaled
selection_strategy: "ucb"          # Upper Confidence Bound for selection

# UCB formula balances:
# - Reward (exploitation): prefer high-reward checkpoints
# - Visit count (exploration): prefer less-visited checkpoints
```

### Expected Behavior
```
Early training:
  - Discovers RIGHT × 40 sequence → checkpoint at step 40
  - Archives this as "frontier" checkpoint
  
Mid training:
  - 30% of episodes start from step-40 checkpoint
  - Explores different actions at the junction
  - Eventually discovers JUMP+LEFT
  - Creates new checkpoint at step 50+
  
Late training:
  - Has checkpoints covering full path
  - Can explore from any difficult section
  - Success rate improves rapidly
```

---

## Solution 2: Temporal Entropy Scheduling (COMPLEMENTARY) 📊

### What It Does
Increase entropy coefficient for **later timesteps** in episode:

```python
# Pseudocode
def get_entropy_coef(timestep_in_episode):
    if timestep_in_episode < 40:
        return 0.01  # Low - beginning is learned
    else:
        return 0.05  # High - explore novel sequences
```

### Why It Helps
- Early steps: Exploit known good sequence (RIGHT × 40)
- Later steps: Explore more aggressively at decision points
- Naturally focuses exploration where needed

### Implementation Difficulty
**HARD** - Requires modifying PPO's entropy calculation to be timestep-aware

### Alternative: Curriculum Entropy
```python
# In reward_config.py
@property
def entropy_coef(self) -> float:
    if self.recent_success_rate < 0.10:
        return 0.05  # High exploration when stuck
    elif self.recent_success_rate < 0.30:
        return 0.03  # Moderate
    return 0.01  # Low when successful
```

**EASIER** - Just pass this to PPO configuration

---

## Solution 3: Action Diversity Bonus (SIMPLE) 🎲

### What It Does
Penalize repeating the same action too many consecutive times:

```python
# In main_reward_calculator.py
def calculate_diversity_bonus(action_history: List[int]) -> float:
    """Bonus for using diverse actions, penalty for repetition."""
    if len(action_history) < 10:
        return 0.0
    
    # Check last 10 actions
    recent = action_history[-10:]
    unique_actions = len(set(recent))
    
    if unique_actions == 1:
        return -0.05  # Penalty for doing same action 10x
    elif unique_actions >= 4:
        return +0.05  # Bonus for diversity
    return 0.0
```

### Pros
- Simple to implement
- Directly addresses "RIGHT × 40" problem
- No hyperparameter tuning needed

### Cons
- May interfere with intentionally repetitive strategies
- Not policy-invariant (changes optimal policy)

---

## Solution 4: RND with (State, Action) Features (MODERATE) 🔬

### What It Does
Current RND uses only **state** for novelty. Modify to use **(state, action)** pairs:

```python
# In rnd_callback.py _extract_features_for_rnd()
features = [
    game_state,           # 41 dims
    reachability,         # 7 dims  
    action_one_hot,       # 6 dims (NEW!)
]
```

### Why It Helps
- Rewards novel **action sequences**, not just state visitation
- Agent gets bonus for trying JUMP+LEFT at the junction
- Even if it's visited that state before with RIGHT

### Configuration
```bash
--enable-rnd \
--rnd-initial-weight 1.0 \  # Moderate weight (vs PBRS 40)
--rnd-grid-size 12
```

### Pros
- Encourages action exploration
- Works with any architecture
- Already implemented (just need to add action features)

### Cons
- Adds complexity
- May compete with PBRS signal
- Needs careful weight balancing

---

## Solution 5: Progressive Entropy (EASIEST) 📈

### What It Does
**Gradually increase entropy as training progresses** (opposite of normal decay):

```python
@property
def entropy_coef(self) -> float:
    """Increase entropy as agent gets stuck to force exploration."""
    # Start conservative, increase if no improvement
    if self.current_timesteps < 500_000:
        return 0.01  # Low initially (learn basics)
    elif self.recent_success_rate < 0.05:
        return 0.05  # INCREASE when stuck (force exploration)
    elif self.recent_success_rate < 0.20:
        return 0.03  # Moderate
    return 0.01  # Reduce when successful
```

### Why It Works
- If agent gets stuck (success < 5% after 500K steps)
- Automatically increases entropy to 0.05
- Forces exploration of novel action sequences
- Reduces back to 0.01 once learning happens

### Implementation
Just add this property to `RewardConfig` class

---

## Recommendation: **Use Go-Explore** (Solution 1) 🏆

### Why Go-Explore is Best

1. **Designed for this exact problem**
   - "Learn beginning, explore ending" is Go-Explore's core use case
   - Archive's partial progress (step 40)
   - Explore from there repeatedly

2. **Already implemented and battle-tested**
   - Your codebase has full Go-Explore with action replay
   - Just needs to be enabled with `--enable-go-explore`

3. **No reward modification needed**
   - Keeps PBRS clean and policy-invariant
   - Exploration is architectural, not reward-based

4. **Proven effective**
   - Go-Explore solved Montezuma's Revenge (1000x harder than your problem)
   - Particularly good for deterministic environments like N++

### Quick Comparison

| Solution | Effectiveness | Complexity | Keeps PBRS Clean |
|----------|--------------|------------|------------------|
| **Go-Explore** | ⭐⭐⭐⭐⭐ | Low (already exists) | ✓ |
| Temporal Entropy | ⭐⭐⭐ | High (PPO modification) | ✓ |
| Action Diversity | ⭐⭐ | Low | ✗ (changes policy) |
| RND + action | ⭐⭐⭐⭐ | Moderate | ✓ |
| Progressive Entropy | ⭐⭐⭐ | Very Low | ✓ |

---

## Recommended Training Command

```bash
python scripts/train_and_compare.py \
    --experiment-name gf-2025-12-18-goexplore \
    --architectures graph_free \
    --single-level '../nclone/test-single-level/006 both flavours of ramp jumping (and the control thereof)' \
    --total-timesteps 20000000 \
    --hardware-profile auto \
    --frame-skip 4 \
    --enable-state-stacking \
    --state-stack-size 4 \
    --use-lstm \
    --lstm-hidden-size 128 \
    --lstm-num-layers 1 \
    --enable-early-stopping \
    --enable-go-explore \
    --checkpoint-selection-strategy ucb \
    --record-eval-videos \
    --num-eval-episodes 10
```

**Key addition**: `--enable-go-explore --checkpoint-selection-strategy ucb`

---

## How Go-Explore Will Help

### Phase 1: Initial Learning (0-500K steps)
```
Agent learns: RIGHT × 40 (gets to junction)
Checkpoint saved: Position at step 40, cumulative reward +0.8
Archive: 1 checkpoint
```

### Phase 2: Junction Exploration (500K-2M steps)
```
30% of episodes: Start from step-40 checkpoint
Agent tries: RIGHT (known), JUMP, JUMP+LEFT, etc.
Eventually: Discovers JUMP+LEFT works!
New checkpoint: Position at step 50, reward +1.2
Archive: 2 checkpoints (step 40, step 50)
```

### Phase 3: Path Completion (2M-10M steps)
```
Episodes mix: spawn starts + checkpoint starts
Agent discovers: Full path from spawn to exit
Success rate: Increases to 20-40%
Archive: 5-10 checkpoints covering critical junctions
```

---

## Alternative: Combine Go-Explore + Progressive Entropy

For maximum effectiveness:

### Add Progressive Entropy to RewardConfig

```python
@property
def entropy_coefficient(self) -> float:
    """Adaptive entropy based on learning progress."""
    if self.current_timesteps < 500_000:
        return 0.02  # Moderate initial exploration
    elif self.recent_success_rate < 0.05:
        return 0.05  # BOOST when stuck in local minimum
    elif self.recent_success_rate < 0.20:
        return 0.03  # Reduce as learning happens
    return 0.01  # Low for exploitation phase
```

### Pass to Trainer

```python
# In architecture_trainer.py, after update_config():
if hasattr(self.reward_config, 'entropy_coefficient'):
    self.model.ent_coef = self.reward_config.entropy_coefficient
```

This gives you:
- **Go-Explore**: Architectural solution (checkpoint + replay)
- **Progressive Entropy**: Adaptive exploration (boosts when stuck)
- **Best of both**: Checkpoints focus where to explore, entropy provides the randomness

---

## Summary

**Your Problem**: Agent learns beginning (RIGHT × 40) but doesn't explore novel actions at critical junction (JUMP+LEFT needed).

**Best Solution**: **Enable Go-Explore**
- Checkpoints progress at step 40
- Future episodes explore from there
- Discovers JUMP+LEFT through repeated junction attempts
- Already implemented, just add `--enable-go-explore`

**Complementary**: Add progressive entropy that increases when stuck (<5% success after 500K steps)

**Expected**: Success rate should improve within 2-3M steps as Go-Explore builds checkpoint archive of good partial paths.

Want me to implement the progressive entropy property as well?
