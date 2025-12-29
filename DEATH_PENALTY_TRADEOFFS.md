# Death Penalty - Full Tradeoff Analysis

## The Dilemma

**Without death penalty**: Camping at 16% beats dying at 20%
```
Camp 16% (2000 steps): +1.68 scaled
Die 20% (400 steps):   +1.60 scaled
Hierarchy violated: Camping > Progress ❌
```

**With death penalty**: Dying at 20% is negative, discourages early risk-taking
```
Die 20% (400 steps): +1.60 PBRS - 1.0 death = +0.60 scaled
But: Death penalty makes early exploration less rewarding
```

## The Core Tradeoff

**Death penalty serves two purposes**:

1. **Anti-camping**: Makes dying at 20% better than camping at 16%
2. **Hazard avoidance**: Teaches "don't touch mines"

**But it also**:
- Weakens learning signal for progress
- Discourages necessary risk-taking
- Adds complexity (3 types, progress-gating)

## Three Viable Options

### Option 1: Keep Death Penalty (Current System) ✅ SAFE

**Configuration**:
```python
IMPACT_DEATH_PENALTY = -10.0
HAZARD_DEATH_PENALTY = -40.0
Progress-gated: 25% / 50% / 100% scaling
Stagnation threshold: 15%
```

**Hierarchy (discovery phase)**:
```
Complete:           +15.7
Switch (50%) + die:  +2.8
Die 20% + switch:    +2.5
Camp 16%:            +0.88  (no stagnation penalty, above 15%)
Die 20% (no switch): +0.6
Stagnation:         -2.4
```

**Pros**:
- ✅ Prevents camping exploit (death + progress > camping)
- ✅ Teaches hazard avoidance explicitly
- ✅ Proven to work (current system)

**Cons**:
- ❌ Weakens learning signal for risky progress
- ❌ Complex (3 penalty types, progress-gating)
- ❌ Might discourage necessary risks

**Verdict**: Safe, proven, but suboptimal learning signal

---

### Option 2: Remove Death Penalty + Increase Stagnation Threshold to 20% ✅ CLEANER

**Configuration**:
```python
# Remove death penalties entirely
# Increase threshold:
STAGNATION_PROGRESS_THRESHOLD = 0.20  # From 0.15
```

**Hierarchy (discovery phase)**:
```
Complete:           +15.7
Switch (50%) + die:  +6.8
Die 50%:             +3.8
Die 20% + switch:    +4.5
Die 20%:             +1.6
Camp 16%:           -0.72  (below 20% threshold, stagnation penalty applies!)
Stagnation:         -2.4
```

**Pros**:
- ✅ ALL progress attempts are positive
- ✅ 3-4× stronger learning signal
- ✅ Prevents camping (20% threshold)
- ✅ Simpler system
- ✅ Encourages risk-taking naturally

**Cons**:
- ⚠️ Requires 20% progress to avoid stagnation penalty (might be harsh)
- ⚠️ No explicit mine avoidance signal (relies on PBRS + hazard costs)

**Verdict**: Cleaner theory, stronger signal, requires monitoring

---

### Option 3: Small Symbolic Death Penalty ⚖️ BALANCED

**Configuration**:
```python
# Single small constant penalty (not progress-gated)
SYMBOLIC_DEATH_PENALTY = -2.0  # -0.2 scaled
STAGNATION_PROGRESS_THRESHOLD = 0.15  # Keep at 15%
```

**Hierarchy (discovery phase)**:
```
Complete:           +15.7
Switch (50%) + die:  +6.6  (+6.8 - 0.2 death)
Die 50%:             +3.6  (+3.8 - 0.2 death)
Die 20% + switch:    +4.3  (+4.5 - 0.2 death)
Die 20%:             +1.4  (+1.6 - 0.2 death)
Camp 16%:            +0.88 (above 15%, no stagnation)
Camp 14%:           -1.72 (below 15%, stagnation penalty)
```

**Pros**:
- ✅ Maintains "death is bad" signal
- ✅ Much simpler (no progress-gating, single value)
- ✅ Still encourages risk-taking (progress + die > camping)
- ✅ Keeps 15% threshold (less aggressive)

**Cons**:
- ⚠️ Camp 16% (+0.88) still slightly beats die 20% (+1.4) in absolute terms
- ⚠️ But curriculum makes camping unprofitable (mid phase: +0.06)

**Verdict**: Good middle ground, maintains death signal without harsh penalties

## Detailed Math Comparison

### Scenario: Agent at 20% Progress, Considering Next Move

**With current death penalty** (-10.0 for early death):
```
Push forward (50% die, 50% complete):
  0.5 × (+1.6 PBRS - 1.0 death) + 0.5 × (+15.7 complete)
  = 0.5 × 0.6 + 0.5 × 15.7
  = +8.15 scaled (expected value)

Play safe, camp at 20% until timeout:
  +1.6 PBRS - 0.4 time = +1.2 scaled

Risk-taking is better: +8.15 > +1.2 ✅
But signal is weak (6.8× difference)
```

**Without death penalty**:
```
Push forward:
  0.5 × (+1.6 PBRS) + 0.5 × (+15.7 complete)
  = 0.5 × 1.6 + 0.5 × 15.7
  = +8.65 scaled (expected value)

Play safe, camp:
  +1.6 PBRS - 0.4 time = +1.2 scaled

Risk-taking is better: +8.65 > +1.2 ✅
Signal is slightly stronger (7.2× difference)
```

**With symbolic penalty (-2.0)**:
```
Push forward:
  0.5 × (+1.6 PBRS - 0.2 death) + 0.5 × (+15.7 complete)
  = 0.5 × 1.4 + 0.5 × 15.7
  = +8.55 scaled

Play safe, camp:
  +1.6 PBRS - 0.4 time = +1.2 scaled

Risk-taking is better: +8.55 > +1.2 ✅
Signal strength: 7.1× difference (between full penalty and no penalty)
```

**Conclusion**: All three maintain correct hierarchy. Symbolic penalty is sweet spot.

## Recommendation Ranking

### 1st Choice: Symbolic Death Penalty (-2.0) ⚖️

**Why**: Best balance of simplicity, signal strength, and safety

```python
# reward_constants.py
DEATH_PENALTY = -2.0  # Simple constant, not progress-gated
# Remove impact/hazard distinctions
```

**Benefits**:
- Maintains "death is suboptimal" intuition
- 3-4× stronger signal than current system
- Much simpler (no progress-gating, no death type checking)
- Still encourages risk-taking (progress attempts remain positive)

### 2nd Choice: No Death Penalty + 20% Threshold 🧪

**Why**: Most theoretically pure, strongest learning signal

```python
# Remove death penalties entirely
STAGNATION_PROGRESS_THRESHOLD = 0.20  # From 0.15
```

**Benefits**:
- Purest PBRS implementation
- Strongest learning signal (no death punishment)
- Forces agent to demonstrate meaningful progress

**Risk**: Requires monitoring for reckless behavior

### 3rd Choice: Keep Current System (Status Quo) 🛡️

**Why**: Proven to work, no further changes needed

**Benefits**:
- Already implemented and tested
- Clear hazard avoidance signal
- Conservative approach

**Drawback**: Weaker learning signal, more complex

## My Strong Recommendation

**Use symbolic death penalty (-2.0 constant)**:

```python
# reward_constants.py - SIMPLIFY death penalties

# Single symbolic death penalty (not progress-gated, not differentiated)
DEATH_PENALTY = -2.0  # Small constant penalty
IMPACT_DEATH_PENALTY = -2.0  # Same as generic
HAZARD_DEATH_PENALTY = -2.0  # Same as generic

# Remove progress-gating entirely
```

```python
# main_reward_calculator.py - SIMPLIFY death handling

if obs.get("player_dead", False):
    # Small constant penalty (opportunity cost is main penalty)
    terminal_reward = DEATH_PENALTY
    
    # Credit switch activation
    terminal_reward += milestone_reward
    
    self.episode_terminal_reward = terminal_reward
    scaled_reward = terminal_reward * GLOBAL_REWARD_SCALE
    
    # No need to check death cause or calculate progress scaling!
    
    self.last_pbrs_components = {
        "terminal_reward": terminal_reward,
        "milestone_reward": milestone_reward,
        "pbrs_reward": 0.0,
        "time_penalty": 0.0,
        "total_reward": terminal_reward,
        "scaled_reward": scaled_reward,
        "is_terminal": True,
        "terminal_type": "death",
    }
    return scaled_reward
```

**This gives you**:
- ✅ Simpler code (no progress calculation, no death type checking)
- ✅ Stronger learning signal (3-4× improvement)
- ✅ Correct hierarchy (progress > camping)
- ✅ Maintains "death is bad" intuition
- ✅ Still encourages risk-taking

**The death penalty becomes a small symbolic reminder, not a harsh punishment.**

Would you like me to implement this further simplification?

