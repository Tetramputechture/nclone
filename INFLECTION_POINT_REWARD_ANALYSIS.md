# Inflection Point Reward Balance Analysis

## Problem Statement

At sharp turn inflection points near hazards, the agent must choose:
- **Option A**: Follow PBRS path (turn away from hazard) → continue progress
- **Option B**: Continue toward hazard (ignore turn) → die at inflection point

We need to ensure Option A is ALWAYS more attractive than Option B.

## Current Reward Structure (Discovery Phase)

### Terminal Rewards
- Completion: **+50.0**
- Switch activation: **+40.0**
- Death penalty: **-8.0**

### PBRS Shaping
- Weight: **40.0** (discovery phase)
- Full path progress (0% → 100%): **+40.0** PBRS
- Per 1% progress: **+0.4** PBRS

### NEW: Anticipatory Guidance (This Implementation)
- Velocity alignment: **±0.1** per step (positive for correct direction, negative for wrong)
- Turn approach bonus: **+0.15** max per step (within 50px of turn waypoint)
- Combined max: **+0.25** per step when approaching turn correctly

## Scenario Analysis: Agent at 30% Progress, Inflection Point Ahead

### Setup
- Agent has made 30% progress along optimal path
- Sharp turn ahead at 35% progress (20px away, ~6-10 steps)
- Hazard located if agent continues straight instead of turning

### Option A: Follow Turn (Correct Path)

**Immediate rewards (approaching turn, 10 steps):**
- PBRS (30% → 35%): +0.05 × 40 = **+2.0**
- Velocity alignment (aligned with multi-hop): +0.1 × 10 = **+1.0**
- Approach bonus (near turn waypoint): +0.15 × 10 = **+1.5**
- **Subtotal: +4.5** over 10 steps

**Continued rewards (35% → 100%):**
- PBRS (35% → 100%): +0.65 × 40 = **+26.0**
- Switch activation: **+40.0**
- Completion: **+50.0**
- **Final total: +120.5**

### Option B: Continue Toward Hazard (Ignore Turn)

**Immediate rewards (before death, ~8 steps):**
- PBRS toward hazard: Variable (depends on if path continues that direction)
  - If hazard is OFF optimal path: **0 to -2.0** (backtracking penalty)
  - If hazard is ON optimal path: **+0.4 to +1.6** (1-4% progress)
- Velocity alignment (misaligned with multi-hop): -0.1 × 8 = **-0.8**
- Approach bonus (not moving toward turn): **0.0**

**Terminal reward:**
- Death penalty: **-8.0**

**Best case total (hazard on path):** +1.6 - 0.8 - 8.0 = **-7.2**
**Worst case total (hazard off path):** -2.0 - 0.8 - 8.0 = **-10.8**

## Reward Differential

| Metric | Follow Turn (A) | Continue to Hazard (B) | Differential |
|--------|-----------------|------------------------|--------------|
| Immediate (10 steps) | +4.5 | -1.2 to +0.8 | +3.7 to +5.7 |
| Final total | +120.5 | -10.8 to -7.2 | +127.7 to +131.3 |

**Conclusion:** Following the turn is **130x better** than dying at inflection point.

## Early Signal Strength (Critical for Learning)

The key question: Does the agent get EARLY enough signal to learn turning is better?

### 30px Before Turn (Pre-Anticipation)
- Multi-hop direction starts "bending" toward turn
- Velocity alignment: **±0.1** per step
  - Continuing straight: **-0.1** (misaligned)
  - Starting to angle toward turn: **+0.1** (aligned)
- **Differential: 0.2 per step** (10 steps = 2.0 cumulative advantage)

### 20px Before Turn (Approach Zone)
- Approach bonus activates: **+0.15** per step
- Velocity alignment: **±0.1** per step
- **Combined differential: 0.35 per step** (6 steps = 2.1 cumulative advantage)

### At Turn Point
- Approach bonus: **+0.15** (max proximity)
- Velocity alignment: **+0.1** (if aligned) or **-0.1** (if misaligned)
- **Differential: 0.25** immediate signal

## Potential Issues

### Issue 1: Velocity Alignment Might Be Too Weak

Current max: **0.1 per step**
- Over 10 steps before death: **±1.0** cumulative
- Death penalty: **-8.0**
- Ratio: **12.5%** of death penalty

If the PBRS reward for going toward hazard (even temporarily) is **+1.6**, then:
- Going toward hazard: +1.6 PBRS - 1.0 alignment = **+0.6**
- Going away (turn): +2.0 PBRS + 1.0 alignment = **+3.0**
- Differential: **+2.4** (5x advantage for turning)

### Issue 2: Death Might Not Be "Felt" Until Too Late

The policy learns Q(s,a) values. At the inflection point:
- If the agent has seen many "RIGHT → death" trajectories with net reward -7.2
- But also seen "RIGHT → progress" trajectories early on with reward +12
- The policy might not learn that THIS specific state requires turning

**Solution:** The velocity alignment penalty accumulates BEFORE death:
- 10 steps of -0.1 misalignment = **-1.0** accumulated
- This is "felt" by the value function before hitting hazard
- Creates gradient that makes continuing-straight less attractive

## Recommended Adjustments

### Option 1: Increase Velocity Alignment Strength

Current: **0.1** max per step
Proposed: **0.2** max per step

**Rationale:**
- Doubles the early signal strength (±2.0 over 10 steps)
- Makes misalignment penalty (−2.0) = **25% of death penalty** (−8.0)
- Creates stronger gradient BEFORE the critical decision point
- Still smaller than PBRS (0.2 vs 0.4 per 1% progress), maintaining hierarchy

### Option 2: Increase Death Penalty at Low Progress

Current: **-8.0** (uniform for <5% success)
Proposed: **-12.0** for deaths in first 50% of path

**Rationale:**
- Early deaths (before switch) get stronger penalty
- Makes "progress+death" even less attractive relative to "turn+continue"
- Differential: -12 vs -8 = **50% stronger deterrent** for premature deaths
- Encourages agent to be more conservative before major milestones

### Option 3: Add Hazard Proximity Penalty (Soft)

When velocity is toward nearby hazard (within 40px), add small penalty:
- Penalty: **-0.05** per step when approaching hazard
- Only applies when moving toward hazard (not static)
- Creates immediate negative feedback before death

## Recommendation

**Implement Option 1 (Increase Velocity Alignment to 0.2)**

This is the cleanest solution because:
1. ✓ Maintains policy invariance (still PBRS-based shaping)
2. ✓ Provides early signal (30-50px before inflection)
3. ✓ Strengthens existing mechanism (velocity alignment already implemented)
4. ✓ No curriculum complexity (uniform across training)
5. ✓ Preserves reward hierarchy (still much smaller than PBRS and terminals)

With 0.2 velocity alignment:
- Turning correctly: +2.0 + 2.0 alignment + 1.5 approach = **+5.5**
- Going straight to death: -10.8 to -9.2 (includes -2.0 alignment penalty)
- **Differential: +14.7 to +15.7** (very clear signal)

## Implementation

```python
# In main_reward_calculator.py:_get_turn_velocity_alignment()
MAX_ALIGNMENT_BONUS = 0.2  # INCREASED from 0.1 for stronger anticipatory guidance
```

This makes velocity alignment **20% of PBRS weight** (0.2 vs 1.0 per 1% progress), which is significant enough to guide but not dominate.
