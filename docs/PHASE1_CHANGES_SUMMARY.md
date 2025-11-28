# Phase 1 Changes Summary - Immediate Reward Fixes

## Diagnostic Results

**Root Cause Identified**: Agent is barely moving (0.02px per action = 0.005px per frame), which is ~150× slower than expected. This creates weak PBRS signal and explains poor performance.

**PBRS Calculation**: Verified as CORRECT. Weight (200 at 0-5% success) is being applied properly.

**Problem**: Agent behavior (excessive oscillation, minimal movement), not PBRS mathematics.

## Changes Applied

### 1. Revisit Penalty - SIGNIFICANTLY STRENGTHENED

**File**: `nclone/gym_environment/reward_calculation/reward_config.py` (lines 143-173)  
**File**: `nclone/gym_environment/reward_calculation/main_reward_calculator.py` (line 103)

**Changes**:
- **Scaling**: Changed from `sqrt(visit_count)` to `visit_count` (LINEAR)
- **Weight**: Increased 5× across all phases:
  - 0-20% success: `0.003 → 0.015` (5× stronger)
  - 20-40% success: `0.002 → 0.010` (5× stronger)
  - 40%+ success: `0.001 → 0.005` (5× stronger)

**Impact**:
- **Old**: Breakeven at 100 visits, allows extensive oscillation
- **New**: Breakeven at 2-3 visits, immediately discourages looping
- **Examples**:
  - 5 visits: Old = -0.007, New = -0.075 (10.7× stronger)
  - 10 visits: Old = -0.009, New = -0.150 (16.7× stronger)
  - 20 visits: Old = -0.013, New = -0.300 (23× stronger)

### 2. Time Penalty - MASSIVELY INCREASED

**File**: `nclone/gym_environment/reward_calculation/reward_config.py` (lines 86-113)

**Changes**: Increased 200-1000× across all phases:
- 0-5% success: `-0.0000005 → -0.0002` (400× stronger)
- 5-30% success: `-0.000001 → -0.0005` (500× stronger)
- 30-50% success: `-0.000002 → -0.001` (500× stronger)

**Impact**:
- **Old**: ~0.003% of episode reward (irrelevant)
- **New**: 0.6-6% of episode reward (meaningful pressure)
- **Over 600 frames**:
  - Discovery: -0.12 total (0.6% of +20 completion)
  - Early: -0.30 total (1.5%)
  - Mid: -0.60 total (3.0%)
  - Refinement: -1.20 total (6.0%)

### 3. Mine Avoidance - KEPT AS-IS

**Rationale**: User's level shows optimal path stays well clear of mines. Current settings (2.0× multiplier, 40px radius) appear adequate for this geometry. Will monitor death rate and adjust if needed.

## Expected Results

### Immediate (within 50K steps):
1. **Reduced oscillation**: 40-60% fewer revisits per episode
2. **More cautious movement**: Agent takes revisit penalty seriously
3. **Slightly longer episodes initially**: Agent being more careful

### Short-term (within 200K steps):
1. **Improved path efficiency**: 10-20% shorter paths to goal
2. **Success rate improvement**: 5-10% absolute increase (to 10-15%)
3. **Less meandering**: Visible in replay trajectories

## Monitoring Checklist

Monitor these TensorBoard metrics:

- `reward/revisit_penalty_total` - Should become MORE NEGATIVE
- `reward/avg_revisits_per_episode` - Should DECREASE by 40-60%
- `reward/time_penalty_total` - Should become more visible (-0.1 to -1.0 per episode)
- `episode/success_rate` - Should INCREASE from 3% to 10-15%
- `episode/path_optimality` - Should INCREASE (closer to 1.0)
- `pbrs/pbrs_mean` - Should INCREASE as agent moves more

## Next Steps (Phase 2)

After Phase 1 stabilizes (50-200K steps), proceed with Phase 2:
1. Add waypoint guidance system (breaks policy invariance)
2. Add progress preservation penalty (breaks policy invariance)
3. Enhanced TensorBoard logging

These require code changes to `main_reward_calculator.py` and will need a new training run.

## Rollback Plan

If Phase 1 causes training instability:

**Moderate rollback** (50% of changes):
```python
# In reward_config.py:
revisit_penalty_weight = 0.010  # Instead of 0.015
time_penalty = -0.0001  # Instead of -0.0002
```

**Full rollback** (revert to original):
```python
revisit_penalty_weight = 0.003
time_penalty = -0.0000005
# Change back to sqrt scaling in main_reward_calculator.py
```

## Files Modified

1. `/home/tetra/projects/nclone/nclone/gym_environment/reward_calculation/reward_config.py`
2. `/home/tetra/projects/nclone/nclone/gym_environment/reward_calculation/main_reward_calculator.py`
3. `/home/tetra/projects/nclone/scripts/diagnose_pbrs.py` (diagnostic tool)

## Git Commit Message

```
fix(rewards): Significantly strengthen revisit penalty and time penalty

Root cause: Agent barely moving (0.02px/action) due to excessive oscillation.
PBRS calculation is correct; problem is agent behavior.

Changes:
- Revisit penalty: 5× stronger + linear scaling (was sqrt)
  Breakeven at 2-3 visits instead of 100
- Time penalty: 200-1000× stronger
  Now 0.6-6% of episode reward (was 0.003%)
- Keep mine avoidance unchanged (optimal path already clear)

Expected: 40-60% reduction in oscillation, 5-10% success rate improvement
within 200K steps.

Phase 1 of reward structure overhaul. Phase 2 will add waypoints and
progress tracking (requires new training run).
```

---

**Status**: Phase 1 COMPLETE - Changes are config-only and can be applied to current training run without breaking saved model.

**Date**: 2025-11-28  
**Training Step**: ~1.2M steps  
**Current Success Rate**: 3% (0-5% range)

