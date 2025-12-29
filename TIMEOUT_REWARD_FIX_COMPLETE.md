# Timeout Reward Fix - Complete

**Issue**: All timeout routes showed exactly -0.40 reward despite different paths  
**Status**: ✅ Fixed

## Problem Diagnosis

**Observed**: Both timeout routes → -0.40 scaled reward

**Breakdown**:
```
-0.40 scaled = -4.0 unscaled
= -0.002 (time penalty) × 4 (frame skip) × 500 (actions) × 0.1 (scale)
= Time penalty ONLY
```

**This means**:
1. PBRS accumulated ≈ 0 (both routes oscillated)
2. Stagnation penalty NOT applied (BUG!)

## Root Cause: Stagnation Check Used "Best" Progress

**BUGGY CODE** (before fix):
```python
# Used CLOSEST distance reached (best progress ever)
closest_distance = self.reward_calculator.closest_distance_this_episode
progress = 1.0 - (closest_distance / combined_path)

if progress < 0.15:
    apply_stagnation_penalty()
```

**The exploit**:
1. Agent explores to 20% progress → `closest_distance` updated
2. Agent oscillates back to spawn
3. `closest_distance` still shows 20% (best ever reached)
4. No stagnation penalty (20% > 15%)
5. Final reward: -0.40 (time only)

**Both routes likely reached 15-20% at some point, avoiding the penalty.**

## Fix Applied ✅

**Use FINAL distance (where agent ended), not best distance**:

```python
# Use FINAL distance to goal (current position)
final_distance = final_obs.get("_pbrs_last_distance_to_goal", None)

if final_distance is None or final_distance == float("inf"):
    # Fallback: use closest if not cached
    final_distance = self.reward_calculator.closest_distance_this_episode

# Calculate progress based on FINAL position
progress = 1.0 - (final_distance / combined_path_distance)

if progress < STAGNATION_PROGRESS_THRESHOLD:
    apply_stagnation_penalty()
```

**Added diagnostic logging**:
```python
logger.info(
    f"[TIMEOUT_EPISODE] "
    f"final_reward={self.current_ep_reward:.3f}, "
    f"pbrs_total={pbrs_total_scaled:.3f}, "
    f"time_total={time_total_scaled:.3f}, "
    f"final_progress={progress:.1%}, "
    f"actions={action_count}"
)
```

## Expected Results After Fix

### Scenario 1: Oscillation Near Spawn (Most Likely for Both Routes)

**Route characteristics**:
- Explores forward and backward repeatedly
- Ends near spawn (final distance ≈ 900px of 1000px total)
- Net PBRS ≈ 0 (oscillation cancels out)

**Reward breakdown**:
```
PBRS accumulated:   ~0.0 unscaled (oscillation)
Time penalty:       -4.0 unscaled
Final progress:     ~10% (ended near spawn)
Stagnation penalty: -20.0 unscaled (< 15% threshold)
──────────────────────────────────────────────
Total:              -24.0 unscaled = -2.40 scaled
```

**Before fix**: -0.40 (bug - no stagnation penalty)  
**After fix**: -2.40 (correct - oscillation punished)

### Scenario 2: Sustained Progress to 25%

**Route characteristics**:
- Consistent forward movement
- Ends at 25% progress (final distance ≈ 750px of 1000px)
- Net PBRS > 0 (forward progress)

**Reward breakdown**:
```
PBRS accumulated:   +20.0 unscaled (25% of 80.0 weight)
Time penalty:       -4.0 unscaled
Final progress:     25% (ended at progress)
Stagnation penalty:  0.0 (≥ 15% threshold)
──────────────────────────────────────────────
Total:              +16.0 unscaled = +1.60 scaled
```

**Before fix**: -0.40 if oscillated back  
**After fix**: +1.60 (correct - sustained progress rewarded)

### Scenario 3: Explore to 30%, Return to 12%

**Route characteristics**:
- Explores deep (30% progress reached)
- Oscillates back to 12% (final distance ≈ 880px)
- Net PBRS ≈ 0 (forward then backward)

**Reward breakdown**:
```
PBRS accumulated:   ~0.0 unscaled (oscillation)
Time penalty:       -4.0 unscaled
Final progress:     12% (ended near spawn)
Stagnation penalty: -20.0 unscaled (< 15% threshold)
──────────────────────────────────────────────
Total:              -24.0 unscaled = -2.40 scaled
```

**Before fix**: -0.40 (bug - best was 30%)  
**After fix**: -2.40 (correct - ended near spawn)

## Why Different Routes Now Have Different Rewards

**Key factors that differentiate rewards**:

1. **Net PBRS accumulated**:
   - Sustained forward: +2.0 to +8.0 scaled
   - Oscillation: ≈ 0.0 scaled
   - Net backward: -1.0 to -4.0 scaled

2. **Final position**:
   - End near spawn (< 15%): -2.0 stagnation penalty
   - End at progress (≥ 15%): No penalty

3. **Episode length**:
   - Longer episodes: More time penalty
   - Shorter episodes: Less time penalty

**Example outcomes**:
```
Route A: Oscillate near spawn, end at 8%
  → -2.40 scaled (PBRS ≈ 0, stagnation penalty)

Route B: Progress to 25%, maintain position
  → +1.60 scaled (PBRS +2.0, no stagnation)

Route C: Progress to 18%, slight oscillation
  → +0.80 scaled (PBRS +1.2, no stagnation)

Route D: Quick exploration to 30%, return to 10%
  → -2.40 scaled (PBRS ≈ 0, stagnation penalty)
```

**Now each route's reward reflects its actual behavior!** ✅

## What This Fixes

### Before Fix (Buggy)
- ❌ All oscillating routes: -0.40 (same reward)
- ❌ Stagnation penalty rarely applied (exploit)
- ❌ No incentive to maintain progress
- ❌ Agent could learn: "Explore to 20%, then oscillate safely"

### After Fix (Correct)
- ✅ Oscillating routes: -2.40 (heavily punished)
- ✅ Sustained progress: +0.5 to +3.0 (rewarded)
- ✅ Strong incentive to maintain progress
- ✅ Agent must learn: "Stay at progress or push forward"

## Verification in Logs

With diagnostic logging added, you'll see:

**For oscillating route**:
```
[TIMEOUT_EPISODE] final_reward=-2.40, pbrs_total=0.02, time_total=-0.40, 
                  final_progress=8%, actions=500
[STAGNATION_TIMEOUT] Applied penalty based on FINAL distance. 
                     final_progress=8% < 15%, penalty=-2.00
```

**For sustained progress route**:
```
[TIMEOUT_EPISODE] final_reward=+1.60, pbrs_total=+2.00, time_total=-0.40, 
                  final_progress=25%, actions=500
[TIMEOUT_NO_PENALTY] Final progress 25% ≥ threshold 15%, no stagnation penalty
```

**Now you can see exactly why rewards differ!** ✅

## Impact on Training

**Before fix**:
- Agent could learn oscillation strategy (always -0.40)
- No differentiation between good and bad timeout episodes
- Weak anti-oscillation signal

**After fix**:
- Oscillation heavily punished (-2.40 vs -0.40 = 6× worse)
- Sustained progress rewarded (+1.60 vs -0.40 = 5× better)
- Strong incentive to maintain forward progress

**The agent will now learn**: "If I can't complete, at least stay at good progress (≥15%)" ✅

## Summary

✅ **Fixed**: Stagnation penalty now uses final position, not best position  
✅ **Added**: Diagnostic logging to show reward breakdown  
✅ **Result**: Different routes will have different rewards based on:
   - Net PBRS (sustained progress vs oscillation)
   - Final position (where they ended)
   - Episode length (time penalty)

**Timeout episodes will now properly reflect the quality of exploration!**

