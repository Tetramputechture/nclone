# Timeout Reward Issue - Fix Explanation

## Problem Identified

**Both timeout routes show exactly -0.40 reward** despite different paths.

## Root Cause Analysis

**-0.40 scaled = -4.0 unscaled = Time penalty only**

This means:
1. ✅ Time penalty correctly applied: -0.002 × 4 frames × 500 actions = -4.0 unscaled
2. ❌ PBRS accumulated ≈ 0 (both routes oscillated)
3. ❌ Stagnation penalty NOT applied (bug in current code)

## Why PBRS ≈ 0 (Both Routes)

Looking at the route visualizations, both agents are **oscillating heavily near spawn**:
- Tangled colorful lines indicate back-and-forth movement
- Forward movement: +PBRS (potential increases)
- Backward movement: -PBRS (potential decreases)
- **Net result**: PBRS ≈ 0 (oscillation cancels out)

This is actually **correct PBRS behavior** - oscillation should give zero net reward!

## Why Stagnation Penalty Wasn't Applied (BUG)

**Current code (BUGGY)**:
```python
# Uses CLOSEST distance reached (best progress ever)
closest_distance = self.reward_calculator.closest_distance_this_episode
progress = 1.0 - (closest_distance / combined_path)

if progress < 0.15:  # Only penalize if never reached 15%
    apply_stagnation_penalty()
```

**The exploit**:
1. Agent explores to 20% progress (updates closest_distance)
2. Agent oscillates back to spawn area
3. closest_distance still shows 20% (best ever reached)
4. No stagnation penalty applied (20% > 15% threshold)
5. Final reward: -0.40 (time only)

**Both routes likely reached 15-20% at some point, so no penalty applied.**

## Fix Applied ✅

**Use FINAL distance (where agent ended), not best distance**:

```python
# Use FINAL distance to goal (current position)
final_distance = final_obs.get("_pbrs_last_distance_to_goal", None)

if final_distance is None or final_distance == float("inf"):
    # Fallback: use closest if not available
    final_distance = self.reward_calculator.closest_distance_this_episode

# Calculate progress based on FINAL position
progress = 1.0 - (final_distance / combined_path_distance)

if progress < STAGNATION_PROGRESS_THRESHOLD:
    apply_stagnation_penalty()
```

## Expected Results After Fix

### Route A (Oscillates near spawn, ends at 5% progress)
```
PBRS accumulated:   ~0.0 unscaled (oscillation)
Time penalty:       -4.0 unscaled
Final progress:     5% (ended near spawn)
Stagnation penalty: -20.0 unscaled (progress < 15%)
──────────────────────────────────────────────
Total:              -24.0 unscaled = -2.40 scaled ← PUNISHED!
```

### Route B (Explores to 25%, maintains position)
```
PBRS accumulated:   +20.0 unscaled (net forward)
Time penalty:       -4.0 unscaled
Final progress:     25% (ended at 25%)
Stagnation penalty:  0.0 (progress ≥ 15%)
──────────────────────────────────────────────
Total:              +16.0 unscaled = +1.60 scaled ← REWARDED!
```

### Route C (Explores to 20%, oscillates back to 8%)
```
PBRS accumulated:   ~0.0 unscaled (oscillation canceled out)
Time penalty:       -4.0 unscaled
Final progress:     8% (ended near spawn)
Stagnation penalty: -20.0 unscaled (progress < 15%)
──────────────────────────────────────────────
Total:              -24.0 unscaled = -2.40 scaled ← PUNISHED!
```

**Now different routes will have different rewards based on**:
1. **Net PBRS** (did they make sustained progress or oscillate?)
2. **Final position** (where did they end up?)

## Why This Fix is Critical

**Before fix** (buggy):
- Oscillation exploit: Reach 20%, return to spawn, avoid penalty
- All oscillating routes get -0.40 (same reward)
- No incentive to maintain progress

**After fix** (correct):
- Oscillation punished: End near spawn → stagnation penalty (-2.40)
- Sustained progress rewarded: End at 25% → no penalty, keep PBRS gains
- Different routes have different rewards based on final position

## Verification

After implementing this fix, timeout routes should show:
- **-2.40 scaled** if agent ended near spawn (progress < 15%)
- **Variable rewards** (+0.5 to +3.0) if agent maintained progress (≥15%)
- **Different rewards for different final positions**

The fix ensures: **Oscillation is punished, sustained progress is rewarded** ✅

## Additional Logging Improvement

Add diagnostic logging to show why rewards differ:

```python
logger.info(
    f"[EPISODE_END] Timeout episode: "
    f"final_dist={final_distance:.0f}px, "
    f"best_dist={self.reward_calculator.closest_distance_this_episode:.0f}px, "
    f"final_progress={progress:.1%}, "
    f"pbrs_total={sum(self.reward_calculator.episode_pbrs_rewards):.2f}, "
    f"reward={self.current_ep_reward:.3f}, "
    f"stagnation_applied={progress < STAGNATION_PROGRESS_THRESHOLD}"
)
```

This will help verify the fix is working correctly in TensorBoard logs.

