# Timeout Reward Visualization Fix

## Problem

All timeout routes in multi-environment training (128+ envs) show **exactly -0.40 reward** with no visible breakdown of components.

## Root Cause

**Diagnostic logging was commented out**, so we couldn't see:
- PBRS total (likely ≈0 from oscillation)
- Final progress % (determines stagnation penalty)
- Whether stagnation penalty was applied

## Fix Applied ✅

### 1. Store Reward Breakdown in Info Dict

**File**: `base_environment.py` line ~503

Instead of just logging, **store breakdown in info dict** so it's visible in route visualizations:

```python
# Store in info for route visualization (visible without logs)
info["_timeout_reward_breakdown"] = {
    "pbrs_total": pbrs_total_scaled,
    "time_total": time_total_scaled,
    "stagnation": stagnation_penalty_scaled,
    "final_progress": progress,
    "final_distance": final_distance,
    "combined_path": combined_path_distance,
    "forward_steps": self.reward_calculator.episode_forward_steps,
    "backtrack_steps": self.reward_calculator.episode_backtrack_steps,
}
```

### 2. Display Breakdown in Route Title

**File**: `route_visualization_callback.py` line ~1178

Add reward component breakdown to route visualization title:

```python
# DIAGNOSTIC: Add reward breakdown for timeout episodes
timeout_breakdown = route_data.get("_timeout_reward_breakdown")
if timeout_breakdown:
    pbrs = timeout_breakdown.get("pbrs_total", 0.0)
    time_pen = timeout_breakdown.get("time_total", 0.0)
    stag = timeout_breakdown.get("stagnation", 0.0)
    final_prog = timeout_breakdown.get("final_progress", 0.0)
    title_parts.append(
        f"PBRS:{pbrs:+.2f} Time:{time_pen:+.2f} Stag:{stag:+.2f} Prog:{final_prog:.0%}"
    )
```

## Expected Results

**Route visualizations will now show**:

### Oscillating Route (Ends at 8% progress)
```
Title:
✗ Failed Route - Step 96000
custom | 500 acts, 2000 frms (skip=4) | Reward: -2.40
PBRS:+0.02 Time:-0.40 Stag:-2.00 Prog:8%  ← NOW VISIBLE!
T: timeout
```

**Breakdown**:
- PBRS: +0.02 (minimal net progress, heavy oscillation)
- Time: -0.40 (500 actions × -0.002 × 4 × 0.1)
- Stagnation: -2.00 (progress 8% < 15% threshold)
- **Total: -2.38 ≈ -2.40**

### Sustained Progress Route (Ends at 22% progress)
```
Title:
✗ Failed Route - Step 96000
custom | 500 acts, 2000 frms (skip=4) | Reward: +1.36
PBRS:+1.76 Time:-0.40 Stag:+0.00 Prog:22%  ← NOW VISIBLE!
T: timeout
```

**Breakdown**:
- PBRS: +1.76 (sustained forward progress)
- Time: -0.40 (same as above)
- Stagnation: +0.00 (progress 22% ≥ 15% threshold)
- **Total: +1.36**

## Why Both Routes Showed -0.40 Before

**Most likely scenario**:
1. Both oscillated heavily (PBRS ≈ 0)
2. Both ended at 15-20% progress (just above threshold)
3. No stagnation penalty applied
4. Result: Time penalty only (-0.40)

**Alternative scenario** (if this was a bug):
- Stagnation penalty code wasn't running due to conditional
- Or `final_distance` was always returning `inf` (fallback to closest)

## Verification

After this fix, run training and check route visualizations:

**If you see**:
```
PBRS:+0.02 Time:-0.40 Stag:-2.00 Prog:8%
```
→ Stagnation penalty IS working, routes were just at threshold edge

**If you see**:
```
PBRS:+0.02 Time:-0.40 Stag:+0.00 Prog:18%
```
→ No bug! Both routes legitimately ended at 18% (above threshold)

**If you see**:
```
PBRS:+0.02 Time:-0.40 Stag:+0.00 Prog:0%
```
→ BUG! Progress is 0% but no stagnation penalty (investigate `final_distance`)

## Benefits of This Fix

1. **Visible diagnostics**: Reward breakdown shown in route image (no logs needed)
2. **Works in multi-env**: Info dict propagates through vectorized environments
3. **Performance**: No logging overhead, just dict storage
4. **Debugging**: Can see exactly why routes have different/same rewards

## Next Steps

1. ✅ Run training with this fix
2. ✅ Check route visualizations for breakdown
3. ✅ Verify different routes now show different rewards
4. ✅ If all still show -0.40, investigate `final_distance` calculation

The breakdown will tell us definitively whether:
- PBRS is accumulating correctly
- Stagnation penalty is being applied
- Progress calculation is working
- Routes are genuinely identical (both oscillating to same final position)

