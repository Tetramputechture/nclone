# CRITICAL: PBRS Always Zero Bug

## Smoking Gun Evidence

**Route shows**:
- Start: Blue dot at spawn (left side)
- End: Red dot at 62% progress (near ramp)
- **PBRS: +0.00** ← IMPOSSIBLE!

**Expected PBRS**:
```
Progress: 0% → 62%
PBRS weight: 80.0 (discovery phase)
Expected: +0.62 × 80.0 × 0.1 = +4.96 scaled
Actual: +0.00 ← BUG!
```

## Possible Causes

### 1. PBRS Not Being Calculated

Check if `pbrs_reward` is always 0 in `calculate_reward()`:

```python
# main_reward_calculator.py line 437
pbrs_reward = self.pbrs_gamma * current_potential - self.prev_potential
```

If `current_potential` and `prev_potential` are always equal, PBRS = 0.

**Diagnosis needed**: Log `current_potential` and `prev_potential` values.

### 2. PBRS Not Being Added to Reward

Check if `reward += pbrs_reward` is executed:

```python
# main_reward_calculator.py line 601
reward += pbrs_reward
```

This looks correct, so likely not the issue.

### 3. PBRS Rewards Not Being Accumulated to Episode Total

Check if `current_ep_reward += reward` includes PBRS:

```python
# base_environment.py line 482 and 616
self.current_ep_reward += reward
```

This should include PBRS since it's added to `reward` before accumulation.

### 4. Episode PBRS List is Empty

Check if `self.episode_pbrs_rewards` is being populated:

```python
# main_reward_calculator.py line 599
self.episode_pbrs_rewards.append(pbrs_reward)
```

Then summed in episode breakdown:

```python
# base_environment.py timeout breakdown
pbrs_total_unscaled = sum(self.reward_calculator.episode_pbrs_rewards)
```

If list is empty, sum = 0!

### 5. Potential is Always Same (Not Changing)

Most likely cause: **`current_potential` calculation is broken or cached incorrectly**.

If potential never changes, PBRS = 0 always:
```
F(s,s') = γ * Φ(s') - Φ(s)
If Φ(s') == Φ(s) always: F = 0 always
```

## Debug Actions Required

### Immediate: Add Logging to Track Potentials

```python
# main_reward_calculator.py, after line 384 (calculate_combined_potential)

logger.warning(
    f"[POTENTIAL_DEBUG] step={self.steps_taken}, "
    f"current={current_potential:.6f}, "
    f"prev={self.prev_potential:.6f if self.prev_potential else 'None'}, "
    f"distance={distance_to_goal:.1f}, "
    f"progress={1.0 - (distance_to_goal / obs.get('_pbrs_combined_path_distance', 1000)):.3f}"
)
```

### Check if Potential Normalization is Broken

```python
# pbrs_potentials.py - objective_distance_potential()

# Expected:
Φ(s) = 1 - (remaining_physics_cost / total_physics_cost)

# At spawn (0% progress): Φ = 1 - 1.0 = 0.0
# At 62% progress: Φ = 1 - 0.38 = 0.62
# PBRS = 0.62 - 0.0 = +0.62 unscaled
```

If this calculation is broken (e.g., division by zero, inf handling), potential could be constant.

### Check If Combined Path Distance is Available

```python
combined_physics_cost = state.get("_pbrs_combined_physics_cost")
if combined_physics_cost is None:
    # BUG: Potential can't be calculated without normalization!
    return 0.0  # This would make all potentials 0!
```

## Most Likely Bug

**Hypothesis**: `_pbrs_combined_physics_cost` or `_pbrs_combined_path_distance` is not being set properly in multi-environment training, causing all potentials to return 0.

This would explain:
- All routes showing PBRS = +0.00
- Agent clearly moving but no reward accumulation
- Same behavior across all environments

## Fix Strategy

1. Add debug logging to track `current_potential` values
2. Verify `_pbrs_combined_physics_cost` is set in observations
3. Check if level cache is being built properly in multi-env setup
4. Verify pathfinding is returning valid distances (not inf)

## Temporary Debug Addition

Add this to main_reward_calculator.py after potential calculation:

```python
# After line 384
if self.steps_taken % 50 == 0:  # Log every 50 steps
    logger.error(
        f"[PBRS_DEBUG] "
        f"potential={current_potential:.4f}, "
        f"prev={self.prev_potential:.4f if self.prev_potential else 0.0}, "
        f"pbrs_this_step={pbrs_reward if 'pbrs_reward' in locals() else 0.0:.4f}, "
        f"pbrs_accumulated={sum(self.episode_pbrs_rewards):.2f}, "
        f"distance={distance_to_goal:.1f if 'distance_to_goal' in locals() else -1}, "
        f"combined_path={obs.get('_pbrs_combined_path_distance', 'MISSING')}"
    )
```

This will show if potentials are being calculated or stuck at 0.

