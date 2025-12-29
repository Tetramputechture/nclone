# Timeout Reward Issue - Diagnosis and Fix

## Problem Observed

Two different timeout routes both show **exactly -0.40 reward** despite different paths.

## Diagnosis

**-0.40 scaled breakdown**:
```
-0.40 scaled = -4.0 unscaled
Time penalty: -0.002 × 4 frames × 500 actions = -4.0 unscaled ✓
PBRS accumulated: ~0.0 unscaled
Stagnation penalty: 0.0 (not applied)
```

**Why is PBRS ≈ 0?**

The agent is **oscillating** - exploring forward then returning to similar distance:
```
Step 1-100:   Move 50px closer → PBRS: +0.5
Step 101-200: Move 30px away → PBRS: -0.3
Step 201-300: Move 20px closer → PBRS: +0.2
...
Net after 500 actions: PBRS ≈ 0.0 (oscillation cancels out)
```

**Why no stagnation penalty?**

Stagnation penalty checks `closest_distance_this_episode` (best distance reached):
```python
progress = 1.0 - (closest_distance / combined_path)
if progress < 0.15:  # Only penalize if never reached 15%
    apply_stagnation_penalty()
```

Both agents likely reached 15%+ progress **at some point**, so no penalty applied even though they ended near spawn.

## The Real Issue

**Current system**: Stagnation penalty based on **best** distance reached  
**Problem**: Agent can explore to 20%, then oscillate back to spawn, avoid penalty

**This creates an exploit**:
1. Make 20% progress quickly (avoid stagnation penalty)
2. Oscillate in safe area for remaining time (PBRS ≈ 0)
3. Final reward: Time penalty only (-0.40)

## Root Cause

The stagnation penalty should be based on **FINAL** position or **NET** progress, not best progress reached.

```python
# CURRENT (exploitable):
closest_distance = self.reward_calculator.closest_distance_this_episode
progress = 1.0 - (closest_distance / combined_path)  # Best progress ever reached

# SHOULD BE (exploit-proof):
final_distance = distance_to_goal  # Current distance (where agent ended)
progress = 1.0 - (final_distance / combined_path)  # Final progress
```

## Fix Option 1: Use Final Distance (Recommended)

**Change stagnation penalty to use final position, not best position**:

```python
# base_environment.py line ~460
# OLD:
closest_distance = self.reward_calculator.closest_distance_this_episode
progress = 1.0 - (closest_distance / combined_path_distance)

# NEW:
final_distance = final_obs.get("_pbrs_last_distance_to_goal", float("inf"))
if final_distance == float("inf"):
    # Fallback to closest if distance not available
    final_distance = self.reward_calculator.closest_distance_this_episode
progress = 1.0 - (final_distance / combined_path_distance)
```

**Effect**:
- Agent ending near spawn: progress = 0-5% → stagnation penalty applies
- Agent ending at 20%: progress = 20% → no penalty (legitimate progress)
- Oscillation no longer exploitable (final position matters)

## Fix Option 2: Require Net Positive PBRS

**Only avoid stagnation penalty if PBRS total is positive**:

```python
# Add condition that PBRS must be positive
pbrs_total = sum(self.reward_calculator.episode_pbrs_rewards)
progress_made = progress >= STAGNATION_PROGRESS_THRESHOLD
net_progress = pbrs_total > 0.5  # Must have made NET progress

if not (progress_made and net_progress):
    apply_stagnation_penalty()
```

**Effect**:
- Agent that oscillates (PBRS ≈ 0): Gets stagnation penalty even if briefly reached 15%
- Agent with consistent forward progress (PBRS > 0): No penalty

## Fix Option 3: Use Net Displacement

**Base stagnation on net displacement from spawn**:

```python
# Calculate net displacement
spawn_distance = obs.get("_displacement_from_spawn", 0.0)
net_progress = spawn_distance / combined_path_distance

if net_progress < STAGNATION_PROGRESS_THRESHOLD:
    apply_stagnation_penalty()
```

**Effect**:
- Agent ending near spawn: Low net displacement → penalty
- Agent ending far from spawn: High net displacement → no penalty

## Recommended Fix: Option 1 (Final Distance)

**Why**: Most direct and aligns with the goal of "reduce distance to goal"

**Implementation**:

```python
# base_environment.py, lines 457-467

# Calculate progress based on FINAL distance to goal (not closest)
final_distance = final_obs.get("_pbrs_last_distance_to_goal", None)

if final_distance is None or final_distance == float("inf"):
    # Fallback: use closest if final distance not available
    final_distance = self.reward_calculator.closest_distance_this_episode

if combined_path_distance > 0 and final_distance != float("inf"):
    # Progress = 1.0 - (final_distance / total_path)
    # This measures where agent ENDED UP, not where it explored
    progress = 1.0 - (final_distance / combined_path_distance)
    progress = max(0.0, min(1.0, progress))
else:
    progress = 0.0

# Apply penalty if final position is below threshold
if progress < STAGNATION_PROGRESS_THRESHOLD:
    timeout_penalty = STAGNATION_TIMEOUT_PENALTY * GLOBAL_REWARD_SCALE
    reward += timeout_penalty
    
    logger.info(
        f"[STAGNATION_TIMEOUT] Applied penalty based on FINAL distance. "
        f"final_progress={progress:.1%} < {STAGNATION_PROGRESS_THRESHOLD:.1%}, "
        f"penalty={timeout_penalty:.3f}, "
        f"final_distance={final_distance:.1f}px, "
        f"combined_path={combined_path_distance:.1f}px"
    )
```

## Expected Results After Fix

### Route A (Explores to 20%, returns to spawn)
```
PBRS accumulated:    ~0.0 (oscillation)
Time penalty:        -0.4
Final distance:      800px (near spawn)
Final progress:      ~5% (not 20%!)
Stagnation penalty:  -2.0 (progress < 15%)
──────────────────────────────────────
Total reward:        -2.4 scaled ← PUNISHED for oscillation!
```

### Route B (Explores to 30%, stays there)
```
PBRS accumulated:    +2.4 (net forward)
Time penalty:        -0.4
Final distance:      600px (30% progress)
Final progress:      30%
Stagnation penalty:  0.0 (progress ≥ 15%)
──────────────────────────────────────
Total reward:        +2.0 scaled ← REWARDED for sustained progress!
```

**Now different routes have different rewards based on where they ended!** ✅

## Why This Matters

**Current bug allows this exploit**:
1. Quickly explore to 20% progress (avoid stagnation check)
2. Oscillate in safe area for 90% of episode (PBRS ≈ 0)
3. End near spawn with -0.40 reward (only time penalty)
4. Repeat forever (never learn to complete)

**After fix**:
1. If agent ends near spawn: -2.4 reward (stagnation penalty)
2. If agent maintains 15%+ progress: Varies based on net PBRS
3. Only completing or staying at progress gives positive returns
4. Oscillation is now properly punished

## Implementation Priority

**HIGH** - This is a critical bug that allows oscillation exploitation.

The fix is simple (5 lines changed) but critical for preventing the agent from learning to oscillate.

