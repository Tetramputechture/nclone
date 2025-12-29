# Timeout Reward Multi-Environment Diagnosis

## Problem

All timeout routes show **exactly -0.40 reward** in multi-environment training (128+ envs), but local single-environment runs work fine.

## What -0.40 Means

```
-0.40 scaled = -4.0 unscaled
= -0.002 (time penalty) × 4 (frame skip) × 500 (actions) × 0.1 (global scale)
= Time penalty ONLY
```

This indicates:
1. PBRS accumulated ≈ 0 (either oscillation or bug)
2. Stagnation penalty = 0 (not applied)

## Hypothesis: enable_logging Affects Reward Calculation

Looking at the environment factory, training environments are created with `enable_logging=False` for performance. Let me check if any reward calculation depends on this flag:

**Files to check**:
1. `main_reward_calculator.py` - Does any reward logic check `enable_logging`?
2. `base_environment.py` - Does stagnation penalty application check logging flag?
3. Wrapper chain - Does any wrapper modify rewards based on logging?

## Key Differences: Local vs Multi-Env

| Aspect | Local Run | Multi-Env (128+) |
|--------|-----------|------------------|
| Logging | Enabled? | Disabled (performance) |
| VecEnv | DummyVecEnv? | SharedMemorySubprocVecEnv |
| Workers | Single process | 128 worker processes |
| Auto-reset | Manual | Automatic (Gymnasium) |

## Possible Root Causes

### 1. Logging Flag Affects Stagnation Check ❓

```python
# base_environment.py line ~505
if truncated and not terminated and self.enable_logging:
    # Diagnostic logging commented out, but...
```

**Wait!** I just realized - the user commented out the diagnostic logging block, but did they also comment out the stagnation penalty application?

Let me re-check...

### 2. Stagnation Penalty Not Applied in Workers ❓

The diagnostic logging is commented out, but the stagnation penalty application (lines 470-490) should still be active. Let me verify the code is correct.

### 3. PBRS Genuinely Summing to Zero ✓ MOST LIKELY

Both routes show heavy oscillation (tangled colored lines near spawn). This means:
- Forward movement: +PBRS
- Backward movement: -PBRS
- Net result: PBRS ≈ 0

**This is correct PBRS behavior!** Oscillation should give zero net reward.

### 4. Stagnation Penalty Should Apply but Doesn't ❓

If both routes ended near spawn (< 15% progress), stagnation penalty should apply:
```
-0.40 (time) + -2.0 (stagnation) = -2.40 total
```

But we're seeing -0.40, which means stagnation penalty is NOT being applied.

**Why?** Two possibilities:
1. Both routes ended at exactly 15-20% progress (just above threshold)
2. `final_distance` calculation is broken in multi-env setup

## Debug Plan

The user commented out my logging, so we can't see the breakdown. But we need to verify:

1. **Is PBRS actually ~0?** (Expected for oscillation)
2. **What is final_progress?** (Should determine stagnation penalty)
3. **Is stagnation penalty code even running?** (Could be a conditional bug)

## Proposed Fix: Add Minimal Diagnostic to Route Title

Instead of relying on logging (which is disabled), add reward breakdown directly to the route visualization title:

```python
# route_visualization_callback.py, in title generation
# Add reward component breakdown from info if available
pbrs_components = info.get("pbrs_components", {})
if pbrs_components:
    pbrs_reward = pbrs_components.get("pbrs_reward", 0.0)
    time_penalty = pbrs_components.get("time_penalty", 0.0)
    
    # Add to title
    title_parts.append(
        f"PBRS: {pbrs_reward:.2f}, Time: {time_penalty:.2f}"
    )
```

This will show in the route image what actually happened with rewards, visible without needing logs.

## Most Likely Scenario

**Both routes are**:
1. Oscillating near spawn (PBRS ≈ 0) ✓
2. Ending at 15-25% final progress (no stagnation penalty) ✓
3. Result: Time penalty only (-0.40) ✓

**This is actually CORRECT behavior** if:
- Both agents explored to ~20% then maintained that distance
- Or both ended exactly at 15-16% (threshold edge case)

The question is: **Why do they show identical -0.40 if they took different paths?**

Answer: Because PBRS depends on **NET distance change**, not path taken:
- Route A: Go 100px forward, 100px back → PBRS = 0
- Route B: Go 50px forward, 30px back, 20px forward → PBRS = 40 scaled
- Route C: Oscillate perfectly → PBRS = 0

If both routes oscillated heavily (visible in tangled lines), both getting PBRS ≈ 0 is expected!

## Action Required

Uncomment the diagnostic logging (even if it's noisy) to see:
- Actual PBRS total
- Actual final progress %
- Whether stagnation penalty was applied

Or implement the route title breakdown so we can see reward components in the visualization without needing logs.

