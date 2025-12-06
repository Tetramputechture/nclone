# Momentum-Aware PBRS Quick Start Guide

## What Was Implemented

Your N++ RL agent can now learn momentum-dependent strategies! The system includes:

1. **Momentum-aware pathfinding** - Paths that preserve momentum are 30% cheaper
2. **Demonstration waypoints** - Extract momentum-building points from expert replays
3. **Multi-stage PBRS** - Routes through waypoints when momentum is needed

## Immediate Usage (No Setup Required)

The momentum-aware pathfinding is **already active** in your environment! Just train as normal:

```python
from nclone.gym_environment import NppEnvironment

env = NppEnvironment(...)
# Momentum-aware PBRS is now active automatically!
```

**What changed:**
- Pathfinding now tracks momentum state
- Paths that build/preserve momentum get 30% cost discount
- PBRS no longer penalizes necessary momentum-building

## Optional: Add Demonstration Waypoints

For levels with complex momentum patterns, extract waypoints from expert demonstrations:

### Step 1: Extract Waypoints

```bash
# Extract from your expert replays
python nclone/tools/extract_momentum_waypoints.py \
    --replay-dir /path/to/expert/replays \
    --output-dir momentum_waypoints_cache \
    --verbose

# This creates: momentum_waypoints_cache/{level_id}.pkl
```

### Step 2: Train (Automatic Loading)

```python
# Waypoints are loaded automatically during reset()
env = NppEnvironment(...)
obs, info = env.reset()
# If waypoints exist for this level, they're now active!
```

### Step 3: Monitor in TensorBoard

Check these metrics during training:
- `_pbrs_potential_change`: Should be positive during momentum-building
- `_pbrs_using_waypoint`: True when waypoint routing active
- `_pbrs_dist_to_waypoint`: Distance to momentum waypoint

## Expected Results

### Before Momentum-Aware PBRS

```
Frame 100: Agent moves left (building momentum)
  - Euclidean distance to goal: increases
  - PBRS reward: -0.05 (PENALIZED for moving away)
  - Agent learns: "Don't move left" ❌

Result: Agent fails to learn momentum-dependent strategies
```

### After Momentum-Aware PBRS

```
Frame 100: Agent moves left (building momentum)
  - Path distance to goal: decreases (momentum-aware path is cheaper)
  - PBRS reward: +0.03 (REWARDED for building momentum)
  - Agent learns: "Build momentum before jumping" ✅

Result: Agent successfully learns momentum-dependent strategies
```

## Verification

### Test Momentum Tracking

```bash
# Run unit tests
pytest nclone/gym_environment/tests/test_momentum_aware_pbrs.py -v

# Should see: 15 passed
```

### Test on Your Level

```python
from nclone.gym_environment import NppEnvironment
import logging

logging.basicConfig(level=logging.DEBUG)

# Load your momentum-dependent level
env = NppEnvironment(custom_map_path="path/to/momentum_level.npp")

obs, _ = env.reset()

# Take actions that build momentum
for _ in range(20):
    obs, reward, done, truncated, info = env.step(1)  # Move left
    print(f"PBRS: {info['pbrs_components']['pbrs_reward']:.4f}")
    # Should see positive or neutral PBRS, not negative!
```

## Configuration (Optional)

### Tune Momentum Preference

Edit `nclone/graph/reachability/pathfinding_algorithms.py`:

```python
# Make momentum preservation more important
MOMENTUM_CONTINUE_MULTIPLIER = 0.5   # Even cheaper (was 0.7)

# Make momentum reversal more expensive
MOMENTUM_REVERSE_MULTIPLIER = 3.0    # More expensive (was 2.5)
```

### Tune Waypoint Detection

Edit `nclone/analysis/momentum_waypoint_extractor.py`:

```python
# Detect more waypoints (more sensitive)
MIN_SPEED_FOR_MOMENTUM = 1.0          # Lower threshold (was 1.5)
SPEED_INCREASE_THRESHOLD = 0.5        # Lower threshold (was 0.8)

# Detect fewer waypoints (less sensitive)
MIN_SPEED_FOR_MOMENTUM = 2.0          # Higher threshold
SPEED_INCREASE_THRESHOLD = 1.2        # Higher threshold
```

## Troubleshooting

### "PBRS still penalizes momentum-building"

**Check:**
1. Verify momentum tracking is working:
   ```python
   # Add debug logging to see momentum inference
   import logging
   logging.getLogger('nclone.graph.reachability').setLevel(logging.DEBUG)
   ```

2. Check if waypoints are loaded:
   ```python
   # After env.reset()
   pbrs_calc = env.reward_calculator.pbrs_calculator
   print(f"Waypoints loaded: {len(pbrs_calc.momentum_waypoints)}")
   ```

### "Waypoints not loading"

**Check:**
1. Cache directory exists: `momentum_waypoints_cache/`
2. Cache file exists: `momentum_waypoints_cache/{level_id}.pkl`
3. Level ID matches between cache and environment

**Fix:**
```bash
# Re-extract waypoints with correct level IDs
python nclone/tools/extract_momentum_waypoints.py \
    --replay-dir /path/to/replays \
    --output-dir momentum_waypoints_cache \
    --verbose
```

## Summary

**What you get:**
- ✅ Momentum-aware pathfinding (automatic, always active)
- ✅ Demonstration waypoints (optional, for complex cases)
- ✅ No PBRS penalty for momentum-building
- ✅ Faster learning on momentum-dependent levels

**What you need to do:**
1. Nothing! Momentum tracking is automatic
2. (Optional) Extract waypoints from demonstrations for complex levels
3. Monitor TensorBoard to verify PBRS behavior

**Ready to train!** 🚀

