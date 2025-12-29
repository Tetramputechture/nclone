# CRITICAL: PBRS Distance Stuck at Spawn Value

## The Smoking Gun

```
Call #1: distance=308.65, player_pos=(312, 444) ← spawn
Call #2: distance=308.65, player_pos=(313, 446) ← moved 2px
Call #3: distance=308.65, player_pos=(312, 446) ← moved more
```

**Distance NEVER changes despite agent moving!**

## Root Cause

The distance is being **cached at spawn value (308.65) and never recalculated**.

This causes:
```
normalized_cost = distance / effective_norm
                = 308.65 / 308.65  
                = 1.0 (always!)

potential = 1.0 - normalized_cost
          = 1.0 - 1.0
          = 0.0 (always!)

PBRS = γ * Φ(s') - Φ(s)
     = 1.0 * 0.0 - 0.0
     = 0.0 (always!)
```

## Where is Distance Cached?

The path_calculator has multiple cache layers:
1. **Step-level cache** - Should be cleared each step
2. **Level cache** - Precomputed distances from BFS
3. **Position cache** - 6px threshold for sub-node interpolation

The bug is likely in **level cache** - it's returning the same distance regardless of player position!

## The Fix

The level cache should return different distances for different player positions. If it's returning spawn distance (308.65) for ALL positions, the cache lookup is broken.

**Check**: `path_calculator.get_distance()` in `path_distance_calculator.py`

The level cache lookup uses:
```python
cached_dist = self.level_cache.get_distance(start_node, goal_pos, goal_id)
```

If `start_node` is always the same (spawn node), distance will always be 308.65!

**Hypothesis**: `find_ninja_node()` is returning spawn node regardless of actual player position!

## Immediate Fix Needed

Add logging to `get_distance()` to show:
- Input player_pos
- Calculated start_node
- Cached distance returned
- Whether start_node changes as player moves

This will reveal if node finding is broken or cache lookup is broken.

