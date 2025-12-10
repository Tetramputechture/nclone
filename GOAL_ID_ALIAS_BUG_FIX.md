# Goal ID Alias Bug Fix - 2025-12-09

## Problem

Training crashed with error:
```
ValueError: Goal node is None for goal_id: switch
```

Despite having a perfectly valid level with switch NOT at (0,0).

## Root Cause

**Goal ID Mismatch:**
- Goals are registered in cache as: `"exit_switch_0"`, `"exit_door_0"`, etc. (specific IDs)
- Code tries to look them up as: `"switch"`, `"exit"` (generic IDs)
- Cache lookup fails → returns None → training crashes

**Where it happens:**
1. `level_data_helpers.py`: Goals registered with specific IDs (`exit_switch_0`)
2. `feature_computation.py` line 377: Code uses generic ID (`"switch"`)
3. `main_reward_calculator.py` line 688: Code looks up generic ID (`"switch"`)
4. Cache returns None → crash

## Solution

Added generic aliases in `path_distance_cache.py` when building the level cache:

```python
# Store goal with specific ID
self._goal_id_to_goal_pos["exit_switch_0"] = goal_node

# Add generic alias for backward compatibility
if goal_id.startswith("exit_switch_"):
    if "switch" not in self._goal_id_to_goal_pos:
        self._goal_id_to_goal_pos["switch"] = goal_node
```

Now both lookups work:
- `level_cache._goal_id_to_goal_pos.get("switch")` → returns goal_node ✅
- `level_cache._goal_id_to_goal_pos.get("exit_switch_0")` → returns goal_node ✅

## Files Modified

**nclone/graph/reachability/path_distance_cache.py**
- Lines 200-215: Added generic alias mappings for "switch" and "exit"

**nclone/gym_environment/entity_extractor.py**
- Lines 98-130: Added validation to fail fast if switch/exit at (0,0)

**nclone/graph/reachability/level_data_helpers.py**
- Lines 43-49, 58-65, 76-83: Changed from warnings to errors for (0,0) positions

## Testing

After fix, the code will:
1. ✅ Fail early with clear error if switch/exit at (0,0) (entity_extractor.py)
2. ✅ Support both generic and specific goal_id lookups
3. ✅ Work with valid levels that have proper entity positions

## Why This Happened

The codebase evolved to use specific goal IDs (`exit_switch_0`) for multi-goal support (multiple switches/exits), but some code still uses generic IDs (`"switch"`) from the old single-goal system. The alias mapping bridges this gap for backward compatibility.

---

**Status:** Fixed ✅  
**Ready for Testing:** Yes - rerun training command


