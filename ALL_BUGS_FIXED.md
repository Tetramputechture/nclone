# All Bugs Fixed - 2025-12-09

## Bug Summary

Fixed 3 critical bugs that were preventing training from starting:

1. ✅ **Position key validation** - Alignment bonus missing validation
2. ✅ **Import efficiency** - GLOBAL_REWARD_SCALE imported in loop
3. ✅ **Goal ID mismatch** - Generic "switch" vs specific "exit_switch_0"

---

## Bug 1: Missing Position Key Validation ✅

**File:** `nclone/gym_environment/reward_calculation/main_reward_calculator.py`

**Problem:** Alignment bonus accessed `player_x`, `player_y` without validation

**Fix:** Added comprehensive validation for all required keys

---

## Bug 2: Import Inside Loop ✅

**File:** `npp_rl/agents/masked_ppo.py`

**Problem:** `GLOBAL_REWARD_SCALE` imported every step (thousands per second)

**Fix:** Moved import to module level

---

## Bug 3: Goal ID Mismatch (CRITICAL) ✅

**Files:**
- `nclone/graph/reachability/path_distance_cache.py`
- `nclone/gym_environment/entity_extractor.py`
- `nclone/graph/reachability/level_data_helpers.py`

**Problem:**
- Goals registered as: `"exit_switch_0"`, `"exit_door_0"` (specific IDs)
- Code looked up as: `"switch"`, `"exit"` (generic IDs)
- Cache returned None → training crashed

**Fix:**
1. Added generic aliases in path_distance_cache.py:
   - `_goal_id_to_goal_pos["switch"] = goal_node` (first exit_switch found)
   - `_goal_id_to_goal_pos["exit"] = goal_node` (first exit_door found)

2. Added validation in entity_extractor.py:
   - Fail fast if switch/exit at (0,0) with clear error

3. Changed level_data_helpers.py:
   - Errors instead of warnings for (0,0) positions

**Result:** Both lookups now work:
- `level_cache._goal_id_to_goal_pos.get("switch")` → goal_node ✅
- `level_cache._goal_id_to_goal_pos.get("exit_switch_0")` → goal_node ✅

---

## All Fixes Complete ✅

**Phase 1 Critical Fixes:**
1. ✅ Entropy coefficient (0.15 → 0.35)
2. ✅ Adaptive boosting (handle negative entropy)
3. ✅ LSTM layers (2 → 1)
4. ✅ Gradient clipping (5.0 → 10.0)
5. ✅ Rollout steps (4096 → 2048)
6. ✅ Action diversity bonus

**Phase 2 Fix 4:**
7. ✅ Velocity-direction alignment bonus

**Bug Fixes:**
8. ✅ Position key validation
9. ✅ Import efficiency
10. ✅ Goal ID mismatch

---

## Ready for Training

All fixes applied and tested. Training should now start successfully:

```bash
cd /home/tetra/projects/npp-rl
./scripts/start_fixed_training.sh
```

Or your quick test command (should now work):
```bash
python scripts/train_and_compare.py \
    --experiment-name "quick_test" \
    --architectures graph_free \
    --single-level '../nclone/test-single-level/006 both flavours of ramp jumping (and the control thereof)' \
    --train-dataset ../nclone/datasets/train \
    --test-dataset ../nclone/datasets/test \
    --total-timesteps 8000 \
    --num-envs 2 \
    --frame-skip 4 \
    --enable-go-explore \
    --use-lstm \
    --enable-state-stacking \
    --enable-demos \
    --demo-path ../nclone/bc_replays_tmp \
    --output-dir experiments
```

---

**Status:** All Bugs Fixed ✅ | Ready for Training ✅


