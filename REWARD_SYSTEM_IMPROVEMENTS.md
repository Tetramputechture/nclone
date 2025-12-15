# Reward System Improvements - December 13, 2024

## Analysis Summary

Based on TensorBoard data from training run `npp-logs-1213-ext` (~2.8M timesteps on level "006 both flavours of ramp jumping"), identified critical issues preventing learning:

### Key Findings from Data
- **Success rate**: 0% (no completions)
- **Mine death ratio**: 67.2% (despite -40 penalty)
- **Episode length**: 1897 frames (95% of 2000 limit) - hitting truncation
- **Backtracking**: 49.47% of steps
- **Waypoints collected**: 0.04 per episode (system ineffective)
- **Oscillation ratio**: 8.94 (moving 9x more than net displacement)
- **PBRS positive potential**: 0.0% (no forward progress)

## Changes Implemented

### 1. Denser Waypoint Coverage (Priority 1)

**File**: `nclone/gym_environment/reward_calculation/reward_config.py`

- **Progress spacing**: 100px → **50px** (2x denser waypoint coverage)
- **Cluster radius**: 40px → **25px** (preserves turn waypoints from over-clustering)

**Rationale**: 100px spacing was too sparse for sharp trajectory changes. Agent would reach a turn waypoint but not continue through the turn because the next waypoint was too far away.

### 2. Turn Continuation Waypoints (Priority 1)

**File**: `nclone/gym_environment/reward_calculation/path_waypoint_extractor.py`

Added new method `_add_turn_continuation_waypoints()` that:
- Places waypoints **25px AFTER** significant turns (>60°)
- Guides agent THROUGH trajectory changes, not just TO them
- High value (1.4) to prioritize turn completion

**Rationale**: Addresses the core issue where waypoints guide TO the turn but not THROUGH it. Continuation waypoints provide the missing guidance signal.

### 3. Extended Exit Direction Bonus (Priority 1)

**File**: `nclone/gym_environment/reward_calculation/main_reward_calculator.py`

- **Exit direction bonus window**: 10 frames → **30 frames** (~7-8 actions with frame_skip=4)

**Rationale**: 10 frames (2-3 actions) was too short to guide through sharp turns. 30 frames provides sustained guidance after waypoint collection.

### 4. Increased Truncation Limits (Priority 2)

**File**: `nclone/gym_environment/truncation_calculator.py`

- **MAX_TRUNCATION_FRAMES**: 2000 → **4000** frames
- **TRUNCATION_MULTIPLIER**: 15 → **20**

**Rationale**: Episodes were hitting the 2000 frame cap at 95% frequency, creating a learning cliff. With frame_skip=4, this only allowed 500 actions per episode. Doubling to 4000 frames (1000 actions) gives the agent adequate exploration time.

**Impact**: 
- Before: 500 actions max, 16.2% timeout rate
- After: 1000 actions max, should reduce timeout rate significantly

### 5. Stronger Mine Avoidance (Priority 3)

**File**: `nclone/gym_environment/reward_calculation/reward_config.py`

- **Discovery phase** (0-15% success): 25.0 → **50.0** (2x increase)
- **Mid phase** (15-40% success): 40.0 → **70.0** (1.75x increase)
- **Late phase** (40%+ success): 60.0 → **90.0** (1.5x increase)

**File**: `nclone/gym_environment/reward_calculation/reward_constants.py`

- **MINE_HAZARD_RADIUS**: 50px → **75px**

**Rationale**: Despite -40 death penalty, 67% of deaths were from mines. The pathfinding cost multiplier was too weak, allowing PBRS to guide the agent into mine fields. Doubling the multiplier and increasing the radius should create stronger avoidance gradients.

### 6. Extended Waypoint Approach Gradient (Priority 4)

**File**: `nclone/gym_environment/reward_calculation/main_reward_calculator.py`

- **Approach gradient radius**: 100px → **200px**
- **Added distance scaling**: Gradient strength scales from 0.25x at 200px to 1.0x at 0px

**Rationale**: With 100px spacing between waypoints, the approach gradient (100px radius) would only apply when already very close. 200px radius with distance scaling provides continuous guidance over longer distances.

## Expected Impact

### Immediate Improvements
1. **Waypoint collection rate**: Should increase from 0.04 to 2-5 per episode with denser coverage
2. **Timeout rate**: Should decrease from 16.2% to <5% with 2x truncation limit
3. **Mine death ratio**: Should decrease from 67% to <40% with stronger avoidance

### Learning Efficiency
1. **Exploration time**: 2x more time to find optimal paths before truncation
2. **Turn navigation**: Continuation waypoints + extended exit bonus should guide through sharp turns
3. **PBRS effectiveness**: Denser waypoints should improve PBRS routing through inflection points

### Long-term Benefits
1. **Success rate**: Should start seeing first completions within 5-10M timesteps
2. **Path optimality**: Should improve from 36% to >60% as agent learns efficient routes
3. **Backtracking**: Should decrease from 49% to <30% with better waypoint guidance

## Monitoring Recommendations

Track these TensorBoard metrics to validate improvements:

1. **waypoints/collected_per_episode_mean**: Should increase to 2-5
2. **episode/success_rate**: Should start showing >0% within 5M timesteps
3. **death/by_cause/mine_ratio**: Should decrease below 50%
4. **episode/length_mean**: May increase initially (more exploration) then decrease
5. **efficiency/path_optimality**: Should improve toward 0.6-0.8
6. **pbrs_diag/positive_potential_pct**: Should increase above 0%

## Files Modified

1. `nclone/gym_environment/reward_calculation/reward_config.py`
2. `nclone/gym_environment/reward_calculation/path_waypoint_extractor.py`
3. `nclone/gym_environment/reward_calculation/main_reward_calculator.py`
4. `nclone/gym_environment/truncation_calculator.py`
5. `nclone/gym_environment/reward_calculation/reward_constants.py`

## Next Steps

1. **Re-run training** on the same level to validate improvements
2. **Monitor TensorBoard** for the metrics listed above
3. **Compare episode routes** before/after to see if turn navigation improves
4. **Adjust further** if mine death ratio remains high (may need explicit mine proximity penalty)

