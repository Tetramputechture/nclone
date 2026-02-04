# Physics Cost Verification: Geometric Distances NO LONGER USED for Rewards

**Date**: 2026-02-03  
**Status**: ✅ Verified - All rewards use physics costs

---

## Executive Summary

The earlier analysis stating "PBRS uses geometric distances" is **now obsolete**. After the 2026-02-03 physics-aware PBRS implementation:

✅ **Primary PBRS**: Uses `get_physics_cost()` with velocity and jump duration  
✅ **Milestone tracking**: Uses cached physics costs (not geometric)  
✅ **Waypoint system**: NOT USED in production (empty list by default)  
⚠️ **Cascading fallback**: Uses geometric distance ONLY when `physics_cost == inf` (intentional recovery gradient)

---

## Complete Reward Flow Verification

### 1. Main PBRS Potential Calculation ✅ PHYSICS COSTS

**File**: `pbrs_potentials.py:237-262`

```python
# PRIMARY: Physics cost with velocity and jump duration
physics_cost = path_calculator.get_physics_cost(
    player_pos, goal_pos,
    adjacency, base_adjacency, level_data, graph_data,
    entity_radius, NINJA_RADIUS,
    ninja_velocity=ninja_velocity,              # ← Velocity-aware
    jump_duration_remaining=jump_duration_remaining,  # ← Jump frame tracking
    goal_id=goal_id,
)

state["_pbrs_physics_cost"] = physics_cost
state["_pbrs_last_distance_to_goal"] = physics_cost  # For milestone tracking
```

**Result**: Main PBRS uses **PHYSICS COSTS** ✅

---

### 2. Milestone Tracking ✅ PHYSICS COSTS

**File**: `main_reward_calculator.py:395-400`

```python
# Milestones use cached distance from PBRS calculator
distance_to_switch = self.pbrs_calculator.get_last_distance_to_goal("switch")
distance_to_exit = self.pbrs_calculator.get_last_distance_to_goal("exit")
milestone_reward = self._check_distance_milestones(
    distance_to_switch, distance_to_exit, switch_activated
)
```

**File**: `pbrs_potentials.py:1345 & 1712-1728`

```python
# After computing objective_distance_potential():
# Cache PHYSICS COST (not geometric distance)
self._cached_distance = state_with_metrics.get("_pbrs_physics_cost", None)

def get_last_distance_to_goal(self, goal_id: str) -> Optional[float]:
    """Returns cached physics cost from last PBRS calculation."""
    if self._last_goal_id == goal_id and self._cached_distance is not None:
        return self._cached_distance  # ← This is physics cost!
    return None
```

**Result**: Milestones use **PHYSICS COSTS** ✅

---

### 3. Position-Based Caching ✅ DISABLED

**File**: `pbrs_potentials.py:1219-1257`

```python
def _get_cached_or_compute_potential(...):
    """Get potential WITHOUT position-based caching (incompatible with velocity-dependent physics costs).
    
    IMPORTANT: Position-based caching is DISABLED for physics-aware PBRS.
    The same position with different velocities will have different physics costs,
    so caching by position alone is incorrect.
    """
    # DISABLED: Position-based caching incompatible with velocity-dependent physics costs
    use_cache = False
```

**Reason**: Position caching would break velocity-aware physics costs. The `path_calculator` already has velocity-bucketed step caching internally.

**Result**: Position cache DISABLED (correct for physics-aware PBRS) ✅

---

### 4. Waypoint System ❌ NOT USED in Production

**File**: `pbrs_potentials.py:1602-1622`

```python
# Use waypoint-aware potential if waypoints available
if self.momentum_waypoints:  # ← This is ALWAYS False (empty list)
    objective_pot = self._get_cached_or_compute_potential_with_waypoints(...)
else:
    # PRODUCTION PATH: Always uses this
    objective_pot = self._get_cached_or_compute_potential(...)
```

**File**: `pbrs_potentials.py:803`

```python
# Initialization
self.momentum_waypoints = momentum_waypoints or []  # ← Defaults to []
```

**File**: `main_reward_calculator.py:129-131`

```python
# PBRSCalculator created WITHOUT momentum_waypoints argument
self.pbrs_calculator = PBRSCalculator(
    path_calculator=path_calculator, reward_config=self.config
)
# momentum_waypoints defaults to [] → waypoint code NEVER executes
```

**Result**: Waypoint system NOT USED ✅ (would use geometric if enabled, but it's disabled)

---

### 5. Cascading Fallback ⚠️ INTENTIONAL Geometric Usage

**File**: `pbrs_potentials.py:268-326`

```python
# === CASCADING FALLBACK FOR UNREACHABLE TRAJECTORIES ===
# Level 1: Physics cost with velocity (primary)
# Level 2: Geometric distance without velocity (recovery path)
# Level 3: Zero potential (truly unreachable)

if physics_cost == float("inf"):
    # Current velocity makes goal unreachable
    # Try geometric distance (best-case with velocity recovery)
    fallback_distance = path_calculator.get_geometric_distance(
        player_pos, goal_pos,
        adjacency, base_adjacency, level_data, graph_data,
        entity_radius, NINJA_RADIUS,
        goal_id=goal_id,
        ninja_velocity=None,  # No velocity constraint for recovery path
    )
    
    if fallback_distance == float("inf"):
        # Truly unreachable - return zero potential
        return 0.0
    
    # Geometric path exists (velocity-independent recovery possible)
    # Provide WEAK gradient (0.2x scale) toward recovery
    ...
    fallback_potential = ... * 0.2  # 20% of normal potential
    return max(0.0, min(1.0, fallback_potential))
```

**Purpose**: When ninja's current velocity makes goal unreachable:
1. **If geometric path exists**: Provide weak (0.2x) gradient toward states where velocity could be recovered
2. **If no geometric path**: Return 0.0 (truly unreachable, be honest)

**Why Geometric Here?**:
- Shows "best case if velocity could be changed" (e.g., landing on platform)
- Respects walls, obstacles, mines (unlike Euclidean)
- Weak 0.2x scale indicates "difficult situation" vs normal 1.0x

**Result**: Geometric ONLY used for recovery gradient when doomed ⚠️ (intentional design)

---

## Summary Table

| Component | Uses | Type | Status |
|-----------|------|------|--------|
| **Primary PBRS potential** | `get_physics_cost()` | Physics cost | ✅ Correct |
| **Milestone tracking** | `get_last_distance_to_goal()` | Physics cost (cached) | ✅ Correct |
| **Position cache** | DISABLED | N/A | ✅ Correct |
| **Waypoint system** | NOT USED | N/A | ✅ Correct |
| **Fallback (doomed trajectories)** | `get_geometric_distance()` | Geometric (0.2x) | ⚠️ Intentional |

---

## Verification Commands

```bash
# Confirm physics costs are used for PBRS
cd /home/tetra/projects/nclone
grep -n "get_physics_cost" nclone/gym_environment/reward_calculation/pbrs_potentials.py
# Should show line 245: physics_cost = path_calculator.get_physics_cost(...)

# Confirm milestones use cached physics cost
grep -n "_cached_distance = state_with_metrics.get" nclone/gym_environment/reward_calculation/pbrs_potentials.py
# Should show line 1345: self._cached_distance = state_with_metrics.get("_pbrs_physics_cost", None)

# Confirm position cache is disabled
grep -n "use_cache = False" nclone/gym_environment/reward_calculation/pbrs_potentials.py  
# Should show line 1257: use_cache = False

# Confirm waypoints not used
grep -n "PBRSCalculator(" nclone/gym_environment/reward_calculation/main_reward_calculator.py
# Should show line 129-131: PBRSCalculator created WITHOUT momentum_waypoints
```

---

## Old Analysis is OBSOLETE

### What the Old Analysis Said (INCORRECT NOW):

> [`pbrs_potentials.py`](nclone/gym_environment/reward_calculation/pbrs_potentials.py) lines 158-200:
>
> - Uses `level_cache.get_geometric_distance()` which returns **geometric distances in pixels**
> - Level cache is built once per level using flood-fill from goals (lines 910-1049)
> - Paths are physics-optimal (pathfinding uses physics costs) but distances are geometric
> - PBRS normalizes by geometric path length: `Φ(d) = 1 - d/k` where d is pixels

### What Actually Happens NOW (CORRECT):

✅ Uses `path_calculator.get_physics_cost()` which returns **physics-weighted costs**  
✅ Paths use physics costs with velocity and jump duration awareness  
✅ PBRS normalizes by physics cost: `Φ(cost) = 1 - cost/baseline_cost`  
✅ Milestones track physics cost reduction (not geometric distance)  
✅ Position caching disabled (incompatible with velocity-dependent costs)  
✅ Cascading fallback provides weak geometric gradient ONLY when physics cost = inf  

---

## Implementation Date Log

### 2026-02-03: Physics-Aware PBRS Implementation
- Added `get_physics_cost()` method to path calculator
- Added `compute_baseline_physics_costs()` for normalization
- Modified `objective_distance_potential()` to use physics costs
- Updated milestone tracking to use cached physics costs
- Disabled position-based caching (velocity incompatibility)
- Added velocity and jump duration propagation through call chain
- Fixed wall contact check (aerial chains reset on wall)
- Added velocity-aware first-edge checks for wall jumps

### Key Fixes
1. **Wall Jump Support**: Added `not src_walled` check so velocity checks only apply in free-fall
2. **Jump Duration Tracking**: Added `jump_duration_remaining` parameter throughout
3. **Aerial Chain Reset**: Fixed to reset on wall contact (not just ground)
4. **Milestone Physics Costs**: Changed `_cached_distance` to store physics cost (not geometric)
5. **Position Cache Disabled**: Removed incompatible position-based caching

---

## Conclusion

✅ **All reward components now use PHYSICS COSTS**  
✅ **Geometric distances ONLY used for recovery gradient (0.2x) when doomed**  
✅ **Old analysis stating "PBRS uses geometric distances" is OBSOLETE**  
✅ **System respects velocity, jump duration, momentum, and trajectory feasibility**  

**Ready for training with physics-aware PBRS!** 🎯
