# Velocity-Aware Jump Duration Logic

**Date**: 2026-02-03  
**Status**: ✅ Corrected Implementation

---

## The Correct Physics Model

### N++ Jump Mechanics

1. **Strong Upward Momentum (yspeed < -5.0)**:
   - Ninja continues rising naturally due to momentum
   - **Jump button hold NOT required**
   - Jump frames don't matter - momentum carries the ninja

2. **Moderate Upward Velocity (-5.0 ≤ yspeed < -0.5)**:
   - Ninja rising but losing momentum
   - **Jump button hold REQUIRED** to sustain rise
   - Jump frames critical - without them, ninja will start falling

3. **Falling/Stationary (yspeed ≥ -0.5)**:
   - Ninja falling or barely rising
   - **Cannot initiate new jump in mid-air**
   - Always impossible regardless of jump frames

---

## Implementation Logic

### Decision Tree for Upward Edges (Airborne)

```
Is ninja falling? (yspeed ≥ -0.5)
├─ YES → velocity_multiplier = inf (cannot jump in mid-air)
└─ NO → Is velocity strong? (yspeed < -5.0)
    ├─ YES → velocity_multiplier = 0.5 (momentum carries, frames don't matter)
    └─ NO → Does ninja have jump frames? (jump_duration_remaining > 0)
        ├─ YES → velocity_multiplier = 1.5 (can hold jump button)
        └─ NO → velocity_multiplier = inf (cannot sustain rise)
```

### Code Implementation

```python
# pathfinding_utils.py:324-343
if dy < 0 and not src_grounded:  # Upward edge while airborne
    if ninja_yspeed >= NINJA_RISING_SLOW_THRESHOLD:  # -0.5
        # CASE 1: Falling - cannot jump
        velocity_multiplier = float("inf")
    
    elif ninja_yspeed < NINJA_RISING_FAST_THRESHOLD:  # -5.0
        # CASE 2: Strong momentum - carries naturally
        # Jump frames irrelevant (momentum-based rise)
        velocity_multiplier = VELOCITY_UPWARD_FAST_MULTIPLIER  # 0.5
    
    elif jump_duration_remaining is not None and jump_duration_remaining <= 0.0:
        # CASE 3: Moderate velocity BUT no jump frames
        # Cannot sustain rise without jump button hold
        velocity_multiplier = float("inf")
    
    else:
        # CASE 4: Moderate velocity AND has jump frames
        # Can hold jump button to sustain rise
        velocity_multiplier = VELOCITY_UPWARD_SLOW_MULTIPLIER  # 1.5
```

---

## Example Scenarios

### Scenario 1: Strong Momentum (Jump Frames Irrelevant)

```python
state = {
    "position": (100, 50),
    "velocity": (0.0, -6.5),  # Strong upward momentum
    "jump_duration_remaining": 0.0,  # EXHAUSTED
    "airborne": True,
    "goal": (100, 0)  # 5 tiles up
}

# Decision: yspeed < -5.0 → CASE 2
edge_cost = 1.0 × 0.5 = 0.5
physics_cost = ~25
potential = 1 - 25/500 = 0.95

# ✅ Momentum carries ninja upward naturally
```

### Scenario 2: Moderate Velocity with Jump Frames

```python
state = {
    "position": (100, 50),
    "velocity": (0.0, -2.0),  # Moderate velocity
    "jump_duration_remaining": 0.6,  # 60% frames remaining
    "airborne": True,
    "goal": (100, 0)  # 5 tiles up
}

# Decision: -5.0 ≤ yspeed < -0.5 AND has frames → CASE 4
edge_cost = 1.0 × 1.5 = 1.5
physics_cost = ~40
potential = 1 - 40/500 = 0.92

# ✅ Can hold jump button to sustain rise
```

### Scenario 3: Moderate Velocity WITHOUT Jump Frames

```python
state = {
    "position": (100, 50),
    "velocity": (0.0, -2.0),  # Moderate velocity
    "jump_duration_remaining": 0.0,  # NO FRAMES LEFT
    "airborne": True,
    "goal": (100, 0)  # 5 tiles up
}

# Decision: -5.0 ≤ yspeed < -0.5 BUT no frames → CASE 3
edge_cost = inf
physics_cost = inf
geometric_dist = 50
potential = 0.2 × (1 - 50/500) = 0.18

# ❌ Cannot sustain rise without jump button hold
```

### Scenario 4: Falling (Always Impossible)

```python
state = {
    "position": (100, 50),
    "velocity": (0.0, 1.0),  # Falling
    "jump_duration_remaining": 1.0,  # FULL FRAMES (doesn't matter)
    "airborne": True,
    "goal": (100, 0)  # 5 tiles up
}

# Decision: yspeed ≥ -0.5 → CASE 1
edge_cost = inf
physics_cost = inf

# ❌ Cannot jump in mid-air
```

---

## PBRS Reward Examples

### Good: Building Strong Momentum

```python
# Agent learns to build strong momentum for upward sections
state_0: position=(100, 80), velocity=(0, -3.0), frames=0.8, Φ=0.85
state_1: position=(100, 70), velocity=(0, -7.0), frames=0.5, Φ=0.92

# Accelerated to strong momentum (yspeed < -5.0)
F(s,s') = 0.99 × 0.92 - 0.85 = +0.06  # POSITIVE REWARD!

# ✅ Agent learns: Build strong momentum for frame-efficient rising
```

### Good: Conserving Jump Frames

```python
# Agent conserves frames for sections needing sustained button hold
state_0: position=(100, 80), velocity=(0, -2.0), frames=0.9, Φ=0.88
state_1: position=(100, 70), velocity=(0, -2.5), frames=0.7, Φ=0.90

# Sustained moderate velocity with jump button hold
F(s,s') = 0.99 × 0.90 - 0.88 = +0.01  # POSITIVE REWARD!

# ✅ Agent learns: Conserve frames when moderate velocity needed
```

### Bad: Exhausting Frames with Moderate Velocity

```python
# Agent wastes frames, ends up with moderate velocity and no frames
state_0: position=(100, 80), velocity=(0, -2.0), frames=0.8, Φ=0.88
state_1: position=(100, 70), velocity=(0, -2.0), frames=0.0, Φ=0.18

# Moderate velocity BUT no frames left → inf cost → recovery gradient
F(s,s') = 0.99 × 0.18 - 0.88 = -0.70  # STRONG NEGATIVE!

# ❌ Agent learns: Don't exhaust frames with only moderate velocity
```

---

## Learning Outcomes

### The Agent Learns TWO Strategies:

#### Strategy A: Momentum-Based Rising
- **Build strong upward momentum** (< -5.0)
- Jump frames become irrelevant
- Efficient for large upward gaps
- Example: Diagonal jumps that build speed

#### Strategy B: Frame-Conserving Rising
- **Maintain moderate velocity** with jump button hold
- **Conserve jump frames** for critical sections
- Efficient for sustained upward climbs
- Example: Vertical shafts requiring careful timing

### The Agent AVOIDS:

- ❌ Exhausting jump frames with only moderate velocity
- ❌ Holding jump unnecessarily on non-critical sections
- ❌ Approaching upward sections without momentum OR frames

---

## Key Constants

```python
NINJA_RISING_SLOW_THRESHOLD = -0.5   # Below this = falling/stationary
NINJA_RISING_FAST_THRESHOLD = -5.0   # Below this = strong momentum
MAX_JUMP_DURATION = 45                # Maximum jump hold frames

VELOCITY_UPWARD_FAST_MULTIPLIER = 0.5    # Strong momentum cost
VELOCITY_UPWARD_SLOW_MULTIPLIER = 1.5    # Moderate velocity cost
```

---

## Comparison: Before vs After

### Before (Incorrect)

| Velocity | Jump Frames | Old Logic | Problem |
|----------|-------------|-----------|---------|
| -6.0 (strong) | 0.0 | Cost = inf | ❌ Wrong! Momentum carries naturally |
| -2.0 (moderate) | 0.6 | Cost = 1.5 | ✅ Correct |
| -2.0 (moderate) | 0.0 | Cost = 1.5 | ❌ Wrong! Cannot sustain |

**Issue**: Blocked ALL upward movement when frames exhausted, even with strong momentum.

### After (Correct)

| Velocity | Jump Frames | New Logic | Correct? |
|----------|-------------|-----------|----------|
| -6.0 (strong) | 0.0 | Cost = 0.5 | ✅ Momentum-based |
| -2.0 (moderate) | 0.6 | Cost = 1.5 | ✅ Can hold jump |
| -2.0 (moderate) | 0.0 | Cost = inf | ✅ Unsustainable |

**Fix**: Jump frames only matter for moderate velocities. Strong momentum works regardless.

---

## Validation Tests

### Test 1: Strong Momentum Without Frames

```python
edge_cost = _calculate_physics_aware_cost(
    src_pos=(100, 50),
    dst_pos=(100, 49),  # Upward
    parent_pos=None,  # First edge
    ninja_velocity=(0.0, -6.5),  # Strong momentum
    jump_duration_remaining=0.0,  # No frames
    physics_cache={(100, 50): {"grounded": False, "walled": False}},
    ...
)
assert edge_cost < 1.0  # Should be cheap (0.5x)
print("✅ Test 1 passed: Strong momentum allows upward movement")
```

### Test 2: Moderate Velocity With Frames

```python
edge_cost = _calculate_physics_aware_cost(
    src_pos=(100, 50),
    dst_pos=(100, 49),  # Upward
    parent_pos=None,
    ninja_velocity=(0.0, -2.0),  # Moderate
    jump_duration_remaining=0.6,  # Has frames
    physics_cache={(100, 50): {"grounded": False, "walled": False}},
    ...
)
assert 1.0 < edge_cost < 2.0  # Should be moderate (1.5x)
print("✅ Test 2 passed: Moderate velocity with frames is possible")
```

### Test 3: Moderate Velocity Without Frames

```python
edge_cost = _calculate_physics_aware_cost(
    src_pos=(100, 50),
    dst_pos=(100, 49),  # Upward
    parent_pos=None,
    ninja_velocity=(0.0, -2.0),  # Moderate
    jump_duration_remaining=0.0,  # No frames!
    physics_cache={(100, 50): {"grounded": False, "walled": False}},
    ...
)
assert edge_cost == float("inf")  # Should be impossible
print("✅ Test 3 passed: Moderate velocity without frames is blocked")
```

---

## Summary

✅ **Physics-Accurate**: Respects N++ momentum vs button-hold mechanics  
✅ **Velocity-Aware**: Jump frames only matter for moderate velocities  
✅ **Encourages Efficiency**: Agent learns both momentum-building and frame-conserving strategies  
✅ **Prevents False Positives**: No rewards for unsustainable trajectories  
✅ **Documentation Complete**: Full technical spec and examples provided  

**The system now correctly models the interaction between velocity, jump duration, and upward movement feasibility!** 🎯
