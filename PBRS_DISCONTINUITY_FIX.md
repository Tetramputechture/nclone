# PBRS Discontinuity Fix - Two-Phase Normalization

## Summary

Fixed a critical bug in the PBRS reward system where switch activation caused a large negative reward penalty (~-20 with weight=40), partially negating the intended switch milestone reward.

## The Bug

### Root Cause

The PBRS potential function used single-phase normalization:
```python
Φ(s) = 1 - (distance_to_goal / combined_path_distance)
```

Where `combined_path_distance = spawn→switch + switch→exit`

### Discontinuity at Switch Activation

When the switch activated:
- **Before**: `goal = switch`, `distance = 0px` → `Φ = 1.0`
- **After**: `goal = exit`, `distance = 500px` → `Φ = 0.5`

This created a sharp **-0.5 potential drop**, resulting in:
```
PBRS = γ × Φ(s') - Φ(s) = 0.99 × 0.5 - 1.0 = -0.505
Weighted (×40): -20.2
```

### Impact on Training

- Switch milestone reward effectively reduced from +40 to ~+20
- Agent received confusing signal: achieving switch gave negative PBRS
- May have caused hesitation near switch or avoided switch activation

## The Fix

### Two-Phase Normalization

Modified potential function to use phase-specific normalization:

**Switch Phase** (goal = switch):
```python
Φ(s) = 0.5 × (1 - distance_to_switch / spawn_to_switch_distance)
# Range: [0.0, 0.5]
```

**Exit Phase** (goal = exit):
```python
Φ(s) = 0.5 + 0.5 × (1 - distance_to_exit / switch_to_exit_distance)
# Range: [0.5, 1.0]
```

### Continuity at Switch Activation

At the switch position:
- **Switch phase**: `distance = 0` → `Φ = 0.5 × (1 - 0/500) = 0.5`
- **Exit phase**: `distance = 500` → `Φ = 0.5 + 0.5 × (1 - 500/500) = 0.5`

Both formulas yield **Φ = 0.5**, ensuring continuity!

```
PBRS = γ × Φ(s') - Φ(s) = 0.99 × 0.5 - 0.5 = -0.005
Weighted (×40): -0.2 (negligible!)
```

## Verification

Run `verify_pbrs_continuity.py` to verify the fix:

```bash
python verify_pbrs_continuity.py
```

### Expected Results

✓ Potential continuous at switch activation (Φ = 0.5 on both sides)
✓ PBRS near zero at switch activation (no large penalty)
✓ Continuity maintained with asymmetric level distances

### Gradient Consistency Note

The gradients per pixel differ slightly between phases due to the `(1-γ)×Φ(s')` oscillation penalty term:
- Switch phase (Φ~0.25): PBRS for 12px ≈ 0.38
- Exit phase (Φ~0.75): PBRS for 12px ≈ 0.18

This is **expected behavior** with γ < 1.0 and does **not violate policy invariance**. The oscillation penalty term naturally scales with potential value, creating slightly weaker gradients in the exit phase (when closer to goal). This is acceptable and may even be desirable.

## Files Modified

1. **pbrs_potentials.py**:
   - Modified `objective_distance_potential()` to use two-phase normalization
   - Updated `_compute_combined_path_distance()` to return phase-specific distances
   - Added cache variables for phase distances

2. **main_reward_calculator.py**:
   - Updated misleading comments claiming continuity without normalization
   - Added diagnostic logging at switch activation to verify fix

## Mathematical Properties Preserved

- **Policy Invariance**: Still holds (Ng et al. 1999 theorem applies for any γ)
- **Markov Property**: No episode history dependencies
- **Dense Rewards**: Sub-pixel gradient still works (0.05-3.33px movements)
- **Oscillation Penalty**: γ=0.99 still penalizes inefficient paths

## Benefits

1. **No discontinuity penalty**: Switch milestone reward fully preserved
2. **Clearer learning signal**: Achieving switch no longer gives negative PBRS
3. **Equal phase contribution**: Both phases contribute equally to total potential range
4. **LSTM-friendly**: Continuous potential function better for temporal learning

## Backward Compatibility

- Requires phase-specific distances in state dict:
  - `_pbrs_spawn_to_switch_distance`
  - `_pbrs_switch_to_exit_distance`
- These are computed and cached automatically by `PBRSCalculator`
- No changes needed to calling code

## Future Considerations

If gradient consistency between phases becomes important:
- Could use γ=1.0 to eliminate oscillation penalty term
- Would lose oscillation penalty (neutral on staying still)
- Trade-off between continuity and oscillation pressure

Current choice (γ=0.99 with two-phase normalization) balances:
- Continuity at switch activation (fixed!)
- Oscillation penalty (preserved)
- Policy invariance (maintained)





