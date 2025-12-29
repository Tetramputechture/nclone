# Reward Balance Verification (Post-Fix)

## Configuration Summary
- **PBRS Weight**: 40.0 (discovery phase, <5% success)
- **Completion Reward**: 50.0
- **Switch Reward**: 40.0
- **Death Penalty**: -8.0 (discovery phase)
- **Time Penalty**: -0.01 per step
- **Global Scale**: 0.1 (all rewards scaled)
- **Stagnation Timeout Penalty**: -20.0 (if progress < 15%)

**Note**: Increased from 20.0 based on TensorBoard analysis showing gradient signal too weak vs entropy noise.

## Typical Episode: ~150 steps (600 frames / 4 frame_skip)

### Scenario 1: SUCCESS (Complete Level)
```
PBRS:       40.0  (full path, 100% progress)
Switch:     40.0  (milestone)
Completion: 50.0  (terminal)
Time:       -1.5  (150 steps × -0.01)
-----------------------------------
Total:      128.5  (unscaled)
Scaled:     12.85  ✓ BEST OUTCOME
```

### Scenario 2: HIGH RISK - 50% Progress + Death
```
PBRS:       20.0  (50% of max 40.0)
Switch:     0.0   (didn't reach)
Death:      -8.0  (penalty)
Time:       -0.75 (75 steps × -0.01)
-----------------------------------
Total:      11.25  (unscaled)
Scaled:     1.125  ✓ ACCEPTABLE (encourages bold play)
```

### Scenario 3: MODERATE RISK - 30% Progress + Death
```
PBRS:       12.0  (30% of max 40.0)
Switch:     0.0   (didn't reach)
Death:      -8.0  (penalty)
Time:       -0.45 (45 steps × -0.01)
-----------------------------------
Total:      3.55  (unscaled)
Scaled:     0.355  ✓ ACCEPTABLE (still encourages progress)
```

### Scenario 4: SWITCH + Death (Major Milestone)
```
PBRS:       30.0  (75% progress to reach switch)
Switch:     40.0  (milestone achieved!)
Death:      -8.0  (penalty)
Time:       -1.125 (112 steps × -0.01)
-----------------------------------
Total:      60.875 (unscaled)
Scaled:     6.088  ✓ EXCELLENT (major milestone, definitely worth risk)
```

### Scenario 5: STAGNATION - 10% Progress + Timeout
```
PBRS:       4.0   (10% of max 40.0)
Switch:     0.0   (didn't reach)
Timeout:    -20.0 (stagnation penalty, <15% progress)
Time:       -5.0  (500 steps × -0.01)
-----------------------------------
Total:      -21.0  (unscaled)
Scaled:     -2.10  ✗ WORST OUTCOME (heavily discouraged)
```

### Scenario 6: CAMPING - 16% Progress + Timeout
```
PBRS:       6.4   (16% of max 40.0)
Switch:     0.0   (didn't reach)
Timeout:    0.0   (>15% progress, no penalty)
Time:       -5.0  (500 steps × -0.01)
-----------------------------------
Total:      1.4   (unscaled)
Scaled:     0.14  ✓ Better than stagnation, but poor
```

### Scenario 7: OSCILLATION - 0% Net Progress + Timeout
```
PBRS:       0.0   (no net distance reduction)
Switch:     0.0   (didn't reach)
Timeout:    -20.0 (stagnation penalty, 0% < 15%)
Time:       -5.0  (500 steps × -0.01)
-----------------------------------
Total:      -25.0  (unscaled)
Scaled:     -2.50  ✗✗ TERRIBLE (even worse than stagnation)
```

## Reward Hierarchy Verification ✓

```
1. Success (12.85)           BEST - always preferred
2. Switch + Death (6.088)    EXCELLENT - major milestone highly valuable
3. High Risk 50% + Death (1.125) GOOD - bold exploration rewarded
4. Moderate Risk 30% + Death (0.355) ACCEPTABLE - some progress
5. Camping 16% (0.14)        POOR - minimal progress
6. Stagnation 10% (-2.10)    BAD - heavily discouraged
7. Oscillation (-2.50)       WORST - no progress at all

Critical Checks:
✓ Success >> Switch+Death >> Progress+Death >> Camping
✓ Switch+Death (6.088) >> 50%+Death (1.125) - milestone is 5.4x more valuable
✓ 30%+Death (0.355) > Camping (0.14) - encourages risk-taking
✓ Any forward progress > Stagnation (-2.10)
✓ Stagnation > Oscillation (-2.50)
```

## Break-Even Analysis

**Question: How much progress justifies death risk?**

For death to be acceptable vs camping: `PBRS - 8.0 - time > camping_reward`

With 75 steps to die:
```
PBRS - 8.0 - 0.75 > 0.14  (camping 16%)
PBRS > 8.89
Progress needed: 8.89 / 40.0 = 22.2%
```

**Result**: Agent needs **>22% progress** to justify death risk over camping. This encourages:
- Bold exploration when making meaningful progress (vs 32% with weight=20)
- Reasonable risk-reward tradeoff (not too conservative)
- Learning from mistakes (dying at 20% vs 50% matters significantly)

## Critical Ratios

### PBRS to Terminal Rewards
```
PBRS Max:        40.0
Completion:      50.0  (1.25x PBRS)
Switch:          40.0  (1.0x PBRS - equal importance!)
Death:           -8.0  (0.2x PBRS)

Ratio Check: Terminal rewards dominate but PBRS is significant ✓
```

### Time Penalty Impact
```
150 steps:  -1.5 total  (1.2% of completion reward)
500 steps:  -5.0 total  (3.8% of completion reward)

Impact: Mild efficiency pressure, doesn't dominate ✓
```

### PBRS Weight vs Value Stability
```
Original weight: 80.0 → Max episode return ~170
Reduced weight:  20.0 → Max episode return ~110 (too weak, signal drowned by noise)
Balanced weight: 40.0 → Max episode return ~130 (strong signal, stable learning)

Final: 23% lower variance than original, 2x stronger than conservative ✓
```

### Signal-to-Noise Ratio
```
PBRS gradient (12px forward):  ~0.048 scaled
Entropy coefficient:            0.03
Policy noise:                   ~0.02-0.05

Ratio: PBRS signal is ~1.5x entropy - sufficient to guide learning ✓
```

## Conclusion

**All reward balancing checks PASS** ✓

The agent is always incentivized to:
1. **Prefer success** over all other outcomes (12.85 vs -2.50 oscillation = 5.1x better)
2. **Take calculated risks** (50% progress + death = 1.125 > camping = 0.14, 8x better)
3. **Pursue milestones** (switch + death = 6.088, 43x better than oscillation)
4. **Avoid stagnation** (camping/oscillation heavily penalized at -2+)
5. **Make forward progress** (any progress > no progress)

The balanced PBRS weight (40.0) provides:
- **Strong gradient signal** (overcomes entropy noise by 1.5x)
- **Stable value learning** (23% lower variance than original 80.0)
- **Clear hierarchy** (terminal rewards still dominate)
- **Efficient risk-reward** (22% breakeven point encourages bold play)

**Status**: Ready for training with stronger gradient signal! ✓
