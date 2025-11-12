# Death Probability Visualization

## Overview

The death probability visualization provides real-time feedback on action safety when navigating near toggled mines. It displays:
1. **Action Masking**: Which actions are blocked due to certain death
2. **Death Probability**: Likelihood of death for each action over N frames
3. **Visual Indicators**: Color-coded bars showing risk levels

## Enabling Visualization

### In Test Environment

Press **D** key during runtime to toggle the death probability visualization.

```bash
python test_environment.py --level 00-00
# Navigate near a toggled mine
# Press D to toggle death probability visualization
```

### In Code

```python
# Enable death probability visualization
env.set_death_probability_debug_enabled(True)

# Configure number of frames to simulate (default: 10)
env.set_death_probability_frames(15)

# Disable it
env.set_death_probability_debug_enabled(False)
```

## Visualization Components

### 1. Compact Static Panel
A **static panel positioned in the furthest corner from the ninja** showing:
- **Title**: "Death Risk (Nf)" where N is frames simulated
- **Overall Death Probability Bar**: Large progress bar showing safest available action
- **Distance Display**: "Dist: Xpx" - distance to nearest mine
- **Action Bars**: 6 compact bars (NO, LT, RT, JP, JL, JR)
  - NO = NOOP, LT = LEFT, RT = RIGHT, JP = JUMP, JL = JUMP+LEFT, JR = JUMP+RIGHT

### 2. Color Coding
- **🟢 Green**: Safe actions (death probability < 50%)
- **🟡 Yellow**: Risky actions (death probability ≥ 50%)
- **🔴 Red**: Masked actions (certain death, 100% probability)

### 3. Overall Death Probability
- **Bar Color**: Matches risk level (green/yellow/red)
- **Value**: Minimum probability among non-masked actions
- **Purpose**: Quick at-a-glance risk assessment
- **Display**: Centered percentage on filled bar

### 4. Smart Positioning
- Panel positioned **80 pixels away** from ninja in opposite quadrant
- Follows ninja while staying out of the way
- Automatically adjusts to stay on screen
- Close enough to read quickly, far enough to not obstruct view

## How It Works

### Probability Calculation
For each of the 6 actions:
1. **Check for certain death**: Uses hybrid predictor's `is_action_deadly()` method
2. **If deadly**: Mark as MASKED, set probability to 100%
3. **If safe**: Simulate forward N frames, check for collision at each frame
4. **Calculate probability**: (frames leading to death) / (total frames simulated)

### Frame Simulation
- Default: 10 frames (configurable 1-30)
- For each frame offset (1 to N):
  - Simulate action for that many frames
  - Check if collision occurs
  - Count collisions across all frame offsets
- Probability = collision_count / total_frames

### Example
With 10 frame simulation:
- Action "RIGHT" simulated for frames 1, 2, 3, ..., 10
- Collision detected at frames 7, 8, 9, 10 (4 out of 10)
- Death probability = 40%
- Displayed as yellow bar (risky but not masked)

## Interpretation

### Panel Layout
```
┌─────────────────────┐
│ Death Risk (10f)    │  ← Title with frame count
│ ███████████ 30%     │  ← Overall death probability (green = safe)
│ Dist: 65px          │  ← Distance to nearest mine
│ NO ████░░ 40        │  ← Action: NOOP, 40% risk
│ LT ██░░░░ 20        │  ← Action: LEFT, 20% risk
│ RT ██████ 60        │  ← Action: RIGHT, 60% risk (yellow)
│ JP ███░░░ 30        │  ← Action: JUMP, 30% risk
│ JL █░░░░░ 10        │  ← Action: JUMP+LEFT, 10% risk (safest!)
│ JR █████████ X      │  ← Action: JUMP+RIGHT, MASKED (red)
└─────────────────────┘
```

### Reading the Display

**Overall Bar (Top)**:
- Shows minimum death probability of safest non-masked action
- Green (<50%): Safe options available
- Yellow (≥50%): All options risky
- Red (100%): All actions masked (trapped!)

**Action Bars**:
- Compact 2-letter codes for space efficiency
- Bar length shows probability (longer = more dangerous)
- Number shows exact percentage (or "X" for masked)
- Color coding: Green (safe), Yellow (risky), Red (deadly)

## Performance Considerations

### Computational Cost
Death probability calculation is **expensive** because it:
- Simulates 6 actions × N frames = 60+ physics simulations
- Each simulation includes collision detection
- Called every frame when visualization is enabled

**Expected Time**: ~5-15ms per calculation depending on:
- Number of frames to simulate (default 10)
- Number of mines in level
- Physics complexity

### When to Use
- ✅ **Development/Debugging**: Understanding action masking behavior
- ✅ **Training Analysis**: Validating safety constraints
- ✅ **Level Testing**: Identifying dangerous areas
- ❌ **Production Training**: Too expensive for real-time use

### Optimization Tips
1. **Reduce frame count**: Use 5-10 frames for faster calculation
2. **Enable only when needed**: Toggle off when not analyzing safety
3. **Combine with mine predictor**: Use both M and D keys for full picture

## Use Cases

### 1. Debugging Action Masking
**Problem**: Agent seems to have valid actions masked
**Solution**: 
- Enable death probability visualization (D key)
- Observe which actions are masked (red bars)
- Check if probabilities match expectations
- Verify mine positions are correct (M key)

### 2. Understanding Risk Gradients
**Problem**: Want to understand how danger increases near mines
**Solution**:
- Move ninja gradually toward a mine
- Watch probability bars increase
- Observe transition from green → yellow → red
- Identify safe maneuvering distance

### 3. Validating Hybrid Predictor
**Problem**: Uncertain if predictor is too conservative/aggressive
**Solution**:
- Check masked actions (100% probability)
- Verify they actually lead to death
- Check non-masked actions have reasonable probabilities
- Adjust MINE_DANGER_THRESHOLD if needed

### 4. Training Debugging
**Problem**: Agent dying to mines despite action masking
**Solution**:
- Enable visualization during episode replay
- Check if death probability was visible before death
- Verify action masking was active
- Look for edge cases or predictor gaps

## Configuration

### Frame Simulation Count
```python
# More frames = more accurate but slower
env.set_death_probability_frames(5)   # Fast (2-5ms)
env.set_death_probability_frames(10)  # Default (5-10ms)
env.set_death_probability_frames(20)  # Accurate (10-20ms)
env.set_death_probability_frames(30)  # Very accurate (15-30ms)
```

### Trade-offs
| Frames | Accuracy | Speed | Use Case |
|--------|----------|-------|----------|
| 5 | Basic | Fast | Quick debugging |
| 10 | Good | Medium | General use (default) |
| 15 | Better | Slower | Detailed analysis |
| 20-30 | Best | Slow | Validation testing |

## Integration with Other Visualizations

### With Mine Predictor (M key)
- Mine predictor shows spatial danger zones
- Death probability shows per-action risk
- Together provide complete safety picture

### With Graph Overlay (V key)
- Graph shows navigation paths
- Death probability shows action constraints
- Helps understand path feasibility near mines

### With Debug Overlay (G key)
- Shows ninja state (velocity, airborne, etc.)
- Death probability explains why actions are risky
- Useful for physics-specific edge cases

## Example Session

```bash
# Start test environment with mine-heavy level
python test_environment.py --level SI-A-00-00

# Navigate ninja near a toggled mine
# Enable mine predictor to see danger zones
Press M

# Enable death probability to see action risks
Press D

# Observe:
# 1. As ninja approaches mine, probabilities increase
# 2. Some actions masked (red, 100%)
# 3. Other actions show gradient (green → yellow)
# 4. Distance to mine shown in panel
# 5. Panel follows ninja around screen

# Try different actions and watch probabilities change
# Move away from mine - probabilities decrease

# Toggle off when done
Press D  # Disable death probability
Press M  # Disable mine predictor
```

## Common Patterns

### Pattern 1: All Actions Safe
```
┌─────────────────────┐
│ Death Risk (10f)    │
│ ░░░░░░░░░░░░ 0%     │  ← Green overall bar (safest = 0%)
│ Dist: 125px         │
│ NO ░░░░░ 0          │  All actions safe
│ LT ░░░░░ 0          │
│ RT ░░░░░ 0          │
│ JP ░░░░░ 0          │
│ JL ░░░░░ 0          │
│ JR ░░░░░ 0          │
└─────────────────────┘
```
**Meaning**: Ninja far from mines, all actions safe

### Pattern 2: Directional Risk
```
┌─────────────────────┐
│ Death Risk (10f)    │
│ ░░░░░░░░░░░░ 0%     │  ← Green overall bar (some safe options)
│ Dist: 45px          │
│ NO ░░░░░ 0          │  ← Safe
│ LT ░░░░░ 0          │  ← Safe
│ RT ████████ 80      │  ← Yellow bar (risky!)
│ JP █░░░░ 10         │
│ JL ░░░░░ 0          │
│ JR █████████ X      │  ← Red bar (masked!)
└─────────────────────┘
```
**Meaning**: Mine to the right, moving right is dangerous

### Pattern 3: Trapped Situation
```
┌─────────────────────┐
│ Death Risk (10f)    │
│ ████████████ 100%   │  ← Red overall bar (trapped!)
│ Dist: 8px           │
│ NO █████████ X      │  All actions masked
│ LT █████████ X      │
│ RT █████████ X      │
│ JP █████████ X      │
│ JL █████████ X      │
│ JR █████████ X      │
└─────────────────────┘
```
**Meaning**: Unavoidable death situation (very rare)

### Pattern 4: Escape Route Available
```
┌─────────────────────┐
│ Death Risk (10f)    │
│ ██░░░░░░░░░░ 20%    │  ← Green overall bar (escape possible!)
│ Dist: 35px          │
│ NO ████████ 80      │  ← Staying dangerous
│ LT ██░░░ 20         │  ← Best escape!
│ RT █████████ X      │  ← Death
│ JP ███████ 70       │  ← Risky
│ JL ███░░ 30         │  ← Second best
│ JR █████████ X      │  ← Death
└─────────────────────┘
```
**Meaning**: LEFT (20%) offers best escape route

## Limitations

1. **Computational Cost**: Too expensive for real-time training
2. **Simulation Horizon**: Only predicts N frames ahead
3. **Deterministic**: Assumes deterministic physics (correct for N++)
4. **Single Threat**: Only considers mine collisions, not other hazards
5. **State Dependent**: Probability changes with ninja state (velocity, position)

## Future Enhancements

Potential improvements:
- Show trajectory preview for masked actions
- Heatmap of safe/unsafe positions
- Historical probability tracking (trend lines)
- Multi-threat analysis (mines + other entities)
- Recommendation system (suggest safest action)
- Performance profiling per calculation

