# Mine Death Predictor Visualization

## Overview

The hybrid mine death predictor now includes comprehensive debug visualization to help understand and validate the three-tier prediction system.

## Enabling Visualization

### In Test Environment

Press **M** key during runtime to toggle the mine predictor debug overlay.

```bash
python test_environment.py --level 00-00
# Press M to toggle mine predictor visualization
```

### In Code

```python
# Enable mine predictor debug visualization
env.set_mine_predictor_debug_enabled(True)

# Disable it
env.set_mine_predictor_debug_enabled(False)
```

## Visualization Components

### 1. Danger Zone Grid Cells (Tier 1)
- **Visual**: Semi-transparent orange squares (24x24 pixels)
- **Purpose**: Shows pre-computed spatial danger zone grid
- **Coverage**: All cells within 80px of any toggled mine
- **Query Speed**: O(1) lookup, handles ~95% of queries

### 2. Danger Zone Radius (Tier 1 Boundary)
- **Visual**: Orange circle outline (80px radius)
- **Purpose**: Shows boundary of Tier 1 spatial pre-filter
- **Meaning**: Outside this radius → ninja definitely safe (Tier 1 returns False)

### 3. Threshold Radius (Tier 2 Boundary)
- **Visual**: Yellow circle outline (30px radius)
- **Purpose**: Shows boundary between Tier 2 and Tier 3
- **Meaning**: 
  - Distance > 30px → Quick distance check (Tier 2)
  - Distance ≤ 30px → Full physics simulation (Tier 3)

### 4. Mine Positions
- **Visual**: Red filled circles (6px radius)
- **Purpose**: Shows exact position of toggled/deadly mines
- **Source**: Filtered to only reachable mines from graph system

### 5. Stats Overlay (Top-Left)
Displays real-time statistics:
- **Build Time**: How long it took to build danger zone grid
- **Mines**: Number of reachable toggled mines
- **Danger Cells**: Number of danger zone grid cells
- **Queries**: Total number of `is_action_deadly()` calls
- **Tier Breakdown**: Distribution of queries across tiers
  - T1: Handled by spatial pre-filter (target: >80%)
  - T2: Handled by distance check (target: ~15%)
  - T3: Handled by full simulation (target: <5%)

### 6. Legend (Top-Right)
Quick reference for visualization elements:
- ● Mine (red)
- ○ Tier 1 (80px) - orange circle
- ○ Tier 2 (30px) - yellow circle
- □ Danger cell - orange square

## Interpreting the Visualization

### Normal Operation
- **Most areas**: No orange overlay (safe, outside danger zones)
- **Near mines**: Orange grid cells appear
- **Close to mines**: Both orange and yellow circles visible
- **Tier stats**: T1 should handle >80% of queries

### Performance Issues
- **Build time > 10ms**: May indicate too many mines or large level
- **T3 queries > 10%**: Distance threshold may be too aggressive
- **Danger cells > 1000**: Very large danger zone (still fast, but memory concern)

### Validation Examples

#### Example 1: Ninja Far from Mines
```
Ninja position: (500, 500)
Nearest mine: (100, 100)
Distance: ~565px

Visualization:
- No orange overlay at ninja position
- Mine visible with circles in corner
- Stats show 100% T1 queries (spatial filter)
```

#### Example 2: Ninja Near Mine
```
Ninja position: (120, 100)
Nearest mine: (100, 100)
Distance: ~20px

Visualization:
- Orange grid cells around ninja
- Yellow circle visible (inside 30px threshold)
- Stats show some T3 queries (full simulation)
```

## Tuning Constants

If visualization reveals issues, you can tune these constants in `physics_constants.py`:

```python
# Tier 1: Spatial danger zone grid
MINE_DANGER_ZONE_RADIUS = 80.0  # Increase for more conservative pre-filter
MINE_DANGER_ZONE_CELL_SIZE = 24  # Cell size (matches N++ tiles)

# Tier 2: Distance threshold
MINE_DANGER_THRESHOLD = 30.0  # Decrease for more Tier 3 queries (more accurate)

# Tier 3: Physics simulation
MINE_DEATH_LOOKAHEAD_FRAMES = 6  # Increase for longer prediction horizon
```

### Trade-offs
- **Larger DANGER_ZONE_RADIUS**: More memory, fewer misses
- **Smaller DANGER_THRESHOLD**: More accurate, slower queries
- **More LOOKAHEAD_FRAMES**: More accurate, slower Tier 3

## Debug Workflow

1. **Enable visualization**: Press `M` in test environment
2. **Navigate to mine**: Move ninja near a toggled mine
3. **Observe tiers**: Watch which tier handles queries
4. **Check stats**: Verify tier distribution matches expectations
5. **Test edge cases**: Try approaching mine from different angles
6. **Validate coverage**: Ensure no gaps in danger zone grid

## Common Issues

### Issue: No visualization appears
- **Cause**: No toggled mines in level
- **Solution**: Use a level with toggle mines (type 1 or 21)

### Issue: Orange grid covers entire level
- **Cause**: Many reachable mines close together
- **Solution**: Normal for mine-heavy levels, no action needed

### Issue: T3 queries > 20%
- **Cause**: DANGER_THRESHOLD too low or ninja stuck near mine
- **Solution**: Increase DANGER_THRESHOLD or verify gameplay behavior

### Issue: Build time > 50ms
- **Cause**: Very large reachable area with many mines
- **Solution**: This should be rare; consider level design

## Performance Expectations

Based on visualization stats, here are target ranges:

| Metric | Target | Acceptable | Concerning |
|--------|--------|------------|------------|
| Build time | <5ms | <10ms | >20ms |
| Danger cells | <500 | <1000 | >2000 |
| T1 queries | >85% | >70% | <50% |
| T2 queries | 10-14% | 5-25% | - |
| T3 queries | <5% | <10% | >20% |
| Memory | <5KB | <20KB | >50KB |

## Integration with Other Debug Overlays

The mine predictor visualization works alongside other debug overlays:

- **Graph overlay (V)**: Shows reachability, mine predictor shows danger zones
- **Exploration (E)**: Shows visited cells, mine predictor shows hazards
- **Grid (C)**: Shows tile boundaries, mine predictor shows danger areas
- **Path-aware (P)**: Shows navigation paths, mine predictor shows obstacles

All overlays can be enabled simultaneously for comprehensive debugging.

## Example Session

```bash
# Start test environment with a mine-heavy level
python test_environment.py --level SI-A-00-00

# Enable graph overlay to see reachability
Press V

# Enable mine predictor visualization
Press M

# Observe:
# - Orange danger zones only appear near reachable mines
# - Stats show build time ~2-3ms
# - As ninja moves, T1/T2/T3 distribution updates
# - Most queries handled by T1 (>90%)

# Toggle off when done
Press M
```

## Future Enhancements

Potential visualization improvements:
- Color-code danger zones by mine count (red = multiple mines)
- Show which action would be masked for current state
- Animate Tier 3 simulation trajectory
- Display distance to nearest mine as ninja moves
- Heatmap of query frequency per grid cell

