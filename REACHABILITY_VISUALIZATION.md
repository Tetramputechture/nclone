# Reachability Visualization System

This document describes the comprehensive reachability analysis and visualization system integrated into the N++ environment test suite.

## Overview

The reachability visualization system provides real-time analysis and visualization of:
- **Reachable positions** from any starting point in the level
- **Subgoals and strategic waypoints** for navigation planning
- **Exploration frontiers** for curiosity-driven reinforcement learning
- **Interactive controls** for dynamic analysis during gameplay

## Command Line Arguments

### Basic Reachability Visualization

```bash
# Enable reachability analysis overlay
python nclone/test_environment.py --visualize-reachability

# Show reachability analysis from current ninja position (updates dynamically)
python nclone/test_environment.py --reachability-from-ninja

# Visualize identified subgoals and strategic waypoints
python nclone/test_environment.py --show-subgoals

# Visualize exploration frontiers for curiosity-driven RL
python nclone/test_environment.py --show-frontiers
```

### Combined Visualization

```bash
# Enable all reachability features
python nclone/test_environment.py --visualize-reachability --reachability-from-ninja --show-subgoals --show-frontiers

# Use with custom map
python nclone/test_environment.py --visualize-reachability --map path/to/custom/map.npp
```

### Screenshot Export

```bash
# Export frame with reachability analysis and quit
python nclone/test_environment.py --export-reachability output.png --visualize-reachability --show-subgoals

# Export with all features enabled
python nclone/test_environment.py --export-reachability analysis.png --visualize-reachability --show-subgoals --show-frontiers --headless
```

## Runtime Controls

When reachability visualization is active, the following keyboard controls are available:

| Key | Action | Description |
|-----|--------|-------------|
| `T` | Toggle reachability overlay | Enable/disable reachability position visualization |
| `N` | Update from ninja position | Recalculate reachability from current ninja location |
| `U` | Toggle subgoal visualization | Show/hide identified subgoals and waypoints |
| `F` | Toggle frontier visualization | Show/hide exploration frontiers |
| `X` | Export screenshot | Save current frame with reachability overlay |
| `R` | Reset environment | Reset the level and clear analysis |

## Visualization Elements

### Reachable Positions
- **Green overlay**: Positions reachable from the analysis starting point
- **Transparency**: Indicates confidence or accessibility level
- **Real-time updates**: When using `--reachability-from-ninja`

### Subgoals and Waypoints
- **Blue markers**: Strategic waypoints for navigation
- **Yellow markers**: Key objectives and subgoals
- **Size variation**: Indicates priority or importance

### Exploration Frontiers
- **Red markers**: Unexplored areas adjacent to reachable positions
- **Gradient intensity**: Shows exploration potential
- **Dynamic updates**: Frontiers change as ninja moves

## Integration with Existing Features

The reachability system integrates seamlessly with existing visualization:

```bash
# Combine with graph visualization
python nclone/test_environment.py --visualize-graph --visualize-reachability

# Use with standalone graph window
python nclone/test_environment.py --standalone-graph --visualize-reachability

# Interactive mode with reachability
python nclone/test_environment.py --interactive-graph --visualize-reachability
```

## Technical Implementation

### Core Components

1. **ReachabilityAnalyzer**: Performs physics-based reachability analysis
2. **SubgoalPlanner**: Identifies strategic waypoints and objectives
3. **FrontierDetector**: Finds exploration opportunities
4. **NppEnvironment Integration**: Renders overlays in real-time

### Performance Considerations

- **Caching**: Reachability analysis is cached until ninja position changes
- **Incremental updates**: Only recalculates when necessary
- **Efficient rendering**: Overlays use optimized drawing routines

### Data Structures

The system uses the following key data structures:

```python
# Reachability state
reachability_state = {
    'reachable_positions': Set[Tuple[int, int]],  # (sub_row, sub_col)
    'switch_states': Dict[int, bool],             # entity_id -> activated
    'unlocked_areas': Set[Tuple[int, int]],       # Areas unlocked by switches
    'subgoals': List[Tuple[int, int, str]]        # (sub_row, sub_col, goal_type)
}

# Subgoals
subgoals = [
    Subgoal(goal_type='key', position=(row, col), priority=100),
    Subgoal(goal_type='switch', position=(row, col), priority=80),
    Subgoal(goal_type='strategic_waypoint', position=(row, col), priority=50)
]

# Frontiers
frontiers = [
    Frontier(position=(row, col), exploration_value=0.8, frontier_type='unexplored'),
    Frontier(position=(row, col), exploration_value=0.6, frontier_type='partially_explored')
]
```

## Use Cases

### 1. Level Design Analysis
```bash
# Analyze level connectivity and identify potential issues
python nclone/test_environment.py --visualize-reachability --export-reachability level_analysis.png --map custom_level.npp --headless
```

### 2. AI Training Visualization
```bash
# Monitor reachability during RL training
python nclone/test_environment.py --visualize-reachability --reachability-from-ninja --show-frontiers
```

### 3. Navigation Planning
```bash
# Visualize optimal paths and waypoints
python nclone/test_environment.py --visualize-reachability --show-subgoals --reachability-from-ninja
```

### 4. Exploration Analysis
```bash
# Study exploration patterns and frontiers
python nclone/test_environment.py --show-frontiers --visualize-reachability
```

## Troubleshooting

### Common Issues

1. **No reachability overlay visible**
   - Ensure `--visualize-reachability` is specified
   - Press `T` to toggle overlay if it's disabled
   - Check that ninja position is valid

2. **Export fails**
   - Use `--headless` flag with `--export-reachability`
   - Ensure output directory is writable
   - Check file extension is supported (.png, .jpg)

3. **Performance issues**
   - Disable unnecessary overlays
   - Use smaller maps for testing
   - Reduce update frequency with manual controls

### Debug Information

When reachability visualization is active, the console shows:
- Number of reachable positions found
- Subgoals identified and their priorities
- Frontiers detected and exploration values
- Performance timing information

## Examples

### Basic Usage
```bash
# Simple reachability visualization
python nclone/test_environment.py --visualize-reachability

# With dynamic updates from ninja position
python nclone/test_environment.py --visualize-reachability --reachability-from-ninja
```

### Advanced Analysis
```bash
# Complete analysis with export
python nclone/test_environment.py \
    --visualize-reachability \
    --show-subgoals \
    --show-frontiers \
    --export-reachability complete_analysis.png \
    --headless
```

### Interactive Exploration
```bash
# Interactive mode with all features
python nclone/test_environment.py \
    --visualize-reachability \
    --reachability-from-ninja \
    --show-subgoals \
    --show-frontiers
```

## API Integration

For programmatic use, the reachability system can be accessed directly:

```python
from nclone.graph.reachability import ReachabilityAnalyzer
from nclone.graph.trajectory_calculator import TrajectoryCalculator
from nclone.graph.subgoal_planner import SubgoalPlanner

# Initialize components
trajectory_calc = TrajectoryCalculator()
analyzer = ReachabilityAnalyzer(trajectory_calc)
planner = SubgoalPlanner()

# Perform analysis
reachability_state = analyzer.analyze_reachability(level_data, start_row, start_col)
subgoals = planner.plan_subgoals(level_data, reachability_state)

# Use with environment
env.set_reachability_debug_enabled(True)
env.set_reachability_data(reachability_state, subgoals)
```

This comprehensive system provides powerful tools for understanding level connectivity, planning navigation strategies, and analyzing exploration opportunities in the N++ environment.