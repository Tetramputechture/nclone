# Subgoal Visualization System

The subgoal visualization system provides real-time debug overlays and image export capabilities for analyzing AI pathfinding and subgoal planning in the nclone environment.

## Features

### Real-time Debug Overlay
- **Live visualization** of subgoals, reachability analysis, and pathfinding data
- **Interactive controls** for toggling visualization modes during runtime
- **Multi-mode display** with basic, detailed, and reachability-focused views

### Image Export System
- **PNG export** of subgoal visualizations with actual level tiles and entities
- **Coordinate system compatibility** with game rendering (Y=0 at top-left, positive Y downward)
- **Comprehensive rendering** including walls, entities, reachable areas, and subgoal markers

### Multi-Switch Logic Support
- **Complex dependency handling** for levels requiring multiple switches to be activated in sequence
- **Locked door state management** with proper collision detection based on switch states
- **Hierarchical completion planning** that identifies optimal switch activation sequences

## Usage

### Command Line Interface

#### Basic Visualization
```bash
# Enable subgoal overlay during interactive play
python -m nclone.test_environment --visualize-subgoals

# Use specific level
python -m nclone.test_environment --map nclone/test_maps/simple-walk --visualize-subgoals
```

#### Export Functionality
```bash
# Export subgoal visualization to PNG
python -m nclone.test_environment --map nclone/test_maps/complex-path-switch-required --visualize-subgoals --export-subgoals /tmp/subgoal_analysis.png

# Different visualization modes
python -m nclone.test_environment --visualize-subgoals --subgoal-mode detailed --export-subgoals output.png
```

### Runtime Controls

When the visualization overlay is active, use these keyboard controls:

- **S** - Toggle subgoal visualization overlay on/off
- **M** - Cycle through visualization modes (basic → detailed → reachability)
- **P** - Update subgoal plan from current ninja position
- **O** - Export current subgoal visualization to screenshot
- **R** - Reset environment and regenerate subgoal plan

### Visualization Modes

#### Basic Mode
- Shows essential subgoals and reachable areas
- Minimal visual clutter for quick analysis
- Suitable for real-time gameplay debugging

#### Detailed Mode (Default)
- Complete subgoal information with labels
- Switch sequences and dependencies
- Reachability boundaries and pathfinding data
- Recommended for comprehensive analysis

#### Reachability Mode
- Focus on reachability analysis results
- Detailed collision detection visualization
- Switch state dependencies and door logic
- Ideal for debugging pathfinding issues

## Technical Implementation

### Core Components

#### SubgoalVisualizationSystem
- **Location**: `nclone/visualization/subgoal_visualization.py`
- **Purpose**: Main visualization controller and rendering coordinator
- **Key Methods**:
  - `render_overlay()` - Real-time overlay rendering
  - `export_visualization()` - PNG export with level content
  - `update_subgoal_plan()` - Refresh subgoal analysis

#### SubgoalPlanner
- **Location**: `nclone/graph/subgoal_planner.py`
- **Purpose**: Multi-switch logic and hierarchical completion planning
- **Key Features**:
  - `create_hierarchical_completion_plan()` - Multi-switch sequence planning
  - `find_switch_sequence()` - Optimal switch activation order
  - Switch dependency analysis and validation

#### Reachability System Integration
- **OpenCV Flood Fill**: Fast reachability analysis with entity collision detection
- **Locked Door Logic**: Proper state handling based on switch activation
- **Tiered Analysis**: Multi-resolution reachability computation for performance

### Coordinate System

The visualization system uses a **top-left origin coordinate system**:
- **Y=0** at the top of the screen
- **Positive Y** values increase downward
- **Positive X** values increase rightward
- **Compatible** with game rendering and Pygame conventions

### Multi-Switch Logic

#### Switch Dependency Resolution
```python
# Example: Level requires switch1 → switch2 → exit
sequence = planner.find_switch_sequence(ninja_pos, exit_pos, switch_states)
# Returns: [('switch1', position1), ('switch2', position2)]
```

#### Locked Door State Handling
- **Closed doors** (switch OFF) block movement and are rendered as solid
- **Open doors** (switch ON) allow passage and are rendered as passable
- **State validation** ensures collision detection matches switch states
- **Multi-door support** for complex levels with multiple locked doors

## Debugging and Troubleshooting

### Common Issues

#### No Subgoals Generated
- **Cause**: Exit switch directly reachable or no valid completion path
- **Solution**: Check level design and switch dependencies
- **Debug**: Use detailed mode to examine reachability analysis

#### Incorrect Door States
- **Cause**: Switch state not properly passed to reachability system
- **Solution**: Verify switch entity IDs match controlled_by attributes
- **Debug**: Enable debug output in SubgoalPlanner

#### Export Images Missing Content
- **Cause**: Level data not properly loaded or entities missing
- **Solution**: Verify map file path and entity definitions
- **Debug**: Check console output for loading errors

### Debug Output

Enable debug output for detailed analysis:
```python
planner = SubgoalPlanner(debug=True)
# Outputs switch sequences, reachability checks, and door state changes
```

### Performance Considerations

- **Tier Selection**: System automatically selects optimal reachability analysis tier
- **Caching**: Reachability results cached for repeated queries
- **Export Optimization**: PNG export uses efficient rendering pipeline
- **Memory Usage**: Large levels may require tier adjustment for performance

## Integration Examples

### Custom Level Analysis
```python
from nclone.visualization.subgoal_visualization import SubgoalVisualizationSystem
from nclone.graph.subgoal_planner import SubgoalPlanner

# Initialize systems
viz_system = SubgoalVisualizationSystem()
planner = SubgoalPlanner()

# Analyze custom level
level_data = load_custom_level("my_level.json")
plan = planner.create_hierarchical_completion_plan(ninja_pos, level_data)

# Export visualization
viz_system.export_visualization("analysis.png", level_data, plan)
```

### Real-time Monitoring
```python
# In game loop
if subgoal_overlay_enabled:
    viz_system.render_overlay(screen, current_level_data, ninja_position)
    
# Update on ninja movement
if ninja_moved:
    viz_system.update_subgoal_plan(ninja_position, current_level_data)
```

## File Structure

```
nclone/
├── visualization/
│   ├── subgoal_visualization.py      # Main visualization system
│   └── subgoal_renderer.py           # Rendering utilities
├── graph/
│   ├── subgoal_planner.py            # Multi-switch logic and planning
│   └── reachability/
│       ├── tiered_system.py          # Reachability analysis coordinator
│       └── opencv_flood_fill.py      # Fast collision detection with door logic
└── test_environment.py               # CLI interface with visualization options
```

## Future Enhancements

- **Interactive editing** of subgoal plans during runtime
- **Performance profiling** overlay for reachability analysis timing
- **Path visualization** showing actual movement trajectories
- **Multi-agent support** for visualizing multiple ninja paths simultaneously
- **Level validation** tools for detecting unreachable areas or broken switch logic