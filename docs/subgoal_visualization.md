# Subgoal Visualization System

The subgoal visualization system provides real-time debug overlays and export functionality for analyzing subgoal planning and reachability in the N++ environment. This system is designed to help with Deep Reinforcement Learning research by visualizing the hierarchical subgoal structure that neural networks can use for navigation.

## Overview

The subgoal visualization system consists of:

- **SubgoalVisualizer**: Core visualization engine with multiple rendering modes
- **Debug Overlay Integration**: Real-time overlay rendering during gameplay
- **Export Functionality**: Static image generation for analysis and documentation
- **Interactive Controls**: Runtime keyboard controls for toggling and configuration

## Key Features

### Visualization Modes

1. **Basic Mode** (`basic`)
   - Simple circular markers for subgoals
   - Minimal visual clutter
   - Best for performance-critical scenarios

2. **Detailed Mode** (`detailed`) - *Default*
   - Subgoal markers with labels and type information
   - Simple line connections between subgoals in execution order
   - Ninja position highlighting
   - Comprehensive information display

3. **Reachability Mode** (`reachability`)
   - All detailed mode features
   - Reachability area highlighting
   - Visual distinction between reachable and unreachable subgoals
   - Integrated with reachability analysis system

### Visual Elements

- **Subgoal Markers**: Color-coded circles indicating different subgoal types
  - Gold Collection subgoals: Gold/yellow markers
  - Exit subgoals: Green markers
  - Switch subgoals: Blue markers
  - Other types: Default purple markers

- **Connections**: Simple lines connecting subgoals in execution order
  - No complex pathfinding visualization (neural network handles pathfinding)
  - Clean, direct connections showing subgoal sequence

- **Ninja Position**: Special marker showing current player position

- **Reachability Areas**: Semi-transparent overlays showing accessible regions

## Usage

### Command Line Arguments

```bash
# Enable subgoal visualization overlay
python -m nclone.test_environment --visualize-subgoals

# Set visualization mode
python -m nclone.test_environment --visualize-subgoals --subgoal-mode detailed

# Export subgoal visualization to image
python -m nclone.test_environment --export-subgoals output.png

# Combine with reachability analysis
python -m nclone.test_environment --visualize-subgoals --visualize-reachability --subgoal-mode reachability
```

### Runtime Controls

When subgoal visualization is enabled, the following keyboard controls are available:

- **S**: Toggle subgoal visualization overlay on/off
- **M**: Cycle through visualization modes (basic → detailed → reachability)
- **P**: Update subgoal plan from current ninja position
- **O**: Export subgoal visualization screenshot (timestamped filename)
- **R**: Reset environment

### Integration with Custom Maps

The system works with custom maps via the `--custom-map-path` argument:

```bash
python -m nclone.test_environment --custom-map-path path/to/map.json --visualize-subgoals --export-subgoals map_analysis.png
```

## API Reference

### SubgoalVisualizer Class

```python
from nclone.graph.subgoal_visualizer import SubgoalVisualizer, SubgoalVisualizationConfig

# Create visualizer with default configuration
visualizer = SubgoalVisualizer()

# Create with custom configuration
config = SubgoalVisualizationConfig(
    subgoal_radius=8,
    ninja_radius=12,
    connection_width=3,
    show_labels=True,
    show_reachability=True
)
visualizer = SubgoalVisualizer(config)
```

### Configuration Options

```python
@dataclass
class SubgoalVisualizationConfig:
    # Visual appearance
    subgoal_radius: int = 6
    ninja_radius: int = 10
    connection_width: int = 2
    
    # Colors (RGBA tuples)
    subgoal_color: Tuple[int, int, int, int] = (255, 100, 255, 200)
    ninja_color: Tuple[int, int, int, int] = (255, 255, 0, 255)
    connection_color: Tuple[int, int, int, int] = (255, 255, 255, 150)
    reachability_color: Tuple[int, int, int, int] = (0, 255, 0, 50)
    
    # Display options
    show_labels: bool = True
    show_connections: bool = True
    show_reachability: bool = False
    animate_subgoals: bool = True
```

### Debug Overlay Integration

```python
# Enable subgoal visualization in debug overlay
debug_overlay.set_subgoal_debug_enabled(True)

# Set visualization mode
debug_overlay.set_subgoal_visualization_mode("detailed")

# Update subgoal data
debug_overlay.set_subgoal_data(subgoals, plan, reachable_positions)

# Export visualization
success = debug_overlay.export_subgoal_visualization("output.png")
```

## Implementation Details

### Design Philosophy

The subgoal visualization system is designed with the following principles:

1. **Neural Network Focus**: The environment provides subgoal structure and reachability information, but does not attempt complex pathfinding visualization. The neural network is responsible for finding optimal paths between subgoals.

2. **Performance Oriented**: Multiple visualization modes allow balancing between information density and performance.

3. **Research Friendly**: Export functionality enables analysis and documentation of subgoal structures for research purposes.

4. **Real-time Debugging**: Interactive controls allow researchers to explore subgoal behavior during live gameplay.

### Technical Architecture

- **Modular Design**: Separate visualization engine that can be integrated into different rendering systems
- **Configuration Driven**: Extensive configuration options for customizing appearance
- **Memory Efficient**: Minimal state storage with on-demand rendering
- **Thread Safe**: Safe for use in multi-threaded environments

### Integration Points

The system integrates with:

- **Reachability Analysis System**: For highlighting accessible areas
- **Subgoal Planning System**: For obtaining subgoal hierarchies and execution plans
- **Debug Overlay Renderer**: For real-time visualization during gameplay
- **Test Environment**: For interactive controls and command-line configuration

## Examples

### Basic Usage

```python
# Create and configure visualizer
visualizer = SubgoalVisualizer()
visualizer.set_mode(SubgoalVisualizationMode.DETAILED)

# Render overlay
surface = visualizer.render_subgoals_overlay(
    surface=game_surface,
    subgoals=current_subgoals,
    ninja_position=(ninja_x, ninja_y),
    reachable_positions=reachable_set,
    current_plan=subgoal_plan,
    tile_x_offset=offset_x,
    tile_y_offset=offset_y,
    adjust=scale_factor
)
```

### Export Functionality

```python
# Export current visualization
success = visualizer.export_subgoal_visualization(
    subgoals=subgoals,
    ninja_position=(x, y),
    level_dimensions=(width, height),
    reachable_positions=reachable_set,
    current_plan=plan,
    filename="analysis.png"
)
```

## Troubleshooting

### Common Issues

1. **No Subgoals Visible**: Ensure subgoal planner is initialized and has generated subgoals
2. **Missing Reachability**: Enable reachability analysis system for reachability mode
3. **Performance Issues**: Switch to basic mode for better performance
4. **Export Failures**: Check file permissions and disk space

### Debug Information

Enable debug logging to see detailed information about subgoal visualization:

```python
import logging
logging.getLogger('nclone.graph.subgoal_visualizer').setLevel(logging.DEBUG)
```

## Future Enhancements

Potential improvements for future versions:

- **Animation System**: Smooth transitions between subgoal states
- **Filtering Options**: Show/hide specific subgoal types
- **Performance Metrics**: Built-in performance monitoring
- **Custom Themes**: Predefined color schemes for different use cases
- **3D Visualization**: Height-based visualization for complex levels

## Contributing

When contributing to the subgoal visualization system:

1. Maintain the separation between visualization and pathfinding logic
2. Ensure new features work across all visualization modes
3. Add appropriate configuration options for customization
4. Include export functionality for new visual elements
5. Update documentation with new features and API changes

## License

This subgoal visualization system is part of the nclone project and follows the same licensing terms.