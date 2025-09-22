# Subgoal Visualization System Usage Guide

## Overview

The subgoal visualization system provides a debug overlay for analyzing AI pathfinding and strategic waypoints in the N++ game environment. It can be used both as a real-time overlay during gameplay and for exporting static analysis images.

## Features

- **Real-time Debug Overlay**: View subgoals, reachable positions, and execution plans during gameplay
- **Export Functionality**: Generate static images of subgoal analysis for documentation and debugging
- **Multiple Visualization Modes**: Basic, detailed, and reachability-focused views
- **Custom Map Support**: Works with both default levels and custom map files
- **Interactive Controls**: Toggle overlays and update plans during runtime

## Command Line Usage

### Basic Export Command

Export a subgoal visualization image from the default level:

```bash
python -m nclone.test_environment --export-subgoals analysis.png
```

### Export with Custom Map

Export subgoal visualization from a specific map file:

```bash
python -m nclone.test_environment --export-subgoals custom_analysis.png --map nclone/test_maps/simple-walk
```

### Interactive Mode with Subgoal Overlay

Run the test environment with real-time subgoal visualization:

```bash
python -m nclone.test_environment --visualize-subgoals
```

### Advanced Options

Combine multiple visualization features:

```bash
python -m nclone.test_environment --visualize-subgoals --subgoal-mode detailed --visualize-reachability
```

## Available Command Line Arguments

### Subgoal-Specific Arguments

- `--visualize-subgoals`: Enable subgoal visualization overlay during gameplay
- `--subgoal-mode {basic,detailed,reachability}`: Set visualization detail level
- `--export-subgoals FILENAME`: Export subgoal visualization to image file and quit

### Map Selection

- `--map PATH`: Load a custom map file (supports binary map format)

### Additional Visualization Options

- `--visualize-reachability`: Show reachability analysis overlay
- `--reachability-from-ninja`: Display reachability from current ninja position
- `--show-subgoals`: Visualize identified subgoals and strategic waypoints
- `--show-frontiers`: Show exploration frontiers for curiosity-driven RL

## Interactive Runtime Controls

When running with `--visualize-subgoals`, the following keyboard controls are available:

- **S**: Toggle subgoal visualization overlay on/off
- **M**: Cycle through visualization modes (basic → detailed → reachability)
- **P**: Update subgoal plan from current ninja position
- **O**: Export current subgoal visualization to screenshot
- **R**: Reset environment to initial state

## Visualization Modes

### Basic Mode
- Shows identified subgoals as colored markers
- Displays basic connectivity between waypoints
- Minimal visual clutter for overview analysis

### Detailed Mode (Default)
- Enhanced subgoal markers with labels
- Execution order visualization
- Reachable position highlighting
- Strategic path indicators

### Reachability Mode
- Focus on reachability analysis
- Detailed connectivity visualization
- Exploration frontier display
- Advanced pathfinding insights

## Output Information

When exporting subgoal visualizations, the system provides detailed feedback:

```
✅ Subgoal visualization exported to analysis.png
   - Subgoals identified: 3
   - Execution order: 5 steps
   - Reachable positions: 127
```

### Status Messages

- **✅ Success**: Export completed successfully
- **⚠️ Warning**: Export completed but with limitations (e.g., no subgoal plan created)
- **❌ Error**: Export failed due to technical issues

## Technical Details

### System Requirements

- Pygame display support (not available in headless mode)
- PIL/Pillow for image export functionality
- Sufficient memory for level analysis and visualization rendering

### File Formats

- **Export Format**: PNG images with RGBA support
- **Map Format**: Binary map files or custom text-based formats
- **Default Resolution**: 1056x600 pixels (adjusts based on level dimensions)

### Performance Considerations

- Subgoal analysis may take several seconds for complex levels
- Export functionality requires full environment initialization
- Memory usage scales with level complexity and reachability analysis depth

## Troubleshooting

### Common Issues

1. **"Debug overlay renderer not available"**
   - Ensure not running in headless mode
   - Verify pygame display initialization succeeded

2. **"No subgoal completion plan created"**
   - Normal for simple levels or when ninja position is unclear
   - Export will still generate basic visualization

3. **"Subgoal planner not initialized"**
   - Check that visualization flags are properly set
   - Verify environment initialization completed successfully

### Debug Information

For detailed troubleshooting, the system provides comprehensive status information:

```
Initializing reachability analysis system...
✅ Reachability analysis system initialized successfully
Initializing subgoal visualization system...
✅ Subgoal visualization system initialized successfully
   - Mode: detailed
   - Overlay enabled: False
```

## Integration with Development Workflow

### Testing and Validation

Use subgoal visualization to:
- Validate AI pathfinding algorithms
- Debug strategic waypoint identification
- Analyze level complexity and reachability
- Document AI behavior for research purposes

### Research Applications

The export functionality is particularly useful for:
- Academic papers and presentations
- Algorithm comparison studies
- Level design analysis
- AI behavior documentation

## Examples

### Basic Usage Example

```bash
# Export analysis of default level
python -m nclone.test_environment --export-subgoals default_level.png

# Export analysis of specific test map
python -m nclone.test_environment --export-subgoals test_map.png --map nclone/test_maps/complex-path-switch-required
```

### Interactive Analysis Example

```bash
# Start interactive session with full visualization
python -m nclone.test_environment --visualize-subgoals --visualize-reachability --subgoal-mode detailed
```

### Batch Analysis Example

```bash
# Export multiple maps for comparison
for map in nclone/test_maps/*; do
    python -m nclone.test_environment --export-subgoals "analysis_$(basename $map).png" --map "$map"
done
```

## Related Documentation

- [Subgoal Visualization Technical Guide](subgoal_visualization.md)
- [Graph System Architecture](FILE_INDEX.md)
- [Reachability Analysis Implementation](TIERED_REACHABILITY_IMPLEMENTATION.md)
- [Test Environment Documentation](../nclone/test_environment.py)

## Version History

- **v1.0**: Initial implementation with basic export functionality
- **v1.1**: Added interactive runtime controls and multiple visualization modes
- **v1.2**: Enhanced custom map support and improved error handling
- **v1.3**: Fixed compatibility issues and improved debug renderer access