# Subgoal Visualization System Guide

## Overview

The subgoal visualization system provides real-time debug overlays and image export capabilities for analyzing AI pathfinding and goal-oriented behavior in the nclone environment. This system integrates with the existing hierarchical graph system and reachability analysis to provide comprehensive visualization of subgoal planning.

## Features

### Real-time Debug Overlay
- **Interactive visualization** during gameplay with toggle controls
- **Multiple visualization modes**: basic, detailed, and reachability-focused
- **Dynamic subgoal updates** based on current ninja position
- **Reachable area highlighting** with flood-fill analysis

### Image Export Capabilities
- **Static visualization export** for analysis and documentation
- **Headless mode support** for automated testing and batch processing
- **Customizable output paths** for organized result storage
- **Multiple level support** with automatic subgoal detection

### Integration with Existing Systems
- **Hierarchical graph system** integration for pathfinding visualization
- **Reachability analysis** using OpenCV flood-fill algorithms
- **Entity recognition** supporting both legacy object and dictionary formats
- **Consolidated LevelData approach** for improved data flow

## Usage

### Command Line Interface

#### Interactive Mode with Real-time Visualization
```bash
# Enable subgoal visualization overlay
python -m nclone.test_environment --visualize-subgoals

# Use specific level with subgoal visualization
python -m nclone.test_environment --map nclone/test_maps/complex-path-switch-required --visualize-subgoals

# Set visualization mode
python -m nclone.test_environment --visualize-subgoals --subgoal-mode detailed
```

#### Export Mode for Static Analysis
```bash
# Export subgoal visualization to image
python -m nclone.test_environment --export-subgoals output.png --headless

# Export with specific level
python -m nclone.test_environment --map nclone/test_maps/minefield --export-subgoals minefield_analysis.png --headless

# Export with specific visualization mode
python -m nclone.test_environment --export-subgoals analysis.png --subgoal-mode reachability --headless
```

### Runtime Controls (Interactive Mode)

| Key | Action |
|-----|--------|
| `S` | Toggle subgoal visualization overlay on/off |
| `M` | Cycle through visualization modes (basic → detailed → reachability) |
| `P` | Update subgoal plan from current ninja position |
| `O` | Export current subgoal visualization to screenshot |
| `R` | Reset environment and regenerate subgoals |

### Visualization Modes

#### Basic Mode
- Shows primary subgoals as colored markers
- Displays execution order with numbered indicators
- Minimal visual clutter for quick overview

#### Detailed Mode (Default)
- Comprehensive subgoal information with labels
- Reachable area highlighting
- Distance and priority indicators
- Entity relationship visualization

#### Reachability Mode
- Focus on reachable vs unreachable areas
- Flood-fill visualization results
- Connectivity analysis between objectives
- Pathfinding constraint visualization

## System Architecture

### Core Components

#### SubgoalPlanner Integration
```python
from nclone.graph.subgoal_planner import SubgoalPlanner
from nclone.graph.reachability.subgoal_integration import ReachabilitySubgoalIntegration

# Create integrated subgoal system
base_planner = SubgoalPlanner()
subgoal_system = ReachabilitySubgoalIntegration(base_planner)

# Generate completion plan
completion_plan = subgoal_system.subgoal_planner.create_hierarchical_completion_plan(
    ninja_position, level_data, entities
)
```

#### Visualization Rendering
```python
from nclone.graph.subgoal_visualizer import SubgoalVisualizer

# Initialize visualizer
visualizer = SubgoalVisualizer()

# Render overlay
visualizer.render_subgoal_overlay(
    surface, subgoals, ninja_position, 
    reachable_positions, mode="detailed"
)
```

### Data Flow Architecture

The system uses a consolidated LevelData approach that eliminates redundant parameters and improves data flow readability:

```python
from nclone.graph.level_data import LevelData, ensure_level_data

# Consolidated data structure
level_data = LevelData(
    tiles=tile_array,
    entities=entity_list,
    player=PlayerState(position=(x, y)),
    switch_states=switch_dict
)

# Backward compatibility support
consolidated_data = ensure_level_data(legacy_data, ninja_position, entities)
```

## Implementation Details

### Subgoal Detection Algorithm

The system uses a simplified completion strategy that prioritizes objectives based on reachability:

1. **Exit Switch Check**: If exit switch is reachable → primary objective
2. **Locked Door Switch**: If exit switch unreachable, find nearest reachable door switch
3. **Exit Door Check**: If switches unavailable, check direct exit door access
4. **Fallback Switch**: Find any reachable switch as fallback objective

### Reachability Analysis

Uses OpenCV-based flood-fill algorithm with ninja radius awareness:

```python
# Collision detection with 10-pixel ninja radius
morphology_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
ninja_aware_mask = cv2.morphologyEx(collision_mask, cv2.MORPH_ERODE, morphology_kernel)

# Flood-fill from ninja position
cv2.floodFill(reachable_mask, None, start_point, 255)
```

### Entity Recognition

Supports both legacy object format and current dictionary format:

```python
def _find_exit_switch(self, entities):
    """Find exit switch supporting both entity formats."""
    for entity in entities:
        # Dictionary format (current)
        if isinstance(entity, dict):
            if entity.get("type") == EntityType.EXIT_SWITCH:
                return entity
        # Object format (legacy)
        elif hasattr(entity, 'type'):
            if entity.type == EntityType.EXIT_SWITCH:
                return entity
    return None
```

## Testing and Validation

### Test Levels

#### complex-path-switch-required
- **Purpose**: Tests switch-dependent pathfinding
- **Features**: Multiple switches, locked doors, complex routing
- **Expected**: 1 subgoal (reach required switch)

#### minefield
- **Purpose**: Tests position-dependent reachability
- **Features**: Scattered enemies, multiple paths
- **Expected**: Variable subgoals based on ninja starting position

### Validation Commands

```bash
# Test subgoal generation
python -c "
from nclone.gym_environment.npp_environment import NppEnvironment
from nclone.graph.subgoal_planner import SubgoalPlanner
import pygame
pygame.init()

env = NppEnvironment(render_mode='rgb_array', custom_map_path='nclone/test_maps/complex-path-switch-required')
planner = SubgoalPlanner()
plan = planner.create_hierarchical_completion_plan((100, 100), env.level_data, env.entities)
print(f'Generated plan: {plan}')
"

# Test visualization export
python -m nclone.test_environment --map nclone/test_maps/complex-path-switch-required --export-subgoals test_output.png --headless
```

## Performance Considerations

### Optimization Features
- **Headless mode** for batch processing without rendering overhead
- **Cached reachability analysis** to avoid redundant flood-fill operations
- **Efficient entity filtering** using dictionary lookups
- **Minimal memory allocation** during real-time visualization

### Performance Metrics
- **Graph building**: ~1704 nodes, 6608 edges for 10x10 test level
- **Reachability analysis**: ~2.7ms for OpenCV flood-fill on typical level
- **Subgoal generation**: <10ms for complex levels with multiple objectives

## Troubleshooting

### Common Issues

#### No Subgoals Generated
- **Cause**: Ninja position in unreachable area or no valid objectives
- **Solution**: Check ninja starting position and level entity configuration
- **Debug**: Use `--subgoal-mode reachability` to visualize reachable areas

#### Incorrect Entity Recognition
- **Cause**: Entity format mismatch (object vs dictionary)
- **Solution**: System automatically handles both formats
- **Debug**: Check entity type constants match EntityType enum values

#### Visualization Not Appearing
- **Cause**: Overlay disabled or wrong visualization mode
- **Solution**: Press `S` to toggle overlay, `M` to cycle modes
- **Debug**: Check console output for initialization messages

### Debug Output

Enable debug mode for detailed analysis:

```python
# Enable debug output in SubgoalPlanner
planner = SubgoalPlanner(debug=True)

# Console output includes:
# - Ninja position validation
# - Entity count and types
# - Reachability analysis results
# - Subgoal generation decisions
```

## Future Enhancements

### Planned Features
- **Multi-objective visualization** for complex completion strategies
- **Path visualization** showing optimal routes to subgoals
- **Performance profiling** overlay for optimization analysis
- **Custom visualization themes** for different analysis needs

### Integration Opportunities
- **RL training visualization** for agent behavior analysis
- **Automated testing** with subgoal validation
- **Level design tools** with subgoal complexity metrics
- **Performance benchmarking** with visualization export

## API Reference

### Key Classes

#### SubgoalPlanner
```python
class SubgoalPlanner:
    def create_hierarchical_completion_plan(
        self, ninja_position, level_data, entities=None, 
        switch_states=None, reachability_analyzer=None
    ) -> Optional[SubgoalPlan]
```

#### SubgoalVisualizer
```python
class SubgoalVisualizer:
    def render_subgoal_overlay(
        self, surface, subgoals, ninja_position, 
        reachable_positions, mode="detailed"
    ) -> None
```

#### LevelData
```python
@dataclass
class LevelData:
    tiles: np.ndarray
    entities: List[Dict[str, Any]]
    player: Optional[PlayerState] = None
    switch_states: Dict[str, bool] = field(default_factory=dict)
```

### Utility Functions

#### ensure_level_data
```python
def ensure_level_data(
    data: Union[LevelData, Dict[str, Any], np.ndarray],
    player_position: Optional[Tuple[float, float]] = None,
    entities: Optional[List[Dict[str, Any]]] = None
) -> LevelData
```

This comprehensive system provides powerful tools for analyzing and debugging AI behavior in the nclone environment, supporting both real-time interactive analysis and automated batch processing workflows.