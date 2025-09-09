# N++ Consolidated Physics-Aware Pathfinding System

This document describes the authoritative pathfinding system for N++, which consolidates all working physics-aware logic into a single, coherent implementation.

## System Architecture

### Core Components

1. **`nclone/pathfinding/`** - Consolidated pathfinding module
   - `CorePathfinder` - Main pathfinding engine using physics-aware movement classification
   - `MovementType` - Enum defining all N++ movement types (WALK, JUMP, FALL, etc.)
   - `PhysicsValidator` - Validates movements against ninja physics constraints

2. **`nclone/visualization/`** - Consolidated visualization module
   - `PathfindingVisualizer` - Creates accurate visual representations of pathfinding results

3. **`nclone/graph/`** - Core graph and movement analysis (existing)
   - `MovementClassifier` - Physics-aware movement classification (the proven system)
   - `LevelData` - Level geometry and entity representation

## Usage

### Basic Pathfinding

```python
from nclone.pathfinding import CorePathfinder
from nclone.graph.level_data import LevelData

# Create pathfinder
pathfinder = CorePathfinder()

# Find path between two points
path_segments = pathfinder.find_path(level_data, start_pos, end_pos)

# Find path through multiple waypoints
waypoints = [ninja_pos, switch_pos, door_pos]
path_segments = pathfinder.find_multi_segment_path(level_data, waypoints)

# Get path summary
summary = pathfinder.get_path_summary(path_segments)
```

### Visualization

```python
from nclone.visualization import PathfindingVisualizer

# Create visualizer
visualizer = PathfindingVisualizer(tile_size=30)

# Visualize a single map
visualizer.visualize_map("simple-walk", level_data, "output.png")

# Create all test visualizations
visualizer.create_all_visualizations()
```

### Complete System Test

```bash
python consolidated_pathfinding_system.py
```

This script runs validation tests on all four test maps and creates visualizations.

## Test Maps and Expected Results

### 1. simple-walk
- **Layout**: 9 tiles wide, single horizontal platform
- **Expected Path**: 2 WALK segments (96px + 72px = 168px total)
- **Movement Types**: WALK only

### 2. long-walk  
- **Layout**: 42 tiles wide, single horizontal platform
- **Expected Path**: 2 WALK segments (936px + 24px = 960px total)
- **Movement Types**: WALK only

### 3. path-jump-required
- **Layout**: Elevated switch requiring jump navigation
- **Expected Path**: JUMP up (99px) + FALL down (76px = 175px total)
- **Movement Types**: JUMP and FALL

### 4. only-jump
- **Layout**: Vertical corridor requiring wall jumping
- **Expected Path**: 2 JUMP segments (48px each = 96px total)
- **Movement Types**: JUMP only

## Physics Validation

The system validates all movements against N++ physics constraints:

- **WALK**: Horizontal movement with <6px vertical change, up to 2000px distance
- **JUMP**: Trajectory-based movement respecting velocity and gravity limits
- **FALL**: Gravity-driven descent with reasonable distance limits
- **WALL_SLIDE**: Vertical movement along walls
- **Special movements**: WALL_JUMP, LAUNCH_PAD, BOUNCE_BLOCK, BOUNCE_CHAIN

## Movement Type Colors (Visualization)

- **WALK**: Green
- **JUMP**: Blue
- **FALL**: Red
- **WALL_SLIDE**: Orange
- **WALL_JUMP**: Cyan
- **LAUNCH_PAD**: Magenta
- **BOUNCE_BLOCK**: Yellow
- **BOUNCE_CHAIN**: Gray

## Files Removed During Consolidation

The following outdated/duplicate files were removed to eliminate confusion:

- `analyze_pathfinding_issue.py`
- `comprehensive_pathfinding_test.py`
- `consolidated_pathfinding_visualization.py`
- `consolidated_physics_pathfinder.py`
- `create_game_accurate_visualization.py`
- `create_pathfinding_visualizations.py`
- `create_physics_accurate_pathfinding.py`
- `create_proper_pathfinding_visualizations.py`
- `debug_pathfinding_analysis.py`
- `enhanced_pathfinding_test.py`
- `final_pathfinding_validation.py`
- `final_physics_accurate_pathfinding.py`
- `fixed_pathfinding_visualization.py`
- `improved_physics_pathfinding_visualization.py`
- `minimal_graph_analysis.py`
- `pathfinding_debug.py`
- `physics_accurate_pathfinding_validation.py`
- `physics_pathfinding_consolidation_summary.py`
- `physics_pathfinding_summary.py`
- `physics_pathfinding_test.py`
- `simple_pathfinding_test.py`
- `simple_visualization_test.py`
- `test_complex_pathfinding.py`
- `test_current_pathfinding.py`
- `test_physics_accurate_pathfinding.py`
- `test_physics_graph.py`
- `visual_pathfinding_validation.py`
- `accurate_doortest_visualization.py`
- Various `*_pathfinding.png` files

## Key Files Retained

- `test_pathfinding_validation.py` - Original validation test (reference)
- `consolidated_pathfinding_system.py` - Main system demonstration
- `simple_physics_validation.py` - Physics validation reference
- `*_consolidated.png` - Generated visualization outputs

## System Benefits

1. **Single Source of Truth**: One authoritative pathfinding implementation
2. **Physics Accuracy**: Uses the proven MovementClassifier system
3. **Complete Validation**: All test maps pass with correct movement types
4. **Clear Visualization**: Accurate visual representation with legend
5. **Maintainable**: Clean, documented, modular architecture
6. **Extensible**: Easy to add new movement types or validation rules

This consolidated system eliminates the confusion of multiple competing implementations and provides a single, reliable pathfinding solution for N++.