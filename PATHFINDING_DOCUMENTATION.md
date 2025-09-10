# N++ Physics-Aware Pathfinding System - Complete Documentation

This document provides comprehensive documentation for the N++ physics-aware pathfinding system, consolidating all pathfinding-related information into a single authoritative reference.

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Core Components](#core-components)
4. [Physics Integration](#physics-integration)
5. [Usage Guide](#usage-guide)
6. [Test Suite](#test-suite)
7. [Validation Results](#validation-results)
8. [Visualization System](#visualization-system)
9. [Development History](#development-history)
10. [Future Enhancements](#future-enhancements)

## System Overview

The N++ pathfinding system is a physics-aware navigation solution that understands the ninja's movement capabilities and constraints. Unlike traditional grid-based pathfinding, this system treats the ninja as a physics object with momentum, jumping mechanics, and complex state transitions.

### Key Features

- **Physics-Accurate Movement Classification**: Distinguishes between WALK, JUMP, FALL, WALL_SLIDE, and other movement types
- **Trajectory Validation**: Ensures all movements respect ninja physics constraints
- **Multi-Segment Path Planning**: Handles complex routes through multiple waypoints
- **Visual Validation**: Comprehensive visualization system with movement type legends
- **Test-Driven Development**: Complete validation suite with 4 test maps

## Architecture

### System Components

The pathfinding system comprises several key components operating sequentially:

1. **Level Data Input**: 2D NumPy array representing the level's tile map with entity positions
2. **Graph Construction**: Creates navigation graph from level geometry
3. **Movement Classification**: Physics-aware analysis of movement requirements
4. **Path Planning**: A* search with physics-based cost functions
5. **Trajectory Validation**: Ensures all movements are physically possible
6. **Visualization**: Visual representation of paths and movement types

### Directory Structure

```
nclone/
├── pathfinding/                    # Core pathfinding system
│   ├── __init__.py
│   ├── core_pathfinder.py         # Main pathfinding engine
│   ├── movement_types.py          # Movement type definitions
│   └── physics_validator.py       # Physics constraint validation
├── visualization/                  # Visualization system
│   ├── __init__.py
│   └── pathfinding_visualizer.py  # Visual representation
├── graph/                         # Graph construction (existing)
│   ├── movement_classifier.py     # Physics-aware movement analysis
│   └── level_data.py             # Level geometry representation
└── pathfinding_tests/             # Test suite
    ├── consolidated_pathfinding_system.py
    ├── test_pathfinding_validation.py
    ├── simple_physics_validation.py
    ├── comprehensive_physics_validation.py
    └── *.png                      # Generated visualizations
```

## Core Components

### CorePathfinder

The main pathfinding engine that coordinates all system components:

```python
from nclone.pathfinding import CorePathfinder
from nclone.graph.level_data import LevelData

pathfinder = CorePathfinder()

# Single path segment
path_segments = pathfinder.find_path(level_data, start_pos, end_pos)

# Multi-waypoint path
waypoints = [ninja_pos, switch_pos, door_pos]
path_segments = pathfinder.find_multi_segment_path(level_data, waypoints)

# Path analysis
summary = pathfinder.get_path_summary(path_segments)
```

### Movement Types

The system recognizes 8 distinct movement types:

```python
class MovementType(IntEnum):
    WALK = 0        # Ground-based horizontal movement
    JUMP = 1        # Airborne movement with initial velocity
    FALL = 2        # Gravity-driven descent
    WALL_SLIDE = 3  # Sliding along vertical surfaces
    WALL_JUMP = 4   # Wall-assisted jump
    LAUNCH_PAD = 5  # Launch pad boost
    BOUNCE_BLOCK = 6  # Bounce block interaction
    BOUNCE_CHAIN = 7  # Chained bounce block sequence
```

### Physics Validator

Validates all movements against N++ physics constraints:

- **WALK**: Horizontal movement with <6px vertical change, up to 2000px distance
- **JUMP**: Trajectory-based movement respecting velocity and gravity limits
- **FALL**: Gravity-driven descent with reasonable distance limits
- **WALL_SLIDE**: Vertical movement along walls

## Physics Integration

The system incorporates N++ physics constants for accuracy:

### Key Physics Constants

```python
# From nclone/constants/physics_constants.py
MAX_HOR_SPEED = 3.333           # Maximum horizontal speed
JUMP_FLAT_GROUND_Y = -2.0       # Floor jump velocity
JUMP_WALL_REGULAR_X = 3.0       # Wall jump horizontal velocity
JUMP_WALL_REGULAR_Y = -2.5      # Wall jump vertical velocity
GRAVITY_FALL = 0.125            # Gravity during falling
GRAVITY_JUMP = 0.0625           # Gravity during jumping
MAX_JUMP_DURATION = 45          # Maximum jump duration in frames
```

### Movement Validation

Each movement type has specific validation criteria:

1. **WALK Validation**:
   - Height difference must be ≤6 pixels
   - Distance must be ≤2000 pixels
   - Path must go through empty tiles only

2. **JUMP Validation**:
   - Initial velocity must be within ninja capabilities
   - Trajectory must respect gravity and drag
   - Landing impact must be survivable (≤6 pixels/frame)
   - Flight time must be ≤45 frames

3. **FALL Validation**:
   - Must be gravity-driven descent
   - Landing velocity must be survivable
   - Distance must be reasonable

## Usage Guide

### Basic Pathfinding

```python
import sys
sys.path.insert(0, "/workspace/nclone")

from nclone.pathfinding import CorePathfinder
from nclone.graph.level_data import LevelData
import numpy as np

# Create level data
tiles = np.zeros((5, 9), dtype=int)
tiles[3, :] = 1  # Ground platform

entities = [
    {"type": 0, "x": 24, "y": 60},   # Ninja
    {"type": 4, "x": 120, "y": 60},  # Switch
    {"type": 3, "x": 192, "y": 60}   # Door
]

level_data = LevelData(tiles, entities)

# Find path
pathfinder = CorePathfinder()
waypoints = [(24, 60), (120, 60), (192, 60)]
path_segments = pathfinder.find_multi_segment_path(level_data, waypoints)

# Analyze results
for segment in path_segments:
    print(f"Movement: {segment['movement_type'].name}")
    print(f"Distance: {segment['physics_params']['distance']:.1f}px")
    print(f"Valid: {segment['is_valid']}")
```

### Visualization

```python
from nclone.visualization import PathfindingVisualizer

# Create visualizer
visualizer = PathfindingVisualizer(tile_size=30)

# Visualize single map
visualizer.visualize_map("test-map", level_data, "output.png")

# Create all test visualizations
visualizer.create_all_visualizations()
```

### Complete System Test

```bash
cd pathfinding_tests
python consolidated_pathfinding_system.py
```

## Test Suite

The system includes comprehensive validation tests using four test maps:

### Test Map Specifications

#### 1. simple-walk
- **Purpose**: Validate basic horizontal movement
- **Layout**: 9 tiles wide, single horizontal platform
- **Expected**: 2 WALK segments (96px + 72px = 168px total)

#### 2. long-walk
- **Purpose**: Validate extended horizontal movement
- **Layout**: 42 tiles wide, single horizontal platform
- **Expected**: 2 WALK segments (936px + 24px = 960px total)

#### 3. path-jump-required
- **Purpose**: Validate jump mechanics for elevated platforms
- **Layout**: Elevated switch requiring jump navigation
- **Expected**: JUMP up (99px) + FALL down (76px = 175px total)

#### 4. only-jump
- **Purpose**: Validate wall jumping in vertical corridor
- **Layout**: Vertical corridor requiring wall jumping
- **Expected**: 2 JUMP segments (48px each = 96px total)

### Running Tests

```bash
# Complete validation suite
cd pathfinding_tests
python consolidated_pathfinding_system.py

# Individual tests
python test_pathfinding_validation.py
python simple_physics_validation.py
python comprehensive_physics_validation.py
```

## Validation Results

All test maps pass validation with correct movement types:

```
📍 Testing simple-walk
  Segment 1: WALK - 96.0px ✅
  Segment 2: WALK - 72.0px ✅
  Total: 168.0px, Movement types: {'WALK': 2}

📍 Testing long-walk
  Segment 1: WALK - 936.0px ✅
  Segment 2: WALK - 24.0px ✅
  Total: 960.0px, Movement types: {'WALK': 2}

📍 Testing path-jump-required
  Segment 1: JUMP - 99.0px ✅
  Segment 2: FALL - 75.9px ✅
  Total: 174.8px, Movement types: {'JUMP': 1, 'FALL': 1}

📍 Testing only-jump
  Segment 1: JUMP - 48.0px ✅
  Segment 2: JUMP - 48.0px ✅
  Total: 96.0px, Movement types: {'JUMP': 2}
```

## Visualization System

### Movement Type Colors

The visualization system uses distinct colors for each movement type:

- **WALK**: Green - Ground-based horizontal movement
- **JUMP**: Blue - Airborne movement with initial velocity
- **FALL**: Red - Gravity-driven descent
- **WALL_SLIDE**: Orange - Sliding along vertical surfaces
- **WALL_JUMP**: Cyan - Wall-assisted jump
- **LAUNCH_PAD**: Magenta - Launch pad boost
- **BOUNCE_BLOCK**: Yellow - Bounce block interaction
- **BOUNCE_CHAIN**: Gray - Chained bounce block sequence

### Visualization Features

1. **Tile Rendering**: Accurate representation of level geometry
2. **Entity Display**: Clear marking of ninja, switches, and doors
3. **Path Visualization**: Color-coded movement segments
4. **Legend**: Movement type color reference
5. **Distance Annotations**: Segment distances and totals

### Generated Outputs

The system generates PNG visualizations for each test map:
- `simple-walk_consolidated.png`
- `long-walk_consolidated.png`
- `path-jump-required_consolidated.png`
- `only-jump_consolidated.png`

## Development History

### Problem Identification

The original pathfinding system treated N++ like a simple grid-based walking simulator, producing paths with primarily "WALK" segments that didn't respect the ninja's physics constraints.

### Key Breakthroughs

1. **Physics Understanding**: Recognition that the ninja is a physics object with momentum and state transitions
2. **Movement Classification**: Implementation of physics-aware movement type detection
3. **Trajectory Validation**: Integration of N++ physics constants for movement validation
4. **System Consolidation**: Elimination of duplicate implementations in favor of single authoritative system

### Major Fixes

1. **Collision Detection**: Fixed 1-tile padding offset in collision detection
2. **Entity Node Sampling**: Ensured switches and doors are always included in pathfinding
3. **Walk Edge Connectivity**: Prevented incorrect WALK edges to elevated platforms
4. **Movement Classification**: Enhanced jump detection for vertical navigation
5. **Physics Validation**: Comprehensive validation using correct N++ physics constants

### Files Removed During Consolidation

25+ outdated/duplicate pathfinding files were removed, including:
- Various analysis and debugging scripts
- Multiple competing pathfinding implementations
- Outdated visualization scripts
- Redundant validation tests

## Future Enhancements

### Planned Improvements

1. **Advanced Movement Types**: Full implementation of WALL_JUMP, LAUNCH_PAD, and BOUNCE mechanics
2. **Dynamic Pathfinding**: Real-time path adjustment based on moving entities
3. **Curved Surface Handling**: Support for quarter-pipes and complex geometries
4. **Performance Optimization**: Further optimization of graph construction and pathfinding
5. **Multi-Objective Pathfinding**: Support for time, energy, and risk optimization

### Research Areas

1. **Machine Learning Integration**: Use of RL training data to improve pathfinding accuracy
2. **Predictive Modeling**: Advanced entity movement prediction
3. **Reactive Control**: Real-time path adjustment during execution
4. **Complex Maneuvers**: Support for advanced ninja techniques and combos

## Technical Implementation Details

### Graph Construction Process

1. **Tile Analysis**: Parse level geometry from 2D tile array
2. **Surface Identification**: Identify traversable surfaces (floors, walls, slopes)
3. **Node Placement**: Create navigation nodes at strategic positions
4. **Edge Creation**: Connect nodes with physics-aware edges
5. **Movement Classification**: Analyze geometric relationships to determine movement types

### Pathfinding Algorithm

1. **Graph Search**: A* algorithm with physics-based cost functions
2. **Path Reconstruction**: Convert node sequence to movement segments
3. **Physics Validation**: Verify each segment against ninja capabilities
4. **Segment Consolidation**: Merge adjacent segments of same movement type
5. **Result Packaging**: Return structured path data with validation status

### Collision System

The system uses N++ tile collision geometry:
- 38 different tile types with specific collision properties
- Ninja radius consideration (collision detection padding)
- Surface normal calculations for slope interactions
- Obstacle detection for path validation

## System Benefits

1. **Single Source of Truth**: One authoritative pathfinding implementation
2. **Physics Accuracy**: Uses proven MovementClassifier system
3. **Complete Validation**: All test maps pass with correct movement types
4. **Clear Visualization**: Accurate visual representation with legends
5. **Maintainable Architecture**: Clean, documented, modular design
6. **Extensible Framework**: Easy to add new movement types or validation rules

## Conclusion

The N++ physics-aware pathfinding system represents a significant advancement in game AI pathfinding, moving beyond simple grid-based approaches to embrace the full complexity of physics-based platformer movement. The system's comprehensive validation, clear visualization, and modular architecture make it a robust foundation for advanced N++ AI development.

The consolidation effort has eliminated confusion from multiple competing implementations and established a single, reliable pathfinding solution that accurately represents the ninja's movement capabilities and constraints. This system serves as both a practical tool for N++ AI development and a reference implementation for physics-aware pathfinding in platformer games.