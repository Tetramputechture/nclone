# N++ Pathfinding Test Suite

This directory contains the consolidated test suite for the N++ physics-aware pathfinding system.

## Current Test Files

### Main Test Script
- **`test_actual_map_files.py`** - Comprehensive test script that loads actual binary test map files from `nclone/test_maps/` directory using the proper map loading chain (NPlayHeadless → Simulator → MapLoader)

### Generated Visualizations (Latest)
- **`corrected_simple-walk_pathfinding.png`** - Simple horizontal walking test with proper entity positioning
- **`corrected_long-walk_pathfinding.png`** - Extended horizontal walking test with proper entity positioning
- **`corrected_path-jump-required_pathfinding.png`** - Jump mechanics test with proper entity positioning
- **`corrected_only-jump_pathfinding.png`** - Vertical wall jumping test with proper entity positioning

## Test Map Specifications

The test suite validates pathfinding using four actual binary test maps from `nclone/test_maps/`:

### 1. simple-walk
- **Purpose**: Basic horizontal movement validation
- **Expected**: WALK segments, ~192px total distance
- **Result**: ✅ 2 WALK segments, 192.0px distance

### 2. long-walk  
- **Purpose**: Extended horizontal movement validation
- **Expected**: WALK segments, ~984px total distance
- **Result**: ✅ 2 WALK segments, 984.0px distance

### 3. path-jump-required
- **Purpose**: Jump mechanics validation for elevated platforms
- **Expected**: JUMP segments for vertical navigation
- **Result**: ✅ 2 JUMP segments, 197.9px distance

### 4. only-jump
- **Purpose**: Wall jumping mechanics in vertical corridors
- **Expected**: JUMP segments for vertical ascent
- **Result**: ✅ 2 JUMP segments, 96.0px distance

## Key Features

### Actual Test Map Loading
- Uses proper map loading chain: `NPlayHeadless` → `Simulator` → `MapLoader`
- Loads actual binary test map files from `nclone/test_maps/` directory
- Follows the same pattern as `base_environment.py` for consistency

### Correct Entity Positioning
- Applies static offset of `-TILE_PIXEL_SIZE` to account for simulation padding
- Entities now appear correctly positioned on their respective tiles
- Ninja (white circle), Switch (yellow dot), and Door (red dot) align properly with tile grid

### Physics-Aware Pathfinding
- All movements respect N++ physics constants from `physics_constants.py`
- Jump trajectories validated for feasibility within ninja capabilities
- Movement types correctly classified based on level geometry

## System Architecture

The consolidated pathfinding system consists of:

- **CorePathfinder** (`nclone/pathfinding/core_pathfinder.py`) - Main pathfinding algorithm using Dijkstra with physics-aware edge weights
- **MovementType** (`nclone/pathfinding/movement_type.py`) - Physics-based movement classification (WALK, JUMP, FALL, etc.)
- **PhysicsValidator** (`nclone/pathfinding/physics_validator.py`) - Validates all movements against N++ physics constraints
- **PathfindingVisualizer** (`nclone/visualization/pathfinding_visualizer.py`) - Generates visualizations using actual test map files

## Running Tests

```bash
# Run the comprehensive test suite
cd pathfinding_tests
python test_actual_map_files.py
```

## Validation Results

All test maps pass validation with correct physics-aware behavior:
- ✅ **simple-walk**: WALK movements on flat platform
- ✅ **long-walk**: WALK movements across full map width  
- ✅ **path-jump-required**: JUMP movements to reach elevated switch
- ✅ **only-jump**: JUMP movements for vertical corridor navigation

The system successfully demonstrates proper understanding of N++ physics constraints and generates realistic pathfinding solutions using actual game map data.