# Task 002: Create Test Maps for Reachability Analysis

## Overview
Create a comprehensive set of test maps specifically designed to validate reachability analysis functionality. These maps will be used for testing the physics-aware reachability system and ensuring it correctly handles all N++ mechanics.

## Context Reference
See [npp-rl comprehensive technical roadmap](../../../npp-rl/docs/comprehensive_technical_roadmap.md) Section 11.2: "Test Data Requirements" and Section 1.2: "Physics-Aware Reachability Analysis Strategy"

## Requirements

### Primary Objectives
1. **Create specialized test maps** for different reachability scenarios
2. **Design maps that test all 33 tile types** and their traversability rules
3. **Include dynamic entity scenarios** (mines, drones, thwumps)
4. **Create switch-door dependency puzzles** for strategic planning validation
5. **Generate performance testing maps** for large-scale reachability analysis

### Map Categories Required

#### Basic Physics Validation Maps
These maps test fundamental physics constraints and tile traversability:

1. **`simple_jump_level.nmap`**
   - **Purpose**: Test basic jump reachability calculations
   - **Layout**: Ground level with platforms at various jump distances
   - **Key Features**:
     - Platforms at max jump distance (reachable)
     - Platforms beyond max jump distance (unreachable)
     - Wall jump opportunities
     - Different tile types (solid, half-tiles, slopes)
   - **Expected Reachability**: ~60% of map reachable from start

2. **`tile_type_comprehensive.nmap`**
   - **Purpose**: Test all 33 tile types for correct traversability
   - **Layout**: Grid layout with each tile type in isolated sections
   - **Key Features**:
     - Empty tiles (type 0) - fully traversable
     - Solid tiles (types 1, 34-37) - blocking
     - Half tiles (types 2-5) - directional traversability
     - Slopes (types 6-9, 18-33) - physics-based traversal
     - Curves (types 10-17) - segment-based collision
   - **Expected Reachability**: Specific patterns based on tile definitions

3. **`obstacle_course.nmap`**
   - **Purpose**: Test trajectory validation through complex geometry
   - **Layout**: Winding path with various obstacles
   - **Key Features**:
     - Narrow passages requiring precise movement
     - Jump sequences with intermediate platforms
     - Wall slide sections
     - Bounce block interactions
   - **Expected Reachability**: Single optimal path through course

#### Dynamic Entity Testing Maps

4. **`drone_patrol.nmap`**
   - **Purpose**: Test dynamic hazard blocking and timing
   - **Layout**: Corridors with patrolling drones
   - **Key Features**:
     - Drones with predictable patrol patterns
     - Safe zones between patrol routes
     - Timing-dependent passages
     - Multiple drone types (regular, mini)
   - **Expected Reachability**: Time-dependent reachable areas

5. **`mine_field.nmap`**
   - **Purpose**: Test static hazard avoidance
   - **Layout**: Area with toggle mines in various states
   - **Key Features**:
     - Active mines (blocking)
     - Inactive mines (passable)
     - Toggling mine sequences
     - Safe paths through mine fields
   - **Expected Reachability**: Clear paths avoiding active mines

6. **`thwump_gauntlet.nmap`**
   - **Purpose**: Test activation-triggered hazards
   - **Layout**: Corridor with thwumps and shove thwumps
   - **Key Features**:
     - Thwumps that activate on approach
     - Shove thwumps with launch trajectories
     - Safe activation distances
     - Timing windows for passage
   - **Expected Reachability**: Careful navigation required

#### Switch-Door Dependency Maps

7. **`locked_door_puzzle.nmap`**
   - **Purpose**: Test basic switch-door relationships
   - **Layout**: Simple level with locked doors and switches
   - **Key Features**:
     - Single door controlled by single switch
     - Switch accessible from start
     - Area behind door only reachable after activation
   - **Expected Reachability**: Expands after switch activation

8. **`complex_multi_switch_level.nmap`**
   - **Purpose**: Test complex switch dependencies and strategic planning
   - **Layout**: Multi-room level with interdependent switches
   - **Key Features**:
     - Multiple doors with different controlling switches
     - Switches that unlock access to other switches
     - Dead-end switches (don't help with level completion)
     - Optimal switch activation sequence required
   - **Expected Reachability**: Complex dependency tree

9. **`switch_puzzle_maze.nmap`**
   - **Purpose**: Test strategic planning for level completion heuristic
   - **Layout**: Maze-like level with exit switch and door
   - **Key Features**:
     - Exit switch behind locked doors
     - Multiple possible switch activation orders
     - Some switches unlock shortcuts
     - Gold collection opportunities
   - **Expected Reachability**: Multiple valid completion strategies

#### Exploration and Curiosity Testing Maps

10. **`maze_with_unreachable_areas.nmap`**
    - **Purpose**: Test curiosity system filtering of unreachable areas
    - **Layout**: Maze with isolated sections
    - **Key Features**:
      - Completely isolated areas (unreachable)
      - Areas reachable only after switch activation (frontier)
      - Clearly reachable exploration areas
      - Dead ends vs. progress paths
    - **Expected Reachability**: Clear distinction between reachable/unreachable

11. **`exploration_test.nmap`**
    - **Purpose**: Test curiosity bonus calculation
    - **Layout**: Open area with various exploration targets
    - **Key Features**:
      - High-value reachable targets (gold, switches)
      - Low-value reachable areas (empty space)
      - Unreachable tempting areas (gold behind walls)
      - Frontier areas (behind doors)
    - **Expected Reachability**: Graduated reachability levels

#### Performance Testing Maps

12. **`large_level.nmap`**
    - **Purpose**: Test reachability analysis performance on full-size levels
    - **Layout**: Full 42x23 tile level (maximum N++ size)
    - **Key Features**:
      - Complex geometry throughout
      - Multiple dynamic entities
      - Several switch-door pairs
      - Realistic level complexity
    - **Expected Reachability**: <100ms analysis time

13. **`memory_test_level.nmap`**
    - **Purpose**: Test memory efficiency during repeated analysis
    - **Layout**: Level designed to stress caching systems
    - **Key Features**:
      - Many similar but distinct areas
      - Frequent switch state changes
      - Complex entity interactions
      - High cache turnover potential
    - **Expected Reachability**: Stable memory usage

#### Edge Case Testing Maps

14. **`physics_challenge.nmap`**
    - **Purpose**: Test edge cases in physics-based reachability
    - **Layout**: Level with extreme physics scenarios
    - **Key Features**:
      - Maximum distance jumps
      - Complex wall jump sequences
      - Launch pad combinations
      - Bounce block chains
    - **Expected Reachability**: Precise physics validation required

15. **`boundary_conditions.nmap`**
    - **Purpose**: Test map boundary and edge conditions
    - **Layout**: Level that uses map edges extensively
    - **Key Features**:
      - Ninja starting at map edges
      - Targets at map boundaries
      - Wrapping or non-wrapping edge behavior
      - Out-of-bounds detection
    - **Expected Reachability**: Proper boundary handling

## Acceptance Criteria

### Functional Requirements
1. **Map Loading**: All maps load correctly using existing `map_loader.py`
2. **Binary Format**: Maps are saved in proper N++ binary format
3. **Entity Placement**: Dynamic entities are correctly positioned and configured
4. **Tile Validation**: All tile types are used correctly according to definitions

### Technical Requirements
1. **File Format**: Maps use `.nmap` extension and binary format
2. **Size Constraints**: Maps fit within N++ level size limits (42x23 tiles max)
3. **Entity Limits**: Respect maximum entity counts per level
4. **Performance**: Large maps load within reasonable time (<1 second)

### Quality Requirements
1. **Documentation**: Each map has clear purpose and expected behavior
2. **Naming Convention**: Descriptive filenames matching their test purpose
3. **Validation Data**: Expected reachability results documented
4. **Visual Clarity**: Maps are visually understandable for debugging

## Test Scenarios

### Map Creation Validation
```bash
# Test map loading
python -c "
from nclone.map_loader import load_map
for map_file in ['simple_jump_level.nmap', 'tile_type_comprehensive.nmap']:
    level_data = load_map(f'test_maps/{map_file}')
    print(f'{map_file}: {level_data.width}x{level_data.height} tiles, {len(level_data.entities)} entities')
"

# Validate tile types
python -c "
import numpy as np
from nclone.map_loader import load_map
level_data = load_map('test_maps/tile_type_comprehensive.nmap')
unique_tiles = np.unique(level_data.tiles)
print(f'Tile types used: {sorted(unique_tiles)}')
assert len(unique_tiles) >= 20, 'Should use many different tile types'
"
```

### Reachability Testing
```bash
# Test reachability analysis on new maps
python -c "
from nclone.graph.reachability_analyzer import ReachabilityAnalyzer
from nclone.graph.trajectory_calculator import TrajectoryCalculator
from nclone.map_loader import load_map

analyzer = ReachabilityAnalyzer(TrajectoryCalculator())

# Test each map
test_maps = [
    'simple_jump_level.nmap',
    'locked_door_puzzle.nmap', 
    'drone_patrol.nmap'
]

for map_name in test_maps:
    level_data = load_map(f'test_maps/{map_name}')
    ninja_start = (50, 400)  # Standard starting position
    
    result = analyzer.analyze_reachability(level_data, ninja_start, {})
    reachable_count = len(result.reachable_positions)
    print(f'{map_name}: {reachable_count} reachable positions')
"
```

### Performance Validation
```bash
# Test performance on large maps
python -c "
import time
from nclone.graph.reachability_analyzer import ReachabilityAnalyzer
from nclone.graph.trajectory_calculator import TrajectoryCalculator
from nclone.map_loader import load_map

analyzer = ReachabilityAnalyzer(TrajectoryCalculator())
level_data = load_map('test_maps/large_level.nmap')

start_time = time.time()
result = analyzer.analyze_reachability(level_data, (50, 550), {})
analysis_time = time.time() - start_time

print(f'Large level analysis time: {analysis_time:.3f}s')
assert analysis_time < 0.1, 'Should complete within 100ms'
"
```

## Implementation Steps

### Phase 1: Map Design and Creation
1. **Set up Map Editor Environment**
   - Ensure N++ level editor is available
   - Document map creation workflow
   - Establish naming conventions

2. **Create Basic Physics Maps**
   ```bash
   # Create maps in order of complexity
   # 1. simple_jump_level.nmap
   # 2. tile_type_comprehensive.nmap  
   # 3. obstacle_course.nmap
   ```

3. **Create Dynamic Entity Maps**
   ```bash
   # Add entity-focused maps
   # 4. drone_patrol.nmap
   # 5. mine_field.nmap
   # 6. thwump_gauntlet.nmap
   ```

### Phase 2: Switch-Door and Strategic Maps
1. **Create Puzzle Maps**
   ```bash
   # Add strategic planning maps
   # 7. locked_door_puzzle.nmap
   # 8. complex_multi_switch_level.nmap
   # 9. switch_puzzle_maze.nmap
   ```

2. **Create Exploration Maps**
   ```bash
   # Add curiosity testing maps
   # 10. maze_with_unreachable_areas.nmap
   # 11. exploration_test.nmap
   ```

### Phase 3: Performance and Edge Case Maps
1. **Create Performance Maps**
   ```bash
   # Add performance testing maps
   # 12. large_level.nmap
   # 13. memory_test_level.nmap
   ```

2. **Create Edge Case Maps**
   ```bash
   # Add edge case maps
   # 14. physics_challenge.nmap
   # 15. boundary_conditions.nmap
   ```

### Phase 4: Documentation and Validation
1. **Document Each Map**
   - Create map documentation file
   - Include expected reachability results
   - Document test scenarios for each map

2. **Validate All Maps**
   - Test loading with map_loader.py
   - Run reachability analysis on each
   - Verify expected behaviors

## Map Documentation Format

For each map, create documentation in `test_maps/README.md`:

```markdown
## simple_jump_level.nmap
- **Purpose**: Basic jump reachability validation
- **Size**: 20x15 tiles
- **Entities**: 0 dynamic entities
- **Key Features**: Platforms at various jump distances
- **Expected Reachability**: 
  - Start position: (50, 400)
  - Reachable positions: ~180 sub-grid cells
  - Unreachable areas: Platforms beyond max jump distance
- **Test Scenarios**:
  - Verify max jump distance constraint
  - Test wall jump detection
  - Validate slope traversal
```

## Success Metrics
- **Map Count**: 15 specialized test maps created
- **Coverage**: All major reachability scenarios covered
- **Performance**: Large maps analyze in <100ms
- **Validation**: All maps load and analyze correctly
- **Documentation**: Complete documentation for each map

## Dependencies
- N++ level editor for map creation
- Existing `map_loader.py` functionality
- Reachability analysis system (from other tasks)

## Estimated Effort
- **Time**: 3-5 days
- **Complexity**: Medium (requires level design skills)
- **Risk**: Low (creating test data)

## Notes
- Maps should be visually clear for debugging purposes
- Consider creating both simple and complex versions of each scenario
- Maps will be used across multiple testing tasks
- Coordinate with npp-rl team for cross-repository testing needs