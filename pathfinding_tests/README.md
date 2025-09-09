# N++ Pathfinding Tests

This directory contains the test suite and validation scripts for the N++ physics-aware pathfinding system.

## Test Files

### Main Test Scripts

- **`consolidated_pathfinding_system.py`** - Primary demonstration and validation script
  - Tests all 4 validation maps (simple-walk, long-walk, path-jump-required, only-jump)
  - Creates visualizations with movement type legends
  - Validates physics accuracy and movement classification

- **`test_pathfinding_validation.py`** - Original validation test (reference implementation)
  - Uses the proven MovementClassifier system
  - Validates movement types against expected results

### Physics Validation

- **`simple_physics_validation.py`** - Basic physics validation with N++ constants
  - Validates jump trajectories and movement capabilities
  - Uses accurate physics constants from `physics_constants.py`

- **`comprehensive_physics_validation.py`** - Complete physics validation suite
  - Comprehensive testing of all movement types
  - Validates against ninja physics constraints

### Generated Visualizations

- **`simple-walk_consolidated.png`** - Simple horizontal walking test
- **`long-walk_consolidated.png`** - Extended horizontal walking test  
- **`path-jump-required_consolidated.png`** - Jump and fall navigation test
- **`only-jump_consolidated.png`** - Vertical wall jumping test

## Running Tests

### Quick Validation
```bash
cd pathfinding_tests
python consolidated_pathfinding_system.py
```

### Individual Tests
```bash
python test_pathfinding_validation.py
python simple_physics_validation.py
python comprehensive_physics_validation.py
```

## Expected Results

All test maps should pass validation with correct movement types:

1. **simple-walk**: 2 WALK segments (96px + 72px = 168px total)
2. **long-walk**: 2 WALK segments (936px + 24px = 960px total)  
3. **path-jump-required**: JUMP up + FALL down (99px + 76px = 175px total)
4. **only-jump**: 2 JUMP segments (48px each = 96px total)

## System Architecture

The tests use the consolidated pathfinding system located in:
- `nclone/pathfinding/` - Core pathfinding logic
- `nclone/visualization/` - Visualization system
- `nclone/graph/` - Graph construction and movement classification

This ensures all tests use the same authoritative pathfinding implementation.