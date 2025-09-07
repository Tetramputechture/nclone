# Graph System Tests

This directory contains tests for the graph system improvements made to nclone. These tests help prevent regressions and ensure the improvements continue to work correctly.

## Test Files

### Core Graph System Tests

#### `test_graph_traversability.py`
Tests the core graph builder improvements:
- Sub-grid resolution for better spatial accuracy
- Raycasting collision detection with ninja radius
- Improved half-tile traversability detection
- Enhanced one-way platform blocking
- Diagonal traversability for slopes
- Graph edge generation and types

#### `test_debug_overlay_renderer.py`
Tests the debug overlay renderer coordinate fixes:
- Sub-grid coordinate conversion for rendering
- Node indexing consistency
- Edge coordinate consistency
- Sub-grid constants validation

### Graph Visualization Fixes Tests

#### `test_graph_fixes_unit_tests.py` ⭐ **MAIN TEST SUITE**
Comprehensive unit tests for the three major graph visualization issues:
- **Issue #1**: Functional edges between switches and doors
- **Issue #2**: Invalid walkable edges in solid tiles  
- **Issue #3**: Ninja pathfinding from solid spawn tiles
- Graph structure integrity validation
- Corridor connections system validation

#### `test_doortest_fixes.py`
Integration tests for the doortest map fixes:
- Functional edge detection and creation
- Solid tile edge validation
- Pathfinding improvements

#### `test_corridor_connections.py`
Tests for the corridor connections system:
- Long-distance pathfinding improvements
- Empty tile cluster connectivity
- Graph fragmentation resolution

#### `test_local_pathfinding.py`
Tests for local pathfinding improvements:
- Ninja escape from solid tiles
- Nearby target reachability
- Path traversability validation

### Specialized Component Tests

#### `test_collision_detection_fix.py`
Tests for collision detection improvements

#### `test_precise_collision.py`
Tests for the precise tile collision system

#### `test_hazard_system.py`
Tests for the hazard classification system

#### `test_pathfinding_functionality.py`
Tests for pathfinding engine functionality

## Running Tests

### Run all tests:
```bash
cd nclone/tests
python run_tests.py
```

### Run specific test file:
```bash
cd nclone
python -m pytest tests/test_graph_traversability.py -v
```

### Run with coverage (if pytest-cov is installed):
```bash
cd nclone
python -m pytest tests/ --cov=nclone.graph --cov-report=html
```

## Requirements

The tests require:
- `pytest` for test execution
- `numpy` for level data creation
- The nclone package modules

Install test dependencies:
```bash
pip install pytest numpy
```

## Test Philosophy

These tests focus on:
1. **Regression Prevention**: Ensuring improvements don't break
2. **Integration Testing**: Testing components work together
3. **Behavior Validation**: Verifying expected behaviors
4. **Edge Case Coverage**: Testing boundary conditions

The tests use realistic N++ level dimensions (23*42) and standard game parameters to ensure they reflect real usage patterns.

## Adding New Tests

When adding new graph-related features:

1. Add test methods to the appropriate test class
2. Use descriptive test names that explain what's being tested
3. Include both positive and negative test cases
4. Test edge cases and boundary conditions
5. Update this README if adding new test files

## Debugging Test Failures

If tests fail:
1. Run with `-v` flag for verbose output
2. Check the specific assertion that failed
3. Verify that constants and dimensions are correct
4. Ensure test level data matches expected format
5. Use `--tb=long` for full tracebacks
