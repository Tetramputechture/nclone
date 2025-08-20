# Graph Traversability Tests

This directory contains tests for the graph traversability improvements made to nclone. These tests help prevent regressions and ensure the improvements continue to work correctly.

## Test Files

### `test_graph_traversability.py`
Tests the core graph builder improvements:
- Sub-grid resolution for better spatial accuracy
- Raycasting collision detection with ninja radius
- Improved half-tile traversability detection
- Enhanced one-way platform blocking
- Diagonal traversability for slopes
- Graph edge generation and types

### `test_debug_overlay_renderer.py`
Tests the debug overlay renderer coordinate fixes:
- Sub-grid coordinate conversion for rendering
- Node indexing consistency
- Edge coordinate consistency
- Sub-grid constants validation

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

The tests use realistic N++ level dimensions (23×42) and standard game parameters to ensure they reflect real usage patterns.

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
