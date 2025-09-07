---
agent: 'CodeActAgent'
---

# Testing Guidelines for nclone

## Test Structure Overview

The nclone repository has a comprehensive testing system organized into specific categories and purposes.

### Primary Test Suite

#### `tests/test_graph_fixes_unit_tests.py` ⭐ **MAIN TEST SUITE**
This is the **most important test file** - it validates the three major graph visualization fixes:

**Critical Tests:**
1. `test_issue1_functional_edges_exist` - Validates switch-door connections
2. `test_issue2_no_invalid_solid_tile_edges` - Ensures no invalid edges in solid tiles
3. `test_issue3_ninja_pathfinding_local` - Tests ninja pathfinding from solid spawn
4. `test_corridor_connections_exist` - Validates long-distance connectivity improvements

**Expected Results:**
- **8/8 tests must pass**
- All tests should complete in under 2 seconds
- No exceptions or errors during execution

```bash
# Run the main test suite
python tests/test_graph_fixes_unit_tests.py

# Expected output:
# Ran 8 tests in X.XXXs
# OK
```

### Test Categories

#### Graph System Tests
- `test_graph_traversability.py`: Core graph builder functionality
- `test_debug_overlay_renderer.py`: Graph visualization system
- `test_corridor_connections.py`: Long-distance pathfinding
- `test_collision_detection_fix.py`: Collision system validation

#### Integration Tests
- `test_doortest_fixes.py`: Complete doortest map validation
- `test_local_pathfinding.py`: Local pathfinding scenarios
- `test_enhanced_traversability_integration.py`: System integration

#### Component Tests
- `test_precise_collision.py`: Collision detection accuracy
- `test_hazard_system.py`: Hazard classification
- `test_pathfinding_functionality.py`: Pathfinding algorithms

### Running Tests

#### Quick Test Commands
```bash
# Main test suite (most important)
python tests/test_graph_fixes_unit_tests.py

# All tests
python tests/run_tests.py

# Specific test file
python tests/test_corridor_connections.py

# With pytest (if available)
python -m pytest tests/test_graph_fixes_unit_tests.py -v
```

#### Test Environment Setup
Tests automatically handle:
- Package path configuration
- Environment initialization
- Headless mode setup (no display required)
- Temporary file cleanup

### Test Development Guidelines

#### When to Write Tests
1. **Always** when fixing bugs - create tests that reproduce the issue
2. **Always** when adding new graph features
3. **Always** when modifying pathfinding algorithms
4. **Always** when changing collision detection logic

#### Test Structure Pattern
```python
def test_specific_functionality(self):
    """Test description explaining what is being validated."""
    # Setup
    env = BasicLevelNoGold(render_mode="rgb_array")
    builder = HierarchicalGraphBuilder()
    
    # Execute
    result = builder.build_graph(env.level_data, env.entities)
    
    # Validate
    self.assertGreater(result.node_count, expected_minimum)
    self.assertEqual(result.edge_types[EdgeType.FUNCTIONAL], expected_count)
```

#### Test Naming Conventions
- `test_issue{N}_{description}` - For specific issue fixes
- `test_{component}_{functionality}` - For component testing
- `test_{integration_scenario}` - For integration testing

### Validation Scripts

#### Primary Validation Tool
```bash
# Comprehensive system validation
python debug/final_validation.py

# Expected output:
# 🎯 ISSUE #1 STATUS: ✅ RESOLVED
# 🎯 ISSUE #2 STATUS: ✅ RESOLVED  
# 🎯 ISSUE #3 STATUS: ✅ RESOLVED
# 📊 OVERALL RESULT: 3/3 issues resolved
```

#### Specialized Validation
```bash
# Graph connectivity analysis
python debug/analyze_graph_fragmentation.py

# Map layout analysis
python debug/analyze_map_layout.py

# Pathfinding validation
python debug/debug_pathfinding.py
```

### Test Data and Fixtures

#### Standard Test Environment
- **Map**: doortest (42x23 tiles)
- **Entities**: 11 entities including switches and doors
- **Ninja Position**: (132, 444) - spawns in solid tile
- **Expected Graph Size**: ~15,470 nodes, ~3,790 edges

#### Test Assertions Patterns
```python
# Graph structure validation
self.assertGreater(graph.node_count, 15000)
self.assertLess(graph.node_count, 16000)

# Edge type validation
functional_edges = [e for e in graph.edges if e.type == EdgeType.FUNCTIONAL]
self.assertEqual(len(functional_edges), 2)

# Pathfinding validation
path = pathfinder.find_path(start, target)
self.assertIsNotNone(path)
self.assertGreater(len(path), 1)
```

### Debugging Test Failures

#### Common Failure Patterns
1. **Import Errors**: Check sys.path configuration in test files
2. **Graph Size Mismatches**: Validate level data loading
3. **Pathfinding Failures**: Check ninja spawn position and connectivity
4. **Edge Count Mismatches**: Verify edge building logic

#### Debugging Commands
```bash
# Run test with verbose output
python tests/test_graph_fixes_unit_tests.py -v

# Debug specific test failure
python -c "
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nclone'))
# Add debugging code here
"
```

### Performance Testing

#### Expected Performance Benchmarks
- **Graph Building**: < 1 second for standard maps
- **Pathfinding**: < 100ms for typical paths
- **Test Suite**: < 5 seconds total execution time

#### Performance Monitoring
```python
import time
start_time = time.time()
# Test code here
execution_time = time.time() - start_time
self.assertLess(execution_time, 1.0)  # Max 1 second
```

### Continuous Integration Considerations

#### Pre-commit Checks
```bash
# Run before committing
make lint                                    # Code quality
python tests/test_graph_fixes_unit_tests.py  # Core functionality
python debug/final_validation.py            # System validation
```

#### Test Coverage Goals
- **Graph System**: 90%+ coverage
- **Pathfinding**: 85%+ coverage  
- **Collision Detection**: 95%+ coverage
- **Edge Building**: 90%+ coverage

### Test Maintenance

#### Regular Test Updates
1. Update test data when map formats change
2. Adjust expected values when algorithms improve
3. Add regression tests for new bug fixes
4. Remove obsolete tests for deprecated features

#### Test Documentation
- Keep test docstrings current
- Document expected behavior changes
- Maintain test data documentation
- Update performance benchmarks