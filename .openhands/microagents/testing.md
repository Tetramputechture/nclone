---
agent: 'CodeActAgent'
---

# Testing Guidelines for nclone

### Test Development Guidelines

#### When to Write Tests
1. **Always** when fixing bugs - create tests that reproduce the issue
2. **Always** when adding new graph features
3. **Always** when changing collision detection logic

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
Tests should always go in the `tests/` directory.
- `test_issue{N}_{description}` - For specific issue fixes
- `test_{component}_{functionality}` - For component testing
- `test_{integration_scenario}` - For integration testing

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

#### Performance Monitoring
```python
import time
start_time = time.time()
# Test code here
execution_time = time.time() - start_time
self.assertLess(execution_time, 1.0)  # Max 1 second
```

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