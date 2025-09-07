---
agent: 'CodeActAgent'
---

# Debugging Guide for nclone

## Debug Directory Structure

The `debug/` directory contains 45+ specialized debugging and analysis scripts organized by functionality.

### Primary Debug Tools

#### `debug/final_validation.py` ⭐ **MAIN VALIDATION TOOL**
**Purpose**: Comprehensive validation of all three major graph issues
**Usage**: `python debug/final_validation.py`
**Expected Output**:
```
🎯 ISSUE #1 STATUS: ✅ RESOLVED (2 functional edges)
🎯 ISSUE #2 STATUS: ✅ RESOLVED (0 invalid solid edges)  
🎯 ISSUE #3 STATUS: ✅ RESOLVED (353 connected nodes)
📊 OVERALL RESULT: 3/3 issues resolved
```

#### `debug/analyze_graph_fragmentation.py`
**Purpose**: Analyzes graph connectivity and identifies fragmentation issues
**Usage**: `python debug/analyze_graph_fragmentation.py`
**Key Metrics**: Connected components, reachability analysis, cluster identification

#### `debug/analyze_map_layout.py`
**Purpose**: Map structure analysis for understanding level geometry
**Usage**: `python debug/analyze_map_layout.py`
**Output**: Empty tile clusters, solid tile patterns, entity positions

### Debug Script Categories

#### Graph Connectivity Analysis
- `debug_connected_components.py`: Component analysis
- `debug_graph_connectivity.py`: Basic connectivity metrics
- `debug_graph_connectivity_detailed.py`: Detailed connectivity analysis
- `debug_isolated_nodes.py`: Identifies disconnected nodes

#### Pathfinding Debugging
- `debug_pathfinding.py`: General pathfinding issue diagnosis
- `debug_ninja_connections.py`: Ninja-specific connectivity analysis
- `debug_path_blocking.py`: Identifies path obstructions
- `debug_target_components.py`: Target reachability analysis

#### Edge System Debugging
- `debug_edge_building.py`: Edge creation process analysis
- `debug_functional_edges.py`: Switch-door connection debugging
- `debug_walkable_edges.py`: Walkable edge validation
- `debug_escape_logic.py`: Ninja escape edge analysis

#### Collision Detection Debugging
- `debug_collision_detection.py`: Comprehensive collision analysis
- `debug_collision_simple.py`: Basic collision testing
- `debug_traversability.py`: Traversability validation

#### Entity and Position Debugging
- `debug_entity_positions.py`: Entity location analysis
- `debug_ninja_spawn.py`: Ninja spawn position debugging
- `debug_node_positions.py`: Graph node positioning
- `debug_coordinate_system.py`: Coordinate system validation

### Common Debugging Workflows

#### Issue #1: Missing Functional Edges
```bash
# Check functional edge creation
python debug/debug_functional_edges.py

# Validate entity positions
python debug/debug_entity_positions.py

# Analyze edge building process
python debug/debug_edge_building.py
```

#### Issue #2: Invalid Solid Tile Edges
```bash
# Check walkable edges in solid tiles
python debug/debug_walkable_edges.py

# Validate collision detection
python debug/debug_collision_detection.py

# Analyze traversability
python debug/debug_traversability.py
```

#### Issue #3: Ninja Pathfinding Problems
```bash
# Analyze ninja connectivity
python debug/debug_ninja_connections.py

# Check escape logic
python debug/debug_escape_logic.py

# Validate pathfinding
python debug/debug_pathfinding.py
```

### Debug Output Interpretation

#### Graph Metrics to Monitor
```
Expected Healthy Values:
- Total nodes: 15,000-16,000
- Total edges: 3,500-4,000
- Ninja connected component: 300+ nodes
- Functional edges: 2 (for doortest map)
- Invalid solid edges: 0
```

#### Pathfinding Success Rates
```
Expected Performance:
- Local pathfinding: 100% success
- Long-distance pathfinding: 40-50% success
- Ninja escape from solid tile: 100% success
```

#### Edge Type Distribution
```
Typical Edge Counts:
- WALK edges: ~2,800-3,200
- JUMP edges: ~400-600
- FALL edges: ~200-400
- FUNCTIONAL edges: 2-10 (map dependent)
```

### Advanced Debugging Techniques

#### Graph Visualization Debugging
```bash
# Enable graph overlay in test environment
python -m nclone.test_environment

# Use keyboard shortcuts:
# - Toggle edge types
# - Zoom in/out
# - Filter by edge color
```

#### Custom Debug Scripts
```python
# Template for custom debugging
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'nclone'))

from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
from nclone.graph.hierarchical_builder import HierarchicalGraphBuilder

def debug_specific_issue():
    env = BasicLevelNoGold(render_mode="rgb_array")
    builder = HierarchicalGraphBuilder()
    graph = builder.build_graph(env.level_data, env.entities)
    
    # Add specific debugging logic here
    print(f"Graph has {len(graph.nodes)} nodes and {len(graph.edges)} edges")

if __name__ == "__main__":
    debug_specific_issue()
```

### Performance Debugging

#### Timing Analysis
```python
import time

def time_graph_building():
    start = time.time()
    # Graph building code
    end = time.time()
    print(f"Graph building took {end - start:.3f} seconds")
```

#### Memory Usage Monitoring
```python
import tracemalloc

tracemalloc.start()
# Code to analyze
current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage: {current / 1024 / 1024:.1f} MB")
print(f"Peak memory usage: {peak / 1024 / 1024:.1f} MB")
```

### Debug Script Development Guidelines

#### Creating New Debug Scripts
1. **Follow naming convention**: `debug_{component}_{specific_issue}.py`
2. **Include docstring**: Explain purpose and expected output
3. **Add to debug/README.md**: Document the new script
4. **Use consistent imports**: Follow existing pattern

#### Debug Script Template
```python
#!/usr/bin/env python3
"""
Debug script for [specific issue description].

Usage: python debug/debug_[component]_[issue].py
Expected output: [description of expected output]
"""

import os
import sys

# Add the nclone package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'nclone'))

from nclone.nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold

def main():
    """Main debugging function."""
    print("=" * 60)
    print("🔍 DEBUGGING [COMPONENT] - [ISSUE]")
    print("=" * 60)
    
    # Debugging logic here
    
    print("✅ Debug analysis complete")

if __name__ == "__main__":
    main()
```

### Integration with Testing

#### Debug-Driven Development
1. **Reproduce issue** with debug script
2. **Create test** that fails with current code
3. **Fix implementation** to pass test
4. **Verify fix** with debug script
5. **Run full validation** with `final_validation.py`

#### Debug Script Validation
```bash
# Ensure debug scripts work correctly
python debug/final_validation.py  # Should show 3/3 resolved
python debug/analyze_graph_fragmentation.py  # Should show healthy metrics
```

### Troubleshooting Debug Scripts

#### Common Issues
1. **Import errors**: Check sys.path configuration
2. **Environment setup**: Ensure headless mode works
3. **File paths**: Use absolute paths for reliability
4. **Memory issues**: Monitor memory usage for large graphs

#### Debug Script Maintenance
- **Regular testing**: Ensure scripts work with current codebase
- **Documentation updates**: Keep README.md current
- **Performance monitoring**: Watch for script execution time increases
- **Output validation**: Verify expected output formats