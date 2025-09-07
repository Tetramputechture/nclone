---
agent: 'CodeActAgent'
---

# Graph System Development Guide

## Core Graph System Architecture

The nclone graph system is the primary architecture for AI pathfinding and navigation. Understanding this system is crucial for effective development.

### Key Components

#### 1. Hierarchical Builder (`nclone/graph/hierarchical_builder.py`)
- **Primary entry point** for graph construction
- Manages multi-resolution graph building (6px, 24px, 96px)
- Integrates all graph subsystems

#### 2. Edge Building (`nclone/graph/edge_building.py`)
- **Core edge creation logic**
- Handles 4 edge types: WALK (green), JUMP (orange), FALL (blue), FUNCTIONAL (yellow)
- Contains corridor connections system for long-distance pathfinding
- **Recently enhanced** with switch-door functional edges and ninja escape logic

#### 3. Pathfinding (`nclone/graph/pathfinding.py`)
- A* algorithm implementation
- Graph-based navigation with multi-resolution support
- Handles pathfinding from solid spawn tiles

#### 4. Collision Detection (`nclone/graph/precise_collision.py`)
- **Critical component** for traversability validation
- 10-pixel ninja radius awareness
- Precise tile-based collision checking

### Development Workflow for Graph Changes

#### 1. Making Changes
```bash
# Edit core graph files
vim nclone/graph/edge_building.py
vim nclone/graph/hierarchical_builder.py
```

#### 2. Testing Changes
```bash
# Run main test suite (8 comprehensive tests)
python tests/test_graph_fixes_unit_tests.py

# Run comprehensive validation
python debug/final_validation.py

# Test specific scenarios
python debug/analyze_graph_fragmentation.py
```

#### 3. Debugging Issues
```bash
# Graph connectivity analysis
python debug/debug_graph_connectivity.py

# Pathfinding debugging
python debug/debug_pathfinding.py

# Edge building analysis
python debug/debug_edge_building.py
```

### Critical Graph System Knowledge

#### Edge Types and Colors
- **WALK (green)**: Standard walkable connections between adjacent nodes
- **JUMP (orange)**: Jump connections for vertical movement
- **FALL (blue)**: Falling connections for gravity-based movement  
- **FUNCTIONAL (yellow)**: Switch-door connections and special interactions

#### Graph Resolution Levels
- **6px**: Fine-grained detail for precise navigation
- **24px**: Medium resolution for balanced performance/accuracy
- **96px**: Coarse resolution for high-level planning

#### Known Issues and Solutions
1. **Functional Edges**: Ensure switch-door connections are created in `build_entity_edges()`
2. **Solid Tile Edges**: Validate traversability before creating walkable edges
3. **Ninja Pathfinding**: Use corridor connections for long-distance navigation

### Testing Requirements

#### Always Run These Tests After Graph Changes
```bash
# Main test suite - MUST PASS
python tests/test_graph_fixes_unit_tests.py

# Validation script - MUST SHOW 3/3 RESOLVED
python debug/final_validation.py
```

#### Expected Test Results
- **8/8 tests passing** in main test suite
- **3/3 issues resolved** in final validation
- **No invalid solid tile edges**
- **Functional edges present** between switches and doors
- **Ninja connectivity improvement** (300+ nodes reachable)

### Performance Considerations

#### Graph Size Expectations
- **Nodes**: ~15,000-16,000 for typical levels
- **Edges**: ~3,500-4,000 including corridor connections
- **Ninja Connectivity**: 300+ nodes in connected component

#### Optimization Guidelines
- Use sub-grid resolution efficiently
- Cache collision detection results
- Minimize redundant edge creation
- Optimize pathfinding with proper heuristics

### Common Development Patterns

#### Adding New Edge Types
1. Define edge type in `nclone/graph/common.py`
2. Add creation logic in `edge_building.py`
3. Update visualization colors in debug overlay
4. Add tests for new edge type

#### Modifying Collision Detection
1. Update `precise_collision.py` methods
2. Test with various tile configurations
3. Validate ninja radius handling
4. Ensure traversability accuracy

#### Enhancing Pathfinding
1. Modify algorithms in `pathfinding.py`
2. Test with fragmented maps
3. Validate long-distance navigation
4. Check performance impact

### Debugging Tools

#### Essential Debug Scripts
- `debug/final_validation.py`: **Primary validation tool**
- `debug/analyze_graph_fragmentation.py`: Connectivity analysis
- `debug/debug_pathfinding.py`: Pathfinding issue diagnosis
- `debug/debug_edge_building.py`: Edge creation debugging

#### Graph Visualization
- Enable graph overlay in test environment
- Use different edge type filters to isolate issues
- Check node connectivity patterns
- Validate edge colors and types

### Integration Points

#### With RL Environment
- Graph provides navigation data for RL agents
- Pathfinding results guide agent actions
- Graph visualization helps debug agent behavior

#### With Game Simulation
- Graph nodes correspond to game world positions
- Edges represent valid movement transitions
- Collision detection ensures game rule compliance