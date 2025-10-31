# Shortest Path Visualization Feature

## Overview

Added the ability to visualize the shortest paths from the ninja to goal entities (switches and exits) in the test environment. This feature helps verify pathfinding accuracy and debug navigation issues.

## Usage

**Keyboard Control**: Press **`P`** to toggle path visualization on/off

When enabled, the system will:
- Calculate the shortest path from ninja's current position to the nearest switch
- Calculate the shortest path from ninja's current position to the nearest exit
- Draw these paths as thick colored lines on the adjacency graph overlay

## Visual Indicators

### Path Colors
- **Bright Green Line** (width: 3px): Path to nearest switch
- **Bright Orange Line** (width: 3px): Path to nearest exit

### How Paths are Drawn
- Paths are rendered as connected line segments between graph nodes
- Each segment connects two adjacent nodes in the shortest path
- Paths update in real-time as the ninja moves

## Technical Implementation

### Algorithm: BFS with Parent Tracking

The implementation uses Breadth-First Search (BFS) to find the shortest path:

```python
def find_shortest_path(start_node, end_node, adjacency):
    """Find shortest path from start to end node using BFS."""
    distances = {start_node: 0}
    parents = {start_node: None}  # Track parent nodes for reconstruction
    queue = deque([start_node])
    
    while queue:
        current = queue.popleft()
        
        if current == end_node:
            # Reconstruct path by backtracking through parents
            path = []
            node = end_node
            while node is not None:
                path.append(node)
                node = parents[node]
            path.reverse()
            return path, distances[end_node]
        
        # Explore neighbors
        for neighbor, cost in adjacency[current]:
            if neighbor not in distances:
                distances[neighbor] = distances[current] + cost
                parents[neighbor] = current
                queue.append(neighbor)
    
    return None, float('inf')  # No path found
```

### Key Features

1. **Node Mapping**: Finds closest graph nodes to entity positions
   - Entities have exact pixel positions
   - Graph nodes are on a grid (e.g., 24px resolution)
   - System finds nearest graph node within 50 pixels

2. **Path Reconstruction**: Uses parent pointers to rebuild path
   - BFS stores parent node for each visited node
   - After reaching goal, backtrack from goal to start
   - Reverse the path to get start-to-goal order

3. **Nearest Goal Selection**: Only shows path to closest goal
   - If multiple switches exist, shows path to nearest one
   - If multiple exits exist, shows path to nearest one
   - Prevents visual clutter from overlapping paths

4. **Real-time Updates**: Recalculates paths every frame
   - Paths update as ninja moves through level
   - Automatically adjusts when ninja gets closer to different goals

## Integration with Existing Controls

The path visualization integrates seamlessly with other debug visualization controls:

| Key | Function | Effect |
|-----|----------|--------|
| **A** | Toggle adjacency graph | Show/hide graph nodes and edges |
| **D** | Toggle distance display | Show/hide switch/exit distance info box |
| **B** | Toggle blocked entities | Show/hide blocked nodes/edges in red |
| **P** | Toggle path visualization | Show/hide shortest paths to goals |

### Recommended Usage Combinations

- **A + P**: Show graph structure with paths overlaid
- **A + D + P**: Complete visualization (graph, distances, paths)
- **P only**: Just show the paths without graph clutter

## Performance Considerations

### Computational Cost
- BFS runs twice per frame (once for switch, once for exit)
- Cost: O(V + E) where V = nodes, E = edges
- Typical graph: ~1000-5000 nodes, ~5000-20000 edges
- Expected performance: < 2ms per frame on modern hardware

### Optimization Opportunities
If performance becomes an issue, consider:

1. **Path Caching**: Cache paths and only recalculate when ninja moves significantly
2. **Spatial Hashing**: Use spatial data structures to speed up "closest node" searches
3. **A* Heuristic**: Use A* instead of BFS for faster pathfinding (though BFS guarantees shortest path in unweighted graphs)
4. **Lazy Updates**: Only recalculate paths every N frames instead of every frame

## Code Structure

### Modified Files

1. **test_environment.py** (+11 lines)
   - Added `show_paths_to_goals` state variable
   - Added 'P' key handler to toggle path display
   - Updated help text to document new control
   - Pass flag to environment via `set_show_paths_to_goals()`

2. **debug_mixin.py** (+7 lines)
   - Added `_show_paths_to_goals` instance variable
   - Added `set_show_paths_to_goals()` setter method
   - Include `show_paths` flag in path_aware info dictionary
   - Update condition to enable path_aware visualization

3. **debug_overlay_renderer.py** (+108 lines)
   - Added `find_shortest_path()` helper function
   - Added path calculation and rendering logic in `_draw_path_aware()`
   - Define path colors (SWITCH_PATH_COLOR, EXIT_PATH_COLOR)
   - Extract `show_paths` flag from path_aware info
   - Find closest nodes to ninja and entities
   - Calculate paths using BFS
   - Render paths as connected line segments

## Testing

To test the shortest path visualization:

```bash
cd nclone
python -m nclone.test_environment --test-path-aware --visualize-adjacency-graph
```

Then:
1. Press **A** to show the adjacency graph
2. Press **P** to show shortest paths to goals
3. Use arrow keys to move ninja around
4. Observe paths updating in real-time

Expected behavior:
- ✅ Bright green path visible from ninja to nearest switch
- ✅ Bright orange path visible from ninja to nearest exit  
- ✅ Paths update smoothly as ninja moves
- ✅ Paths take efficient routes through graph
- ✅ No flickering or visual artifacts

## Debugging Path Issues

If paths look incorrect:

1. **Check Node Connectivity**: Press **A** to see if graph edges are correct
2. **Check Entity Positions**: Verify switches/exits are at expected locations
3. **Check Distance Threshold**: Ensure entity is within 50 pixels of a graph node
4. **Check Graph Building**: Verify graph is built correctly for current level

### Common Issues

**Problem**: Path shows infinity (∞) or doesn't appear
- **Cause**: No path exists between ninja and goal
- **Solution**: Check if switch/exit is reachable in the adjacency graph

**Problem**: Path takes unexpected route
- **Cause**: Graph edge costs don't match expected traversal difficulty
- **Solution**: Review edge building logic in graph builder

**Problem**: Path flickers or jumps around
- **Cause**: Multiple nodes equally close to ninja (ties)
- **Solution**: Consider adding deterministic tie-breaking

## Future Enhancements

Potential improvements to path visualization:

1. **Multi-Goal Paths**: Show paths to all switches/exits, not just nearest
2. **Path Cost Display**: Show total path cost/distance on path
3. **Alternative Paths**: Highlight 2nd-best path with different color
4. **Waypoint Markers**: Draw circles at path waypoints for clarity
5. **Animated Path**: Animate path traversal to show direction
6. **Path Comparison**: Compare current path vs optimal historical path
7. **Jump vs Walk Segments**: Color-code path segments by movement type
8. **Path History**: Show ghost trail of recently-taken paths

## Related Documentation

- `FLICKERING_FIX_SUMMARY.md`: Documentation of earlier visualization fixes
- `docs/sim_mechanics_doc.md`: Core simulation mechanics
- `graph/hierarchical_builder.py`: Graph building system
- `debug_overlay_renderer.py`: Visualization rendering logic

---

**Branch**: `fix/path-visualization-controls`  
**Commit**: `09af1ba`  
**Date**: 2025-10-30  
**Status**: ✅ Implemented and pushed
