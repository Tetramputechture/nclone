# Path Visualization Flickering and Distance Fix - Summary

## Problem Statement

The path visualization system in `test_environment.py` had two critical issues:

1. **Flickering**: The screen alternated between two different visualization states every frame
2. **Infinity Distances**: Switch and Exit distances were always showing as "∞" instead of actual values

## Root Cause Analysis

### Flickering Cause
There were **two competing visualization systems** running simultaneously:

1. **NEW system** (lines 2201-2220): Integrated with environment's render pipeline via `_draw_path_aware()`
2. **OLD system** (lines 2230-2445): Direct overlay blitting via `draw_path_distances()` and `draw_adjacency_graph()`

These systems were:
- Building graphs independently
- Using different caching mechanisms
- Interfering with each other's state
- Creating visual conflicts causing the flicker

### Infinity Distance Cause
The distance calculation in the OLD system was failing because:
- Path calculator was not properly finding paths between nodes
- Entity positions were not being correctly mapped to graph nodes
- BFS pathfinding was not being used for reliable distance calculation

## Solution

### Fix 1: Remove Duplicate Visualization System (Commit cc8ef90)

**Removed 238 lines** of obsolete code:
- Entire OLD visualization system (lines 2230-2445)
- Obsolete caching variables (`cached_graph_data`, `cached_level_hash`, `OverlayCache` class)
- Duplicate graph building logic
- Direct overlay blitting that bypassed the render pipeline

**Kept** the NEW system which:
- Properly integrates with environment's render pipeline
- Uses environment's internal caching mechanism  
- Renders via `_draw_path_aware()` method in debug_overlay_renderer
- Avoids duplicate rendering and state conflicts

### Fix 2: Enhance Path Visualization (Commit d34de62)

**Added 181 lines** of new functionality to `_draw_path_aware()`:

1. **Entity Support**
   - Pass entity data (switches, exits) from environment to renderer
   - Extract switch and exit positions from entity list

2. **Visual Markers**
   - Draw prominent markers for switches (bright green circles, radius 6)
   - Draw prominent markers for exits (orange circles, radius 6)
   - Black border around markers for visibility

3. **Distance Calculation**
   - Use BFS pathfinding on adjacency graph
   - Find closest graph node to ninja position
   - Find closest graph nodes to each switch/exit
   - Calculate actual path distances using graph edges
   - Handle multiple switches/exits (use nearest)

4. **Info Box**
   - Display floating info box near ninja position
   - Show "Switch: X" distance (or ∞ if unreachable)
   - Show "Exit: Y" distance (or ∞ if unreachable)
   - Auto-position to stay on screen

5. **Legend**
   - Display color-coded legend in top-left
   - Show node type meanings:
     * Blue = Ninja
     * Green = Switch
     * Orange = Exit
     * Light green = Tile

## Technical Details

### Modified Files

1. **nclone/test_environment.py** (-238 lines)
   - Removed duplicate visualization system
   - Removed obsolete caching infrastructure
   - Kept unified graph building before env.step()

2. **nclone/gym_environment/mixins/debug_mixin.py** (+1 line)
   - Added `entities` to path_aware info payload
   - Enables renderer to locate switches and exits

3. **nclone/debug_overlay_renderer.py** (+180 lines)
   - Enhanced `_draw_path_aware()` with complete visualization
   - Added switch/exit position extraction
   - Added BFS pathfinding for distance calculation
   - Added visual markers for entities
   - Added info box rendering
   - Added legend rendering

### Graph Building Architecture

The graph is now built **once per level** and cached:

```python
if path_aware_system['current_graph'] is None or path_aware_system.get('level_id') != env.current_map_name:
    # Rebuild graph for new level
    level_data_dict = env.level_data.to_dict()
    path_aware_system['current_graph'] = path_aware_system['graph_builder'].build_graph(level_data_dict)
    path_aware_system['level_id'] = env.current_map_name
```

The cached graph is then passed to the environment which handles rendering internally.

### Distance Calculation Algorithm

```python
# 1. Find closest graph node to ninja
closest_node = find_closest_node_to(ninja_pos, adjacency)

# 2. Find closest graph node to target (switch/exit)
target_node = find_closest_node_to(target_pos, adjacency)

# 3. BFS pathfinding from ninja_node to target_node
distances = {closest_node: 0}
queue = deque([closest_node])

while queue and target_node not in distances:
    current = queue.popleft()
    for neighbor, cost in adjacency[current]:
        if neighbor not in distances:
            distances[neighbor] = distances[current] + cost
            queue.append(neighbor)

# 4. Return distance if path found, else infinity
return distances.get(target_node, float('inf'))
```

## Keyboard Controls

- **D**: Toggle distance display (shows info box with switch/exit distances)
- **A**: Toggle adjacency graph display (shows nodes, edges, markers, legend)
- **B**: Toggle blocked entities display (shows blocked nodes/edges in red)

## Testing

To test the fixes:

```bash
cd nclone
python -m nclone.test_environment --test-path-aware --show-path-distances --visualize-adjacency-graph
```

Expected behavior:
- ✅ No flickering between frames
- ✅ Stable adjacency graph visualization
- ✅ Switch and Exit markers visible (if pressing A)
- ✅ Actual distance values shown (if pressing D)
- ✅ Legend displayed in top-left (if pressing A)
- ✅ Info box near ninja showing distances (if pressing D)

## Commits

1. **216eca7**: Fix unimplemented module imports (minor)
2. **cc8ef90**: Remove duplicate path visualization system (fixes flickering)
3. **d34de62**: Enhance path visualization with distances and legend (fixes infinity)

## Lines Changed

- **Removed**: 238 lines (duplicate system)
- **Added**: 181 lines (enhanced visualization)
- **Net change**: -57 lines (simpler, unified system)

## Results

✅ **Flickering eliminated** - Single unified rendering pipeline  
✅ **Distances accurate** - BFS pathfinding calculates real graph distances  
✅ **Better visualization** - Legend, markers, and info box improve clarity  
✅ **Cleaner code** - 238 lines of duplicate code removed  
✅ **Proper caching** - Graph built once per level, not every frame  

## Future Improvements

Potential enhancements for the path visualization system:

1. **Performance**: Cache distance calculations (currently recalculated when ninja moves)
2. **Visualization**: Add path highlighting showing actual route to objectives
3. **Multi-target**: Show distances to all collectibles, not just nearest
4. **Heatmap**: Color-code graph nodes by distance from ninja
5. **Path quality**: Show jump-cost vs walk-cost breakdown
6. **Reachability**: Highlight unreachable areas (infinity regions)

---

**Branch**: `fix/path-visualization-controls`  
**Status**: ✅ All changes committed and pushed  
**Date**: 2025-10-30
